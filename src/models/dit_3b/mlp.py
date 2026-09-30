# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


def get_mlp(mlp_type: Optional[str] = "normal"):
    if mlp_type == "normal":
        return MLP
    elif mlp_type == "swiglu":
        return SwiGLUMLP


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        self.proj_in = ops.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = ops.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    # Token count threshold for activating chunked forward.
    # Below this, the single-expression path is faster with negligible VRAM impact.
    # Above this, chunking prevents ~2-3 GB activation spikes per block.
    CHUNK_THRESHOLD = 8192

    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = ops.Linear(dim, hidden_dim, bias=False)
        self.proj_out = ops.Linear(hidden_dim, dim, bias=False)
        self.proj_in = ops.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = x.shape[0]

        if seq_len <= self.CHUNK_THRESHOLD:
            # Fast path: single expression, no chunking overhead
            return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))

        # Chunked path: split tokens to cap peak VRAM from gate/up/hidden
        # Each chunk holds at most CHUNK_THRESHOLD tokens worth of (L_chunk, hidden_dim)
        # tensors, reducing peak from 3 * L * hidden_dim to 3 * chunk_size * hidden_dim
        num_chunks = (seq_len + self.CHUNK_THRESHOLD - 1) // self.CHUNK_THRESHOLD
        output = torch.empty(seq_len, x.shape[-1], dtype=x.dtype, device=x.device)

        for i in range(num_chunks):
            start = i * self.CHUNK_THRESHOLD
            end = min(start + self.CHUNK_THRESHOLD, seq_len)
            x_chunk = x[start:end]
            # gate, up, hidden are allocated only for this chunk then freed
            output[start:end] = self.proj_out(
                F.silu(self.proj_in_gate(x_chunk)) * self.proj_in(x_chunk)
            )

        return output
