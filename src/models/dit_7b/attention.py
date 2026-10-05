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

import torch
import torch.nn.functional as F

# Import flash/sage attn with automatic fallback from compatibility layer
from ...optimization.compatibility import (
    call_flash_attn_2_varlen, call_flash_attn_3_varlen,
    call_sage_attn_2_varlen, call_sage_attn_3_varlen,
    call_sparge_attn_varlen
)

from torch import nn


def pytorch_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False):
    """
    VRAM-optimized PyTorch variable-length attention.
    
    Pre-allocates a single output buffer and writes each window's result directly
    into the correct slice, eliminating the 2x VRAM spike from output_splits list
    + torch.cat that plagued the original implementation.
    
    Also caches cu_seqlens CPU conversion to avoid per-window CPU-GPU sync stalls.
    """
    total_len, num_heads, head_dim = q.shape
    
    # Pre-allocate output buffer — single allocation, no duplication
    output = torch.empty_like(q)
    
    # Convert cu_seqlens to CPU once (avoid repeated CPU-GPU sync per window)
    cu_q_cpu = cu_seqlens_q.long().cpu()
    cu_k_cpu = cu_seqlens_k.long().cpu()
    num_seqs = len(cu_q_cpu) - 1
    
    drop_p = dropout_p if not deterministic else 0.0
    
    for i in range(num_seqs):
        q_start, q_end = cu_q_cpu[i].item(), cu_q_cpu[i + 1].item()
        k_start, k_end = cu_k_cpu[i].item(), cu_k_cpu[i + 1].item()
        
        # Reshape: (seq, heads, dim) → (1, heads, seq, dim) for SDPA
        q_i = q[q_start:q_end].permute(1, 0, 2).unsqueeze(0)
        k_i = k[k_start:k_end].permute(1, 0, 2).unsqueeze(0)
        v_i = v[k_start:k_end].permute(1, 0, 2).unsqueeze(0)
        
        out_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i,
            dropout_p=drop_p,
            is_causal=causal,
        )
        
        # Write directly into pre-allocated buffer — no intermediate list
        output[q_start:q_end] = out_i.squeeze(0).permute(1, 0, 2)
    
    return output


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    """
    Variable-length attention with configurable backend.
    
    Supported backends:
    - sdpa: PyTorch SDPA (fully compilable, always available)
    - flash_attn_2: Flash Attention 2 (Ampere+)
    - flash_attn_3: Flash Attention 3 (Hopper+)
    - sageattn_2: SageAttention 2
    - sageattn_3: SageAttention 3 (Blackwell/RTX 50xx)
    - spargeattn: SpargeAttn-hswq (block-sparse on SageAttention2++ kernels)
    
    All non-SDPA backends use @torch._dynamo.disable wrapper (C++ extensions).
    """

    def __init__(self, attention_mode: str = 'sdpa', compute_dtype: torch.dtype = None, sparge_topk: float = 0.5):
        """
        Initialize with specified attention backend.
        
        Args:
            attention_mode: 'sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3', or 'spargeattn'
            compute_dtype: Compute dtype for attention (set by pipeline, defaults to None for auto-detection)
            sparge_topk: KV block keep ratio for the spargeattn backend (default 0.5)
        """
        super().__init__()
        self.attention_mode = attention_mode
        self.compute_dtype = compute_dtype
        self.sparge_topk = sparge_topk

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        
        # Convert to pipeline compute_dtype if configured (handles FP8 → fp16/bf16)
        if self.compute_dtype is not None and q.dtype != self.compute_dtype:
            q = q.to(self.compute_dtype)
            k = k.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        
        if self.attention_mode == 'flash_attn_3':
            return call_flash_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'flash_attn_2':
            return call_flash_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_3':
            return call_sage_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_2':
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'spargeattn':
            return call_sparge_attn_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                sparge_topk=self.sparge_topk, **kwargs
            )
        else:
            # PyTorch SDPA
            return pytorch_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )