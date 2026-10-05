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

from functools import lru_cache
from typing import Tuple
import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn

from ...common.cache import Cache


class RotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="pixel",
            max_freq=256,
        )
        # 1. Set model.requires_grad_(True) after model creation will make
        #    the `requires_grad=False` for rope freqs no longer hold.
        # 2. Even if we don't set requires_grad_(True) explicitly,
        #    FSDP is not memory efficient when handling fsdp_wrap
        #    with mixed requires_grad=True/False.
        # With above consideration, it is easier just remove the freqs
        # out of nn.Parameters when `learned_freq=False`
        freqs = self.rope.freqs
        del self.rope.freqs
        self.rope.register_buffer("freqs", freqs.data)

    @lru_cache(maxsize=128)
    def get_axial_freqs(self, *dims):
        return self.rope.get_axial_freqs(*dims)


class RotaryEmbedding3d(RotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)

    def forward(
        self,
        q: torch.FloatTensor,  # b h l d
        k: torch.FloatTensor,  # b h l d
        size: Tuple[int, int, int],
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        T, H, W = size
        freqs = self.get_axial_freqs(T, H, W)
        q = rearrange(q, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        k = rearrange(k, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        q = rearrange(q, "b h T H W d -> b h (T H W) d")
        k = rearrange(k, "b h T H W d -> b h (T H W) d")
        return q, k


def _rope_apply_inplace(t, freqs):
    """In-place interleaved rotary for (h, L, d); equivalent to
    apply_rotary_emb(freqs, t) when rot_dim == d, without its transient buffers."""
    rot = freqs.shape[-1]
    d = t.shape[-1]
    if rot != d or d % 2 != 0:
        return apply_rotary_emb(freqs, t)
    cos = freqs.cos().to(t.dtype).unsqueeze(0)
    sin = freqs.sin().to(t.dtype).unsqueeze(0)
    if not t.is_contiguous():
        t = t.contiguous()
    t2 = t.view(*t.shape[:-1], d // 2, 2)
    t_even = t2[..., 0].clone()
    t_odd = t2[..., 1].clone()
    t2[..., 0].copy_(t_even * cos[..., 0::2] - t_odd * sin[..., 0::2])
    t2[..., 1].copy_(t_odd * cos[..., 1::2] + t_even * sin[..., 1::2])
    return t


class NaRotaryEmbedding3d(RotaryEmbedding3d):
    def forward(
        self,
        q: torch.FloatTensor,  # L h d
        k: torch.FloatTensor,  # L h d
        shape: torch.LongTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        freqs = cache("rope_freqs_3d", lambda: self.get_freqs(shape))
        freqs = freqs.to(device=q.device, dtype=q.dtype)
        q = rearrange(q, "L h d -> h L d")
        k = rearrange(k, "L h d -> h L d")
        # __W4A8_MEM__: apply rotary in-place instead of apply_rotary_emb's
        # (t*cos, rotate_half, t*sin, sum, cat) chain, which transiently holds
        # ~5x a full q/k buffer. rot_dim == d here (no left/right remainder), so
        # the cat only concatenates empty slices. Same interleaved-pair math.
        q = _rope_apply_inplace(q, freqs)
        k = _rope_apply_inplace(k, freqs)
        q = rearrange(q, "h L d -> L h d")
        k = rearrange(k, "h L d -> L h d")
        return q, k

    @torch._dynamo.disable  # Disable compilation: shape.tolist() is data-dependent and causes graph breaks
    def get_freqs(
        self,
        shape: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Generate RoPE frequencies for video and text with adaptive dimensions.
    
        Note: This method uses @torch._dynamo.disable because it requires
        data-dependent control flow (.tolist()) that cannot be symbolically
        traced by torch.compile. The cache() wrapper in the forward pass memoizes
        results to reduce recomputation overhead.
        """
        freq_list = []
        for f, h, w in shape.tolist():
            freqs = self.get_axial_freqs(f, h, w)
            freq_list.append(freqs.view(-1, freqs.size(-1)))
        return torch.cat(freq_list, dim=0)
