"""
SeedVR2 W4A8 (asym_w4a8_int8) native inference via ComfyUI comfy.ops construction-time injection.

W4A8 safetensors carry ``comfy_quant`` (format ``asym_w4a8_int8``) plus
``weight_s_rel`` (FP8 per-group scale) and ``weight_s_channel`` (FP32 per-channel scale).
Native VRAM-saving load requires Linear modules that already implement
``_load_from_state_dict`` → ``comfy.ops._load_quantized_module`` at
``load_state_dict`` time. That is provided directly by ``comfy.ops.mixed_precision_ops``
using ComfyUI's native AsymW4A8Int8Layout (QUANT_ALGOS["asym_w4a8_int8"]).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import torch


def checkpoint_is_w4a8(checkpoint_path: Optional[str]) -> bool:
    """True if safetensors has at least one ``*.comfy_quant`` with format asym_w4a8_int8."""
    if not checkpoint_path:
        return False
    path = str(checkpoint_path)
    if not (path.endswith(".safetensors") or path.endswith(".sft")):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".comfy_quant"):
                    continue
                raw = f.get_tensor(key)
                if raw.dtype != torch.uint8:
                    continue
                conf = json.loads(raw.numpy().tobytes())
                fmt = conf.get("format", "")
                if fmt in ("asym_w4a8_int8", "w4a8"):
                    return True
    except Exception:
        return False
    return False


def _mark_dit_locked(o):
    """Wrap an ops.Linear so quantized weights get flagged as locked.

    The flag stops comfy_kitchen.QuantizedTensor from silently fp16-expanding
    the weight on any unimplemented op (see base.py). Only INT8/NVFP4/W4A8 packed
    weights (int8/uint8 storage) opt in; VAE and other models never do.
    """
    import torch as _t
    class _LockedLinear(o):
        def forward(self, input, *args, **kwargs):
            w = getattr(self, 'weight', None)
            try:
                d = getattr(w, '_qdata', None)
                if d is not None and getattr(d, 'dtype', None) in (_t.int8, _t.uint8):
                    w._dit_quant_locked = True
            except Exception:
                pass
            return super().forward(input, *args, **kwargs)
    return _LockedLinear


def get_w4a8_mixed_precision_ops(compute_dtype: torch.dtype = torch.float16) -> Any:
    """
    W4A8 ops: core-native LOADING + self convrot FORWARD.

    ComfyUI core supports the asym_w4a8_int8 *pack* natively (its
    _load_quantized_module builds the AsymW4A8Int8Layout QuantizedTensor), but its
    forward dispatch cannot do the ConvRot rotation. So:
      - loading goes through core's mixed_precision_ops -> DisTorch2 CPU offload
        works exactly like INT8 (weight stays on the donor device, streamed per
        forward via comfy_cast_weights).
      - compute is our own _forward (self convrot kernel), installed below.
    """
    import comfy.ops as comfy_ops

    # W4A8-only memory-efficient convrot forward (GEMM side only).
    install_efficient_w4a8_forward()

    ops = comfy_ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=False,
        disabled=[],
    )

    _BaseLinear = ops.Linear
    _act_dtype = compute_dtype if compute_dtype in (torch.float16, torch.bfloat16) else torch.float16

    class Linear(_BaseLinear):
        def forward(self, input, *args, **kwargs):
            if (
                isinstance(input, torch.Tensor)
                and getattr(self, "quant_format", None) in ("asym_w4a8_int8", "w4a8")
                and not getattr(self, "_full_precision_mm", False)
                and input.dtype not in (torch.float16, torch.bfloat16)
            ):
                input = input.to(dtype=_act_dtype)
            return super().forward(input, *args, **kwargs)

        def _forward(self, input, weight, bias):
            # __W4A8_SELF_FORWARD__: compute with OUR OWN convrot kernel. Core has
            # already streamed the (offloaded) weight onto the compute device via
            # forward_comfy_cast_weights, so DisTorch2 offload is untouched here.
            p = getattr(weight, "_params", None)
            if (
                type(weight).__name__ == "QuantizedTensor"
                and getattr(weight, "_layout_cls", None) == _LAYOUT
                and p is not None
            ):
                return _efficient_w4a8_linear(
                    input,
                    weight._qdata,
                    p.scale,
                    getattr(p, "s_channel", None),
                    codebook=getattr(p, "codebook", None),
                    correction=getattr(p, "correction", None),
                    bias=bias,
                    group_size=int(getattr(p, "group_size", 16)),
                    convrot_groupsize=int(getattr(p, "convrot_groupsize", 256)),
                    out_dtype=getattr(p, "orig_dtype", input.dtype),
                )
            return torch.nn.functional.linear(input, weight, bias)

    ops.Linear = _mark_dit_locked(Linear)
    return ops

# ============================================================================
# W4A8 memory-efficient forward (__W4A8_EFFICIENT_FORWARD__)
# ----------------------------------------------------------------------------
# The stock comfy_kitchen cuda ``w4a8_int8_linear`` (convrot path) unconditionally
# preallocates ``xq = torch.empty(m, k, int8)`` and then, on the fast_act branch,
# REASSIGNS xq/xs from ``quantize_int8_rowwise_convrot64`` - the preallocated
# m*k int8 buffer (1.06 GiB at m=92664,k=12288; 0.28 GiB at k=3072) is never used.
# INT8 (``int8_linear``) does not do this, so W4A8 wastes ~1 GiB per large linear
# during SeedVR2 Phase 2 and can overflow VRAM. Measured bit-exact vs the stock
# chunked path (max abs diff 0.0).
#
# This module (W4A8-only) installs a dedicated forward that skips the unused
# prealloc and otherwise runs the identical fused-convrot-quant + chunked-gemm
# path. distorch2 / INT8 / NVFP4 code paths are untouched.
_W4A8_EFF_INSTALLED = False


def _efficient_w4a8_linear(x, qdata, s_rel, s_channel, codebook=None,
                           correction=None, bias=None, group_size=16,
                           convrot_groupsize=256, out_dtype=None):
    """W4A8 convrot GEMM without the unused m*k int8 preallocation.

    Delegates to the stock kernel for shapes it does not handle (small m / non
    fast_act), so behavior matches upstream exactly where it matters.
    """
    import comfy_kitchen.backends.cuda as cb
    from comfy_kitchen.backends.cuda import (
        _wrap_for_dlpack, DTYPE_TO_CODE, _int4_int8_weight_chunk_cols,
        _convrot_fused_shared_memory_fits, quantize_int8_rowwise_convrot64,
        _gemm_vector_arg,
    )
    orig = getattr(cb, "_stock_w4a8_int8_linear", cb.w4a8_int8_linear)

    if out_dtype is None:
        out_dtype = x.dtype
    n = int(qdata.shape[0])
    k = int(qdata.shape[1]) * 2
    if x.shape[-1] != k:
        return orig(x, qdata, s_rel, s_channel, codebook, correction, bias,
                    group_size, convrot_groupsize, out_dtype)

    x_2d = x.reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    fast_act = (
        m >= 512
        and correction is None
        and s_rel.dtype == torch.float8_e4m3fn
        and convrot_groupsize == 256
        and k % 256 == 0
        and getattr(cb, "_W4A8_CHUNKED", True)
        and 256 <= k <= cb._CONVROT_FUSED_MAX_K
        and _convrot_fused_shared_memory_fits(x_2d, k, convrot_groupsize)
    )
    if not fast_act:
        return orig(x, qdata, s_rel, s_channel, codebook, correction, bias,
                    group_size, convrot_groupsize, out_dtype)

    out = torch.empty(m, n, dtype=out_dtype, device=x.device)
    output_dtype_code = DTYPE_TO_CODE[out_dtype]
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    bias_arg = _gemm_vector_arg(bias, x.device, out_dtype) if bias is not None else None
    s_rel_u8 = s_rel.view(torch.uint8)  # wrap per kernel call (dlpack capsule is single-use)

    # __W4A8_MEM__: cap the activation-side int8 buffer. The stock kernel needs a
    # full m*k int8 activation (xq); at m=92664,k=12288 that is ~1.08 GiB. Row-wise
    # (and convrot64) quantization is independent per row, so processing the
    # activation in row chunks is bit-exact and shrinks xq to chunk_m*k.
    # Chunk only the ACTIVATION/GEMM side; weight tensors are untouched.
    _CHUNK_ROWS = 2048
    if m <= _CHUNK_ROWS:
        chunk_cols = _int4_int8_weight_chunk_cols(m, n)
        workspace = torch.empty(min(chunk_cols, n), k, dtype=torch.int8, device=x.device)
        xq, xs = quantize_int8_rowwise_convrot64(x_2d, convrot_groupsize)
        used = cb._C.w4a8_codebook_gemm_chunked(
            _wrap_for_dlpack(xq), _wrap_for_dlpack(qdata), _wrap_for_dlpack(s_rel_u8),
            _wrap_for_dlpack(codebook) if codebook is not None else None,
            _wrap_for_dlpack(s_channel), _wrap_for_dlpack(xs.reshape(m)),
            _wrap_for_dlpack(bias_arg) if bias_arg is not None else None,
            _wrap_for_dlpack(workspace), _wrap_for_dlpack(out),
            group_size, chunk_cols, output_dtype_code, stream_ptr,
        )
        if not used:
            return orig(x, qdata, s_rel, s_channel, codebook, correction, bias,
                        group_size, convrot_groupsize, out_dtype)
    else:
        used_any = False
        # Allocate the int8 workspace ONCE (sized for the chunk) and reuse it,
        # instead of re-allocating per chunk (removes per-chunk peak growth).
        _wc = _int4_int8_weight_chunk_cols(_CHUNK_ROWS, n)
        workspace = torch.empty(min(_wc, n), k, dtype=torch.int8, device=x.device)
        for start in range(0, m, _CHUNK_ROWS):
            end = min(start + _CHUNK_ROWS, m)
            x_chunk = x_2d[start:end]
            mc = end - start
            chunk_cols = _int4_int8_weight_chunk_cols(mc, n)
            xq, xs = quantize_int8_rowwise_convrot64(x_chunk, convrot_groupsize)
            out_chunk = out[start:end]
            used = cb._C.w4a8_codebook_gemm_chunked(
                _wrap_for_dlpack(xq), _wrap_for_dlpack(qdata), _wrap_for_dlpack(s_rel_u8),
                _wrap_for_dlpack(codebook) if codebook is not None else None,
                _wrap_for_dlpack(s_channel), _wrap_for_dlpack(xs.reshape(mc)),
                _wrap_for_dlpack(bias_arg) if bias_arg is not None else None,
                _wrap_for_dlpack(workspace), _wrap_for_dlpack(out_chunk),
                group_size, chunk_cols, output_dtype_code, stream_ptr,
            )
            if not used:
                return orig(x, qdata, s_rel, s_channel, codebook, correction, bias,
                            group_size, convrot_groupsize, out_dtype)
            used_any = True
            if correction is not None:
                groups = k // group_size
                sx = xq.view(mc, groups, group_size).sum(-1, dtype=torch.int32).to(out_dtype)
                sx = sx * xs.to(out_dtype)
                out_chunk.addmm_(sx, correction.to(out_dtype))
        if not used_any:
            return orig(x, qdata, s_rel, s_channel, codebook, correction, bias,
                        group_size, convrot_groupsize, out_dtype)
        return out.reshape(*x.shape[:-1], n)

    if correction is not None:
        groups = k // group_size
        sx = xq.view(m, groups, group_size).sum(-1, dtype=torch.int32).to(out_dtype)
        sx = sx * xs.to(out_dtype)
        out.addmm_(sx, correction.to(out_dtype))
    return out.reshape(*x.shape[:-1], n)


def install_efficient_w4a8_forward() -> bool:
    """Route the W4A8 layout's linear op through the efficient forward. Idempotent."""
    global _W4A8_EFF_INSTALLED
    if _W4A8_EFF_INSTALLED:
        return True
    try:
        import comfy_kitchen.tensor.w4a8_int8 as _tw
        import comfy_kitchen.backends.cuda as _cb
        if not hasattr(_cb, "_stock_w4a8_int8_linear"):
            _cb._stock_w4a8_int8_linear = _cb.w4a8_int8_linear
        _stock = _tw.w4a8_int8_linear

        def _patched(*args, **kwargs):
            try:
                return _efficient_w4a8_linear(*args, **kwargs)
            except Exception:
                return _stock(*args, **kwargs)

        _patched._seedvr2_efficient = True
        _tw.w4a8_int8_linear = _patched
        _W4A8_EFF_INSTALLED = True
        return True
    except Exception:
        return False


def prepare_w4a8_state_dict_for_comfy_ops(state: dict) -> dict:
    """
    Prepare state_dict for W4A8 (asym_w4a8_int8) loading under ComfyUI comfy.ops._load_quantized_module.

    1. Moves ``*.comfy_quant`` tensors to CPU in-place so .numpy().tobytes() works.
    2. Ensures ``weight`` tensor is stored as torch.int8 storage.
    3. Ensures ``weight_s_rel`` is available (aliasing from ``weight_scale`` if present).
    4. Ensures ``weight_s_channel`` is available (1D [N] per-channel scale).
    """
    for key in list(state.keys()):
        if key.endswith(".comfy_quant"):
            val = state[key]
            if torch.is_tensor(val):
                val_cpu = val.cpu()
                try:
                    cq_dict = json.loads(val_cpu.numpy().tobytes())
                    modified = False
                    if cq_dict.get("format") in ("asym_w4a8_int8", "w4a8"):
                        if not cq_dict.get("convrot", False):
                            cq_dict["convrot"] = True
                            modified = True
                        if "convrot_groupsize" not in cq_dict:
                            cq_dict["convrot_groupsize"] = 256
                            modified = True
                        params = cq_dict.get("params")
                        if not isinstance(params, dict):
                            params = {}
                            cq_dict["params"] = params
                            modified = True
                        if not params.get("convrot", False):
                            params["convrot"] = True
                            modified = True
                        if "convrot_groupsize" not in params:
                            params["convrot_groupsize"] = 256
                            modified = True
                        if "group_size" not in params:
                            params["group_size"] = 16
                            modified = True
                        if modified:
                            new_bytes = json.dumps(cq_dict).encode("utf-8")
                            val_cpu = torch.from_numpy(np.frombuffer(new_bytes, dtype=np.uint8).copy())
                except Exception:
                    pass
                state[key] = val_cpu

            prefix = key[:-len(".comfy_quant")]
            w_key = f"{prefix}.weight"

            # Ensure weight is int8
            if w_key in state and torch.is_tensor(state[w_key]):
                w_t = state[w_key]
                if w_t.dtype == torch.uint8:
                    state[w_key] = w_t.view(torch.int8)

            # Ensure weight_s_rel is present
            s_rel_key = f"{prefix}.weight_s_rel"
            s_alt_key = f"{prefix}.weight_scale"
            if s_rel_key not in state and s_alt_key in state:
                state[s_rel_key] = state[s_alt_key]

            # Ensure weight_s_channel is present (1D shape (N,) as required by comfy_kitchen AsymW4A8Int8Layout)
            s_ch_key = f"{prefix}.weight_s_channel"
            if s_ch_key not in state and w_key in state:
                out_features = state[w_key].shape[0]
                state[s_ch_key] = torch.ones((out_features,), dtype=torch.float32)
            elif s_ch_key in state and torch.is_tensor(state[s_ch_key]) and state[s_ch_key].ndim == 2:
                # Squeeze (N, 1) to (N,) to match comfy_kitchen AsymW4A8Int8Layout expected shape (n,)
                state[s_ch_key] = state[s_ch_key].squeeze(-1)

    return state
