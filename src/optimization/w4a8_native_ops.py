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
    Return ComfyUI native ``comfy.ops.mixed_precision_ops`` for W4A8 DiT loads.

    Empty ``quant_config``: layers with ``comfy_quant`` become QuantizedTensor
    managed natively by ComfyUI's AsymW4A8Int8Layout; unmarked layers load as plain
    compute_dtype Parameters.
    """
    import comfy.ops as comfy_ops

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

    ops.Linear = _mark_dit_locked(Linear)
    return ops


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
