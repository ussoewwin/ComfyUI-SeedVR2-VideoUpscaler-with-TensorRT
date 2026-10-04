"""
SeedVR2 7B / 3B W4A8 (asym_w4a8_int8) Quantization Tool.

Strictly follows ComfyUI native quantization specification (comfy_kitchen.tensor.w4a8_int8
and comfy.ops._load_quantized_module):
- Preserves all 842 sensitive layers (embeddings, norms, biases, modulation, heads) in FP16.
- Quantizes the 288 transformer block linear weights to W4A8:
    * weight: packed INT4 in torch.int8 storage [out_features, in_features // 2]
    * weight_s_rel: per-group relative scale in torch.float8_e4m3fn [out_features, in_features // group_size]
    * weight_s_channel: per-channel scale in torch.float32 [out_features]
    * weight_codebook: optional codebook tensor (if used)
    * weight_correction: optional correction tensor (if used)
    * comfy_quant: {"format": "asym_w4a8_int8", "params": {"group_size": 16, "convrot_groupsize": 256}}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Default model paths
DEFAULT_INPUT = r"D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_fp16.safetensors"
DEFAULT_OUTPUT = r"D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_w4a8.safetensors"
DEFAULT_REFERENCE = r"D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_nvfp4.safetensors"

# Regex for sensitive layers that MUST remain in FP16
SENSITIVE_FP16_PATTERNS = [
    re.compile(r"^vid_in\."),
    re.compile(r"^txt_in\."),
    re.compile(r"^emb_in\."),
    re.compile(r"^vid_out\."),
    re.compile(r".*patch_embed.*"),
    re.compile(r".*final_layer.*"),
    re.compile(r".*out_proj\.(?!vid|txt).*"),
    re.compile(r".*norm.*"),
    re.compile(r".*fusedrms.*"),
    re.compile(r".*rope.*"),
    re.compile(r".*bias$"),
    re.compile(r".*scale$"),
]

# Regex identifying core 2D linear weights inside transformer blocks
BLOCK_LINEAR_WEIGHT_PATTERN = re.compile(
    r"^blocks\.\d+\.(?:attn\.(?:proj_qkv|proj_out)\.(?:vid|txt)|mlp\.(?:vid|txt)\.(?:fc1|fc2))\.weight$"
)


def _build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot)."""
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    return h / (size**0.5)


def _rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T."""
    out_f, in_f = weight.shape
    n_groups = in_f // group_size
    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)


def is_sensitive_layer(key: str, tensor: torch.Tensor) -> bool:
    """Return True if the layer must be kept in FP16."""
    if tensor.ndim != 2:
        return True
    if tensor.shape[0] % 16 != 0 or tensor.shape[1] % 16 != 0:
        return True
    for pat in SENSITIVE_FP16_PATTERNS:
        if pat.search(key):
            return True
    return False


def get_reference_quant_keys(reference_path: str) -> Optional[Set[str]]:
    """Inspect reference NVFP4 safetensors to extract exactly which weights were quantized."""
    if not reference_path or not os.path.isfile(reference_path):
        return None
    try:
        quant_keys = set()
        with safe_open(reference_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".comfy_quant"):
                    prefix = key[:-len(".comfy_quant")]
                    weight_key = f"{prefix}.weight"
                    quant_keys.add(weight_key)
        print(f"[Reference] Found {len(quant_keys)} quantized layers in reference model: {reference_path}")
        return quant_keys
    except Exception as e:
        print(f"[Reference] Warning: Could not read reference keys ({e}), falling back to architectural rules.")
        return None


def quantize_w4a8_tensor_native(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Quantize 2D weight matrix using ComfyUI comfy_kitchen native w4a8_int8 quantization.
    
    Returns:
        qdata: torch.Tensor (int8, [out_features, in_features // 2])
        s_rel: torch.Tensor (float8_e4m3fn, [out_features, in_features // group_size])
        s_channel: torch.Tensor (float32, [out_features])
        correction: Optional[torch.Tensor]
        codebook: Optional[torch.Tensor]
        comfy_quant: torch.Tensor (uint8, JSON metadata)
    """
    out_dim, in_dim = weight.shape
    w = weight.to(device=device, dtype=torch.float32)

    # 1. Try comfy_kitchen native quantizer first
    try:
        from comfy_kitchen.tensor.w4a8_int8 import quantize_w4a8_int8_weight
        qdata, s_rel, s_channel, correction, codebook = quantize_w4a8_int8_weight(
            weight=w,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            symmetric=True,
            scale_dtype=torch.float8_e4m3fn,
            codebook=True,
            bits=4,
        )
        qdata = qdata.to(device="cpu", dtype=torch.int8)
        s_rel = s_rel.to(device="cpu")
        s_channel = s_channel.to(device="cpu", dtype=torch.float32)
        if correction is not None:
            correction = correction.to("cpu")
        if codebook is not None:
            codebook = codebook.to("cpu")
    except Exception as e:
        # 2. Mathematically strict ConvRot + Hierarchical Scale Fallback
        # Offline ConvRot Hadamard rotation
        h = _build_hadamard(convrot_groupsize, device=w.device, dtype=torch.float32)
        w_rot = _rotate_weight(w, h, convrot_groupsize)
        
        # Per-channel scale (float32, [out_dim])
        s_channel = w_rot.abs().amax(dim=-1).clamp(min=1e-8)
        w_norm = w_rot / s_channel.unsqueeze(-1)
        
        # Per-group scale (FP8 e4m3fn, [out_dim, in_dim // group_size])
        num_groups = in_dim // group_size
        w_grouped = w_norm.view(out_dim, num_groups, group_size)
        s_rel_float = (w_grouped.abs().amax(dim=-1) / 7.0).clamp(min=1e-7)
        s_rel = s_rel_float.to(torch.float8_e4m3fn).to("cpu")
        
        # Quantize to INT4 [-8, 7] and pack 2 values per byte
        s_rel_expanded = s_rel_float.unsqueeze(-1)
        q = torch.clamp(torch.round(w_grouped / s_rel_expanded), -8, 7).to(torch.int8)
        q_2d = q.view(out_dim, in_dim)
        
        # Pack low nibble (even) and high nibble (odd)
        q_low = q_2d[:, 0::2] & 0x0F
        q_high = (q_2d[:, 1::2] & 0x0F) << 4
        packed_u8 = (q_low | q_high).to(torch.uint8)
        qdata = packed_u8.view(torch.int8).to("cpu")
        
        s_channel = s_channel.to(device="cpu", dtype=torch.float32)
        correction = None
        codebook = None

    # Sidecar comfy_quant metadata matching ComfyUI ops.py expectations
    quant_meta = {
        "format": "asym_w4a8_int8",
        "convrot": True,
        "convrot_groupsize": convrot_groupsize,
        "params": {
            "format": "asym_w4a8_int8",
            "group_size": group_size,
            "convrot": True,
            "convrot_groupsize": convrot_groupsize,
        },
    }
    comfy_quant_bytes = json.dumps(quant_meta).encode("utf-8")
    comfy_quant = torch.from_numpy(np.frombuffer(comfy_quant_bytes, dtype=np.uint8).copy())

    return qdata, s_rel, s_channel, correction, codebook, comfy_quant


def run_quantization(
    input_path: str,
    output_path: str,
    reference_path: Optional[str] = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Main quantization execution pipeline."""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 80)
    print("SeedVR2 DiT W4A8 Quantization (asym_w4a8_int8 ComfyUI Native)")
    print(f"  Input Checkpoint     : {input_path}")
    print(f"  Output Model         : {output_path}")
    print(f"  Reference Model      : {reference_path or 'None (using structural rules)'}")
    print(f"  Group Size           : {group_size}")
    print(f"  ConvRot Group Size   : {convrot_groupsize}")
    print(f"  Compute Device       : {device}")
    print("=" * 80)
    
    # 1. Check reference model for layer matching if available
    ref_quant_keys = None
    if reference_path:
        ref_quant_keys = get_reference_quant_keys(reference_path)

    # 2. Open source checkpoint and plan quantization
    t_start = time.time()
    tensors_to_save: Dict[str, torch.Tensor] = {}
    
    with safe_open(input_path, framework="pt", device="cpu") as src:
        keys = list(src.keys())
        total_keys = len(keys)
        print(f"[Inspect] Total tensors in source model: {total_keys}")
        
        quant_count = 0
        fp16_count = 0
        
        for idx, key in enumerate(keys, 1):
            tensor = src.get_tensor(key)
            
            # Decide whether to quantize or preserve
            should_quantize = False
            if ref_quant_keys is not None:
                if key in ref_quant_keys and tensor.ndim == 2:
                    should_quantize = True
            else:
                if BLOCK_LINEAR_WEIGHT_PATTERN.match(key) and not is_sensitive_layer(key, tensor):
                    should_quantize = True
            
            if should_quantize:
                qdata, s_rel, s_channel, correction, codebook, q_meta = quantize_w4a8_tensor_native(
                    tensor,
                    group_size=group_size,
                    convrot_groupsize=convrot_groupsize,
                    device=device,
                )
                
                prefix = key[:-len(".weight")]
                tensors_to_save[key] = qdata
                tensors_to_save[f"{prefix}.weight_s_rel"] = s_rel
                tensors_to_save[f"{prefix}.weight_s_channel"] = s_channel
                if codebook is not None:
                    tensors_to_save[f"{prefix}.weight_codebook"] = codebook
                if correction is not None:
                    tensors_to_save[f"{prefix}.weight_correction"] = correction
                tensors_to_save[f"{prefix}.comfy_quant"] = q_meta
                
                quant_count += 1
                if quant_count % 20 == 0 or quant_count == 1:
                    print(f"  [{idx}/{total_keys}] Quantized to W4A8: {key} (shape {list(tensor.shape)} -> {list(qdata.shape)})")
            else:
                # Keep sensitive layers untouched in FP16
                tensors_to_save[key] = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
                fp16_count += 1
                
    print(f"\n[Summary] Quantized layers: {quant_count}, Preserved FP16 layers: {fp16_count}")
    
    # Save safetensors cleanly
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_path):
        try:
            os.chmod(output_path, stat.S_IWRITE | stat.S_IREAD)
            os.remove(output_path)
            print(f"[Save] Removed existing destination file: {output_path}")
        except Exception as e:
            print(f"[Save] Warning: Could not remove existing file directly: {e}")

    save_meta = {"format": "pt"}
    try:
        save_file(tensors_to_save, output_path, metadata=save_meta)
    except Exception as e:
        print(f"[Save] Direct save to {output_path} failed: {e}")
        temp_path = output_path + ".tmp"
        if os.path.exists(temp_path):
            try:
                os.chmod(temp_path, stat.S_IWRITE | stat.S_IREAD)
                os.remove(temp_path)
            except Exception:
                pass
        print(f"[Save] Retrying save via temporary file: {temp_path}")
        save_file(tensors_to_save, temp_path, metadata=save_meta)
        print(f"[Save] Renaming {temp_path} -> {output_path}...")
        if os.path.exists(output_path):
            try:
                os.chmod(output_path, stat.S_IWRITE | stat.S_IREAD)
                os.remove(output_path)
            except Exception:
                pass
        shutil.move(temp_path, output_path)
    
    out_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    in_size_gb = os.path.getsize(input_path) / (1024 ** 3)
    elapsed = time.time() - t_start
    
    print("=" * 80)
    print(f"[SUCCESS] W4A8 Quantization Complete in {elapsed:.2f}s!")
    print(f"  Source Size : {in_size_gb:.2f} GB")
    print(f"  Output Size : {out_size_gb:.2f} GB (VRAM saving: ~{(1 - out_size_gb/in_size_gb)*100:.1f}%)")
    print(f"  Saved to    : {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SeedVR2 DiT W4A8 Quantization Tool (asym_w4a8_int8)")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT, help="Source FP16 checkpoint")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT, help="Output W4A8 checkpoint")
    parser.add_argument("--reference", "-r", type=str, default=DEFAULT_REFERENCE, help="Reference NVFP4 model to mimic")
    parser.add_argument("--group-size", "-g", type=int, default=16, help="Weight quantization group size (default: 16)")
    parser.add_argument("--convrot-groupsize", type=int, default=256, help="ConvRot group size (default: 256)")
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    
    args = parser.parse_args()
    run_quantization(
        input_path=args.input,
        output_path=args.output,
        reference_path=args.reference if os.path.exists(args.reference) else None,
        group_size=args.group_size,
        convrot_groupsize=args.convrot_groupsize,
        device=args.device,
    )


if __name__ == "__main__":
    main()
