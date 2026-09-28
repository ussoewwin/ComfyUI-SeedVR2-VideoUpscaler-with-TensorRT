"""Cloud GPU ONNX export worker for RTX 5090 (32GB VRAM, Blackwell sm_120).

Traces the full static ONNX (e.g. 85f x 512x512) directly on the GPU in seconds
to minutes, avoiding the slow CPU trace and the local 16GB VRAM limit.

OOM strategy (--device auto, default):
    1) GPU trace with no Conv3d chunking (original behaviour);
    2) on CUDA OOM: retry ON GPU with Conv3d chunking 4 GB, then 2 GB
       (lowers the peak while staying on the fast GPU path);
    3) only if all GPU attempts fail: CPU float16 trace (system RAM, CUDA
       hidden). The CPU path is extremely slow for large 512-tile traces
       (hours) and should be considered a last resort.

Use --gpu-conv-limit-gb to fix the GPU path's Conv3d chunk size directly, and
--device cuda / --device cpu to force a specific path.

The produced ONNX is GPU-independent. Build the engine with cloud_build_engine.py
on any Blackwell (sm_120) GPU, or locally on the RTX 5060 Ti.

Usage:
    python tools/cloud_export_gpu.py --repo <custom_node_root> \
        --kind encoder --frames 85 --output <onnx_path> [--model ema_vae_fp16.safetensors] \
        [--device auto|cuda|cpu] [--gpu-conv-limit-gb 0] [--cpu-conv-limit-gb 16]
"""

from __future__ import annotations

import argparse
import gc
import os
import inspect
import sys
import time
from pathlib import Path

# Promote timely VRAM release during the trace (frees cached blocks as they exceed 80%).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,expandable_segments:True")

import torch
import yaml


def _find_vae_file(model_name: str, model_dir: Path) -> Path:
    candidates = [
        model_dir / model_name,
        model_dir / "SEEDVR2" / model_name,
        Path("models") / "SEEDVR2" / model_name,
        Path("models") / "vae" / model_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate VAE file {model_name}")


class _EncoderModule(torch.nn.Module):
    """Standalone copy (avoids importing src.interfaces which needs comfy_api)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(video, memory_state=MemoryState.DISABLED)
        if self.quant_conv is not None:
            hidden = self.quant_conv(hidden, memory_state=MemoryState.DISABLED)
        return hidden


class _DecoderModule(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, memory_state=MemoryState.DISABLED)


def configure_fixed_vae(vae: torch.nn.Module) -> None:
    if hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(float("inf"))
            if hasattr(module, "set_memory_device"):
                module.set_memory_device(None)
        if hasattr(module, "slicing"):
            module.slicing = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="custom node root directory")
    parser.add_argument("--kind", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ema_vae_fp16.safetensors", help="VAE filename")
    parser.add_argument("--model-dir", default=None, help="VAE model directory (default: <repo>/models)")
    parser.add_argument("--tile", type=int, default=256, choices=[256, 512],
                        help="spatial tile size for the ONNX (256 = 1/4 memory; engine tile must match)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto = GPU first (with chunked GPU retries), fall back to CPU fp16 (system RAM) "
                             "on CUDA OOM; cuda = GPU only; cpu = CPU only (CUDA hidden)")
    parser.add_argument("--gpu-conv-limit-gb", type=float, default=0.0,
                        help="Conv3d chunk size (GB) for the GPU trace (0 = no chunking). "
                             "OOM retries try 4 GB then 2 GB when this is 0.")
    parser.add_argument("--cpu-conv-limit-gb", type=float, default=16.0,
                        help="Conv3d chunk size (GB) for the CPU trace path only (default 16, like "
                             "export_onnx_worker.py). 0 = no chunking.")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        print("ERROR: --device cuda requested but CUDA is not available", flush=True)
        return 2
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {props.total_memory / 2**30:.1f} GiB, arch: sm_{props.major}{props.minor}", flush=True)
        if args.device != "cpu" and not (props.major == 12 and props.minor == 0):
            print("WARNING: not sm_120 (Blackwell). Large traces may OOM on smaller VRAM.", flush=True)
    elif args.device != "cpu":
        print("NOTE: CUDA not available; using CPU fp16 (system RAM).", flush=True)

    use_gpu = (args.device != "cpu") and cuda_available
    load_dev = "cuda" if use_gpu else "cpu"

    sys.path.insert(0, args.repo)

    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL
    from src.models.video_vae_v3.modules.types import MemoryState
    from src.utils.debug import Debug
    from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
    from tools.onnx_export_utils import _portable_export, _cuda_disabled, _force_cpu_fp16
    from safetensors.torch import load_file
    # Expose to module-level helpers (configure_fixed_vae / _EncoderModule.forward)
    import sys as _sys
    _mod = _sys.modules[__name__]
    _mod.MemoryState = MemoryState
    _mod.InflatedCausalConv3d = InflatedCausalConv3d

    repo_path = Path(args.repo)
    vae_config_path = repo_path / "src" / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
    with open(vae_config_path, "r", encoding="utf-8") as f:
        vae_kwargs = yaml.safe_load(f)

    sig = inspect.signature(VideoAutoencoderKL.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    filtered_kwargs = {k: v for k, v in vae_kwargs.items() if k in valid_params}

    print("Instantiating VideoAutoencoderKL on meta device...", flush=True)
    with torch.device("meta"):
        vae = VideoAutoencoderKL(**filtered_kwargs)

    vae_dir = Path(args.model_dir) if args.model_dir else (repo_path / "models")
    vae_file = _find_vae_file(args.model, vae_dir)
    print(f"Loading VAE weights from {vae_file} to {load_dev.upper()}...", flush=True)
    state_dict = load_file(str(vae_file), device=load_dev)
    vae.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    vae = vae.to(device=load_dev, dtype=torch.float16).eval()
    configure_fixed_vae(vae)

    # Keep the reference graph (no conv/norm chunking) unless a chunk size is requested.
    # NOTE: chunked graphs differ from the Studio-compatible unchunked reference; engines
    # built from a chunked ONNX should be validated before production use.
    from src.models.video_vae_v3.modules.global_config import set_norm_limit
    set_norm_limit(float("inf"))
    _dbg = Debug(enabled=False)

    def _apply_conv_limits(value: float) -> None:
        for _m in vae.modules():
            if isinstance(_m, InflatedCausalConv3d):
                _m.set_memory_limit(value)
            _m.debug = _dbg

    def _free_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_oom(exc: BaseException) -> bool:
        oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),) if isinstance(t, type))
        return isinstance(exc, oom_types) or "out of memory" in str(exc).lower()

    _apply_conv_limits(float("inf"))
    _free_cuda()

    cpu_conv_limit = float(args.cpu_conv_limit_gb) if (args.cpu_conv_limit_gb and args.cpu_conv_limit_gb > 0) else float("inf")

    frames = ((args.frames - 1) // 4) * 4 + 1
    lat_frames = (frames - 1) // 4 + 1
    # Use the user-chosen tile size for the decoder too (--tile), not a hardcoded 256.
    dec_tile_px = args.tile
    dec_lat_tile = dec_tile_px // 8

    t0 = time.perf_counter()
    enc_tile = args.tile
    if args.kind == "encoder":
        mod = _EncoderModule(vae).eval().to(device=load_dev, dtype=torch.float16)
        dummy = torch.zeros((1, 3, frames, enc_tile, enc_tile), dtype=torch.float16, device=load_dev)
        stem_suffix = f"tile{enc_tile}"
    else:
        mod = _DecoderModule(vae.decoder).eval().to(device=load_dev, dtype=torch.float16)
        dummy = torch.zeros((1, 16, lat_frames, dec_lat_tile, dec_lat_tile), dtype=torch.float16, device=load_dev)
        stem_suffix = f"tile_{dec_tile_px}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def _trace_gpu(limit_gb: float) -> None:
        lim = float(limit_gb) if (limit_gb and limit_gb > 0) else float("inf")
        _apply_conv_limits(lim)
        txt = "no chunking" if lim == float("inf") else f"Conv3d chunk {lim:g} GB"
        print(f"Exporting {frames}f {args.kind} ONNX on GPU (legacy tracer; {txt})...", flush=True)
        _free_cuda()
        with torch.inference_mode():
            _portable_export(mod, (dummy,), output, legacy=True)

    def _export_cpu() -> None:
        limit_txt = "no chunking" if cpu_conv_limit == float("inf") else f"Conv3d chunk {cpu_conv_limit:g} GB"
        print(f"[CPU-offload] Tracing {frames}f {args.kind} ONNX on CPU fp16 (system RAM, CUDA hidden; {limit_txt})...", flush=True)
        print("[CPU-offload] NOTE: the CPU path is very slow for large 512-tile traces (can take hours).", flush=True)
        _apply_conv_limits(cpu_conv_limit)
        _free_cuda()
        cpu_module, cpu_args = _force_cpu_fp16(mod, (dummy,))
        with _cuda_disabled():
            _portable_export(cpu_module, cpu_args, output, legacy=True)
        del cpu_module, cpu_args
        _free_cuda()

    if use_gpu:
        produced = False
        try:
            _trace_gpu(args.gpu_conv_limit_gb)
            produced = True
        except RuntimeError as exc:
            if args.device == "cuda" or not _is_oom(exc):
                raise
            print(f"[GPU-offload] GPU trace hit OOM ({type(exc).__name__}).", flush=True)
            if args.gpu_conv_limit_gb and args.gpu_conv_limit_gb > 0:
                print("[GPU-offload] a Conv3d chunk size was already set; not retrying at other sizes.", flush=True)
            else:
                for retry_gb in (4.0, 2.0):
                    try:
                        print(f"[GPU-offload] retrying on GPU with Conv3d chunk {retry_gb:g} GB...", flush=True)
                        _trace_gpu(retry_gb)
                        produced = True
                        break
                    except RuntimeError as exc2:
                        if not _is_oom(exc2):
                            raise
                        print(f"[GPU-offload] retry with {retry_gb:g} GB still OOM.", flush=True)
        if not produced:
            print("[CPU-offload] all GPU attempts failed; switching to CPU fp16 (system RAM).", flush=True)
            try:
                vae.to("cpu")
            except Exception:
                pass
            _export_cpu()
    else:
        _export_cpu()

    print(f"WORKER-OK {args.kind} {frames}f -> {output} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
