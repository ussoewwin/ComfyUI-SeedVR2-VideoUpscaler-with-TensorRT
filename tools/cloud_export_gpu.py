"""Cloud GPU ONNX export worker for RTX 5090 (32GB VRAM, Blackwell sm_120).

Traces a static ONNX graph for the TensorRT VAE engines.

OOM strategy (--device auto, default):
    1) legacy exporter (torch.jit tracer) on GPU, no Conv3d chunking - the
       reference graph used by all existing engines.  NOTE: the legacy tracer
       keeps every intermediate activation alive, so 85f x 512 needs ~33GB and
       does not fit in a 31GB card;
    2) on CUDA OOM: automatically re-runs itself in a FRESH PROCESS with
       --export-mode dynamo (torch.export based exporter).  Fake-tensor tracing
       needs almost no GPU memory, so large 512-tile graphs can be exported on
       a 5090;
    3) last resort: CPU float16 trace (system RAM, CUDA hidden).  This is
       extremely slow for large 512-tile graphs (hours) - avoid if possible.

Use --export-mode to force a specific exporter, --gpu-conv-limit-gb to set a
Conv3d chunk size for the GPU path, and --device cuda / --device cpu to force
a specific device.

The produced ONNX is GPU-independent. Build the engine with cloud_build_engine.py
on any Blackwell (sm_120) GPU, or locally on the RTX 5060 Ti.

Usage:
    python tools/cloud_export_gpu.py --repo <custom_node_root> \
        --kind encoder --frames 85 --output <onnx_path> [--model ema_vae_fp16.safetensors] \
        [--tile 512] [--device auto|cuda|cpu] [--export-mode legacy|dynamo]
"""

from __future__ import annotations

import argparse
import gc
import os
import inspect
import subprocess
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
                        help="auto = GPU first (fresh-process fallbacks), then CPU fp16 (system RAM); "
                             "cuda = GPU only; cpu = CPU only (CUDA hidden)")
    parser.add_argument("--export-mode", choices=["legacy", "dynamo"], default="legacy",
                        help="legacy = torch.jit tracer (reference graph; keeps all activations alive "
                             "while tracing); dynamo = torch.export based exporter (fake tensors, "
                             "almost no GPU memory)")
    parser.add_argument("--gpu-conv-limit-gb", type=float, default=0.0,
                        help="Conv3d chunk size (GB) for the GPU trace (0 = no chunking)")
    parser.add_argument("--cpu-conv-limit-gb", type=float, default=16.0,
                        help="Conv3d chunk size (GB) for the CPU trace path only (default 16)")
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

    def _trace_gpu() -> None:
        lim = float(args.gpu_conv_limit_gb) if (args.gpu_conv_limit_gb and args.gpu_conv_limit_gb > 0) else float("inf")
        _apply_conv_limits(lim)
        txt = "no chunking" if lim == float("inf") else f"Conv3d chunk {lim:g} GB"
        print(f"Exporting {frames}f {args.kind} ONNX on GPU ({args.export_mode} exporter; {txt})...", flush=True)
        _free_cuda()
        with torch.inference_mode():
            _portable_export(mod, (dummy,), output, legacy=(args.export_mode == "legacy"))

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
        oom_hit = False
        try:
            _trace_gpu()
        except RuntimeError as exc:
            if args.device == "cuda" or not _is_oom(exc):
                raise
            oom_hit = True
        produce_note = ""

        if oom_hit:
            print("[GPU-offload] GPU trace hit CUDA OOM.", flush=True)
            if args.export_mode != "dynamo":
                # Re-run in a fresh process: the failed legacy trace keeps its activation
                # tensors alive via the exception traceback / tracer state, so retrying in
                # the same process would immediately fail again.
                child_cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--repo", args.repo, "--kind", args.kind, "--frames", str(args.frames),
                    "--output", str(output), "--model", args.model, "--tile", str(args.tile),
                    "--device", args.device, "--export-mode", "dynamo",
                    "--cpu-conv-limit-gb", str(args.cpu_conv_limit_gb),
                ]
                if args.model_dir:
                    child_cmd += ["--model-dir", args.model_dir]
                if args.gpu_conv_limit_gb and args.gpu_conv_limit_gb > 0:
                    child_cmd += ["--gpu-conv-limit-gb", str(args.gpu_conv_limit_gb)]
                print("[GPU-offload] retrying in a fresh process with the dynamo (torch.export) exporter "
                      "- fake-tensor tracing needs almost no GPU memory...", flush=True)
                res = subprocess.run(child_cmd)
                return res.returncode
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
