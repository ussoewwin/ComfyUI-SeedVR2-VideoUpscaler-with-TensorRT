"""Cloud GPU ONNX export worker for RTX 5090 (32GB VRAM, Blackwell sm_120).

Traces a static ONNX graph for the TensorRT VAE engines.

OOM strategy (--device auto, default):
    1) GPU trace with the default PyTorch allocator (reference graph);
    2) on CUDA OOM: re-run in a FRESH PROCESS with --allocator managed.  That
       swaps PyTorch's CUDA allocator for one backed by cudaMallocManaged (UVM),
       so VRAM overflows are paged into system RAM - a 31GB RTX 5090 with plenty
       of host RAM can then trace graphs whose working set exceeds VRAM.  The
       graph stays the SAME legacy (reference) graph;
    3) last resort: CPU float16 trace (system RAM, CUDA hidden).  Very slow.

--export-mode dynamo is available but NOT used automatically: the torch.export
based exporter decomposes ops differently from the reference exporter, so its
graphs are not guaranteed to match the reference engines.

Usage:
    python tools/cloud_export_gpu.py --repo <custom_node_root> \
        --kind encoder --frames 85 --output <onnx_path> [--model ema_vae_fp16.safetensors] \
        [--tile 512] [--device auto|cuda|cpu] [--allocator default|managed] \
        [--export-mode legacy|dynamo]
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

_MANAGED_ALLOC_SRC = r"""
#include <cuda_runtime.h>
#include <cstddef>

extern "C" void* seedvr2_managed_malloc(std::size_t size, int device, cudaStream_t stream) {
    (void)device; (void)stream;
    void* p = nullptr;
    if (cudaMallocManaged(&p, size) != cudaSuccess) {
        return nullptr;
    }
    return p;
}

extern "C" void seedvr2_managed_free(void* ptr, std::size_t size, int device, cudaStream_t stream) {
    (void)size; (void)device; (void)stream;
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}
"""


def _install_managed_allocator() -> bool:
    """Swap PyTorch's CUDA allocator for one backed by cudaMallocManaged (UVM).

    VRAM overflows are then paged into system RAM, so a trace whose working set
    exceeds the card's VRAM still completes (this is the code path that "fully
    uses" an RTX 5090 together with a large amount of host RAM).  cudaMallocManaged
    is a CUDA *runtime* call, so a plain C++ compiler suffices (no nvcc needed).
    """
    import ctypes

    so_path: str | None = None

    # 1) Let torch build the tiny shared library (it knows the CUDA include/lib paths).
    try:
        from torch.utils.cpp_extension import load_inline

        mod = load_inline(
            name="seedvr2_managed_alloc",
            cpp_sources=_MANAGED_ALLOC_SRC,
            with_cuda=True,
            verbose=False,
            extra_cflags=["-O1"],
        )
        so_path = str(Path(mod.__file__))
    except Exception as exc:
        print(f"[MEM] cpp_extension build failed ({exc}); trying direct gcc...", flush=True)

    # 2) Fallback: compile with gcc against discovered CUDA runtime headers/libs.
    if so_path is None:
        try:
            import glob as _glob
            import site as _site
            import subprocess as _sp
            import tempfile

            search_roots: list[Path] = []
            try:
                for sp in _site.getsitepackages():
                    search_roots.append(Path(sp))
            except Exception:
                pass
            try:
                search_roots.append(Path(_site.getusersitepackages()))
            except Exception:
                pass
            search_roots.append(Path(torch.__file__).resolve().parent.parent)

            include_dirs: list[str] = []
            lib_dirs: list[str] = []
            for root in search_roots:
                include_dirs += _glob.glob(str(root / "nvidia" / "*" / "include"))
                lib_dirs += _glob.glob(str(root / "nvidia" / "*" / "lib"))
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home:
                include_dirs.append(str(Path(cuda_home) / "include"))
                lib_dirs.append(str(Path(cuda_home) / "lib64"))

            src_file = Path(tempfile.mkdtemp()) / "seedvr2_managed_alloc.cpp"
            src_file.write_text(_MANAGED_ALLOC_SRC, encoding="utf-8")
            out_so = src_file.with_suffix(".so")
            cmd = ["gcc", "-shared", "-fPIC", "-O1", "-o", str(out_so), str(src_file)]
            for d in include_dirs:
                cmd += ["-I", d]
            for d in lib_dirs:
                cmd += ["-L", d]
            cmd += ["-lcudart"]
            r = _sp.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out_so.exists():
                so_path = str(out_so)
            else:
                print(f"[MEM] gcc build failed: {(r.stderr or '').strip()[:400]}", flush=True)
        except Exception as exc:
            print(f"[MEM] direct gcc path failed: {exc}", flush=True)

    if so_path is None:
        return False

    try:
        lib = ctypes.CDLL(so_path)
        lib.seedvr2_managed_malloc
        lib.seedvr2_managed_free
    except Exception as exc:
        print(f"[MEM] allocator symbols not found in {so_path}: {exc}", flush=True)
        return False

    try:
        from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator

        alloc = CUDAPluggableAllocator(so_path, "seedvr2_managed_malloc", "seedvr2_managed_free")
        change_current_allocator(alloc)
        print(f"[MEM] managed-memory allocator ACTIVE ({so_path}) - VRAM overflow now spills into system RAM", flush=True)
        return True
    except Exception as exc:
        print(f"[MEM] change_current_allocator failed: {exc}", flush=True)
        return False


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
                        help="auto = GPU first (managed-memory retry, then CPU), cuda = GPU only, cpu = CPU only")
    parser.add_argument("--allocator", choices=["default", "managed"], default="default",
                        help="managed = cudaMallocManaged allocator: VRAM overflows spill into system RAM "
                             "(use with plenty of host RAM to trace graphs larger than VRAM)")
    parser.add_argument("--export-mode", choices=["legacy", "dynamo"], default="legacy",
                        help="legacy = torch.jit tracer (reference graph); dynamo = torch.export based "
                             "(different op decomposition - not used automatically)")
    parser.add_argument("--gpu-conv-limit-gb", type=float, default=0.0,
                        help="Conv3d chunk size (GB) for the GPU trace (0 = no chunking)")
    parser.add_argument("--cpu-conv-limit-gb", type=float, default=16.0,
                        help="Conv3d chunk size (GB) for the CPU trace path only (default 16)")
    args = parser.parse_args()

    if args.allocator == "managed":
        if not _install_managed_allocator():
            print("[MEM] WARNING: managed allocator unavailable; continuing with the default allocator.", flush=True)

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
        print(f"Exporting {frames}f {args.kind} ONNX on GPU ({args.export_mode} exporter; {txt}; "
              f"allocator={args.allocator})...", flush=True)
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

        if oom_hit:
            print("[GPU-offload] GPU trace hit CUDA OOM.", flush=True)
            if args.allocator != "managed":
                # Re-run in a fresh process with the managed-memory allocator: VRAM overflows
                # are paged into system RAM, so a trace working set larger than VRAM can still
                # complete while keeping the SAME legacy (reference) graph.
                child_cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--repo", args.repo, "--kind", args.kind, "--frames", str(args.frames),
                    "--output", str(output), "--model", args.model, "--tile", str(args.tile),
                    "--device", args.device, "--allocator", "managed",
                    "--export-mode", args.export_mode,
                    "--cpu-conv-limit-gb", str(args.cpu_conv_limit_gb),
                ]
                if args.model_dir:
                    child_cmd += ["--model-dir", args.model_dir]
                if args.gpu_conv_limit_gb and args.gpu_conv_limit_gb > 0:
                    child_cmd += ["--gpu-conv-limit-gb", str(args.gpu_conv_limit_gb)]
                print("[GPU-offload] retrying in a fresh process with the managed (VRAM+system RAM) allocator...", flush=True)
                res = subprocess.run(child_cmd)
                return res.returncode
            print("[CPU-offload] GPU trace still OOM with the managed allocator; falling back to CPU fp16.", flush=True)
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
