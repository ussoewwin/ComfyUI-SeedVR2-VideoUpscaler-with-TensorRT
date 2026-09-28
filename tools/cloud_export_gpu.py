"""Cloud GPU ONNX export worker for RTX 5090 (32GB VRAM, Blackwell sm_120).

Traces a static ONNX graph for the TensorRT VAE engines.

OOM strategy (--device auto, default):
    1) GPU trace with the default PyTorch allocator (reference graph);
    2) on CUDA OOM: free EVERYTHING this process holds, then re-run in a FRESH
       PROCESS with --allocator managed.  That swaps PyTorch's CUDA allocator
       for one backed by cudaMallocManaged (UVM), so VRAM overflows are paged
       into system RAM - a 31GB RTX 5090 with plenty of host RAM can then trace
       graphs whose working set exceeds VRAM.  The graph stays the SAME legacy
       (reference) graph;
    3) last resort: CPU float16 trace (system RAM, CUDA hidden).  Very slow.

The managed allocator library is built WITHOUT any CUDA headers or link-time
CUDA libraries: it resolves cudaMallocManaged/cudaFree at runtime via dlopen,
so a plain C compiler is the only requirement.

If --allocator managed cannot be built, the process FAILS FAST (no silent CPU
fallback) so the cause is visible in the log.

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

# No CUDA headers, no link-time CUDA libs: cudaMallocManaged/cudaFree are
# resolved at runtime with dlopen, so this builds on any machine with a C
# compiler and works with whichever libcudart the process already has loaded
# (falls back to dlopen("libcudart.so.13"/".so.12"/".so")).
_MANAGED_ALLOC_SRC = r"""
#include <cstddef>
#include <dlfcn.h>

typedef struct CUstream_st* cudaStream_t;
typedef int (*seedvr2_malloc_managed_fn)(void**, std::size_t, unsigned int);
typedef int (*seedvr2_free_fn)(void*);

static seedvr2_malloc_managed_fn seedvr2_p_malloc = nullptr;
static seedvr2_free_fn seedvr2_p_free = nullptr;

static void seedvr2_resolve(void) {
    if (seedvr2_p_malloc != nullptr && seedvr2_p_free != nullptr) {
        return;
    }
    seedvr2_p_malloc = (seedvr2_malloc_managed_fn)dlsym(RTLD_DEFAULT, "cudaMallocManaged");
    seedvr2_p_free = (seedvr2_free_fn)dlsym(RTLD_DEFAULT, "cudaFree");
    if (seedvr2_p_malloc != nullptr && seedvr2_p_free != nullptr) {
        return;
    }
    const char* names[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so", nullptr};
    for (int i = 0; names[i] != nullptr; ++i) {
        void* h = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
        if (h == nullptr) {
            continue;
        }
        if (seedvr2_p_malloc == nullptr) {
            seedvr2_p_malloc = (seedvr2_malloc_managed_fn)dlsym(h, "cudaMallocManaged");
        }
        if (seedvr2_p_free == nullptr) {
            seedvr2_p_free = (seedvr2_free_fn)dlsym(h, "cudaFree");
        }
        if (seedvr2_p_malloc != nullptr && seedvr2_p_free != nullptr) {
            return;
        }
    }
}

extern "C" void* seedvr2_managed_malloc(std::size_t size, int device, cudaStream_t stream) {
    (void)device; (void)stream;
    seedvr2_resolve();
    if (seedvr2_p_malloc == nullptr) {
        return nullptr;
    }
    void* p = nullptr;
    if (seedvr2_p_malloc(&p, size, 1u /* cudaMemAttachGlobal */) != 0) {
        return nullptr;
    }
    return p;
}

extern "C" void seedvr2_managed_free(void* ptr, std::size_t size, int device, cudaStream_t stream) {
    (void)size; (void)device; (void)stream;
    seedvr2_resolve();
    if (seedvr2_p_free != nullptr && ptr != nullptr) {
        seedvr2_p_free(ptr);
    }
}
"""


def _install_managed_allocator() -> bool:
    """Swap PyTorch's CUDA allocator for one backed by cudaMallocManaged (UVM).

    VRAM overflows are then paged into system RAM, so a trace whose working set
    exceeds the card's VRAM still completes (this is the code path that "fully
    uses" an RTX 5090 together with a large amount of host RAM).
    """
    import ctypes

    so_path: str | None = None

    # 1) Compile the tiny library directly (no headers, no CUDA libs needed).
    try:
        import tempfile

        src_dir = Path(tempfile.mkdtemp())
        src_file = src_dir / "seedvr2_managed_alloc.cpp"
        src_file.write_text(_MANAGED_ALLOC_SRC, encoding="utf-8")
        out_so = src_dir / "seedvr2_managed_alloc.so"
        last_err = ""
        for cc in ("gcc", "cc", "g++", "clang++"):
            r = subprocess.run(
                [cc, "-shared", "-fPIC", "-O2", "-o", str(out_so), str(src_file), "-ldl"],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and out_so.exists():
                so_path = str(out_so)
                print(f"[MEM] built allocator library with {cc}: {so_path}", flush=True)
                break
            last_err = (r.stderr or "").strip()[:300]
        if so_path is None:
            print(f"[MEM] direct compile failed: {last_err}", flush=True)
    except Exception as exc:
        print(f"[MEM] direct compile path failed: {exc}", flush=True)

    # 2) Fallback: torch's cpp_extension (finds headers/libs on its own).
    if so_path is None:
        try:
            from torch.utils.cpp_extension import _get_build_directory, load_inline

            try:
                load_inline(
                    name="seedvr2_managed_alloc",
                    cpp_sources=_MANAGED_ALLOC_SRC,
                    is_python_module=False,
                    with_cuda=True,
                    verbose=False,
                    extra_cflags=["-O2", "-fvisibility=default"],
                    extra_ldflags=["-ldl"],
                )
            except Exception as exc:
                print(f"[MEM] cpp_extension build/import failed ({exc}); checking build dir...", flush=True)
            try:
                build_dir = Path(_get_build_directory("seedvr2_managed_alloc", verbose=False))
                sos = sorted(build_dir.glob("**/*.so"))
                if sos:
                    so_path = str(sos[-1])
                    print(f"[MEM] found built allocator library: {so_path}", flush=True)
            except Exception as exc:
                print(f"[MEM] could not locate build dir: {exc}", flush=True)
        except Exception as exc:
            print(f"[MEM] cpp_extension unavailable ({exc})", flush=True)

    if so_path is None:
        return False

    try:
        lib = ctypes.CDLL(so_path)
        lib.seedvr2_managed_malloc
        lib.seedvr2_managed_free
    except Exception as exc:
        print(f"[MEM] allocator symbols not found in {so_path}: {exc}", flush=True)
        return False

    # Sanity check that the CUDA runtime symbols can be resolved (informational).
    try:
        probe = ctypes.CDLL(so_path)
        probe.seedvr2_managed_malloc
    except Exception:
        pass

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
            print("ERROR: --allocator managed requested but the allocator could not be built. "
                  "Aborting (no CPU fallback). See the [MEM] lines above.", flush=True)
            return 3

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
                # The failed trace keeps a large amount of GPU memory referenced; free it all
                # BEFORE handing over to the child, otherwise the child starts with almost no
                # free VRAM (the "Initial CUDA memory" line in its log would show that).
                try:
                    del mod
                    del dummy
                    del vae
                except Exception:
                    pass
                for _ in range(3):
                    gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    free_b, _total_b = torch.cuda.mem_get_info()
                    print(f"[GPU-offload] parent released GPU memory: {free_b / 2**30:.2f} GiB free now", flush=True)
                except Exception:
                    pass
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
