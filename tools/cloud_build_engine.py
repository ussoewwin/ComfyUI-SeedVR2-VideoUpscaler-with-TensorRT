"""Cloud TensorRT engine builder for large-batch VAE engines.

Builds a .rtxplan from a static ONNX graph.

Memory strategy: a large graph can need far more working memory than the GPU has
VRAM (e.g. the 185f tile512 encoder needs ~46 GiB of workspace to build).  When
the normal attempts fail, the builder retries with a managed-memory
(cudaMallocManaged / UVM) GPU allocator installed into TensorRT itself
(IGpuAllocator): every device allocation of the build then spills into system
RAM, so a 31GB RTX 5090 together with host RAM can build graphs whose
build-time working set exceeds VRAM.

!!! CRITICAL !!!
TensorRT engines are GPU-architecture specific. An engine built on a cloud GPU
will NOT run on a local RTX 5060 Ti. If your cloud GPU is a different
architecture, use this script ONLY to build the ONNX (see export_onnx_worker.py)
and build the engine locally instead.

Usage:
    python cloud_build_engine.py <onnx_path> --output <engine_path> [--workspace-gb 24] [--min-ws] [--managed]

Setup (first time):
    pip install -r cloud_requirements.txt
"""

from __future__ import annotations

import argparse
import ctypes
import math
import re
import subprocess
import sys
import time
from pathlib import Path


def _diagnose_required_workspace_gb(onnx_path: str) -> float | None:
    """Run one quick build with a verbose logger and read TRT's 'Need <bytes>' figure.

    TRT reports large-build failures as:
        (foreignNode) [pass.cpp] Exceeded mem budget of <budget>. Need <bytes>
        Try increasing the workspace size with IBuilderConfig::setMemoryPoolLimit.
    This helper extracts <bytes> so the caller can retry with a sufficient size.
    """
    code = (
        "import sys\n"
        "import tensorrt_rtx as trt\n"
        "logger = trt.Logger(trt.Logger.VERBOSE)\n"
        "builder = trt.Builder(logger)\n"
        "network = builder.create_network()\n"
        "parser = trt.OnnxParser(network, logger)\n"
        "if not parser.parse_from_file(sys.argv[1]):\n"
        "    raise SystemExit('parse failed')\n"
        "config = builder.create_builder_config()\n"
        "config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)\n"
        "builder.build_serialized_network(network, config)\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", code, onnx_path],
            capture_output=True, text=True, timeout=1800,
        )
        out = (res.stdout or "") + "\n" + (res.stderr or "")
        m = re.search(r"Need (\d+)", out)
        if m:
            return int(m.group(1)) / 2**30
    except Exception:
        pass
    return None


def _load_cudart():
    """Load libcudart through ctypes (it is already loaded by torch in practice)."""
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so", "libcudart.so.13.0", "libcudart.so.12.0"):
        try:
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            lib.cudaMallocManaged
            lib.cudaFree
            lib.cudaMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
            lib.cudaMallocManaged.restype = ctypes.c_int
            lib.cudaFree.argtypes = [ctypes.c_void_p]
            lib.cudaFree.restype = ctypes.c_int
            return lib
        except Exception:
            continue
    raise RuntimeError("libcudart not found (tried libcudart.so.13/.so.12/.so)")


def _build_managed(onnx_path: str, ws_gb: float):
    """Build with a TensorRT IGpuAllocator backed by cudaMallocManaged (VRAM + host RAM)."""
    import tensorrt_rtx as trt

    cudart = _load_cudart()

    class _ManagedAlloc(trt.IGpuAllocator):  # type: ignore[misc]
        def __init__(self):
            trt.IGpuAllocator.__init__(self)
            self._base = {}

        def allocate(self, size, alignment, flags):  # noqa: D102
            try:
                a = max(int(alignment), 1)
                raw = ctypes.c_void_p()
                if cudart.cudaMallocManaged(ctypes.byref(raw), ctypes.c_size_t(int(size) + a), 1) != 0:
                    return 0
                if not raw.value:
                    return 0
                aligned = (raw.value + a - 1) // a * a
                self._base[int(aligned)] = raw.value
                return int(aligned)
            except Exception:
                return 0

        def deallocate(self, memory):  # noqa: D102
            try:
                base = self._base.pop(int(memory), None)
                if base is not None:
                    cudart.cudaFree(ctypes.c_void_p(base))
            except Exception:
                pass

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder.set_gpu_allocator(_ManagedAlloc())
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        print(f"ERROR: TensorRT could not parse {onnx_path}:\n{errors}", flush=True)
        return None
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_gb * (1 << 30)))
    return builder.build_serialized_network(network, config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path, help="input static ONNX graph")
    parser.add_argument("--output", type=Path, required=True, help="output .rtxplan path")
    parser.add_argument("--workspace-gb", type=float, default=24.0,
                        help="TensorRT workspace limit in GB (default 24)")
    parser.add_argument("--min-ws", action="store_true",
                        help="binary-search the smallest workspace that still builds (minimizes runtime VRAM)")
    parser.add_argument("--managed", action="store_true",
                        help="skip the normal attempts and build with the managed (VRAM+system RAM) allocator")
    args = parser.parse_args()

    import torch
    import tensorrt_rtx as trt

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available on this machine", flush=True)
        return 2

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {props.total_memory / 2**30:.1f} GiB, arch: {props.major}.{props.minor}", flush=True)
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        print(f"Free VRAM at build start: {free_b / 2**30:.2f} GiB", flush=True)
    except Exception:
        pass
    print(f"TensorRT-RTX: {trt.__version__}", flush=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        print(f"ERROR: TensorRT could not parse {args.onnx}:\n{errors}", flush=True)
        return 1

    config = builder.create_builder_config()

    def _try_build(ws_gb: float):
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_gb * (1 << 30)))
            return builder.build_serialized_network(network, config)
        except Exception as exc:
            print(f"WARNING: build raised an exception at {ws_gb:g} GB: {exc}", flush=True)
            return None

    t0 = time.perf_counter()
    blob = None
    need_gb = None
    if args.min_ws:
        lo, hi = 2.0, args.workspace_gb
        blob = _try_build(hi)
        if blob is None:
            while blob is None and hi <= 48:
                hi += 2.0
                blob = _try_build(hi)
        if blob is None:
            print("ERROR: engine build failed even at high allocation", flush=True)
            return 1
        while hi - lo > 0.5:
            mid = (lo + hi) / 2
            b = _try_build(mid)
            if b is not None:
                blob = b
                hi = mid
            else:
                lo = mid
        print(f"Min allocation: {hi:.1f} GB (smallest that builds)", flush=True)
    else:
        if args.managed:
            print("INFO: --managed requested; using the managed (VRAM+system RAM) allocator directly.", flush=True)
        else:
            blob = _try_build(args.workspace_gb)
            if blob is None:
                # TRT accepts or rejects a build based on the *budget* (the configured pool),
                # not on how much VRAM happens to be free: "Exceeded mem budget of X. Need Y".
                # Retry with a spread of sizes: larger graphs often need more than the
                # configured value, smaller ones may build with less.
                for ws_try in (24.0, 28.0, 32.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0):
                    if ws_try == args.workspace_gb:
                        continue
                    print(f"WARNING: build failed at {args.workspace_gb:g} GB; retrying with {ws_try:g} GB...", flush=True)
                    blob = _try_build(ws_try)
                    if blob is not None:
                        print(f"NOTE: engine built with adjusted allocation {ws_try:g} GB.", flush=True)
                        break
                    print(f"WARNING: build failed at {ws_try:g} GB as well.", flush=True)

        if blob is None:
            # Ask TRT (verbose run) how much it actually needs, then retry with that size.
            need_gb = _diagnose_required_workspace_gb(str(args.onnx))
            if need_gb is not None:
                print(f"DIAG: TRT reports this graph needs ~{need_gb:.1f} GiB of workspace to build.", flush=True)
                for extra in (2.0, 4.0, 6.0):
                    ws_try = math.ceil(need_gb) + extra
                    if ws_try > 30.0:
                        continue  # cannot fit in 31GB VRAM with the default allocator
                    print(f"WARNING: retrying with the required allocation {ws_try:g} GB...", flush=True)
                    blob = _try_build(ws_try)
                    if blob is not None:
                        print(f"NOTE: engine built at {ws_try:g} GB.", flush=True)
                        break
                    print(f"WARNING: build failed at {ws_try:g} GB as well.", flush=True)
            else:
                print("DIAG: could not determine the required workspace size from a verbose build.", flush=True)

        if blob is None:
            # Managed-memory build: TensorRT's own allocations spill into system RAM.
            if need_gb is None:
                need_gb = 32.0
            ws_try = min(64.0, math.ceil(need_gb) + 4.0)
            print(f"INFO: retrying with the managed (cudaMallocManaged / VRAM+system RAM) allocator at {ws_try:g} GB...", flush=True)
            try:
                blob = _build_managed(str(args.onnx), ws_try)
            except Exception as exc:
                print(f"WARNING: managed build failed: {exc}", flush=True)
            if blob is not None:
                print(f"NOTE: engine built with the managed allocator at {ws_try:g} GB.", flush=True)
            else:
                print("WARNING: managed build failed as well.", flush=True)
    dt = time.perf_counter() - t0

    if blob is None:
        print("ERROR: engine build failed (the build needs more memory than was granted, "
              "or the graph is unsupported)", flush=True)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"OK: {args.output} ({blob.nbytes / 2**20:.1f} MiB in {dt:.1f}s)", flush=True)
    print("WARNING: this engine is GPU-architecture specific; it may not run on other GPUs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
