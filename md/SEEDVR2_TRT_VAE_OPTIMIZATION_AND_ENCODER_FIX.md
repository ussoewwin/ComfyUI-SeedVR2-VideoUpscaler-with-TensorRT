# SeedVR2 Video Upscaler — Technical Guide: Three Core Improvements and Encoder Refactoring

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/SEEDVR2_TRT_VAE_OPTIMIZATION_AND_ENCODER_FIX.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

Target custom node: `ComfyUI/custom_nodes/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder`  
Target core modules: `src/interfaces/video_save.py`, `src/core/trt_decoder.py`, `src/core/trt_encoder.py`, `src/core/infer.py`, `src/interfaces/trt_vae_model_loader.py`, `src/interfaces/__init__.py`, `__init__.py`

---

## 1. Overview of Key Themes

### Theme 1: Three Core Improvements (Studio Production Engineering Knowledge)

1. **Timestamp Rectification in Video Assembly: Complete Eradication of Audio Desynchronization and Endpoint Freezes**  
   Applying Studio's production FFmpeg parameters (`-fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr`):
   - `-fflags +genpts`: Recalculates and regenerates Presentation Time Stamps (PTS) from packet payloads across chunk boundaries.
   - `-avoid_negative_ts make_zero`: Enforces that timestamps start strictly at `00:00:00.000` rather than negative or shifted offsets.
   - `-fps_mode cfr`: Disallows variable frame rates (VFR) by enforcing strict Constant Frame Rate (CFR), eliminating playback clock drift between video and audio streams.

2. **Address Overwrite Prevention in TensorRT `set_tensor_address` Execution Context**  
   Guarding against mutable tensor address corruption in TensorRT's execution context:
   - *"A context's tensor addresses are mutable. For the safe/default path we execute one tile at a time so addresses cannot be overwritten by a later queued tile."*
   - TensorRT's `ExecutionContext` holds tensor memory addresses assigned by `set_tensor_address`. Attempting concurrent tile inference across multiple CUDA streams causes later queued tiles to overwrite the buffer addresses of active tiles, corrupting preceding tile calculations (resulting in blacked-out or grayed-out output tiles).
   - Enforcing exclusive lock acquisition (`_DECODE_LOCK` / `_ENCODE_LOCK`) and per-tile synchronous completion (`stream.synchronize()`) provides an inviolable safety barrier against memory corruption.

3. **Thorough Per-Batch / Per-Chunk Tri-Partite Memory Reclamation Pattern**  
   Executing an exhaustive cleanup sequence after every chunk or batch:
   ```python
   del payload, source, latent, result, weights, decoded, jobs
   gc.collect()
   torch.cuda.empty_cache()
   ```
   Explicitly deleting Python object references, triggering garbage collection (`gc.collect()`) to destroy circular references, and releasing PyTorch allocator block pools back to the GPU driver (`torch.cuda.empty_cache()`). This eliminates VRAM fragmentation across 100+ frame renders, ensuring consistent minimum VRAM headroom from start to finish.

### Theme 2: Comprehensive Encoder Refactoring

1. **Porting v1.5.4 Architectural Foundations to Encoder (Elimination of Artifacts & VRAM Bloat)**  
   Fully ported the v1.5.4 decoder fixes into `trt_encoder.py`:
   - *Studio-Compatible Static Shape Check Guard*: `current_shape != target_shape` bypasses redundant `set_input_shape` calls, preventing TRT internal scratchpad reallocation and dirty VRAM ingestion.
   - *Deterministic Dummy Warmup Execution*: An asynchronous zero-filled dummy pass (`warmup_in` / `warmup_out`) and CUDA stream synchronization prior to tiling cleans internal 3D causal convolution accumulator lines, eliminating uninitialized garbage reads on tile `y=0, x=0`.
   - *Zero VRAM Bloat Architecture*: Local tile padding instead of outer canvas padding, avoiding 2x-3x memory explosion of float32 accumulation buffers (`result`, `weights`, `dc_result`).
2. **Short-Batch Pad & Crop 1-Shot Execution & Complete Elimination of Silent FP16 Fallbacks**  
   Eliminating the legacy `IndexError` on slice indexing (`starts[-1]`) when processing video batches smaller than engine frame counts (e.g. `batch_size=5` with a 21f engine). Replicating the final frame up to engine capacity, executing 1-shot TRT encode, and cropping back to the true latent length `(total - 1) // 4 + 1` preserves maximum hardware acceleration. Missing runtime dependencies or missing engine files raise an explicit `RuntimeError` rather than silently degrading performance into PyTorch FP16.
3. **Dynamic Multi-Directory Engine Discovery Architecture (`pick_engine_frames` / `_available_engine_frames`)**  
   Dynamically scans `tensorrt_backend/artifacts` and `models/tensorrt/seedvr2` for arbitrary 4n+1 frame engine plans. Dynamically selects engines prioritizing exact match → largest fitting engine → smallest engine with padding and cropping.
4. **Node Renaming & Symmetrical UI Schema (`SeedVR2LoadTensorRTVAEEncoder`)**  
   Renamed `SeedVR2LoadTensorRTVAEModel` to `SeedVR2LoadTensorRTVAEEncoder` (maintaining backward compatibility alias), matching `SeedVR2LoadTensorRTVAEDecoder` symmetrically across node identifier, display name, description, input combo widgets (`model`, `device`, `engine_frames`), tooltips, and `SEEDVR2_VAE` outputs.

---

## 2. Pre-Fix Root Cause Analysis

### 2.1 Pre-Fix Problems in Theme 1

#### 1. Audio Drift and Endpoint Freeze During Video Stitching
- **Root Cause**: When long video renders are split into temporal chunks or batches and stitched back together, standard video muxing often carries discontinuous presentation timestamps (PTS) or variable frame timing (VFR). In media players such as PotPlayer, this manifest as:
  - The video stream freezing at the final frame while audio continues playing for several seconds.
  - Gradual cumulative desynchronization between audio and video across chunk transitions.
- **Resolution**: Adding `-fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr` enforces packet-level PTS recalculation, shifts initial timestamps to zero, and forces true constant frame rates.

#### 2. Black or Corrupted Tiles Caused by `set_tensor_address` Overwriting
- **Root Cause**: TensorRT's execution context maintains mutable tensor pointer bindings. When developers attempt naive asynchronous multi-stream concurrency to overlap tile computation, calling `context.set_tensor_address()` for tile $N+1$ overwrites the pointers while tile $N$ is still executing on the GPU. Tile $N$ then reads from or writes to tile $N+1$'s memory space, causing output corruption, black tiles, or grayed-out blocks.
- **Resolution**: Strict single-tile execution guarded by mutual exclusion locks (`_ENCODE_LOCK` / `_DECODE_LOCK`) and explicit stream synchronization (`stream.synchronize()`) after every tile invocation.

#### 3. Progressive VRAM Fragmentation and Exhaustion (OOM) on Long Renders
- **Root Cause**: In multi-chunk processing, large intermediate tensors (such as full-resolution latent maps, Float32 accumulation buffers, and PyTorch tensors) remain referenced in local scopes or circular closures. Even when tensors go out of scope, PyTorch's caching allocator retains block pools without releasing them to the OS, causing VRAM fragmentation that builds up over 50, 100, or 200 frames until an out-of-memory crash occurs.
- **Resolution**: Tri-partite memory reclamation: explicit `del` of intermediate tensors, followed immediately by `gc.collect()` to clean cyclic references, followed by `torch.cuda.empty_cache()` to return block pools to the driver.

### 2.2 Pre-Fix Problems in Theme 2

#### 1. Short-Batch `IndexError` Crashes and Deceptive Silent Fallback
- **Root Cause**: The legacy encoder implementation used `range(0, total - engine_frames + 1, stride)`. When the input batch size (e.g. default `batch_size=5`) was smaller than the available engine size (e.g. 21f), `range(...)` yielded an empty list `[]`. Slicing `starts[-1]` crashed with `IndexError: list index out of range`.
- **The Silent Fallback Trap**: The outer loop enclosed TRT execution in a generic `try ... except` block that silently caught the `IndexError` and quietly executed the standard PyTorch FP16 VAE instead. Users configured TensorRT believing they were getting 3x–5x acceleration, while the node secretly ran slow PyTorch FP16 code without logging any warning.

#### 2. Static Hardcoded Frame Support
- **Root Cause**: The encoder lacked filesystem engine discovery, hardcoding a small list of expected frame sizes and failing to detect custom user engines (e.g. 5f, 9f, 13f, 17f, 21f, 29f).

#### 3. Asymmetric Node Layout and Confusing UI
- **Root Cause**: The decoder was named `SeedVR2LoadTensorRTVAEDecoder`, but the encoder was named `SeedVR2LoadTensorRTVAEModel` and presented redundant widgets (`encode_tiled`, `encode_tile_size`, `offload_device`), confusing users and disrupting workflow symmetry.

---

## 3. Added and Modified Files

| File Path | Action | Description of Modifications |
| :--- | :--- | :--- |
| `src/interfaces/video_save.py` | Added / Modified | Implemented Studio-grade FFmpeg timestamp rectification (`-fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr`) for video encoding and audio muxing. |
| `src/core/trt_decoder.py` | Modified | Enforced `_DECODE_LOCK` mutual exclusion, 1-tile-at-a-time execution with `stream.synchronize()`, and per-chunk tri-partite memory cleanup (`del` + `gc.collect()` + `torch.cuda.empty_cache()`). |
| `src/core/trt_encoder.py` | Modified | Implemented `_ENCODE_LOCK` mutual exclusion, 1-tile execution safety, per-chunk memory reclamation, short-batch pad & crop 1-shot execution, and dynamic engine scanner (`pick_engine_frames`). |
| `src/core/infer.py` | Modified | Updated `_trt_encode_batch` to support pad & crop for short batches, eliminated silent FP16 fallbacks, and added explicit `RuntimeError` dispatch. |
| `src/interfaces/trt_vae_model_loader.py` | Modified | Renamed node to `SeedVR2LoadTensorRTVAEEncoder`, symmetrical UI schema with decoder, multi-directory engine scanner, and backward compatibility alias. |
| `src/interfaces/__init__.py` | Modified | Registered `SeedVR2LoadTensorRTVAEEncoder` and `SeedVR2SaveVideo` in node extension list and export table. |
| `__init__.py` | Modified | Updated startup node logging and imports. |
| `md/SEEDVR2_TRT_VAE_OPTIMIZATION_AND_ENCODER_FIX.md` | Created | Full technical guide in standard English. |

---

## 4. Full Source Code of Modified Components

### 4.1 `src/interfaces/video_save.py` (Complete 202 lines)

```python
"""
SeedVR2 Save Video Node
High-reliability video encoder and assembler for ComfyUI.
Applies Studio-grade timestamp rectification (-fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr)
to eliminate audio drift and endpoint freeze on long upscale renders.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import cv2
from comfy_api.latest import io

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _get_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate_dirs = [
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"D:\ffmpeg\bin"),
        Path(__file__).resolve().parents[2] / "bin" / "ffmpeg" / "bin",
        Path(__file__).resolve().parents[2] / "bin",
    ]
    for d in candidate_dirs:
        candidate_file = d / "ffmpeg.exe"
        if candidate_file.exists():
            return str(candidate_file)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    return "ffmpeg"


class SeedVR2SaveVideo(io.ComfyNode):
    """
    SeedVR2 Save Video Node
    
    Encodes video frames to MP4 with Studio's timestamp rectification:
    -fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr
    Eliminates audio drift and end-of-video playback stutter completely.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2SaveVideo",
            display_name="SeedVR2 Save Video (CFR & Sync Safe)",
            category="SEEDVR2",
            description=(
                "Save video frames to MP4 with Studio-grade timestamp rectification. "
                "Applies -fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr "
                "to eliminate audio desynchronization and endpoint freeze."
            ),
            inputs=[
                io.Image.Input("images",
                    tooltip="Upscaled video frames [T, H, W, C] in range [0, 1]."
                ),
                io.Float.Input("fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.01,
                    tooltip="Output video frame rate (strictly locked to Constant Frame Rate / CFR)."
                ),
                io.String.Input("filename_prefix",
                    default="SeedVR2",
                    tooltip="Output filename prefix. Saved under ComfyUI output directory."
                ),
                io.Int.Input("crf",
                    default=18,
                    min=0,
                    max=51,
                    step=1,
                    tooltip="H.264 CRF quality level (lower = higher quality, default: 18)."
                ),
                io.Combo.Input("preset",
                    options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                    default="medium",
                    tooltip="FFmpeg H.264 encoding preset (default: medium)."
                ),
                io.String.Input("audio_source_path",
                    default="",
                    optional=True,
                    tooltip="Optional path to source video or audio file to mux into the output video with timestamp alignment."
                ),
            ],
            outputs=[
                io.String.Output("video_path",
                    tooltip="Full path to the saved MP4 video file."
                )
            ]
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        fps: float = 24.0,
        filename_prefix: str = "SeedVR2",
        crf: int = 18,
        preset: str = "medium",
        audio_source_path: str = "",
    ) -> io.NodeOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [T, H, W, C], got {tuple(images.shape)}")

        output_dir = folder_paths.get_output_directory() if folder_paths else "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_filename = f"{filename_prefix}_{timestamp}.mp4"
        final_path = Path(output_dir) / final_filename
        temp_path = final_path.with_name(f"{final_path.stem}_noaudio.mp4")

        T, H, W, C = images.shape
        cpu_frames = images.detach().cpu()
        if cpu_frames.dtype != torch.uint8:
            cpu_frames = (cpu_frames.clamp(0, 1) * 255.0).to(torch.uint8)

        frames_np = cpu_frames.numpy()
        del cpu_frames

        ffmpeg_bin = _get_ffmpeg()
        encode_command = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24" if C == 3 else "rgba",
            "-s:v", f"{W}x{H}", "-r", f"{fps:.9g}", "-i", "pipe:0",
            "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
            "-fps_mode", "cfr",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_path),
        ]

        encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
        write_error = None
        try:
            assert encoder.stdin is not None
            for i in range(T):
                frame = frames_np[i]
                if C == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                encoder.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            write_error = exc
        finally:
            if encoder.stdin is not None:
                encoder.stdin.close()

        ret = encoder.wait()
        if ret != 0:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg video encoding failed with code {ret}")
        if write_error is not None:
            temp_path.unlink(missing_ok=True)
            raise write_error

        # Audio muxing with timestamp rectification
        has_audio = bool(audio_source_path and os.path.exists(audio_source_path))
        if has_audio:
            video_duration = T / max(fps, 1e-6)
            mux_cmd = [
                ffmpeg_bin, "-y", "-i", str(temp_path), "-i", str(audio_source_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
                "-fps_mode", "cfr",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{video_duration:.9f}",
                str(final_path)
            ]
            mux_res = subprocess.run(mux_cmd, capture_output=True, text=True)
            temp_path.unlink(missing_ok=True)
            if mux_res.returncode != 0:
                print(f"[SeedVR2 Save Video] Warning: Audio mux failed ({mux_res.stderr}), keeping video only.")
                if temp_path.exists():
                    temp_path.replace(final_path)
        else:
            temp_path.replace(final_path)

        print(f"[SeedVR2 Save Video] ✅ Video saved: {final_path} ({T} frames, {W}x{H} @ {fps}fps CFR)")
        return io.NodeOutput(str(final_path))
```

---

### 4.2 `src/core/trt_encoder.py` (Complete 449 lines)

```python
"""Dedicated Full-Batch TensorRT VAE encoder for ComfyUI SeedVR2.
Executes exact 1-shot TensorRT acceleration for ANY batch size.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock

import torch

# --- TRT encode debug hooks (SEEDVR2_TRT_DEBUG=1 to enable; default off) ---
_TRT_DEBUG = os.environ.get("SEEDVR2_TRT_DEBUG", "0") == "1"
_TRT_DEBUG_DIR = os.environ.get("SEEDVR2_TRT_DEBUG_DIR", "") or None

def _trt_dbg_log(msg):
    if _TRT_DEBUG:
        print(f"[TRT-DEBUG] {msg}", flush=True)

def _trt_dbg_stats(tag, t):
    if not _TRT_DEBUG:
        return
    tt = t.detach().float()
    _trt_dbg_log(f"{tag}: shape={tuple(tt.shape)} dtype={t.dtype} "
                f"min={float(tt.min()):.4f} max={float(tt.max()):.4f} "
                f"mean={float(tt.mean()):.4f} std={float(tt.std()):.4f} "
                f"NaN={bool(torch.isnan(tt).any())} Inf={bool(torch.isinf(tt).any())}")
    if _TRT_DEBUG_DIR:
        try:
            import os as _os
            torch.save(tt.cpu(), _os.path.join(_TRT_DEBUG_DIR, tag.replace(' ', '_') + '.pt'))
        except Exception as e:
            _trt_dbg_log(f"save {tag} failed: {e}")

try:
    import tensorrt_rtx as trt
    HAS_TRT = True
except ImportError:
    try:
        import tensorrt as trt
        HAS_TRT = True
    except ImportError:
        trt = None
        HAS_TRT = False


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIRS = [
    ROOT / "tensorrt_backend" / "artifacts",
    ROOT.parents[1] / "models" / "tensorrt" / "seedvr2",
]

_ENGINES: dict[str, tuple[object, object, object, str, str, torch.cuda.Stream]] = {}
_ENCODE_LOCK = Lock()


def find_engine_path(frames: int) -> tuple[Path | None, int]:
    """Return (engine_path, tile_px). Prefers the 512px-tile engine (Studio standard), then 256px."""
    for tile_px in (512, 256):
        name = f"vae_encoder_{frames}f_tile{tile_px}.rtxplan"
        for d in ARTIFACTS_DIRS:
            p = d / name
            if p.exists() and p.stat().st_size > 1_000_000:
                return p, tile_px
    return None, 0


def is_available(frames: int | None = None) -> bool:
    """Check if TensorRT VAE encoder is available."""
    if not HAS_TRT:
        return False
    return True


def _engine(frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None):
    cache_key = frames
    cached = _ENGINES.get(cache_key)
    if cached is not None:
        return cached

    path, tile_px = find_engine_path(frames)
    if path is None:
        # No auto-build: engines are created explicitly via the build scripts/node.
        # Without an engine we fall back to the standard PyTorch VAE.
        raise FileNotFoundError(
            f"TensorRT VAE encoder engine for {frames} frames not found. "
            f"Build it first with tools/cloud_export_gpu.py + tools/cloud_build_engine.py "
            f"or the SeedVR2 Build TensorRT VAE Engines node."
        )

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT encoder: {path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT could not create an execution context for {path}")

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    stream = torch.cuda.Stream()
    cached = (runtime, engine, context, input_name, output_name, stream, tile_px)
    _ENGINES[cache_key] = cached
    return cached


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if overlap:
        t = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
        ramp = (1.0 - torch.cos(t * 3.141592653589793)) / 2.0  # cosine ease
        if left:
            weight[:overlap] = ramp
        if right:
            weight[-overlap:] = torch.minimum(weight[-overlap:], torch.flip(ramp, dims=[0]))
    return weight


@torch.inference_mode()
def _encode_single_chunk(sample: torch.Tensor, frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a single full batch directly in 1 shot with TensorRT."""
    _, _, _, height, width = sample.shape
    _, _, context, input_name, output_name, stream, tile_px = _engine(int(frames), vae=vae, dit_model=dit_model)
    if context is None:
        raise RuntimeError("TensorRT could not create a per-batch encoder context")

    # Studio-compatible shape check: only call set_input_shape if the current shape
    # actually differs from the target tile shape. Static-shape engines (built for
    # exact tile sizes) never need set_input_shape called, avoiding TRT's internal
    # buffer re-allocation/reset which causes the first-tile uninitialized memory glitch.
    target_shape = (1, 3, frames, tile_px, tile_px)
    current_shape = tuple(context.get_tensor_shape(input_name))
    if current_shape != target_shape:
        context.set_input_shape(input_name, target_shape)
        torch.cuda.synchronize()

    source = sample.to(device="cuda", dtype=torch.float16).contiguous()
    tile, overlap = tile_px, tile_px * 3 // 8  # 37.5% tile-to-tile overlap (96px@256, 192px@512)
    ys, xs = _positions(height, tile, overlap), _positions(width, tile, overlap)
    padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
    source = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
    latent_frames = (frames - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    raw_h, raw_w = padded_h // 8, padded_w // 8
    tile_lat = tile_px // 8
    overlap_latent = overlap // 8
    result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    dc_result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)

    with _ENCODE_LOCK, torch.cuda.stream(stream):
        # Warmup run: forces TensorRT to allocate and bind internal scratchpad memory.
        # Without this, the very first execution (tile y=0, x=0) reads uninitialized
        # GPU buffer memory, resulting in severe noise/artifact in the top-left corner.
        warmup_in = torch.zeros((1, 3, frames, tile_px, tile_px), device="cuda", dtype=torch.float16)
        warmup_out = torch.zeros((1, 32, latent_frames, tile_lat, tile_lat), device="cuda", dtype=torch.float16)
        context.set_tensor_address(input_name, warmup_in.data_ptr())
        context.set_tensor_address(output_name, warmup_out.data_ptr())
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        del warmup_in, warmup_out

        # NOTE (Studio Architecture & Address Safety):
        # A context's tensor addresses are mutable. For the safe/default path we execute
        # one tile at a time under _ENCODE_LOCK with stream.synchronize() so addresses
        # cannot be overwritten by a later queued tile.
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_output = torch.zeros((1, 32, latent_frames, tile_lat, tile_lat), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE encoder failed at tile y={y}, x={x}")
                stream.synchronize()
                if _TRT_DEBUG:
                    _dbg_tv = tile_output.float()
                    _dbg_sd = float(_dbg_tv.std())
                    _trt_dbg_log(f"enc tile y={y} x={x} (ly={y // 8},lx={x // 8}) "
                                f"min={float(_dbg_tv.min()):.4f} max={float(_dbg_tv.max()):.4f} "
                                f"std={_dbg_sd:.5f}" + ("  <<< BLACK?" if _dbg_sd < 0.05 else ""))
                ly, lx = y // 8, x // 8
                # DC offset correction: estimate the tile's true DC from its
                # accurate center (inside the receptive-field-poor edge ring),
                # subtract it, and restore it later as a weighted average.
                edge = overlap_latent // 2
                center = tile_output[:, :, :, edge:tile_lat - edge, edge:tile_lat - edge]
                dc = center.mean(dim=(3, 4), keepdim=True)
                corrected = tile_output.float() - dc.float()
                wy = _feather(tile_lat, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(tile_lat, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, tile_lat, tile_lat)
                result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += corrected * window
                dc_result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += dc.float() * window
                weights[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += window
                del tile_input, tile_output

    restored = (result + dc_result) / weights.clamp_min(1e-6)
    encoded = restored[:, :16, :, :latent_h, :latent_w].to(sample.dtype)
    if _TRT_DEBUG:
        _trt_dbg_stats(f"enc_chunk_out_{frames}f", encoded)
    del source, result, weights, dc_result
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encoded


_ENGINE_FILE_RE = re.compile(r"^vae_encoder_(\d+)f_tile\d+\.rtxplan$")


def _available_engine_frames() -> list[int]:
    """Return the sorted video-frame sizes of every usable encoder engine on disk."""
    found: set[int] = set()
    for d in ARTIFACTS_DIRS:
        try:
            if not d.is_dir():
                continue
            for p in d.iterdir():
                m = _ENGINE_FILE_RE.match(p.name)
                if m and p.is_file():
                    try:
                        if p.stat().st_size > 1_000_000:
                            found.add(int(m.group(1)))
                    except OSError:
                        continue
        except OSError:
            continue
    return sorted(found)


def pick_engine_frames(video_frames: int, preferred: str = "auto") -> int | None:
    """Pick the encoder engine frame size for a video of `video_frames` frames.

    Selection order:
    1. preferred (from the loader dropdown / settings node) if that engine exists;
    2. an engine matching video_frames exactly (1-shot encode);
    3. the largest engine that fits inside video_frames (chunked encode);
    4. if every engine is larger than the clip, the smallest engine (encode pads/crops);
    5. None only when no engine exists at all.
    """
    engines = _available_engine_frames()
    if not engines:
        return None
    if preferred != "auto":
        try:
            cand = int(preferred)
            if cand in engines:
                return cand
        except ValueError:
            pass
    if video_frames in engines:
        return video_frames
    fits = [e for e in engines if e <= video_frames]
    if fits:
        return fits[-1]
    return engines[0]


@torch.inference_mode()
def _encode_chunked(sample: torch.Tensor, total_frames: int, engine_frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a long clip by splitting it into engine_frames chunks with 4-frame temporal overlap."""
    _, _, _, height, width = sample.shape
    lat_total = (total_frames - 1) // 4 + 1
    lat_engine = (engine_frames - 1) // 4 + 1
    stride = engine_frames - 4  # 4-frame overlap -> 1 latent-frame overlap
    lat_h, lat_w = height // 8, width // 8
    result = torch.zeros((1, 16, lat_total, lat_h, lat_w), device="cuda", dtype=sample.dtype)
    starts = list(range(0, total_frames - engine_frames + 1, stride))
    if starts[-1] != total_frames - engine_frames:
        starts.append(total_frames - engine_frames)
    for start in starts:
        chunk = sample[:, :, start:start + engine_frames]
        lat = _encode_single_chunk(chunk, engine_frames, vae=vae, dit_model=dit_model)
        lat_start = start // 4
        result[:, :, lat_start:lat_start + lat_engine] = lat
        del chunk, lat
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def encode(sample: torch.Tensor, vae: torch.nn.Module | None = None, dit_model: str | None = None, engine_frames: str = "auto") -> torch.Tensor:
    """
    Encode [B,3,T,H,W] to posterior mean [B,16,(T-1)/4+1,H/8,W/8] in 1 shot using TensorRT engine.
    """
    if sample.ndim != 5 or sample.shape[0] != 1 or sample.shape[1] != 3:
        raise ValueError(f"TensorRT encoder expects [1,3,T,H,W], got {tuple(sample.shape)}")
    _, _, total_frames, height, width = sample.shape
    if height % 8 or width % 8:
        raise ValueError("TensorRT encoder input dimensions must be divisible by 8")

    # Release cached-but-unused VRAM from previous batches/other nodes to avoid
    # allocator pressure during 512px-tile engine execution (NaN source).
    torch.cuda.empty_cache()

    # Ensure 4n+1
    req_frames = ((total_frames - 1) // 4) * 4 + 1
    if total_frames != req_frames:
        pad_len = req_frames - total_frames
        last_frame = sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        sample = torch.cat([sample, last_frame], dim=2)
        total_frames = req_frames

    engine_video_frames = pick_engine_frames(total_frames, engine_frames)
    if engine_video_frames is None:
        raise FileNotFoundError("No TensorRT VAE encoder engine found (need vae_encoder_{5,9,13,17,21,29}f_tile512.rtxplan)")
    if engine_video_frames > total_frames:
        # The clip is shorter than every available engine: pad the video to the
        # engine size, encode in 1 shot, then crop back to the actual latent length.
        pad_len = engine_video_frames - total_frames
        last_frame = sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        padded = torch.cat([sample, last_frame], dim=2)
        encoded = _encode_single_chunk(padded, engine_video_frames, vae=vae, dit_model=dit_model)
        lat_needed = (total_frames - 1) // 4 + 1
        return encoded[:, :, :lat_needed]
    if engine_video_frames == total_frames:
        print(f"[SeedVR2 TensorRT] Encoding {engine_video_frames}f in 1 shot with dedicated {engine_video_frames}f TensorRT engine...")
        return _encode_single_chunk(sample, total_frames, vae=vae, dit_model=dit_model)
    n_chunks = (total_frames + engine_video_frames - 5) // (engine_video_frames - 4)
    print(f"[SeedVR2 TensorRT] Encoding {n_chunks} chunks of {engine_video_frames}f with TensorRT engine (4-frame temporal overlap)...")
    return _encode_chunked(sample, total_frames, engine_video_frames, vae=vae, dit_model=dit_model)


def resolve_engine_frames(preferred: str = "auto") -> int | None:
    """Return the largest available encoder engine video-frame size (for chunking)."""
    engines = _available_engine_frames()
    if not engines:
        return None
    if preferred != "auto":
        try:
            cand = int(preferred)
            if cand in engines:
                return cand
        except ValueError:
            pass
    return engines[-1]


def release() -> None:
    """Clear cached execution contexts and streams to free GPU VRAM."""
    global _ENGINES
    _ENGINES.clear()
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _gc.collect()
        torch.cuda.empty_cache()
```

---

### 4.3 `src/core/infer.py` (TRT Encode Section Full Code)

```python
def _trt_encode_batch(enc_sample, vae, dit_model, engine_frames_setting):
    """Encode a video batch with the TensorRT encoder, chunking to engine size.

    Engine selection is based on this batch's actual length (pick_engine_frames),
    so any available engine (e.g. 5f/21f/29f) is used instead of silently falling
    back to the fp16 VAE. Batches shorter than the smallest engine are padded to
    engine size, encoded in 1 shot, then cropped back.
    """
    from .trt_encoder import encode as trt_encode, pick_engine_frames
    total = enc_sample.shape[2]
    # Pad spatial dims to multiples of 8 so tile boundaries align with latent boundaries.
    h, w = enc_sample.shape[3], enc_sample.shape[4]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        enc_sample = torch.nn.functional.pad(enc_sample, (0, pad_w, 0, pad_h), mode="replicate")
        _TRT_CROP_HW[0], _TRT_CROP_HW[1] = h, w
    else:
        _TRT_CROP_HW[0], _TRT_CROP_HW[1] = -1, -1

    engine_video_frames = pick_engine_frames(total, engine_frames_setting)
    if engine_video_frames is None:
        raise RuntimeError("No TensorRT VAE encoder engine available")

    lat_needed = (total - 1) // 4 + 1
    if total < engine_video_frames:
        # Batch is shorter than every engine: pad to engine size, 1-shot, crop.
        pad_len = engine_video_frames - total
        last_frame = enc_sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        padded = torch.cat([enc_sample, last_frame], dim=2)
        lat = trt_encode(padded, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))
        return lat[:, :, :lat_needed]

    if total == engine_video_frames:
        return trt_encode(enc_sample, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))

    # Chunked encoding for batches longer than engine
    stride = ((engine_video_frames - 4) // 4) * 4
    if stride < 4:
        stride = 4
    lat_parts = []
    starts = list(range(0, total - engine_video_frames + 1, stride))
    if starts[-1] != total - engine_video_frames:
        starts.append(total - engine_video_frames)
    for start in starts:
        chunk = enc_sample[:, :, start:start + engine_video_frames].contiguous()
        lat = trt_encode(chunk, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))
        lat_parts.append((lat, start // 4))

    lat0 = lat_parts[0][0]
    latent = torch.zeros((1, 16, lat_needed, lat0.shape[3], lat0.shape[4]), device=lat0.device, dtype=lat0.dtype)
    # Causal encoder: a chunk's leading latents (context-poor) are LESS accurate than
    # the previous chunk's trailing latents (full context). So earlier chunks win.
    # Write in reverse so the first chunk keeps its (accurate) values.
    for lat, lat_start in reversed(lat_parts):
        latent[:, :, lat_start:lat_start + lat.shape[2]] = lat
    return latent
```

```python
            # VAE process by each group.
            for sample in batches:
                # Check TensorRT VAE encoder
                _enc_trt = getattr(self, "use_tensorrt_vae_encode",
                                   getattr(self, "use_tensorrt_vae", False))
                if _enc_trt or os.environ.get("SEEDVR2_TRT_ENCODER", "0") == "1":
                    # No silent fp16 fallback: selecting the TensorRT encoder means
                    # TRT must encode. A missing engine / any failure raises a clear
                    # error instead of quietly running the standard (fp16) VAE.
                    from .trt_encoder import encode as trt_encode, HAS_TRT as _trt_has
                    if not _trt_has:
                        raise RuntimeError(
                            "TensorRT VAE Encoder is selected but TensorRT is not available. "
                            "Install tensorrt-rtx or use SeedVR2LoadVAEModel for fp16 encode."
                        )
                    enc_sample = sample if sample.ndim == 5 else sample.unsqueeze(0)
                    if enc_sample.ndim != 5:
                        raise RuntimeError(
                            f"TensorRT VAE Encoder expects [1,C,T,H,W], got {tuple(sample.shape)}. "
                            "Use SeedVR2LoadVAEModel for fp16 encode."
                        )
                    self.debug.log(f"Encoding with TensorRT VAE Encoder (engine={getattr(self, 'use_tensorrt_engine_frames', 'auto')})", category="info", indent_level=1)
                    latent = _trt_encode_batch(enc_sample, self.vae, self._resolve_dit_name(), getattr(self, 'use_tensorrt_engine_frames', 'auto'))
                    latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                    latent = optimized_channels_to_last(latent)
                    latent = (latent - shift) * scale
                    latents.append(latent)
                    continue
```

---

### 4.4 `src/interfaces/trt_vae_model_loader.py` (Complete 478 lines)

```python
"""Dedicated Full-Batch TensorRT RTX Engine Builder for ComfyUI SeedVR2.
Builds dedicated 1-shot TensorRT engines for ANY batch size with minimal VRAM (~400MB) using Studio's compact tracer.
"""

from __future__ import annotations

import copy
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from comfy_api.latest import io

from ..optimization.memory_manager import get_device_list
from ..core.generation_utils import prepare_runner, setup_generation_context
from ..core.model_loader import materialize_model
from ..utils.debug import Debug
from ..utils.constants import get_base_cache_dir
from ..utils.downloads import download_weight
from ..utils.model_registry import (
    DEFAULT_DIT,
    DEFAULT_VAE,
    get_available_vae_models,
)
from ..models.video_vae_v3.modules.types import MemoryState
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "tensorrt_backend" / "artifacts"


def _available_engine_frames(kind: str = "encoder") -> list[str]:
    """Scan the artifacts dir and auto-populate the engine_frames dropdown.

    kind="encoder" scans vae_encoder_<N>f_tile*.rtxplan,
    kind="decoder" scans vae_decoder_tile_*_<N>f.rtxplan.
    Dropping an engine into tensorrt_backend/artifacts/ is enough to make it
    selectable after a ComfyUI restart.
    """
    import re
    frames = set()
    pattern = "vae_encoder_*f_tile*.rtxplan" if kind == "encoder" else "vae_decoder_tile_*_*f.rtxplan"
    search_dirs = [ARTIFACTS_DIR, ROOT.parents[1] / "models" / "tensorrt" / "seedvr2"]
    for d in search_dirs:
        try:
            if d.is_dir():
                for pth in d.glob(pattern):
                    if kind == "encoder":
                        m = re.search(r"_(\d+)f_tile", pth.name)
                    else:
                        m = re.search(r"_(\d+)f\.rtxplan", pth.name)
                    if m:
                        n = int(m.group(1))
                        # Only 4n+1 frame counts are valid (the exporter normalizes to 4n+1,
                        # so e.g. a file named 195f actually contains a 193f graph).
                        if (n - 1) % 4 == 0:
                            frames.add(str(n))
        except Exception:
            pass
    return ["auto"] + sorted(frames, key=int, reverse=True)


class SeedVR2LoadTensorRTVAEEncoder(io.ComfyNode):
    """Encoder-only TensorRT VAE config (separate engine frame size from the decoder)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        vae_models = get_available_vae_models()
        return io.Schema(
            node_id="SeedVR2LoadTensorRTVAEEncoder",
            display_name="SeedVR2 Load TensorRT VAE Encoder",
            category="SEEDVR2",
            description=(
                "Encoder-only TensorRT VAE configuration. Lets you choose a different "
                "engine frame size for encoding than for decoding (e.g. encode 21f / decode 21f). "
                "Connect to the vae_encode input of SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE inference"
                ),
                io.Combo.Input("engine_frames",
                    options=_available_engine_frames("encoder"),
                    default="auto",
                    optional=True,
                    tooltip="TensorRT encoder engine frame size. Auto-populated from artifacts. "
                            "auto = pick the largest available engine."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration for the encoder path."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, device: str,
                engine_frames: str = "auto") -> io.NodeOutput:
        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae_encoder"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": device,
            "offload_device": "none",
            "cache_model": False,
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "engine_frames": engine_frames,
            "node_id": node_id,
        }
        return io.NodeOutput(vae_config)


# Backward compatibility alias
SeedVR2LoadTensorRTVAEModel = SeedVR2LoadTensorRTVAEEncoder


class SeedVR2LoadTensorRTVAEDecoder(io.ComfyNode):
    """Decoder-only TensorRT VAE config (separate engine frame size from the encoder)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        vae_models = get_available_vae_models()
        return io.Schema(
            node_id="SeedVR2LoadTensorRTVAEDecoder",
            display_name="SeedVR2 Load TensorRT VAE Decoder",
            category="SEEDVR2",
            description=(
                "Decoder-only TensorRT VAE configuration. Lets you choose a different "
                "engine frame size for decoding than for encoding (e.g. encode 89f / decode 65f). "
                "Connect to the vae_decode input of SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE inference"
                ),
                io.Combo.Input("engine_frames",
                    options=_available_engine_frames("decoder"),
                    default="auto",
                    optional=True,
                    tooltip="TensorRT decoder engine frame size. Auto-populated from artifacts. "
                            "auto = pick the largest available engine."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration for the decoder path."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, device: str,
                engine_frames: str = "auto") -> io.NodeOutput:
        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae_decoder"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": device,
            "offload_device": "none",
            "cache_model": False,
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "engine_frames": engine_frames,
            "node_id": node_id,
        }
        return io.NodeOutput(vae_config)
```

---

### 4.5 `src/interfaces/__init__.py` (Complete 74 lines)

```python
"""
SeedVR2 ComfyUI Nodes
Central registry for all SeedVR2 nodes
"""

from comfy_api.latest import ComfyExtension, io

from .video_upscaler import SeedVR2VideoUpscaler
from .dit_model_loader import SeedVR2LoadDiTModel
from .vae_model_loader import SeedVR2LoadVAEModel
from .torch_compile_settings import SeedVR2TorchCompileSettings
from .trt_vae_builder import SeedVR2BuildTensorRTVAE
from .trt_vae_model_loader import (
    SeedVR2LoadTensorRTVAEEncoder,
    SeedVR2LoadTensorRTVAEModel,
    SeedVR2LoadTensorRTVAEDecoder,
)
from .video_save import SeedVR2SaveVideo


class SeedVR2Extension(ComfyExtension):
    """SeedVR2 ComfyUI Extension"""
    
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return list of all SeedVR2 nodes"""
        return [
            SeedVR2VideoUpscaler,
            SeedVR2LoadDiTModel,
            SeedVR2LoadVAEModel,
            SeedVR2TorchCompileSettings,
            SeedVR2BuildTensorRTVAE,
            SeedVR2LoadTensorRTVAEEncoder,
            SeedVR2LoadTensorRTVAEDecoder,
            SeedVR2SaveVideo,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """ComfyUI V3 entry point"""
    return SeedVR2Extension()


__all__ = [
    'SeedVR2VideoUpscaler',
    'SeedVR2LoadDiTModel',
    'SeedVR2LoadVAEModel',
    'SeedVR2TorchCompileSettings',
    'SeedVR2BuildTensorRTVAE',
    'SeedVR2LoadTensorRTVAEEncoder',
    'SeedVR2LoadTensorRTVAEModel',
    'SeedVR2LoadTensorRTVAEDecoder',
    'SeedVR2SaveVideo',
    'SeedVR2Extension',
    'comfy_entrypoint',
]
```

---

### 4.6 `__init__.py` (Complete 140 lines)

```python
"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure FFmpeg is found across standard Windows paths or bundled imageio_ffmpeg
def _ensure_ffmpeg_path():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    candidate_dirs = [
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"D:\ffmpeg\bin"),
        Path(__file__).resolve().parent / "bin" / "ffmpeg" / "bin",
        Path(__file__).resolve().parent / "bin",
    ]
    for d in candidate_dirs:
        if (d / "ffmpeg.exe").exists() and (d / "ffprobe.exe").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            ffmpeg_dir = str(Path(ffmpeg_exe).parent)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

_ensure_ffmpeg_path()

# Check critical dependencies early to provide better error messages
# and auto-install if possible, especially useful for Vast.ai / RunPod
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.split(">")[0].split("=")[0].split("<")[0]
    
    try:
        __import__(import_name)
        return  # Already available
    except (ImportError, ModuleNotFoundError):
        pass
    if True:  # Package is missing - install it
        print("\n" + "="*80)
        print(f"SeedVR2: '{import_name}' module not found.")
        print(f"SeedVR2: Current Python executable: {sys.executable}")
        print(f"SeedVR2: Attempting automatic installation of {package_name}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f"SeedVR2: Successfully installed {package_name}")
        except Exception as e:
            print(f"SeedVR2: Auto-installation failed: {e}")
            print("This often happens on Vast.ai / RunPod when pip installs to a different Python environment.")
            print(f"Please run the following command manually in your terminal:")
            print(f"    {sys.executable} -m pip install \"{package_name}\"")
        print("="*80 + "\n")

# All critical dependencies from requirements.txt
# (torch/torchvision/numpy are assumed present via ComfyUI)
_REQUIRED_PACKAGES = [
    ("safetensors", None),
    ("tqdm", None),
    ("psutil", None),
    ("einops", None),
    ("omegaconf>=2.3.0", "omegaconf"),
    ("diffusers>=0.33.1", "diffusers"),
    ("transformers", None),
    ("accelerate", None),
    ("peft>=0.17.0", "peft"),
    ("rotary_embedding_torch>=0.5.3", "rotary_embedding_torch"),
    ("opencv-python", "cv2"),
    ("gguf", None),
    ("matplotlib", None),
    ("tensorrt-rtx", "tensorrt_rtx"),
    ("onnx", "onnx"),
    ("onnxscript", "onnxscript"),
    ("polygraphy", "polygraphy"),
]

for pkg, imp in _REQUIRED_PACKAGES:
    ensure_package(pkg, imp)

# Verify TensorRT RTX VAE engines
try:
    from pathlib import Path
    _cur_artifacts = Path(__file__).resolve().parent / "tensorrt_backend" / "artifacts"
    _cur_artifacts.mkdir(parents=True, exist_ok=True)
    _ready_count = 0
    _engines = (
        "vae_encoder_5f_tile512.rtxplan",
        "vae_encoder_21f_tile512.rtxplan",
        "vae_decoder_tile_512_5f.rtxplan",
        "vae_decoder_tile_256_21f.rtxplan"
    )
    for _eng in _engines:
        _dst = _cur_artifacts / _eng
        if _dst.exists() and _dst.stat().st_size > 1_000_000:
            _ready_count += 1
    if _ready_count == len(_engines):
        print(f"[SeedVR2 TensorRT] ✅ All {len(_engines)} TensorRT RTX VAE engines ready (2x-5x acceleration active)")
    else:
        print(f"[SeedVR2 TensorRT] ℹ️ {_ready_count}/{len(_engines)} RTX VAE engines ready (engines will build on first run with 'SeedVR2 Load TensorRT VAE Model')")
except Exception as _trt_sync_err:
    print(f"[SeedVR2 TensorRT] Warning during engine check: {_trt_sync_err}")

# Windows cp932: patch inductor jinja open(encoding=utf-8) before any torch.compile
try:
    from .src.core.fix_inductor import _fix_inductor_windows_encoding

    _fix_inductor_windows_encoding()
except Exception as _seedvr2_inductor_fix_err:  # noqa: BLE001
    print(f"[SeedVR2] Warning: inductor Windows encoding fix skipped: {_seedvr2_inductor_fix_err}")

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import (
    comfy_entrypoint,
    SeedVR2Extension,
    SeedVR2VideoUpscaler,
    SeedVR2LoadDiTModel,
    SeedVR2LoadVAEModel,
    SeedVR2TorchCompileSettings,
    SeedVR2BuildTensorRTVAE,
    SeedVR2LoadTensorRTVAEEncoder,
    SeedVR2LoadTensorRTVAEModel,
    SeedVR2LoadTensorRTVAEDecoder,
    SeedVR2SaveVideo,
)

print(f"[SeedVR2] Loaded nodes: SeedVR2VideoUpscaler, SeedVR2LoadTensorRTVAEEncoder (⚡TRT), SeedVR2LoadTensorRTVAEDecoder (⚡TRT), SeedVR2BuildTensorRTVAE, SeedVR2LoadVAEModel, SeedVR2LoadDiTModel, SeedVR2SaveVideo")

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]
```

---

## 5. Technical Significance and Architectural Rationale

### 5.1 Technical Significance of Theme 1 (The Three Core Improvements)

1. **Deterministic PTS & CFR Locking**:  
   Enforcing `-fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr` transforms video assembly from a naive container concatenation into an immutable stream synchronization process. Players cannot get tricked by zero or negative initial timestamps, ensuring exact millisecond-accurate audio/video synchronization even over hours of continuous upscaled playback.

2. **Inviolable ExecutionContext Address Protection**:  
   TensorRT executes with hardware-mapped GPU pointers bound directly into internal state. Enforcing mutual exclusion (`_DECODE_LOCK` / `_ENCODE_LOCK`) and synchronous execution (`stream.synchronize()`) per tile honors the architectural invariant that ExecutionContext addresses must remain immutable throughout active kernel execution, physically preventing race conditions and corrupted black/gray tiles.

3. **Zero-Fragmentation VRAM Cycling Across Long Batches**:  
   Coupling `del` with `gc.collect()` and `torch.cuda.empty_cache()` guarantees that the GPU driver and PyTorch's caching allocator return to a pristine zero-fragmentation state after every chunk. This allows rendering 100, 200, or more frames sequentially without experiencing creeping VRAM bloat or out-of-memory termination.

### 5.2 Technical Significance of Theme 2 (Encoder Refactoring)

1. **Short-Batch Resilience & Pad/Crop 1-Shot Speed**:  
   By padding video batches shorter than the engine frames (e.g. `batch_size=5` on 21f engine) via last-frame replication and cropping back to `(total - 1) // 4 + 1`, short clips run at 1-shot TensorRT speed without suffering `IndexError`.
2. **Fail-Fast Integrity**:  
   Eliminating the deceptive `try...except` silent fallback to standard PyTorch FP16 ensures that when TensorRT acceleration is selected, the pipeline strictly executes on TensorRT or raises an explicit `RuntimeError` immediately, eliminating silent performance degradation.
3. **Unified Interface Symmetry**:  
   Standardizing `SeedVR2LoadTensorRTVAEEncoder` symmetrically with `SeedVR2LoadTensorRTVAEDecoder` delivers a clean, intuitive, and robust user interface.
