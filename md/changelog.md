<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/zhmd/changelogzh.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# Changelog

## v1.7.1 — 2026-10-28
- **Summary:** TensorRT RTX dependency stack update:
  - **`tensorrt-rtx` 1.6.1.120 → 1.7.1.116:** Updated the TensorRT RTX runtime to 1.7.1. Engines built with 1.6.1 cannot be deserialized by 1.7.1 (`Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed`), so existing `.rtxplan` engines must be rebuilt after upgrading.
  - **`onnx` 1.22.0 → 1.23.2:** Updated ONNX to 1.23.2.
  - **`onnxscript` 0.7.1 → 0.7.2:** Updated onnxscript to 0.7.2.
  - **`polygraphy` 0.53.4 → 0.53.6:** Updated Polygraphy to 0.53.6.
  - **Pinned Versions Enforced in `install.py`:** `ensure_package()` previously only checked whether the module could be imported, so the `requirements-windows-cu132.txt` pins were silently ignored when an older build was already installed. The installer now passes the pinned versions (`tensorrt-rtx==1.7.1.116`, `onnx==1.23.2`, `onnxscript==0.7.2`, `polygraphy==0.53.6`) explicitly.
- **Technical Details:** See [v1.7.1 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.7.1) for complete explanation

## v1.7.0 — 2026-10-23
- **Summary:** Added `alignment_mode` resolution grid alignment selector to SeedVR2 Video Upscaler:
  - **`alignment_mode` Input Selector:** Introduced a dedicated combo input (`pad (lossless)`, `crop_16`, `crop_32`, `crop_64`) on `SeedVR2 Video Upscaler` to control how non-standard resolutions and arbitrary aspect ratios are aligned to the model's spatial divisible grid:
    - **`pad (lossless)` (Default):** Applies reflection padding (`DivisiblePad`) up to a 16-multiple without altering the native image content, restoring original dimensions losslessly after decoding.
    - **`crop_16` (Original SeedVR2 Behavior):** Truncates odd remainder dimensions to the nearest multiple of 16 using center-crop (`DivisibleCrop(16)`), completely eliminating padding-induced edge artifacts on custom aspect ratios by eliminating padding altogether.
    - **`crop_32` / `crop_64`:** Center-crops odd dimensions to multiples of 32 or 64 for strict alignment constraints.
- **Technical Details:** See [v1.7.0 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.7.0) for complete explanation

## v1.6.9 — 2026-10-23
- **Summary:** Systematic VRAM contamination elimination & deep memory reclamation architecture:
  - **Pre- and Post-Execution ComfyUI Model & Cache Purge:** Integrated explicit model eviction and cache drainage (`mm.cleanup_models_gc()`, `mm.cleanup_models()`, `mm.soft_empty_cache(force=True)`, and `torch.cuda.synchronize()`) before and after upscaler execution in `video_upscaler.py`. Completely purges lingering resident models and tensor remnants left behind by predecessor nodes in the workflow.
  - **Per-Batch Deep Memory Cleansing Across All 4 Phases:** Embedded a four-tier deep purge sequence (`_gc.collect()` → `mm.soft_empty_cache(force=True)` → `torch.cuda.synchronize()` → `torch.cuda.empty_cache()`) immediately after every single batch completion across all four pipeline phases in `generation_phases.py` (Phase 1 Encode, Phase 2 DiT Upscale, Phase 3 VAE Decode, and Phase 4 Post-process), eliminating inter-batch memory fragmentation and cumulative tensor retention.
  - **TensorRT VAE ComfyUI Memory Manager Synchronization:** Harmonized PyTorch CUDA caching allocator purges with ComfyUI's model management cache drainage (`mm.soft_empty_cache(force=True)`) after every chunk in `trt_decoder.py` and `trt_encoder.py`.
  - **Core Memory Manager Unification:** Integrated `mm.soft_empty_cache(force=True)` directly into `clear_memory()` within `src/optimization/memory_manager.py`.
- **Technical Details:** See [v1.6.9 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.9) for complete explanation

## v1.6.8 — 2026-10-23
- **Summary:** Removal of the SpargeAttn-hswq (`spargeattn`) attention backend from the DiT loader nodes:
  - **`spargeattn` Option Removed:** The `spargeattn` attention backend produced vertical seam noise in the upscaled output at high target resolutions, while `sdpa`, `flash_attn_2`, `flash_attn_3`, `sageattn_2`, and `sageattn_3` were unaffected. The backend has therefore been withdrawn, and `spargeattn` removed from the `attention_mode` combo on both DiT loaders (`SeedVR2 (Down)Load DiT Model` and `SeedVR2 (Down)Load DiT Model with Distorch2`).
  - **`sparge_topk` Control Removed:** The associated `sparge_topk` KV keep-ratio input, together with its configuration wiring, has been removed from both loader nodes.
  - **Remaining Attention Backends:** `sdpa` / `flash_attn_2` / `flash_attn_3` / `sageattn_2` / `sageattn_3`.
- **Technical Details:** See [v1.6.8 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.8) for complete explanation

## v1.6.7 — 2026-10-12
- **Summary:** Elimination of right/bottom-edge checkerboard noise via optional adaptive safe padding & TensorRT VAE cross-stream synchronization:
  - **Optional Adaptive Safe Padding for DiT 720P Window Attention:** Resolved an issue where specific portrait/custom aspect ratios (such as 560×938 2x upscaling) generated runt window slivers (e.g. width remainder of only 2 latent pixels against the 23-pixel window size). Isolated slivers lacked spatial context, causing localized self-attention degradation and checkerboard noise along the right edge and bottom-right corner.
  - **`safe_padding` Node Switch on SeedVR2 Video Upscaler:** Exposed a dedicated `safe_padding` boolean input (default: `False`). When enabled, `DivisiblePad` dynamically evaluates DiT 720P window geometry and pads an additional 16-multiple safe margin when runt slivers (< 6 latent px) are detected, seamlessly cropping back to target dimensions after decoding. Standard resolutions (1280x720, 1920x1080) and memory-constrained environments can keep it disabled (`False`) to ensure strictly minimal VRAM usage.
  - **TensorRT VAE Cross-Stream Synchronization:** Implemented explicit CUDA stream synchronization (`wait_stream`) between the PyTorch caller stream and the dedicated TensorRT execution stream in `trt_decoder.py` and `trt_encoder.py`. Guarantees memory ordering between pre-allocated tile buffers and asynchronous tile accumulation before final division, eliminating race conditions across tile boundaries.
- **Technical Details:** See [v1.6.7 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.7) for complete explanation

## v1.6.6 — 2026-10-11
- **Summary:** Full native support for ConvRot W4A8 DiT models & DisTorch2 compatibility scope:
  - **Full Support for 3B / 7B / 7B Sharp ConvRot W4A8 Models:** Complete support across model registry, automated download pipeline, and native inference for `seedvr2_3b_convrot_w4a8.safetensors`, `seedvr2_7b_convrot_w4a8.safetensors`, and `seedvr2_7b_sharp_convrot_w4a8.safetensors`. The model loader dropdown exclusively registers canonical `convrot` model names to eliminate redundant redownloads, with backwards-compatible alias resolution for legacy workflows.
  - **W4A8 DisTorch2 Compatibility Scope:** Asymmetric W4A8 quantized models are strictly supported via the standard DiT loader (`SeedVR2 (Down)Load DiT Model`). DisTorch2 (`SeedVR2 (Down)Load DiT Model with Distorch2`) does not support W4A8 models due to blockwise CPU RAM streaming and weight-dispatch constraints specific to asymmetric 4-bit packed weights.
- **Technical Details:** See [v1.6.6 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.6) for complete explanation

## v1.6.5 — 2026-10-10
- **Summary:** TensorRT VAE fallback elimination & engine build ONNX auto-cleanup:
  - **Eliminate PyTorch VAE Fallback in SeedVR2 Video Upscaler:** Fixed an issue where connecting only one VAE input (`vae_decode` or `vae_encode`) left the other as an empty configuration, unintentionally causing the encoder to drop back to the standard PyTorch FP16 VAE and triggering massive CUDA out-of-memory errors on high resolutions (e.g. 4208px). The node now automatically mirrors configurations between encode and decode if only one is connected. Furthermore, when TensorRT is active on either endpoint, both encoding and decoding are strictly locked to TensorRT with synchronized runner flags, completely eliminating silent FP16 fallbacks.
  - **Automatic Intermediate ONNX Cleanup on Engine Build:** In the `SeedVR2 Build TensorRT VAE Engines` node and loader compilation routines, intermediate `.onnx` and companion `.onnx.data` files are now automatically purged immediately after the TensorRT `.rtxplan` engine has been successfully generated, saving gigabytes of disk space per build.
  - **Node Display Name Cleanup:** Streamlined node title to `SeedVR2 Video Upscaler` by removing legacy version suffixes from node registration.
- **Technical Details:** See [v1.6.5 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.5) for complete explanation

## v1.6.4 — 2026-10-10
- **Summary:** Updated auto-download repository for ConvRot INT8 and w4a8 DiT models:
  - **ConvRot INT8 & w4a8 Model Registry Migration:** Updated the auto-download source repository for all 3B / 7B / 7B sharp ConvRot INT8 and asym w4a8 INT8 DiT models to [`ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8`](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8).
  - **Direct Root-Level File Resolution:** Aligned remote download filenames directly with root-level repository assets (`seedvr2_3b_convrot_int8.safetensors`, `seedvr2_7b_convrot_int8.safetensors`, `seedvr2_7b_sharp_convrot_int8.safetensors`, and respective w4a8 counterparts) with SHA256 integrity pinning, removing unnecessary remote subfolder indirection.
- **Technical Details:** See [v1.6.4 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.4) for complete explanation

## v1.6.3 — 2026-10-09
- **Summary:** Dynamic `triton-windows` resolution & unpinning:
  - **Dynamic Triton resolution:** Removed hardcoded version pin (`triton-windows==3.5.1.post24`) from `requirements-windows-cu132.txt`. The installer and runtime setup now dynamically resolve and install the latest available `triton-windows` release from PyPI, automatically adapting to newer PyTorch runtimes and CUDA environments without version locks.
- **Technical Details:** See [v1.6.3 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.3) for complete explanation

## v1.6.2 — 2026-10-08
- **Summary:** norm bf16 mode now available on the standard (legacy) DiT loader:
  - **norm_bf16 on SeedVR2 (Down)Load DiT Model:** The `norm_bf16` switch, previously exclusive to
    the DisTorch2 loader (v1.6.0), is now implemented on the standard `SeedVR2 (Down)Load DiT Model`
    node as well. OFF (default) = stock fp32 norm path (quality-priority; behaviour unchanged).
    ON = run RMS/QK norm in bf16 during Phase 2, saving significant resident VRAM at the cost of
    bf16 rounding (per-pixel PSNR ~37–39 dB vs fp32).
  - Both loaders share the same switch semantics and the same DiT-side bf16 norm path, so the
    behaviour is identical regardless of which loader node is used.
- **Technical Details:** See [v1.6.2 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.2) for complete explanation

## v1.6.1 — 2026-10-05
- **Summary:** SpargeAttn speed work since v1.6.0 — the spargeattn backend now ships its full per-call path without the fixed overheads that had kept it slower than SageAttention2:
  - **Sage2++ fp16-accumulate path:** fixed the ragged-window path and routed spargeattn through the SageAttention2++ fp16-accumulate kernel (the evolved form), instead of falling through to the slower f32-accumulate kernel.
  - **Per-equal-length window batching (zero-copy):** consecutive windows of equal length are batched into the stock API call, removing per-window Python/launch overhead without changing results.
  - **Per-call D2H sync elimination:** removed the per-call host synchronization in the spargeattn fast path (plan-cache keyed by shape/numel), so the fast path issues zero device-to-host syncs.
  - **Variable-length (varlen) delegation:** spargeattn varlen handling now delegates to the SpargeAttn-hswq library's new `varlen` entry point (>= 1.2.1). Uniform windows keep a single zero-copy batched launch; mixed-length windows are bucketed by identical length inside the library (one launch per distinct length, no padding waste, no per-sequence launches). The fixed-length premise of the kernels is never violated.
  - **Note:** an experimental persistent Triton-cache feature was tried and reverted; the spargeattn path is left in its measured-correct state (no persistent-cache instrumentation).
- **Technical Details:** See [v1.6.1 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.1) for complete explanation

## v1.6.0 — 2026-10-03
- **Summary:** New DisTorch2 DiT loader node and Phase 2 VRAM controls:
  - **SeedVR2 (Down)Load DiT Model with Distorch2:** New node that hosts the entire (quantized) DiT
    in system RAM and streams it to the compute device during denoising, using a vendored DisTorch2
    backend (verbatim copy of ComfyUI-MultiGPU by pollockjj, GPL-3.0; see the Credits/License
    sections). Adds virtual-VRAM budget, donor device, per-block allocation string, and model-eject
    controls, with the same `SEEDVR2_DIT` output type as the standard loader.
  - **norm_bf16 toggle:** New per-node switch for RMS/QK norm precision during Phase 2.
    OFF (default) = stock fp32 path, bit-identical output. ON = bf16 norm path, saving resident
    VRAM at the cost of bf16 rounding (per-pixel PSNR ~37–39 dB vs fp32).
  - **emb_repeat_nocache toggle:** Disable the emb_repeat cache during Phase 2 (recompute each use;
    saves resident VRAM; output bit-identical).
  - **Model registry:** Removed the fp8 DiT entries (3B / 7B / 7B sharp); default DiT is now
    `seedvr2_7b_int8_convrot.safetensors`.
- **Technical Details:** See [v1.6.0 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.6.0) for complete explanation

Fork release history.

## v1.5.8 — 2026-10-02
- **Summary:** TensorRT VAE weight-precision optimization (decoder + encoder):
  - **Weights-Only FP16 Conversion:** TRT VAE Decoder/Encoder non-engine fallback paths now cast
    engine weights to FP16 while keeping `result` accumulation buffers in FP32 and preserving the
    4D output contract, cutting weight-resident VRAM without touching output precision.
  - **Verified Parity:** Repeated-run stability confirmed on RTX 5060 Ti — decoder PSNR 84–85 dB,
    encoder PSNR 72.4 dB against the FP32 baseline.
- **Technical Details:** See [v1.5.8 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.8) for complete explanation

## v1.5.7 — 2026-09-30
- **Summary:** DiT execution VRAM spike mitigation and activation memory stabilization:
  - **SDPA Output Buffer Pre-allocation:** Replaced slice accumulation in a Python list and final `torch.cat()` with a single pre-allocated output buffer and direct slice writes, eliminating the 2x VRAM duplication spike and unnecessary CPU-GPU synchronizations in `pytorch_varlen_attention`.
  - **Chunked SwiGLU MLP Forward:** Implemented sequence chunking along the token dimension for sequences exceeding 8,192 tokens in `SwiGLUMLP`, capping simultaneous intermediate `gate`, `up`, and `hidden` tensor allocations and slashing peak activation VRAM from ~2.0 GB down to ~330 MB per transformer block with bit-exact parity.
  - **Memory-Efficient Gathering via `torch.index_select`:** Replaced advanced fancy indexing `torch.cat([vid, txt])[tgt_idx]` with dedicated CUDA kernel `torch.index_select` across text token replication in Swin Window Attention, avoiding implicit buffer duplicates.
  - **Euler Condition Buffer In-Place Reuse:** Pre-allocated the 33-channel condition tensor buffer once prior to the sampling loop, reusing it via in-place slice writes across all diffusion steps to eliminate repetitive per-step dynamic allocations and allocator fragmentation.
- **Technical Details:** See [v1.5.7 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.7) for complete explanation

## v1.5.6 — 2026-09-29
- **Summary:** Download-policy correction and model-registry updates:
  - **Loader-Only Model Downloads:** Removed all forced auto-downloads. Models are now downloaded only when a model is selected in a loader node and the file is missing — the installer no longer pre-downloads the default models on install/update, and the TensorRT VAE engine build no longer fetches the default DiT ([#2](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/issues/2)).
  - **Six ConvRot INT8 / NVFP4 DiT Models Registered:** `seedvr2_3b_int8_convrot`, `seedvr2_3b_nvfp4`, `seedvr2_7b_int8_convrot`, `seedvr2_7b_nvfp4`, `seedvr2_7b_sharp_int8_convrot`, and `seedvr2_7b_sharp_nvfp4` (hosted on `Comfy-Org/SeedVR2`, SHA256-pinned) added to the model registry and download lists.
  - **GGUF Entries Removed:** GGUF (Q4_K_M / Q8_0) DiT entries removed from the model registry.
- **Technical Details:** See [v1.5.6 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.6) for complete explanation

## v1.5.5 — 2026-09-28
- **Summary:** Production reliability improvements and comprehensive TensorRT VAE Encoder refactoring:
  - **FFmpeg CFR & Timestamp Rectification:** Eliminated video/audio desynchronization and endpoint freeze during video assembly.
  - **ExecutionContext Address Safety:** Enforced lock and stream synchronization to prevent memory overwrite and black tile corruption.
  - **Deterministic Memory Cleanup:** Applied tri-partite reclamation (`del`, `gc`, `empty_cache`) across loops to prevent VRAM fragmentation.
  - **TensorRT VAE Encoder Refactoring:** Ported v1.5.4 artifact/VRAM fixes, added short-batch pad & crop 1-shot execution, removed silent FP16 fallbacks, and symmetrically added `SeedVR2LoadTensorRTVAEEncoder`.
- **Technical Details:** See [v1.5.5 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.5) for complete explanation



## v1.5.4 — 2026-09-27
- **Summary:** Completely resolved top-left mosaic/checkerboard decoding artifacts in the TensorRT VAE Decoder on landscape videos without VRAM inflation:
  - **Studio-Compatible Static Shape Check:** Only invoke `context.set_input_shape` when the current shape actually differs from the target tile shape. Static-shape engines now bypass redundant reconfigurations, preventing TRT's internal scratchpad buffer reallocations that previously ingested dirty VRAM.
  - **Deterministic Dummy Warmup Execution:** Executed a single zero-filled dummy inference pass before the spatial tiling loop. This forces TensorRT to sanitize all internal convolution workspaces and temporal accumulator lines, eliminating dirty memory reads on the first tile (`y=0, x=0`).
  - **Zero VRAM Bloat Architecture:** Avoided spatial outer padding that would otherwise inflate float32 accumulation buffers (`result` and `weights`) by 2x–3x VRAM, preserving native resolution decoding speed and minimal VRAM consumption.
- **Technical Details:** See [v1.5.4 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.4) for complete explanation

## v1.5.3 — 2026-09-09
- **Summary:** TensorRT VAE Encoder activation attempt unsuccessful; FP16 VAE encoding kept unchanged:
  - **TensorRT VAE Encoder:** `SeedVR2LoadTensorRTVAEModel` was registered during an activation attempt and has been removed again — the TensorRT encoder's top-left tiling artifact could not be resolved at either 256px or 512px tile size. `SeedVR2LoadTensorRTVAEDecoder` (decode-only TRT) and `SeedVR2BuildTensorRTVAE` remain available. A batched one-shot variant of the FP16 encode path was also evaluated and reverted (it reproduced the blur on FP16); the per-frame loop remains the FP16 encode implementation.
- **Technical Details:** See [v1.5.3 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.3) for complete explanation

## v1.5.2 — 2026-09-08

- **Summary:** Fixed a black-out (blank tile) regression in the TensorRT VAE Decoder by correcting engine selection:
  - **Engine Selection Overhaul:** `pick_engine_frames` now scans the artifact directories for engines that actually exist (e.g. 25f / 29f / 41f / 61f) instead of the hardcoded `(video_frames, 29, 21, 5)` list, so downloaded engines are actually used instead of silently falling back to the PyTorch VAE.
  - **No Silent Fallback:** `resolve_engine_frames` now returns the largest engine on disk; clips shorter than the smallest engine are padded, decoded in one shot, and cropped back instead of falling back.
  - **Per-Batch Engine Selection:** `_trt_decode_batch` selects the engine from the actual batch length (with pad + crop for short batches).
- **Technical Details:** See [v1.5.2 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.2) for complete explanation

## v1.5.1 — 2026-09-05

- **Summary:** Comprehensive installer and runtime reliability overhaul:
  - **Zero-Intervention Automated Installation:** Resolved dependency installation failures in `install.py` by automatically installing `requirements.txt` into the host Python environment without requiring external batch files.
  - **Attention Backend Unification & PyTorch SDPA Standard:** Removed brittle wheel auto-downloaders for FlashAttention 2 and SageAttention 2, standardizing on PyTorch native SDPA fallback (`attention_mode: sdpa`) when custom attention packages are absent ([#1](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/issues/1)).
  - **Comprehensive FFmpeg PATH Discovery:** Implemented proactive multi-candidate directory search and `imageio_ffmpeg` fallback across runtime module loading (`__init__.py`), `install.py`, `scripts/verify_install.py`, and CLI to eliminate video export crashes.
  - **Decoder Engine Tile Size Specification:** Documented mandatory `tile_size: 256` constraint for TensorRT VAE Decoder engine compilation to prevent spatial dimension mismatch errors during inference.
  - **Full 64-bit Seed Range:** Expanded seed input range to full 64-bit (`0..0xffffffffffffffff`) matching ComfyUI core nodes (KSampler) and removed obsolete NumPy random seed dependency ([PR #635](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/pull/635)).
- **Technical Details:** See [v1.5.1 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/releases/tag/v1.5.1) for complete explanation

## v1.5 — 2026-09-03

- **Summary:** Added TensorRT VAE Decoder support with dedicated loader node (`SeedVR2LoadTensorRTVAEDecoder`), multi-tile engine support (256px/512px, 4n+1 frame sizes), cached execution context reuse, decoupled encode/decode configurations, and automatic safe fallback to PyTorch VAE on missing engines.
- **Technical Details:** See [v1.5 Release Notes](../zhmd/v1.5.md) for complete explanation

## v1.4 — 2026-07-31

- **Summary:** Register 3B HSWQ INT8 ConvRot and NVFP4 DiT packs in `MODEL_REGISTRY` (same native VRAM path as 7B).
- **Technical Details:** See [v1.4 Release Notes](../zhmd/v1.4.md) for complete explanation

## v1.3 — 2026-07-28

- **Summary:** Windows torch.compile / inductor runtime improvements: parallel inductor compile on win32, shut down compile workers after each phase’s first batch to free CUDA contexts, run-scoped `cudnn.benchmark`, and more uniform VAE temporal slices to reduce compile shape variants.
- **Technical Details:** See [v1.3 Release Notes](../zhmd/v1.3.md) for complete explanation

## v1.2 — 2026-07-28

- **Summary:** Native NVFP4 loading for SeedVR2 DiT via construction-time `comfy.ops.mixed_precision_ops`, plus Windows / inductor fixes so FP16 VAE `torch.compile` no longer fails on cp932 decode or `aten.bmm` fallback+decomp asserts. INT8 path from v1.1 remains available.
- **Technical Details:** See [v1.2 Release Notes](../zhmd/v1.2.md) for complete explanation

## v1.1 — 2026-07-27

- **Summary:** Native INT8 loading for SeedVR2 DiT (`int8_tensorwise` + `comfy_quant` / `weight_scale`) via construction-time `comfy.ops.mixed_precision_ops`, so INT8 packs stay quantized through `load_state_dict` instead of expanding to full FP16 (VRAM reduction). DiT only; VAE remains FP16.
- **Technical Details:** See [v1.1 Release Notes](../zhmd/v1.1.md) for complete explanation

## v1.0 — 2026-04-05

- **Summary:** Auto-install missing SeedVR2 dependencies into the active ComfyUI Python (`sys.executable`) at node load, addressing `ModuleNotFoundError` (e.g. `diffusers`, `rotary_embedding_torch`) on cloud templates such as Vast.ai / RunPod where terminal `pip` and ComfyUI’s venv diverge.
- **Technical Details:** See [v1.0 Release Notes](../zhmd/v1.0.md) for complete explanation
