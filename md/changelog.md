<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/zhmd/changelogzh.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# Changelog

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
