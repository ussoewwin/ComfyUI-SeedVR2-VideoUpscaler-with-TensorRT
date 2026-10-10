<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/zhmd/v1.6.5.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

SeedVR2 Video Upscaler v1.6.5 resolves a critical stability failure in the VAE inference pipeline where unintended fallback to standard PyTorch FP16 VAE occurred during multi-stage or single-VAE workflow configurations, leading to catastrophic CUDA Out Of Memory (OOM) errors at high resolutions. Additionally, this release introduces automated post-build cleanup for intermediate ONNX graph artifacts in TensorRT engine builders.

---

## 1. Complete Elimination of Unintended PyTorch FP16 VAE Fallbacks

### Root Cause & Vulnerability Analysis
In previous releases, `SeedVR2VideoUpscaler` exposed separate inputs for `vae_encode` and `vae_decode`. When workflows only wired one of the inputs (e.g. connecting a TensorRT VAE Loader solely to `vae_decode` while leaving `vae_encode` unconnected), `encode_cfg` evaluated to an empty dictionary (`{}`). Consequently:
- `runner.use_tensorrt_vae_encode` evaluated to `False`.
- A legacy flag decoupling policy (`# TRT decoder + FP16 encoder must keep the encoder on the FP16 path`) allowed the encoder to silently decouple from TensorRT.
- Phase 1 (VAE Encoding) silently fell back to PyTorch FP16 VAE (`VideoAutoencoderKLWrapper`), completely bypassing the dedicated TensorRT 1-shot engine.

At high resolutions (such as `2160x2160px` padded to `4208x4208px` in 2-stage upscaling pipelines), PyTorch VAE's un-tiled 3D causal convolutions (`InflatedCausalConv3d`) attempted to allocate upwards of 30.69 GiB (26.47 GiB allocated + 4.22 GiB requested) during slice concatenation, immediately causing CUDA OOM on standard 16GB GPUs.

### Technical Remediation
1. **Bidirectional Configuration Mirroring**:
   If either `vae_encode` or `vae_decode` is provided, the node now automatically mirrors the configuration to both endpoints:
   ```python
   encode_cfg = dict(vae_encode) if vae_encode is not None else (dict(vae_decode) if vae_decode is not None else {})
   decode_cfg = dict(vae_decode) if vae_decode is not None else (dict(vae_encode) if vae_encode is not None else {})
   ```
2. **Strict TensorRT Synchronization**:
   If TensorRT is active or requested on either endpoint, both `encode_cfg` and `decode_cfg` are strictly locked to TensorRT:
   ```python
   _is_trt = bool(
       encode_cfg.get("use_tensorrt_vae", False)
       or decode_cfg.get("use_tensorrt_vae", False)
       or encode_cfg.get("vae_backend") == "tensorrt"
       or decode_cfg.get("vae_backend") == "tensorrt"
   )
   if _is_trt:
       encode_cfg["use_tensorrt_vae"] = True
       encode_cfg["vae_backend"] = "tensorrt"
       decode_cfg["use_tensorrt_vae"] = True
       decode_cfg["vae_backend"] = "tensorrt"
   
   runner.use_tensorrt_vae_encode = _is_trt
   runner.use_tensorrt_vae_decode = _is_trt
   runner.use_tensorrt_vae = _is_trt
   ```
3. **Hardened Infer Checks (`src/core/infer.py`)**:
   In `VideoDiffusionInfer.vae_encode()` and `vae_decode()`, the TensorRT active checks (`_enc_trt` and `_dec_trt`) now verify all runner-level TRT indicators, guaranteeing that any active TRT workflow directly invokes the dedicated TensorRT engine with zero silent dropbacks to PyTorch FP16.

---

## 2. Automated Cleanup of Intermediate ONNX Artifacts upon Engine Build

During dedicated TensorRT RTX engine compilation via `SeedVR2BuildTensorRTVAE` (`src/interfaces/trt_vae_builder.py`) or dynamic loader builds (`src/interfaces/trt_vae_model_loader.py`):
- Tracing large video batch graphs previously created temporary `.onnx` and companion `.onnx.data` files in `tensorrt_backend/artifacts/`.
- While necessary as input for `cloud_build_engine.py`, these ONNX files are completely redundant once the final `.rtxplan` engine is built and verified.
- **Automated Immediate Purge**: Immediately upon successful engine serialization, `onnx_path` and `onnx_data` are automatically unlinked and safely removed from disk. This prevents multi-gigabyte temporary graphs from cluttering disk storage.

---

## 3. Node Registration & Display Name Standardization

- Cleaned up node registration display names in `SeedVR2VideoUpscaler` by removing legacy hardcoded version identifiers, standardizing to `SeedVR2 Video Upscaler`.
- Preserved full backward compatibility with existing workflows and saved graph JSON schemas.

---

## 4. Modified Files & Verification

- `src/interfaces/video_upscaler.py`: Bidirectional VAE config mirroring, strict TRT flag locking.
- `src/core/infer.py`: Strict multi-flag TRT encoder/decoder checks.
- `src/interfaces/trt_vae_builder.py`: Post-build ONNX and ONNX data unlinking.
- `src/interfaces/trt_vae_model_loader.py`: Post-build ONNX unlinking in runtime build paths.
- `md/changelog.md` & `zhmd/changelogzh.md`: Bilingual changelog synchronization.
