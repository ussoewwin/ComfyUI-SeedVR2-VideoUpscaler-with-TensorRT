<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/zhmd/v1.5.6.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Part 1: Loader-Only Auto-Download (Download Policy Fix)

### 1. Pre-Fix Problems and Failure Modes

#### 1.1 Forced Model Downloads on Every Install/Update
- **Problem**: `install.py` ran a default-model pre-download step on every install/update (via `scripts/download_models.py`), unconditionally fetching `seedvr2_ema_3b_fp8_e4m3fn.safetensors` and `ema_vae_fp16.safetensors`.
- **Symptom**: Users who neither want nor use those files had them re-downloaded on every single update — the only workaround was to keep the files forever, wasting disk space and bandwidth ([#2](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/issues/2)).

#### 1.2 Default DiT Fetch During TensorRT VAE Engine Build
- **Problem**: When the loader-selected VAE file was missing, `src/interfaces/trt_vae_model_loader.py` called `download_weight(DEFAULT_DIT, model, ...)`, which also pulled the default 3B FP8 DiT model.
- **Symptom**: Building a TensorRT VAE engine could download an unrequested default DiT.

### 2. New Download Policy (Fixed)

> **Auto-download happens only when a model is selected in a loader node and the file is missing.**

No other automatic downloads exist:

| Trigger | Download behavior |
| :--- | :--- |
| Installation / update | **None** |
| Running the upscaler node | Only the node-selected DiT / VAE models, if missing |
| TensorRT VAE engine build | Only the loader-selected VAE, if missing |
| CLI (`inference_cli.py`) | Only the models selected via CLI arguments |

### 3. Fixed and Modified Files

| File | Status | Change |
| :--- | :--- | :--- |
| `install.py` | Modified | Removed `download_default_models()` and its step-4 invocation; no pre-download on install/update |
| `scripts/install.ps1` | Modified | Removed the forced default-model download block |
| `src/interfaces/trt_vae_model_loader.py` | Modified | Downloads only the loader-selected VAE (never the default DiT) |
| `src/utils/downloads.py` | Modified | `download_weight()` accepts `dit_model` / `vae_model` optionally; only explicitly passed names are processed (`None` = skip) |

### 4. Key Code Changes

```python
# install.py — step 4 policy change
# 4. Default models: intentionally NOT pre-downloaded.
#    Downloads happen only when a model is selected in a loader node and is missing.
```

```python
# src/interfaces/trt_vae_model_loader.py — selected VAE only
if not (model_dir / model).exists():
    print(f"[SeedVR2 TensorRT] Downloading {model} to {model_dir}...")
    # Download only the model selected in the loader (never the default DiT).
    download_weight(vae_model=model, model_dir=str(model_dir))
```

```python
# src/utils/downloads.py — optional slots
def download_weight(dit_model: Optional[str] = None, vae_model: Optional[str] = None, model_dir: Optional[str] = None, debug=None) -> bool:
    ...
    files_to_download = [
        (name, MODEL_REGISTRY.get(name))
        for name in (dit_model, vae_model)
        if name
    ]
```

---

## Part 2: Six ConvRot INT8 / NVFP4 DiT Models Registered

Registered six quantized DiT packs in `MODEL_REGISTRY` (hosted on [`Comfy-Org/SeedVR2`](https://huggingface.co/Comfy-Org/SeedVR2), `diffusion_models/`, SHA256-pinned). They are selectable and auto-downloadable under the policy above:

| Model | URL |
| :--- | :--- |
| `seedvr2_3b_int8_convrot.safetensors` | https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_3b_int8_convrot.safetensors |
| `seedvr2_3b_nvfp4.safetensors` | https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_3b_nvfp4.safetensors |
| `seedvr2_7b_int8_convrot.safetensors` | https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_7b_int8_convrot.safetensors |
| `seedvr2_7b_nvfp4.safetensors` | https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_7b_nvfp4.safetensors |
| `seedvr2_7b_sharp_int8_convrot.safetensors` | https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_7b_sharp_int8_convrot.safetensors |
| `seedvr2_7b_sharp_nvfp4.safetensors` | https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_7b_sharp_nvfp4.safetensors |

---

## Part 3: GGUF DiT Entries Removed

- Removed the GGUF (Q4_K_M / Q8_0) DiT entries from `MODEL_REGISTRY`: `seedvr2_ema_3b-Q4_K_M.gguf`, `seedvr2_ema_3b-Q8_0.gguf`, `seedvr2_ema_7b-Q4_K_M.gguf`, `seedvr2_ema_7b_sharp-Q4_K_M.gguf`.

---

*Full changelog: [md/changelog.md](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/md/changelog.md)*
