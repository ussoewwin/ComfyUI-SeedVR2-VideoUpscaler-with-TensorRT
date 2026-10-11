---
license: apache-2.0
tags:
- seedvr2
- dit
- video-upscaler
- comfyui
- tensorrt
- quantization
- w4a8
- int4
- int8
- convrot
- asym_w4a8_int8
base_model: ByteDance-Seed/SeedVR2
pipeline_tag: video-to-video
---

# SeedVR2 DiT — ConvRot INT8 & W4A8 Quantization (`asym_w4a8_int8` & `int8_tensorwise`)

This repository provides the official **ConvRot INT8** and **ConvRot W4A8 (`asym_w4a8_int8`)** quantized diffusion transformer (DiT) weights for [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) / [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT).

It brings the original **15.35 GB** FP16 7B DiT down to **4.44 GB** (~71.1% memory & disk reduction) for W4A8, and **8.53 GiB** VRAM footprint for INT8, enabling native low-bit inference directly inside ComfyUI with comprehensive VRAM savings.

---

## Available Models & Checkpoint Specifications

| Model Checkpoint | Architecture | Precision / Layout | Disk Size | Peak VRAM | Hugging Face Direct URL |
|:---|:---|:---|:---:|:---:|:---|
| **`seedvr2_7b_convrot_w4a8.safetensors`** | 7B DiT Base | `asym_w4a8_int8` + ConvRot | **4.44 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_convrot_w4a8.safetensors) |
| **`seedvr2_7b_sharp_convrot_w4a8.safetensors`** | 7B DiT Sharp | `asym_w4a8_int8` + ConvRot | **4.44 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_sharp_convrot_w4a8.safetensors) |
| **`seedvr2_3b_convrot_w4a8.safetensors`** | 3B DiT Base | `asym_w4a8_int8` + ConvRot | **2.21 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_3b_convrot_w4a8.safetensors) |
| **`seedvr2_7b_convrot_int8.safetensors`** | 7B DiT Base | `int8_tensorwise` + ConvRot | **8.12 GB** | **8.53 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_convrot_int8.safetensors) |
| **`seedvr2_7b_sharp_convrot_int8.safetensors`** | 7B DiT Sharp | `int8_tensorwise` + ConvRot | **8.12 GB** | **8.53 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_7b_sharp_convrot_int8.safetensors) |
| **`seedvr2_3b_convrot_int8.safetensors`** | 3B DiT Base | `int8_tensorwise` + ConvRot | **4.08 GB** | **5.69 GiB** | [Download](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8/resolve/main/seedvr2_3b_convrot_int8.safetensors) |

---

## Technical Specifications (W4A8 Architecture)

| Parameter | Specification |
|:---|:---|
| **Base Architectures** | SeedVR2 7B / 3B NaDiT (ByteDance Seed) |
| **Quantization Format** | ComfyUI Native `asym_w4a8_int8` (`AsymW4A8Int8Layout`) |
| **Weight Precision** | INT4 packed into `torch.int8` storage (`[out_features, in_features // 2]`) |
| **Orthogonal Rotation** | **ConvRot** normalized Hadamard transform ($H_{256} \cdot H_{256}^T = I$) |
| **ConvRot Group Size** | `256` channels |
| **Weight Group Size** | `16` channels |
| **Scale Hierarchy** | **2-Level**: Per-channel scale (FP32) $\times$ Per-group relative scale (FP8 `e4m3fn`) |
| **Quantized Layers (7B)** | **288** Core Transformer Block Attention & MLP 2D Projection Weights |
| **Preserved Layers (7B)** | **842** Sensitive Layers strictly kept in **FP16** (Norms, Biases, RoPE, Embeddings, Output Heads) |
| **VRAM Reduction (7B)** | **~71.1%** weight footprint reduction |

---

## Trajectory Comparison & Empirical Benchmarks

Extensive multi-seed trajectory benchmarking across 25 deterministic seeds (`42` to `99999`) using [`benchmark/seedvr2_w4a8_traj_compare.py`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/benchmark/seedvr2_w4a8_traj_compare.py) and [`seedvr2_int8_traj_compare.py`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/benchmark/seedvr2_int8_traj_compare.py):

### Multi-Seed Summary (25 Seeds)

| Model & Quantization Mode | Inference Wall Time | Peak VRAM | Latent Cosine (Mean) | Latent Cosine Range | Same-Image Verdict | Bifurcation Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **7B ConvRot INT8** | **24.51s** | **8.53 GiB** | **0.99892** | 0.99870 ~ 0.99902 | **25/25** | **0/25** |
| **7B ConvRot W4A8** | **14.32s** | **5.69 GiB** | **0.98672** | 0.98579 ~ 0.98759 | **25/25** | **0/25** |
| **7B Sharp ConvRot INT8** | **24.50s** | **8.53 GiB** | **0.99766** | 0.99707 ~ 0.99796 | **25/25** | **0/25** |
| **7B Sharp ConvRot W4A8** | **11.18s** | **5.69 GiB** | **0.97682** | 0.97505 ~ 0.97789 | N/A (High Sharpness) | **0/25** |
| **3B ConvRot INT8** | **9.55s** | **5.69 GiB** | **0.99724** | 0.99679 ~ 0.99750 | **25/25** | **0/25** |
| **3B ConvRot W4A8** | **9.02s** | **5.69 GiB** | **0.97093** | 0.96981 ~ 0.97230 | N/A | **0/25** |
| *Baseline FP16 7B* | *129.41s* | *16.04 GiB* | *1.00000* | *1.00000* | *Reference* | *0* |

- **Zero Mode Collapse / Bifurcation**: All seeds exhibit 0/25 bifurcation, indicating strict trajectory alignment without semantic collapse or runaway trajectories.
- **Speed & Memory**: 7B W4A8 achieves a **~9.0x speedup** (14.32s vs 129.41s) and cuts resident VRAM down to **5.69 GiB** on consumer RTX hardware.

---

## Mathematical Formulation: ConvRot & 2-Level Hierarchical Scaling

### 1. ConvRot Orthogonal Hadamard Rotation
DiT linear layers typically suffer from extreme activation and weight channel outliers that cause catastrophic precision degradation when truncated to 4 bits. ConvRot resolves this offline by applying an orthogonal Hadamard transformation matrix $H \in \mathbb{R}^{256 \times 256}$ along the reduction channel dimension $K$:

$$H_4 = \begin{pmatrix} 1 & 1 & 1 & -1 \\ 1 & 1 & -1 & 1 \\ 1 & -1 & 1 & 1 \\ -1 & 1 & 1 & 1 \end{pmatrix}, \quad H_{256} = \frac{1}{\sqrt{256}} \left( H_4 \otimes H_4 \otimes H_4 \otimes H_4 \right)$$

$$W_{rot} = W_{grouped} \cdot H_{256}^T$$

Because $H$ is orthogonal ($H \cdot H^T = I$), the dot product with the rotated activation $X_{rot} = X \cdot H$ is mathematically invariant:

$$X_{rot} \cdot W_{rot}^T = (X \cdot H)(W \cdot H^T)^T = X \cdot H \cdot H^T \cdot W^T = X \cdot W^T$$

### 2. Hierarchical Scaling & Group Quantization
1. **Per-Channel Scale ($s_{channel}$)**:
   $$s_{channel} = \max_{j} |W_{rot, \cdot, j}| \in \mathbb{R}^{out\_features} \quad (\text{FP32})$$
   $$W_{norm} = \frac{W_{rot}}{s_{channel}}$$

2. **Per-Group Relative Scale ($s_{rel}$)**:
   Over blocks of $group\_size = 16$:
   $$s_{rel} = \frac{\max_{k \in group} |W_{norm, \cdot, k}|}{7.0} \in \mathbb{R}^{out\_features \times (in\_features / 16)} \quad (\text{torch.float8\_e4m3fn})$$

3. **INT4 Quantization & Packing**:
   $$q = \text{clamp}\left( \text{round}\left( \frac{W_{norm}}{s_{rel}} \right), -8, 7 \right) \in \text{INT8}$$
   Packed into bytes (even column into low nibble, odd column into high nibble).

---

## ComfyUI Native VRAM-Saving Integration

Unlike naive loader implementations that dequantize weights into FP16 upon loading, this model runs through ComfyUI's native **`comfy.ops.mixed_precision_ops`** and **`comfy_kitchen.tensor.w4a8_int8`**:

- Injected directly at DiT construction time (`create_object` on `torch.device("meta")`).
- `comfy.ops._load_quantized_module` populates `QuantizedTensor` directly on the target GPU.
- Matmuls execute through native Tensor Core INT8 / dequant GEMM without intermediate full FP16 weight expansion in VRAM.

---

## Installation & Usage in ComfyUI

### 1. Model Placement
Place the downloaded `.safetensors` file into your ComfyUI models directory:

```text
ComfyUI/
└── models/
    └── SEEDVR2/
        ├── seedvr2_7b_convrot_w4a8.safetensors
        ├── seedvr2_7b_sharp_convrot_w4a8.safetensors
        ├── seedvr2_3b_convrot_w4a8.safetensors
        ├── seedvr2_7b_convrot_int8.safetensors
        └── seedvr2_3b_convrot_int8.safetensors
```

### 2. ComfyUI Node Setup
In the custom node repository [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT):
1. In the standard **`SeedVR2 (Down)Load DiT Model`** node, select your desired model (e.g. `seedvr2_7b_convrot_w4a8.safetensors`).
2. Connect to the **`SeedVR2 Video Upscaler`** node.
3. Run inference — enjoy substantial VRAM reductions with preserved FP16 precision on sensitive layers.

> **Important Compatibility Note**:
> W4A8 models are strictly supported via the standard DiT loader (**`SeedVR2 (Down)Load DiT Model`**). The DisTorch2 loader (**`SeedVR2 (Down)Load DiT Model with Distorch2`**) does **not** support W4A8 models due to blockwise CPU RAM streaming and weight-dispatch constraints specific to asymmetric 4-bit packed weights. For DisTorch2 offloading, use the ConvRot INT8 models.

---

## Acknowledgements & References

- **Base Architecture**: [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- **ComfyUI Integration**: [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) & [AInVFX](https://www.youtube.com/@AInVFX)
- **TensorRT & Quantization Fork**: [ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT)
- **Civitai Resources**: Community models available on [civitai.red](https://civitai.red)
