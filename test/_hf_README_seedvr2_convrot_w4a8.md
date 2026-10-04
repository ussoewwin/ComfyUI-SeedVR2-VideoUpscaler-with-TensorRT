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

# SeedVR2 7B DiT — ConvRot W4A8 Quantization (`asym_w4a8_int8`)

This repository provides the official **ConvRot W4A8 (`asym_w4a8_int8`)** quantized diffusion transformer (DiT) weights for [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) / [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT).

It brings the original **15.35 GB** FP16 7B DiT down to **4.44 GB** (~71.1% memory & disk reduction), enabling native low-bit inference directly inside ComfyUI with full VRAM savings.

---

## Technical Specifications

| Parameter | Specification |
|:---|:---|
| **Base Architecture** | SeedVR2 7B NaDiT (ByteDance Seed) |
| **Quantization Format** | ComfyUI Native `asym_w4a8_int8` (`AsymW4A8Int8Layout`) |
| **Weight Precision** | INT4 packed into `torch.int8` storage (`[out_features, in_features // 2]`) |
| **Orthogonal Rotation** | **ConvRot** normalized Hadamard transform ($H_{256} \cdot H_{256}^T = I$) |
| **ConvRot Group Size** | `256` channels |
| **Weight Group Size** | `16` channels |
| **Scale Hierarchy** | **2-Level**: Per-channel scale (FP32) $\times$ Per-group relative scale (FP8 `e4m3fn`) |
| **Quantized Layers** | **288** Core Transformer Block Attention & MLP 2D Projection Weights |
| **Preserved Layers** | **842** Sensitive Layers strictly kept in **FP16** (Norms, Biases, RoPE, Embeddings, Output Heads) |
| **Model File Size** | **4.44 GB** (FP16 original: 15.35 GB) |
| **VRAM Footprint Reduction** | **~71.1%** |

---

## Mathematical Formulation: ConvRot & 2-Level Hierarchical Scaling

### 1. ConvRot Orthogonal Hadamard Rotation
DiT and LLM linear layers typically suffer from extreme activation and weight channel outliers that cause catastrophic precision degradation when truncated to 4 bits. ConvRot resolves this offline by applying an orthogonal Hadamard transformation matrix $H \in \mathbb{R}^{256 \times 256}$ along the reduction channel dimension $K$:

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

### 1. Download Model
Download [`seedvr2_7b_convrot_w4a8.safetensors`](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-w4a8/resolve/main/seedvr2_7b_convrot_w4a8.safetensors) and place it into your ComfyUI models directory:

```text
ComfyUI/
└── models/
    └── SEEDVR2/
        └── seedvr2_7b_convrot_w4a8.safetensors
```

### 2. Node Selection
In the custom node [ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT):
1. In **`SeedVR2 Load DiT Model`** or **`SeedVR2 (Down)Load DiT Model with Distorch2`**, select:
   `seedvr2_7b_convrot_w4a8.safetensors`
2. Connect to the **`SeedVR2 Video Upscaler`** node.
3. Run inference — enjoy ~71% VRAM reduction with preserved FP16 precision on sensitive layers.

---

## Verification & Trajectory Benchmarking

The deterministic trajectory divergence between FP16 baseline and ConvRot W4A8 was verified using [`benchmark/seedvr2_w4a8_traj_compare.py`](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/benchmark/seedvr2_w4a8_traj_compare.py):
- **Checkpoint Validation**: Verified `w4a8_layers=288`, `group_size=16`, `convrot=True`, `convrot_groupsize=256`.
- **Per-Step Latent Cosine**: Confirmed strong trajectory alignment without bifurcation or mode collapse.
- **Fidelity**: Preserves fine textures and high-frequency details across all upscale steps.

---

## Acknowledgements & References

- **Base Architecture**: [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- **ComfyUI Integration**: [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) & [AInVFX](https://www.youtube.com/@AInVFX)
- **TensorRT & Quantization Fork**: [ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT)
