# SeedVR2 7B Quantization Benchmark Test Results

Deterministic per-step latent trajectory divergence and image quality benchmark comparing **FP16 reference** vs **ConvRot INT8** vs **ConvRot W4A8** vs **NVFP4** on the SeedVR2 7B architecture.

**Source:** `benchmark/benchmark result.txt`  
**Evaluation Scripts:**
- `benchmark/seedvr2_int8_traj_compare.py`
- `benchmark/seedvr2_w4a8_traj_compare.py`
- `benchmark/seedvr2_nvfp4_traj_compare.py`

**Evaluation Protocol:** 25-seed deterministic trajectory analysis (seeds: 42, 137, 256, 512, 777, 1024, 1337, 2048, 3141, 4096, 5555, 6891, 7777, 8192, 9999, 12345, 20240, 32768, 44444, 54321, 65536, 71828, 88888, 94103, 99999)

---

## 1. Summary Comparison (Quantization Formats vs FP16 Reference)

### Cross-Format Overview

| Quantization Format | GEMM Mode / Kernel | Latent Cosine (↑) | Latent MSE (↓) | Image SSIM (↑) | Image MSE (↓) | Wall Time (s) | Speedup (↑) | Peak VRAM | VRAM Saving | Same-Image Rate | Bifurcated Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 Reference** | Native `torch.float16` | 1.00000 | 0.000 | 1.000000 | 0.000000 | ~121–131s | 1.00x | 16.04–16.07 GiB | baseline | 25/25 (100%) | 0/25 (0%) |
| **ConvRot INT8** | Native INT8 (`int8_tensorwise`, dequant GEMM) | **0.99892** | **1.800e-03** | **0.994682** | **0.920831** | **24.51s** | **4.95x** | **8.53 GiB** | **−46.8%** | **25/25 (100%)** | **0/25 (0%)** |
| **ConvRot W4A8** | Native W4A8 (`asym_w4a8_int8` + ConvRot, SM80+) | **0.98672** | **2.234e-02** | **0.976833** | **4.567560** | **14.32s** | **9.04x** | **5.69 GiB** | **−64.5%** | **25/25 (100%)** | **0/25 (0%)** |
| **NVFP4** | Stock ComfyUI `mixed_precision` (dequant GEMM) | **0.96430** | **5.967e-02** | **0.952514** | **8.923874** | **22.76s** | **5.77x** | **5.72 GiB** | **−64.4%** | **0/25 (0%)** | **0/25 (0%)** |

*Note: Wall time and Peak VRAM measured on seed 99999.*

---

## 2. Detailed Results per Quantization Format

### 2.1. SeedVR2 7B ConvRot INT8

#### Architecture Profile & Execution Configuration
- **GEMM Mode:** Native INT8 (`int8_tensorwise`, dequant GEMM)
- **Forward Statistics:** INT8 forward statistics are not exposed by the HSWQ nodes package
- **Performance:** Wall time: 24.51s (vs FP16: 121.36s, **4.95x speedup**) | Peak VRAM: 8.53 GiB (vs FP16: 16.04 GiB, **46.8% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.99892** | Pristine trajectory preservation |
| **Min Latent Cosine** (↑ better) | **0.99870** (Seed 512) | No catastrophic drops across any seed |
| **Max Latent Cosine** (↑ better) | **0.99902** (Seed 65536) | Near bit-identical reconstruction |
| **Mean Latent MSE** (↓ better) | **1.79988e-03** | Extremely low latent error |
| **Mean Image SSIM** (↑ better) | **0.994682** (Min: 0.994423, Max: 0.994946) | Exceptional structural similarity |
| **Mean Image MSE** (↓ better) | **0.920831** (Min: 0.840793, Max: 1.026345) | Sub-1.0 pixel reconstruction error |
| **Max Step Drop** (↓ better) | **0.0000** | Zero single-step trajectory divergence |
| **Same-Image Rate** | **25/25 (100%)** | All 25 seeds generate the identical image |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events |

#### Side-by-Side per Seed (ConvRot INT8)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.99890 | 1.828e-03 | 0.0000 | 0.972780 | 0.994574 | `same-image` |
| **137** | 0.99884 | 1.935e-03 | 0.0000 | 0.946700 | 0.994561 | `same-image` |
| **256** | 0.99900 | 1.679e-03 | 0.0000 | 0.960883 | 0.994746 | `same-image` |
| **512** | 0.99870 | 2.197e-03 | 0.0000 | 0.948945 | 0.994530 | `same-image` |
| **777** | 0.99898 | 1.718e-03 | 0.0000 | 0.885440 | 0.994901 | `same-image` |
| **1024** | 0.99890 | 1.823e-03 | 0.0000 | 0.877075 | 0.994580 | `same-image` |
| **1337** | 0.99896 | 1.735e-03 | 0.0000 | 0.872996 | 0.994771 | `same-image` |
| **2048** | 0.99891 | 1.826e-03 | 0.0000 | 0.945959 | 0.994671 | `same-image` |
| **3141** | 0.99898 | 1.692e-03 | 0.0000 | 0.981082 | 0.994760 | `same-image` |
| **4096** | 0.99874 | 2.129e-03 | 0.0000 | 1.014225 | 0.994511 | `same-image` |
| **5555** | 0.99894 | 1.756e-03 | 0.0000 | 0.873702 | 0.994768 | `same-image` |
| **6891** | 0.99892 | 1.829e-03 | 0.0000 | 0.941817 | 0.994708 | `same-image` |
| **7777** | 0.99879 | 2.062e-03 | 0.0000 | 0.934760 | 0.994723 | `same-image` |
| **8192** | 0.99893 | 1.782e-03 | 0.0000 | 0.855408 | 0.994643 | `same-image` |
| **9999** | 0.99889 | 1.838e-03 | 0.0000 | 0.929711 | 0.994423 | `same-image` |
| **12345** | 0.99898 | 1.703e-03 | 0.0000 | 0.891295 | 0.994789 | `same-image` |
| **20240** | 0.99889 | 1.877e-03 | 0.0000 | 0.871635 | 0.994946 | `same-image` |
| **32768** | 0.99901 | 1.654e-03 | 0.0000 | 0.910722 | 0.994828 | `same-image` |
| **44444** | 0.99895 | 1.767e-03 | 0.0000 | 1.026345 | 0.994593 | `same-image` |
| **54321** | 0.99897 | 1.719e-03 | 0.0000 | 0.899787 | 0.994659 | `same-image` |
| **65536** | 0.99902 | 1.616e-03 | 0.0000 | 1.005343 | 0.994623 | `same-image` |
| **71828** | 0.99900 | 1.665e-03 | 0.0000 | 0.902072 | 0.994655 | `same-image` |
| **88888** | 0.99890 | 1.836e-03 | 0.0000 | 0.840793 | 0.994774 | `same-image` |
| **94103** | 0.99896 | 1.747e-03 | 0.0000 | 0.909740 | 0.994674 | `same-image` |
| **99999** | 0.99899 | 1.685e-03 | 0.0000 | 0.921366 | 0.994649 | `same-image` |
| **Mean** | **0.99892** | **1.800e-03** | **0.0000** | **0.920831** | **0.994682** | **25/25 same-image (100%)** |

---

### 2.2. SeedVR2 7B ConvRot W4A8

#### Architecture Profile & Execution Configuration
- **GEMM Mode:** Native W4A8 (`asym_w4a8_int8`, `AsymW4A8Int8Layout` + ConvRot)
- **Weight Storage:** INT4 packed storage (`torch.int8 [out_features, in_features // 2]`)
- **ConvRot Rotation:** Hadamard orthogonal rotation active (`convrot_groupsize=256`)
- **Scale Hierarchy:** FP8 group scale (`weight_s_rel`) + FP32 channel scale (`weight_s_channel`)
- **Compute Kernel:** `comfy_kitchen` `w4a8_int8_linear` / dequant GEMM on SM80+
- **Performance:** Wall time: 14.32s (vs FP16: 129.41s, **9.04x speedup**) | Peak VRAM: 5.69 GiB (vs FP16: 16.04 GiB, **64.5% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.98672** | High-fidelity retention under 4-bit weights |
| **Min Latent Cosine** (↑ better) | **0.98579** (Seed 4096) | Stable floor across random seeds |
| **Max Latent Cosine** (↑ better) | **0.98759** (Seed 71828) | Consistent high cosine fidelity |
| **Mean Latent MSE** (↓ better) | **2.234e-02** | Low error accumulation |
| **Mean Image SSIM** (↑ better) | **0.976833** (Min: 0.974796, Max: 0.978883) | High visual and structural preservation |
| **Mean Image MSE** (↓ better) | **4.567560** (Min: 4.274974, Max: 4.859007) | Tightly bounded pixel variance |
| **Max Step Drop** (↓ better) | **0.0000** | Zero trajectory instability |
| **Same-Image Rate** | **25/25 (100%)** | All 25 seeds maintain the identical semantic composition |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events |

#### Side-by-Side per Seed (ConvRot W4A8)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.98691 | 2.197e-02 | 0.0000 | 4.656818 | 0.975971 | `same-image` |
| **137** | 0.98659 | 2.242e-02 | 0.0000 | 4.549381 | 0.976359 | `same-image` |
| **256** | 0.98672 | 2.245e-02 | 0.0000 | 4.668989 | 0.975866 | `same-image` |
| **512** | 0.98610 | 2.366e-02 | 0.0000 | 4.436874 | 0.978089 | `same-image` |
| **777** | 0.98679 | 2.244e-02 | 0.0000 | 4.472575 | 0.978023 | `same-image` |
| **1024** | 0.98654 | 2.251e-02 | 0.0000 | 4.687104 | 0.975206 | `same-image` |
| **1337** | 0.98676 | 2.219e-02 | 0.0000 | 4.498112 | 0.976748 | `same-image` |
| **2048** | 0.98705 | 2.172e-02 | 0.0000 | 4.502431 | 0.977148 | `same-image` |
| **3141** | 0.98701 | 2.174e-02 | 0.0000 | 4.584469 | 0.977068 | `same-image` |
| **4096** | 0.98579 | 2.413e-02 | 0.0000 | 4.460091 | 0.977680 | `same-image` |
| **5555** | 0.98648 | 2.262e-02 | 0.0000 | 4.499983 | 0.977004 | `same-image` |
| **6891** | 0.98627 | 2.333e-02 | 0.0000 | 4.618887 | 0.976871 | `same-image` |
| **7777** | 0.98629 | 2.349e-02 | 0.0000 | 4.468643 | 0.978430 | `same-image` |
| **8192** | 0.98712 | 2.152e-02 | 0.0000 | 4.478792 | 0.976634 | `same-image` |
| **9999** | 0.98639 | 2.272e-02 | 0.0000 | 4.705870 | 0.975314 | `same-image` |
| **12345** | 0.98679 | 2.218e-02 | 0.0000 | 4.610132 | 0.976453 | `same-image` |
| **20240** | 0.98617 | 2.345e-02 | 0.0000 | 4.274974 | 0.978883 | `same-image` |
| **32768** | 0.98709 | 2.177e-02 | 0.0000 | 4.523461 | 0.977324 | `same-image` |
| **44444** | 0.98677 | 2.238e-02 | 0.0000 | 4.677377 | 0.976211 | `same-image` |
| **54321** | 0.98717 | 2.144e-02 | 0.0000 | 4.669692 | 0.976485 | `same-image` |
| **65536** | 0.98697 | 2.157e-02 | 0.0000 | 4.859007 | 0.974796 | `same-image` |
| **71828** | 0.98759 | 2.076e-02 | 0.0000 | 4.523142 | 0.976238 | `same-image` |
| **88888** | 0.98690 | 2.204e-02 | 0.0000 | 4.348898 | 0.977563 | `same-image` |
| **94103** | 0.98714 | 2.171e-02 | 0.0000 | 4.537492 | 0.976604 | `same-image` |
| **99999** | 0.98658 | 2.258e-02 | 0.0000 | 4.675994 | 0.975849 | `same-image` |
| **Mean** | **0.98672** | **2.234e-02** | **0.0000** | **4.567560** | **0.976833** | **25/25 same-image (100%)** |

---

### 2.3. SeedVR2 7B NVFP4

#### Architecture Profile & Execution Configuration
- **GEMM Mode:** Stock ComfyUI `mixed_precision` (dequant GEMM)
- **Patch Status:** No HSWQ TC patch applied; `QuantizedTensor` stays packed, matmul dequantizes
- **Performance:** Wall time: 22.76s (vs FP16: 131.44s, **5.77x speedup**) | Peak VRAM: 5.72 GiB (vs FP16: 16.07 GiB, **64.4% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.96430** | Bounded drift under stock dequant GEMM |
| **Min Latent Cosine** (↑ better) | **0.96264** (Seed 12345) | Zero catastrophic collapse |
| **Max Latent Cosine** (↑ better) | **0.96589** (Seed 71828) | Predictable trajectory floor |
| **Mean Latent MSE** (↓ better) | **5.967e-02** | Moderate divergence without bifurcation |
| **Mean Image SSIM** (↑ better) | **0.952514** (Min: 0.948723, Max: 0.958304) | Consistent image structure |
| **Mean Image MSE** (↓ better) | **8.923874** (Min: 8.305307, Max: 9.280363) | Bounded pixel drift |
| **Max Step Drop** (↓ better) | **0.0000** | Zero discontinuous step drops |
| **Same-Image Rate** | **0/25 (0%)** | Gradual trajectory drift across all runs |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events (retains base composition) |

#### Side-by-Side per Seed (NVFP4)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.96471 | 5.888e-02 | 0.0000 | 8.937768 | 0.951307 | `drifted (different image)` |
| **137** | 0.96376 | 6.032e-02 | 0.0000 | 8.812521 | 0.952986 | `drifted (different image)` |
| **256** | 0.96505 | 5.876e-02 | 0.0000 | 8.905413 | 0.952172 | `drifted (different image)` |
| **512** | 0.96442 | 6.007e-02 | 0.0000 | 8.519649 | 0.956933 | `drifted (different image)` |
| **777** | 0.96347 | 6.151e-02 | 0.0000 | 8.782884 | 0.953874 | `drifted (different image)` |
| **1024** | 0.96519 | 5.772e-02 | 0.0000 | 9.280363 | 0.948931 | `drifted (different image)` |
| **1337** | 0.96421 | 5.974e-02 | 0.0000 | 8.884372 | 0.952592 | `drifted (different image)` |
| **2048** | 0.96532 | 5.773e-02 | 0.0000 | 8.867726 | 0.952872 | `drifted (different image)` |
| **3141** | 0.96506 | 5.799e-02 | 0.0000 | 8.877296 | 0.952960 | `drifted (different image)` |
| **4096** | 0.96283 | 6.270e-02 | 0.0000 | 8.803332 | 0.954634 | `drifted (different image)` |
| **5555** | 0.96344 | 6.063e-02 | 0.0000 | 9.099228 | 0.952041 | `drifted (different image)` |
| **6891** | 0.96374 | 6.106e-02 | 0.0000 | 8.910255 | 0.952832 | `drifted (different image)` |
| **7777** | 0.96268 | 6.350e-02 | 0.0000 | 8.701624 | 0.957000 | `drifted (different image)` |
| **8192** | 0.96586 | 5.668e-02 | 0.0000 | 8.922103 | 0.952269 | `drifted (different image)` |
| **9999** | 0.96400 | 5.992e-02 | 0.0000 | 8.984280 | 0.951138 | `drifted (different image)` |
| **12345** | 0.96264 | 6.229e-02 | 0.0000 | 9.182448 | 0.950477 | `drifted (different image)` |
| **20240** | 0.96406 | 6.041e-02 | 0.0000 | 8.305307 | 0.958304 | `drifted (different image)` |
| **32768** | 0.96488 | 5.881e-02 | 0.0000 | 8.872224 | 0.953372 | `drifted (different image)` |
| **44444** | 0.96507 | 5.887e-02 | 0.0000 | 8.906494 | 0.952785 | `drifted (different image)` |
| **54321** | 0.96499 | 5.824e-02 | 0.0000 | 9.072032 | 0.951635 | `drifted (different image)` |
| **65536** | 0.96435 | 5.869e-02 | 0.0000 | 9.260098 | 0.948723 | `drifted (different image)` |
| **71828** | 0.96589 | 5.672e-02 | 0.0000 | 8.884585 | 0.951045 | `drifted (different image)` |
| **88888** | 0.96399 | 6.015e-02 | 0.0000 | 8.705837 | 0.952893 | `drifted (different image)` |
| **94103** | 0.96465 | 5.941e-02 | 0.0000 | 9.174470 | 0.949827 | `drifted (different image)` |
| **99999** | 0.96323 | 6.138e-02 | 0.0000 | 9.244542 | 0.949242 | `drifted (different image)` |
| **Mean** | **0.96430** | **5.967e-02** | **0.0000** | **8.923874** | **0.952514** | **25/25 drifted (100%)** |

---

## 3. Performance, Speedup & VRAM Footprint Analysis

### Inference Latency Comparison (Wall Time on Seed 99999)
- **FP16 Reference:** ~121.36s – 131.44s (baseline)
- **ConvRot INT8:** 24.51s (**4.95x speedup**)
- **ConvRot W4A8:** 14.32s (**9.04x speedup**) — Fastest execution profile
- **NVFP4 (Stock Dequant):** 22.76s (**5.77x speedup**)

### GPU Memory Consumption (Peak VRAM on Seed 99999)
- **FP16 Reference:** ~16.04 – 16.07 GiB (requires 24GB VRAM class GPU for comfortable headroom)
- **ConvRot INT8:** 8.53 GiB (**−46.8% memory reduction**, operates comfortably within 10–12GB VRAM cards)
- **ConvRot W4A8:** 5.69 GiB (**−64.5% memory reduction**, operates comfortably on 6–8GB VRAM cards)
- **NVFP4:** 5.72 GiB (**−64.4% memory reduction**, operates comfortably on 6–8GB VRAM cards)

---

## 4. Key Findings and Trajectory Analysis

1. **Zero Catastrophic Bifurcations Across All Formats (0/25 = 0%):**
   Unlike standard text-to-image diffusion models where 4-bit quantization can trigger sudden Step 11 bifurcations into distinct attractor basins, all evaluated SeedVR2 7B quantization formats exhibited `max-drop = 0.0000` across all 25 tested seeds. Trajectory progression remained strictly smooth and deterministic.
2. **ConvRot INT8 Delivers Near-Lossless Parity:**
   Reaching a mean latent cosine of **0.99892** and image SSIM of **0.994682**, ConvRot INT8 achieves full `same-image` classification on 25/25 seeds. It cuts inference time by ~5x and VRAM by ~47% with negligible perceptual variation from FP16.
3. **ConvRot W4A8 Is the Optimal High-Efficiency Sweet Spot:**
   By combining INT4 packed storage, FP8 group scales, FP32 channel scales, and Hadamard orthogonal rotation (`convrot_groupsize=256`), ConvRot W4A8 achieves **14.32s wall time (9.04x speedup)** and **5.69 GiB peak VRAM**. It maintains **0.98672 mean latent cosine** and **0.976833 image SSIM**, achieving 25/25 `same-image` verdict while slashing VRAM by ~65%.
4. **NVFP4 Operates in Bounded Drift Mode Under Dequant GEMM:**
   Without hardware Tensor Core W4A4 patches applied in stock ComfyUI, NVFP4 performs on-the-fly dequantization. While all seeds maintain structural coherence (`drifted` verdict, 0 bifurcations), latent cosine sits at **0.96430**, making ConvRot W4A8 noticeably superior in both speed (14.32s vs 22.76s) and fidelity (0.98672 vs 0.96430).

---

## 5. Metric Definitions & Benchmark Protocol

- **Final Latent Cosine (`lat-cos`):** Cosine similarity between the final denoised latent of the FP16 reference and the quantized model. Values $\ge 0.98$ represent `same-image` parity.
- **Final Latent MSE (`lat-mse`):** Mean Squared Error of the final denoised latent tensor against the FP16 reference.
- **Max Step Drop (`max-drop`):** Maximum single-step cosine drop between consecutive sampling steps, measuring discontinuous trajectory collapse.
- **Image MSE (`img-MSE`):** Pixel-space Mean Squared Error between the decoded upscaled output of FP16 and the quantized model.
- **Image SSIM (`img-SSIM`):** Structural Similarity Index Measure between decoded image outputs. Closer to 1.0 indicates higher visual fidelity.
- **Trajectory Verdict:**
  - `same-image`: per-seed final latent cosine $\ge 0.98$, yielding virtually indistinguishable image outputs.
  - `drifted (different image)`: gradual, continuous trajectory shift without sudden discontinuous steps.
  - `bifurcated @step N`: sudden trajectory jump at step $N$ (`max-drop > 0.05`), resulting in an entirely different attractor basin.
- **Protocol:** Deterministic multi-seed trajectory evaluation across 25 dispersed seeds, fixed cuDNN deterministic mode, matching noise generators and scheduling parameters.
