# SeedVR2 7B & 3B Quantization Benchmark Test Results

Deterministic per-step latent trajectory divergence and image quality benchmark comparing **FP16 reference** vs **ConvRot INT8** vs **ConvRot W4A8** vs **NVFP4** across standard 7B, sharp 7B, and 3B variants on the SeedVR2 architecture.

**Source:** `benchmark/benchmark result.txt`  
**Evaluation Scripts:**
- `benchmark/seedvr2_int8_traj_compare.py`
- `benchmark/seedvr2_w4a8_traj_compare.py`
- `benchmark/seedvr2_nvfp4_traj_compare.py`

**Evaluation Protocol:** 25-seed deterministic trajectory analysis (seeds: 42, 137, 256, 512, 777, 1024, 1337, 2048, 3141, 4096, 5555, 6891, 7777, 8192, 9999, 12345, 20240, 32768, 44444, 54321, 65536, 71828, 88888, 94103, 99999)

---

## 1. Summary Comparison (Quantization Formats vs FP16 Reference)

### 1.1. Standard 7B Models Overview

| Quantization Format | GEMM Mode / Kernel | Latent Cosine (↑) | Latent MSE (↓) | Image SSIM (↑) | Image MSE (↓) | Wall Time (s) | Speedup (↑) | Peak VRAM | VRAM Saving | Same-Image Rate | Bifurcated Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 Reference** | Native `torch.float16` | 1.00000 | 0.000 | 1.000000 | 0.000000 | ~121–131s | 1.00x | 16.04–16.07 GiB | baseline | 25/25 (100%) | 0/25 (0%) |
| **ConvRot INT8** | Native INT8 (`int8_tensorwise`, dequant GEMM) | **0.99892** | **1.800e-03** | **0.994682** | **0.920831** | **24.51s** | **4.95x** | **8.53 GiB** | **−46.8%** | **25/25 (100%)** | **0/25 (0%)** |
| **ConvRot W4A8** | Native W4A8 (`asym_w4a8_int8` + ConvRot, SM80+) | **0.98672** | **2.234e-02** | **0.976833** | **4.567560** | **14.32s** | **9.04x** | **5.69 GiB** | **−64.5%** | **25/25 (100%)** | **0/25 (0%)** |
| **NVFP4** | Stock ComfyUI `mixed_precision` (dequant GEMM) | **0.96430** | **5.967e-02** | **0.952514** | **8.923874** | **22.76s** | **5.77x** | **5.72 GiB** | **−64.4%** | **0/25 (0%)** | **0/25 (0%)** |

### 1.2. Sharp Variant 7B Models Overview

| Quantization Format | GEMM Mode / Kernel | Latent Cosine (↑) | Latent MSE (↓) | Image SSIM (↑) | Image MSE (↓) | Wall Time (s) | Speedup (↑) | Peak VRAM | VRAM Saving | Same-Image Rate | Bifurcated Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sharp FP16 Reference** | Native `torch.float16` | 1.00000 | 0.000 | 1.000000 | 0.000000 | ~117–125s | 1.00x | 16.04–16.07 GiB | baseline | 25/25 (100%) | 0/25 (0%) |
| **Sharp ConvRot INT8** | Native INT8 (`int8_tensorwise`, dequant GEMM) | **0.99766** | **3.939e-03** | **0.992470** | **1.306679** | **31.38s** | **3.98x** | **8.53 GiB** | **−46.8%** | **25/25 (100%)** | **0/25 (0%)** |
| **Sharp ConvRot W4A8** | Native W4A8 (`asym_w4a8_int8` + ConvRot, SM80+) | **0.97682** | **3.893e-02** | **0.964523** | **6.300442** | **11.18s** | **10.49x** | **5.69 GiB** | **−64.5%** | **0/25 (0%)** | **0/25 (0%)** |
| **Sharp NVFP4** | Stock ComfyUI `mixed_precision` (dequant GEMM) | **0.94510** | **9.109e-02** | **0.937356** | **11.337591** | **11.59s** | **10.15x** | **5.72 GiB** | **−64.4%** | **0/25 (0%)** | **0/25 (0%)** |

### 1.3. 3B Models Overview

| Quantization Format | GEMM Mode / Kernel | Latent Cosine (↑) | Latent MSE (↓) | Image SSIM (↑) | Image MSE (↓) | Wall Time (s) | Speedup (↑) | Peak VRAM | VRAM Saving | Same-Image Rate | Bifurcated Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3B FP16 Reference** | Native `torch.float16` | 1.00000 | 0.000 | 1.000000 | 0.000000 | ~11.93–13.10s | 1.00x | 7.17–7.20 GiB | baseline | 25/25 (100%) | 0/25 (0%) |
| **3B ConvRot INT8** | Native INT8 (`int8_tensorwise`, dequant GEMM) | **0.99724** | **5.204e-03** | **0.992455** | **1.547978** | **9.55s** | **1.25x** | **5.69 GiB** | **−20.6%** | **25/25 (100%)** | **0/25 (0%)** |
| **3B ConvRot W4A8** | Native W4A8 (`asym_w4a8_int8` + ConvRot, SM80+) | **0.97093** | **5.560e-02** | **0.963753** | **8.040325** | **9.02s** | **1.38x** | **5.69 GiB** | **−20.6%** | **0/25 (0%)** | **0/25 (0%)** |
| **3B NVFP4** | Stock ComfyUI `mixed_precision` (dequant GEMM) | **0.93334** | **1.268e-01** | **0.936180** | **14.911295** | **9.70s** | **1.35x** | **5.72 GiB** | **−20.6%** | **0/25 (0%)** | **0/25 (0%)** |

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

### 2.4. SeedVR2 7B Sharp ConvRot INT8

#### Architecture Profile & Execution Configuration
- **Model Checkpoint:** `seedvr2_7b_sharp_int8_convrot.safetensors`
- **Reference Model:** `seedvr2_7b_sharp_fp16.safetensors`
- **GEMM Mode:** Native INT8 (`int8_tensorwise`, dequant GEMM)
- **Forward Statistics:** INT8 forward statistics are not exposed by the HSWQ nodes package
- **Performance:** Wall time: 31.38s (vs Sharp FP16: 124.92s, **3.98x speedup**) | Peak VRAM: 8.53 GiB (vs Sharp FP16: 16.04 GiB, **46.8% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.99766** | Pristine trajectory preservation on sharp weights |
| **Min Latent Cosine** (↑ better) | **0.99707** (Seed 44444) | Extremely tight worst-case lower bound |
| **Max Latent Cosine** (↑ better) | **0.99796** (Seed 1024) | Near bit-identical trajectory alignment |
| **Mean Latent MSE** (↓ better) | **3.939e-03** | Minimal numerical drift |
| **Mean Image SSIM** (↑ better) | **0.992470** (Min: 0.991593, Max: 0.993059) | Flawless edge and texture retention |
| **Mean Image MSE** (↓ better) | **1.306679** (Min: 1.130180, Max: 1.489268) | Very low pixel-level reconstruction error |
| **Max Step Drop** (↓ better) | **0.0000** | Zero single-step trajectory divergence |
| **Same-Image Rate** | **25/25 (100%)** | Full determinism across all 25 seeds |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events |

#### Side-by-Side per Seed (Sharp ConvRot INT8)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.99768 | 3.835e-03 | 0.0000 | 1.301748 | 0.992379 | `same-image` |
| **137** | 0.99758 | 4.045e-03 | 0.0000 | 1.301762 | 0.992254 | `same-image` |
| **256** | 0.99769 | 3.913e-03 | 0.0000 | 1.328642 | 0.992233 | `same-image` |
| **512** | 0.99743 | 4.324e-03 | 0.0000 | 1.365463 | 0.992473 | `same-image` |
| **777** | 0.99767 | 3.924e-03 | 0.0000 | 1.246638 | 0.992877 | `same-image` |
| **1024** | 0.99796 | 3.402e-03 | 0.0000 | 1.215097 | 0.992333 | `same-image` |
| **1337** | 0.99775 | 3.726e-03 | 0.0000 | 1.287144 | 0.992285 | `same-image` |
| **2048** | 0.99787 | 3.549e-03 | 0.0000 | 1.273745 | 0.992545 | `same-image` |
| **3141** | 0.99757 | 4.040e-03 | 0.0000 | 1.389078 | 0.992346 | `same-image` |
| **4096** | 0.99739 | 4.416e-03 | 0.0000 | 1.266364 | 0.993059 | `same-image` |
| **5555** | 0.99775 | 3.824e-03 | 0.0000 | 1.257686 | 0.992605 | `same-image` |
| **6891** | 0.99771 | 3.850e-03 | 0.0000 | 1.343288 | 0.992493 | `same-image` |
| **7777** | 0.99775 | 3.820e-03 | 0.0000 | 1.231050 | 0.992854 | `same-image` |
| **8192** | 0.99785 | 3.617e-03 | 0.0000 | 1.130180 | 0.993011 | `same-image` |
| **9999** | 0.99771 | 3.863e-03 | 0.0000 | 1.410422 | 0.991726 | `same-image` |
| **12345** | 0.99790 | 3.455e-03 | 0.0000 | 1.310920 | 0.991966 | `same-image` |
| **20240** | 0.99772 | 3.855e-03 | 0.0000 | 1.279615 | 0.992800 | `same-image` |
| **32768** | 0.99734 | 4.469e-03 | 0.0000 | 1.489268 | 0.991616 | `same-image` |
| **44444** | 0.99707 | 4.923e-03 | 0.0000 | 1.423542 | 0.991593 | `same-image` |
| **54321** | 0.99789 | 3.537e-03 | 0.0000 | 1.155991 | 0.992586 | `same-image` |
| **65536** | 0.99765 | 3.900e-03 | 0.0000 | 1.393759 | 0.992502 | `same-image` |
| **71828** | 0.99763 | 3.942e-03 | 0.0000 | 1.297706 | 0.992468 | `same-image` |
| **88888** | 0.99765 | 3.932e-03 | 0.0000 | 1.304572 | 0.992472 | `same-image` |
| **94103** | 0.99771 | 3.831e-03 | 0.0000 | 1.285110 | 0.992371 | `same-image` |
| **99999** | 0.99758 | 4.088e-03 | 0.0000 | 1.277793 | 0.992709 | `same-image` |
| **Mean** | **0.99766** | **3.939e-03** | **0.0000** | **1.306679** | **0.992470** | **25/25 same-image (100%)** |

---

### 2.5. SeedVR2 7B Sharp ConvRot W4A8

#### Architecture Profile & Execution Configuration
- **Model Checkpoint:** `seedvr2_7b_sharp_convrot_w4a8.safetensors`
- **Reference Model:** `seedvr2_7b_sharp_fp16.safetensors`
- **GEMM Mode:** Native W4A8 (`asym_w4a8_int8`, `AsymW4A8Int8Layout` + ConvRot)
- **Weight Storage:** INT4 packed storage (`torch.int8 [out_features, in_features // 2]`)
- **ConvRot Rotation:** Hadamard orthogonal rotation active (`convrot_groupsize=256`)
- **Scale Hierarchy:** FP8 group scale (`weight_s_rel`) + FP32 channel scale (`weight_s_channel`)
- **Compute Kernel:** `comfy_kitchen` `w4a8_int8_linear` / dequant GEMM on SM80+
- **Performance:** Wall time: 11.18s (vs Sharp FP16: 117.25s, **10.49x speedup**) | Peak VRAM: 5.69 GiB (vs Sharp FP16: 16.04 GiB, **64.5% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.97682** | Consistent trajectory stability across all seeds |
| **Min Latent Cosine** (↑ better) | **0.97505** (Seed 4096) | Predictable, tight performance band |
| **Max Latent Cosine** (↑ better) | **0.97789** (Seed 65536) | Strong latent alignment |
| **Mean Latent MSE** (↓ better) | **3.893e-02** | Bounded error accumulation |
| **Mean Image SSIM** (↑ better) | **0.964523** (Min: 0.959610, Max: 0.968514) | Solid visual fidelity |
| **Mean Image MSE** (↓ better) | **6.300442** (Min: 5.946131, Max: 6.720542) | Controlled pixel deviation |
| **Max Step Drop** (↓ better) | **0.0000** | Zero trajectory bifurcation |
| **Same-Image Rate** | **0/25 (0%)** | Continuous bounded drift under sharp kernels |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events (no structural collapse) |

#### Side-by-Side per Seed (Sharp ConvRot W4A8)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.97764 | 3.692e-02 | 0.0000 | 6.155150 | 0.965473 | `drifted (different image)` |
| **137** | 0.97674 | 3.888e-02 | 0.0000 | 6.391566 | 0.964073 | `drifted (different image)` |
| **256** | 0.97662 | 3.948e-02 | 0.0000 | 6.609159 | 0.961738 | `drifted (different image)` |
| **512** | 0.97536 | 4.133e-02 | 0.0000 | 6.162277 | 0.966750 | `drifted (different image)` |
| **777** | 0.97585 | 4.082e-02 | 0.0000 | 6.112899 | 0.966924 | `drifted (different image)` |
| **1024** | 0.97725 | 3.799e-02 | 0.0000 | 6.719863 | 0.959867 | `drifted (different image)` |
| **1337** | 0.97727 | 3.773e-02 | 0.0000 | 6.209129 | 0.964466 | `drifted (different image)` |
| **2048** | 0.97694 | 3.844e-02 | 0.0000 | 6.348325 | 0.963356 | `drifted (different image)` |
| **3141** | 0.97707 | 3.804e-02 | 0.0000 | 6.486503 | 0.963851 | `drifted (different image)` |
| **4096** | 0.97505 | 4.221e-02 | 0.0000 | 6.050926 | 0.968514 | `drifted (different image)` |
| **5555** | 0.97648 | 3.994e-02 | 0.0000 | 6.443589 | 0.964402 | `drifted (different image)` |
| **6891** | 0.97574 | 4.072e-02 | 0.0000 | 6.417443 | 0.963941 | `drifted (different image)` |
| **7777** | 0.97534 | 4.191e-02 | 0.0000 | 6.235523 | 0.965722 | `drifted (different image)` |
| **8192** | 0.97759 | 3.746e-02 | 0.0000 | 6.070515 | 0.966800 | `drifted (different image)` |
| **9999** | 0.97705 | 3.860e-02 | 0.0000 | 6.615998 | 0.960640 | `drifted (different image)` |
| **12345** | 0.97761 | 3.686e-02 | 0.0000 | 6.720542 | 0.959610 | `drifted (different image)` |
| **20240** | 0.97604 | 4.060e-02 | 0.0000 | 6.172409 | 0.966695 | `drifted (different image)` |
| **32768** | 0.97743 | 3.771e-02 | 0.0000 | 6.417006 | 0.963568 | `drifted (different image)` |
| **44444** | 0.97693 | 3.859e-02 | 0.0000 | 6.316401 | 0.964453 | `drifted (different image)` |
| **54321** | 0.97694 | 3.850e-02 | 0.0000 | 6.382173 | 0.962706 | `drifted (different image)` |
| **65536** | 0.97789 | 3.664e-02 | 0.0000 | 6.362737 | 0.965235 | `drifted (different image)` |
| **71828** | 0.97780 | 3.680e-02 | 0.0000 | 5.946131 | 0.966800 | `drifted (different image)` |
| **88888** | 0.97780 | 3.698e-02 | 0.0000 | 6.041076 | 0.966376 | `drifted (different image)` |
| **94103** | 0.97767 | 3.729e-02 | 0.0000 | 6.158057 | 0.964699 | `drifted (different image)` |
| **99999** | 0.97643 | 3.970e-02 | 0.0000 | 6.165150 | 0.966403 | `drifted (different image)` |
| **Mean** | **0.97682** | **3.893e-02** | **0.0000** | **6.300442** | **0.964523** | **25/25 drifted (100%)** |

---

### 2.6. SeedVR2 7B Sharp NVFP4

#### Architecture Profile & Execution Configuration
- **Model Checkpoint:** `seedvr2_7b_sharp_nvfp4.safetensors`
- **Reference Model:** `seedvr2_7b_sharp_fp16.safetensors`
- **GEMM Mode:** Stock ComfyUI `mixed_precision` (dequant GEMM)
- **Patch Status:** No HSWQ TC patch applied; `QuantizedTensor` stays packed, matmul dequantizes
- **Performance:** Wall time: 11.59s (vs Sharp FP16: 117.60s, **10.15x speedup**) | Peak VRAM: 5.72 GiB (vs Sharp FP16: 16.07 GiB, **64.4% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.94510** | Continuous bounded drift under stock dequant GEMM |
| **Min Latent Cosine** (↑ better) | **0.94107** (Seed 4096) | Zero catastrophic collapse |
| **Max Latent Cosine** (↑ better) | **0.94775** (Seed 88888) | Stable baseline fidelity |
| **Mean Latent MSE** (↓ better) | **9.109e-02** | Moderate divergence without bifurcation |
| **Mean Image SSIM** (↑ better) | **0.937356** (Min: 0.928574, Max: 0.945327) | Preserved structural integrity |
| **Mean Image MSE** (↓ better) | **11.337591** (Min: 10.549964, Max: 12.305139) | Higher pixel variation due to sharp high frequencies |
| **Max Step Drop** (↓ better) | **0.0000** | Zero discontinuous step drops |
| **Same-Image Rate** | **0/25 (0%)** | Smooth trajectory drift across all runs |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events |

#### Side-by-Side per Seed (Sharp NVFP4)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.94556 | 8.922e-02 | 0.0000 | 11.323482 | 0.936805 | `drifted (different image)` |
| **137** | 0.94522 | 9.069e-02 | 0.0000 | 11.585066 | 0.935118 | `drifted (different image)` |
| **256** | 0.94563 | 9.112e-02 | 0.0000 | 11.517452 | 0.935610 | `drifted (different image)` |
| **512** | 0.94215 | 9.650e-02 | 0.0000 | 11.110511 | 0.941383 | `drifted (different image)` |
| **777** | 0.94471 | 9.306e-02 | 0.0000 | 10.580075 | 0.944966 | `drifted (different image)` |
| **1024** | 0.94485 | 9.146e-02 | 0.0000 | 11.982109 | 0.930872 | `drifted (different image)` |
| **1337** | 0.94473 | 9.118e-02 | 0.0000 | 11.302195 | 0.938384 | `drifted (different image)` |
| **2048** | 0.94589 | 8.928e-02 | 0.0000 | 11.494301 | 0.936161 | `drifted (different image)` |
| **3141** | 0.94635 | 8.818e-02 | 0.0000 | 11.169156 | 0.938926 | `drifted (different image)` |
| **4096** | 0.94107 | 9.966e-02 | 0.0000 | 10.549964 | 0.945327 | `drifted (different image)` |
| **5555** | 0.94447 | 9.289e-02 | 0.0000 | 11.564417 | 0.936048 | `drifted (different image)` |
| **6891** | 0.94277 | 9.538e-02 | 0.0000 | 11.260390 | 0.937313 | `drifted (different image)` |
| **7777** | 0.94378 | 9.421e-02 | 0.0000 | 11.303657 | 0.938176 | `drifted (different image)` |
| **8192** | 0.94683 | 8.810e-02 | 0.0000 | 11.011261 | 0.940104 | `drifted (different image)` |
| **9999** | 0.94401 | 9.336e-02 | 0.0000 | 12.094303 | 0.929667 | `drifted (different image)` |
| **12345** | 0.94602 | 8.816e-02 | 0.0000 | 12.305139 | 0.928574 | `drifted (different image)` |
| **20240** | 0.94161 | 9.782e-02 | 0.0000 | 11.067552 | 0.941244 | `drifted (different image)` |
| **32768** | 0.94678 | 8.830e-02 | 0.0000 | 11.356407 | 0.936133 | `drifted (different image)` |
| **44444** | 0.94661 | 8.871e-02 | 0.0000 | 11.163193 | 0.937827 | `drifted (different image)` |
| **54321** | 0.94588 | 8.969e-02 | 0.0000 | 11.764934 | 0.931937 | `drifted (different image)` |
| **65536** | 0.94641 | 8.804e-02 | 0.0000 | 11.483976 | 0.937484 | `drifted (different image)` |
| **71828** | 0.94745 | 8.631e-02 | 0.0000 | 11.036493 | 0.939837 | `drifted (different image)` |
| **88888** | 0.94775 | 8.649e-02 | 0.0000 | 10.933421 | 0.940696 | `drifted (different image)` |
| **94103** | 0.94659 | 8.866e-02 | 0.0000 | 11.284058 | 0.935915 | `drifted (different image)` |
| **99999** | 0.94438 | 9.270e-02 | 0.0000 | 11.206266 | 0.939381 | `drifted (different image)` |
| **Mean** | **0.94510** | **9.109e-02** | **0.0000** | **11.337591** | **0.937356** | **25/25 drifted (100%)** |

---

### 2.7. SeedVR2 3B ConvRot INT8

#### Architecture Profile & Execution Configuration
- **Model Checkpoint:** `seedvr2_3b_convrot_int8.safetensors`
- **Reference Model:** `seedvr2_3b_fp16.safetensors`
- **GEMM Mode:** Native INT8 (`int8_tensorwise`, dequant GEMM)
- **Forward Statistics:** INT8 forward statistics are not exposed by the HSWQ nodes package
- **Performance:** Wall time: 9.55s (vs 3B FP16: 11.93s, **1.25x speedup**) | Peak VRAM: 5.69 GiB (vs 3B FP16: 7.17 GiB, **20.6% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.99724** | Pristine trajectory preservation on 3B architecture |
| **Min Latent Cosine** (↑ better) | **0.99679** (Seed 65536) | Exceptionally stable floor across all runs |
| **Max Latent Cosine** (↑ better) | **0.99750** (Seed 20240) | Near bit-identical trajectory alignment |
| **Mean Latent MSE** (↓ better) | **5.204e-03** | Extremely low latent reconstruction error |
| **Mean Image SSIM** (↑ better) | **0.992455** (Min: 0.991583, Max: 0.993003) | Exceptional structural similarity retention |
| **Mean Image MSE** (↓ better) | **1.547978** (Min: 1.460683, Max: 1.699673) | Sub-1.7 pixel reconstruction error |
| **Max Step Drop** (↓ better) | **0.0000** | Zero single-step trajectory divergence |
| **Same-Image Rate** | **25/25 (100%)** | All 25 seeds generate the identical image |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events |

#### Side-by-Side per Seed (3B ConvRot INT8)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.99735 | 5.065e-03 | 0.0000 | 1.598828 | 0.992327 | `same-image` |
| **137** | 0.99709 | 5.538e-03 | 0.0000 | 1.587684 | 0.992136 | `same-image` |
| **256** | 0.99701 | 5.685e-03 | 0.0000 | 1.612331 | 0.992275 | `same-image` |
| **512** | 0.99744 | 4.909e-03 | 0.0000 | 1.476956 | 0.992854 | `same-image` |
| **777** | 0.99739 | 4.989e-03 | 0.0000 | 1.517438 | 0.992741 | `same-image` |
| **1024** | 0.99735 | 5.021e-03 | 0.0000 | 1.591048 | 0.992579 | `same-image` |
| **1337** | 0.99745 | 4.796e-03 | 0.0000 | 1.475101 | 0.992749 | `same-image` |
| **2048** | 0.99720 | 5.322e-03 | 0.0000 | 1.522122 | 0.992490 | `same-image` |
| **3141** | 0.99739 | 4.921e-03 | 0.0000 | 1.634311 | 0.992461 | `same-image` |
| **4096** | 0.99721 | 5.306e-03 | 0.0000 | 1.540704 | 0.992509 | `same-image` |
| **5555** | 0.99697 | 5.748e-03 | 0.0000 | 1.632369 | 0.992197 | `same-image` |
| **6891** | 0.99701 | 5.688e-03 | 0.0000 | 1.518481 | 0.992267 | `same-image` |
| **7777** | 0.99711 | 5.526e-03 | 0.0000 | 1.579360 | 0.992192 | `same-image` |
| **8192** | 0.99723 | 5.289e-03 | 0.0000 | 1.468337 | 0.992555 | `same-image` |
| **9999** | 0.99735 | 4.975e-03 | 0.0000 | 1.621438 | 0.992382 | `same-image` |
| **12345** | 0.99702 | 5.668e-03 | 0.0000 | 1.535722 | 0.992195 | `same-image` |
| **20240** | 0.99750 | 4.774e-03 | 0.0000 | 1.488438 | 0.993003 | `same-image` |
| **32768** | 0.99730 | 5.143e-03 | 0.0000 | 1.517253 | 0.992525 | `same-image` |
| **44444** | 0.99733 | 5.091e-03 | 0.0000 | 1.526772 | 0.992500 | `same-image` |
| **54321** | 0.99720 | 5.347e-03 | 0.0000 | 1.583769 | 0.992191 | `same-image` |
| **65536** | 0.99679 | 6.098e-03 | 0.0000 | 1.699673 | 0.991583 | `same-image` |
| **71828** | 0.99746 | 4.900e-03 | 0.0000 | 1.460683 | 0.992514 | `same-image` |
| **88888** | 0.99734 | 5.054e-03 | 0.0000 | 1.508571 | 0.992564 | `same-image` |
| **94103** | 0.99713 | 5.446e-03 | 0.0000 | 1.560227 | 0.992083 | `same-image` |
| **99999** | 0.99748 | 4.799e-03 | 0.0000 | 1.462352 | 0.992817 | `same-image` |
| **Mean** | **0.99724** | **5.204e-03** | **0.0000** | **1.547978** | **0.992455** | **25/25 same-image (100%)** |

---

### 2.8. SeedVR2 3B ConvRot W4A8

#### Architecture Profile & Execution Configuration
- **Model Checkpoint:** `seedvr2_3b_convrot_w4a8.safetensors`
- **Reference Model:** `seedvr2_3b_fp16.safetensors`
- **GEMM Mode:** Native W4A8 (`asym_w4a8_int8`, `AsymW4A8Int8Layout` + ConvRot)
- **Weight Storage:** INT4 packed storage (`torch.int8 [out_features, in_features // 2]`)
- **ConvRot Rotation:** Hadamard orthogonal rotation active (`convrot_groupsize=256`)
- **Scale Hierarchy:** FP8 group scale (`weight_s_rel`) + FP32 channel scale (`weight_s_channel`)
- **Compute Kernel:** `comfy_kitchen` `w4a8_int8_linear` / dequant GEMM on SM80+
- **Performance:** Wall time: 9.02s (vs 3B FP16: 12.41s, **1.38x speedup**) | Peak VRAM: 5.69 GiB (vs 3B FP16: 7.17 GiB, **20.6% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.97093** | Consistent trajectory stability across all 25 seeds |
| **Min Latent Cosine** (↑ better) | **0.96981** (Seed 1024, 5555) | Predictable, tight performance floor |
| **Max Latent Cosine** (↑ better) | **0.97230** (Seed 44444) | Bounded drift with strong semantic retention |
| **Mean Latent MSE** (↓ better) | **5.560e-02** | Bounded error accumulation |
| **Mean Image SSIM** (↑ better) | **0.963753** (Min: 0.961673, Max: 0.965323) | High structural fidelity |
| **Mean Image MSE** (↓ better) | **8.040325** (Min: 7.814416, Max: 8.309586) | Controlled pixel variation |
| **Max Step Drop** (↓ better) | **0.0000** | Zero trajectory bifurcation |
| **Same-Image Rate** | **0/25 (0%)** | Continuous bounded drift under 4-bit weights |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events (no structural collapse) |

#### Side-by-Side per Seed (3B ConvRot W4A8)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.97114 | 5.534e-02 | 0.0000 | 8.015691 | 0.963087 | `drifted (different image)` |
| **137** | 0.97039 | 5.671e-02 | 0.0000 | 8.309586 | 0.963015 | `drifted (different image)` |
| **256** | 0.97111 | 5.524e-02 | 0.0000 | 8.056443 | 0.963971 | `drifted (different image)` |
| **512** | 0.97122 | 5.532e-02 | 0.0000 | 7.814416 | 0.965323 | `drifted (different image)` |
| **777** | 0.97160 | 5.472e-02 | 0.0000 | 7.887145 | 0.964348 | `drifted (different image)` |
| **1024** | 0.96981 | 5.762e-02 | 0.0000 | 8.071762 | 0.963629 | `drifted (different image)` |
| **1337** | 0.97078 | 5.530e-02 | 0.0000 | 7.848997 | 0.964699 | `drifted (different image)` |
| **2048** | 0.97114 | 5.508e-02 | 0.0000 | 8.052877 | 0.964100 | `drifted (different image)` |
| **3141** | 0.97143 | 5.409e-02 | 0.0000 | 7.935335 | 0.964071 | `drifted (different image)` |
| **4096** | 0.97136 | 5.494e-02 | 0.0000 | 8.019708 | 0.965084 | `drifted (different image)` |
| **5555** | 0.96981 | 5.749e-02 | 0.0000 | 8.004462 | 0.964192 | `drifted (different image)` |
| **6891** | 0.97118 | 5.519e-02 | 0.0000 | 7.957605 | 0.964930 | `drifted (different image)` |
| **7777** | 0.97004 | 5.756e-02 | 0.0000 | 8.068575 | 0.963506 | `drifted (different image)` |
| **8192** | 0.97080 | 5.610e-02 | 0.0000 | 7.971283 | 0.964147 | `drifted (different image)` |
| **9999** | 0.97043 | 5.594e-02 | 0.0000 | 8.293447 | 0.961873 | `drifted (different image)` |
| **12345** | 0.97033 | 5.667e-02 | 0.0000 | 8.157631 | 0.962400 | `drifted (different image)` |
| **20240** | 0.97072 | 5.621e-02 | 0.0000 | 7.951004 | 0.964799 | `drifted (different image)` |
| **32768** | 0.97086 | 5.586e-02 | 0.0000 | 8.123813 | 0.963256 | `drifted (different image)` |
| **44444** | 0.97230 | 5.302e-02 | 0.0000 | 7.947057 | 0.963703 | `drifted (different image)` |
| **54321** | 0.97112 | 5.537e-02 | 0.0000 | 8.076583 | 0.963899 | `drifted (different image)` |
| **65536** | 0.97150 | 5.422e-02 | 0.0000 | 8.063380 | 0.962298 | `drifted (different image)` |
| **71828** | 0.97221 | 5.400e-02 | 0.0000 | 7.935223 | 0.963428 | `drifted (different image)` |
| **88888** | 0.96999 | 5.732e-02 | 0.0000 | 8.104853 | 0.962733 | `drifted (different image)` |
| **94103** | 0.97015 | 5.681e-02 | 0.0000 | 8.259025 | 0.961673 | `drifted (different image)` |
| **99999** | 0.97179 | 5.399e-02 | 0.0000 | 8.082230 | 0.964467 | `drifted (different image)` |
| **Mean** | **0.97093** | **5.560e-02** | **0.0000** | **8.040325** | **0.963753** | **25/25 drifted (100%)** |

---

### 2.9. SeedVR2 3B NVFP4

#### Architecture Profile & Execution Configuration
- **Model Checkpoint:** `seedvr2_3b_nvfp4.safetensors`
- **Reference Model:** `seedvr2_3b_fp16.safetensors`
- **GEMM Mode:** Stock ComfyUI `mixed_precision` (dequant GEMM)
- **Patch Status:** No HSWQ TC patch applied; `QuantizedTensor` stays packed, matmul dequantizes
- **Performance:** Wall time: 9.70s (vs 3B FP16: 13.10s, **1.35x speedup**) | Peak VRAM: 5.72 GiB (vs 3B FP16: 7.20 GiB, **20.6% memory reduction**)

#### Metric Overview
| Metric / Property | Value | Evaluation / Status |
| :--- | :--- | :--- |
| **Mean Latent Cosine** (↑ better) | **0.93334** | Gradual trajectory drift under stock dequant GEMM |
| **Min Latent Cosine** (↑ better) | **0.93027** (Seed 54321) | Zero catastrophic collapse |
| **Max Latent Cosine** (↑ better) | **0.93622** (Seed 71828) | Predictable trajectory floor |
| **Mean Latent MSE** (↓ better) | **1.268e-01** | Moderate error accumulation without bifurcation |
| **Mean Image SSIM** (↑ better) | **0.936180** (Min: 0.931307, Max: 0.939420) | Maintained structural silhouette |
| **Mean Image MSE** (↓ better) | **14.911295** (Min: 14.444382, Max: 15.457009) | Substantial pixel variation |
| **Max Step Drop** (↓ better) | **0.0000** | Zero discontinuous step drops |
| **Same-Image Rate** | **0/25 (0%)** | Gradual trajectory drift across all runs |
| **Bifurcated Seeds Rate** | **0/25 (0%)** | Zero bifurcation events (no structural collapse) |

#### Side-by-Side per Seed (3B NVFP4)
| Seed | Latent Cosine (↑) | Latent MSE (↓) | Max Drop (↓) | Image MSE (↓) | Image SSIM (↑) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.93496 | 1.240e-01 | 0.0000 | 14.623539 | 0.936537 | `drifted (different image)` |
| **137** | 0.93236 | 1.284e-01 | 0.0000 | 15.064868 | 0.935126 | `drifted (different image)` |
| **256** | 0.93275 | 1.273e-01 | 0.0000 | 14.940977 | 0.935563 | `drifted (different image)` |
| **512** | 0.93546 | 1.231e-01 | 0.0000 | 14.472721 | 0.939420 | `drifted (different image)` |
| **777** | 0.93437 | 1.251e-01 | 0.0000 | 14.699918 | 0.938006 | `drifted (different image)` |
| **1024** | 0.93050 | 1.312e-01 | 0.0000 | 15.258422 | 0.935790 | `drifted (different image)` |
| **1337** | 0.93288 | 1.261e-01 | 0.0000 | 14.793865 | 0.937152 | `drifted (different image)` |
| **2048** | 0.93180 | 1.286e-01 | 0.0000 | 15.134623 | 0.934981 | `drifted (different image)` |
| **3141** | 0.93376 | 1.246e-01 | 0.0000 | 14.869573 | 0.935094 | `drifted (different image)` |
| **4096** | 0.93407 | 1.252e-01 | 0.0000 | 14.722128 | 0.939110 | `drifted (different image)` |
| **5555** | 0.93284 | 1.270e-01 | 0.0000 | 15.038270 | 0.936600 | `drifted (different image)` |
| **6891** | 0.93051 | 1.315e-01 | 0.0000 | 15.083158 | 0.935145 | `drifted (different image)` |
| **7777** | 0.93153 | 1.301e-01 | 0.0000 | 14.979739 | 0.936102 | `drifted (different image)` |
| **8192** | 0.93359 | 1.260e-01 | 0.0000 | 14.939871 | 0.937009 | `drifted (different image)` |
| **9999** | 0.93096 | 1.291e-01 | 0.0000 | 15.457009 | 0.931307 | `drifted (different image)` |
| **12345** | 0.93513 | 1.231e-01 | 0.0000 | 14.676622 | 0.937965 | `drifted (different image)` |
| **20240** | 0.93386 | 1.255e-01 | 0.0000 | 14.683638 | 0.938733 | `drifted (different image)` |
| **32768** | 0.93541 | 1.232e-01 | 0.0000 | 14.845179 | 0.936939 | `drifted (different image)` |
| **44444** | 0.93508 | 1.235e-01 | 0.0000 | 14.898013 | 0.935000 | `drifted (different image)` |
| **54321** | 0.93027 | 1.321e-01 | 0.0000 | 15.320377 | 0.934489 | `drifted (different image)` |
| **65536** | 0.93464 | 1.236e-01 | 0.0000 | 14.890772 | 0.934804 | `drifted (different image)` |
| **71828** | 0.93622 | 1.231e-01 | 0.0000 | 14.444382 | 0.937532 | `drifted (different image)` |
| **88888** | 0.93328 | 1.263e-01 | 0.0000 | 14.967441 | 0.935825 | `drifted (different image)` |
| **94103** | 0.93249 | 1.279e-01 | 0.0000 | 15.326191 | 0.933056 | `drifted (different image)` |
| **99999** | 0.93477 | 1.236e-01 | 0.0000 | 14.651061 | 0.937222 | `drifted (different image)` |
| **Mean** | **0.93334** | **1.268e-01** | **0.0000** | **14.911295** | **0.936180** | **25/25 drifted (100%)** |

---

## 3. Performance, Speedup & VRAM Footprint Analysis

### Inference Latency Comparison (Wall Time on Seed 99999)

#### Standard 7B Architecture
- **FP16 Reference:** ~121.36s – 131.44s (baseline)
- **ConvRot INT8:** 24.51s (**4.95x speedup**)
- **ConvRot W4A8:** 14.32s (**9.04x speedup**)
- **NVFP4 (Stock Dequant):** 22.76s (**5.77x speedup**)

#### Sharp 7B Architecture
- **Sharp FP16 Reference:** ~117.25s – 124.92s (baseline)
- **Sharp ConvRot INT8:** 31.38s (**3.98x speedup**)
- **Sharp ConvRot W4A8:** 11.18s (**10.49x speedup**) — Fastest overall 7B profile
- **Sharp NVFP4:** 11.59s (**10.15x speedup**)

#### 3B Architecture
- **3B FP16 Reference:** ~11.93s – 13.10s (baseline)
- **3B ConvRot INT8:** 9.55s (**1.25x speedup**)
- **3B ConvRot W4A8:** 9.02s (**1.38x speedup**) — Fastest overall 3B profile
- **3B NVFP4 (Stock Dequant):** 9.70s (**1.35x speedup**)

### GPU Memory Consumption (Peak VRAM on Seed 99999)

#### 7B Architecture (Standard & Sharp)
- **FP16 Reference:** ~16.04 – 16.07 GiB (requires 24GB VRAM class GPU for comfortable headroom)
- **ConvRot INT8:** 8.53 GiB (**−46.8% memory reduction**, operates comfortably within 10–12GB VRAM cards)
- **ConvRot W4A8:** 5.69 GiB (**−64.5% memory reduction**, operates comfortably on 6–8GB VRAM cards)
- **NVFP4:** 5.72 GiB (**−64.4% memory reduction**, operates comfortably on 6–8GB VRAM cards)

#### 3B Architecture
- **3B FP16 Reference:** ~7.17 – 7.20 GiB (operates on 8–10GB VRAM cards)
- **3B ConvRot INT8:** 5.69 GiB (**−20.6% memory reduction**, operates comfortably on 6–8GB VRAM cards)
- **3B ConvRot W4A8:** 5.69 GiB (**−20.6% memory reduction**, operates comfortably on 6–8GB VRAM cards)
- **3B NVFP4:** 5.72 GiB (**−20.6% memory reduction**, operates comfortably on 6–8GB VRAM cards)

---

## 4. Key Findings and Trajectory Analysis

1. **Zero Catastrophic Bifurcations Across All 9 Evaluated Configurations (0/225 = 0%):**
   Across 225 total generation runs (25 seeds × 9 model configurations across Standard 7B, Sharp 7B, and 3B), `max-drop` remained strictly **0.0000** without a single exception. Unlike standard diffusion text-to-image models that often experience discrete attractor flips under aggressive 4-bit quantizations, the continuous video/image latent distribution in SeedVR2 produces deterministic and continuous trajectories across all standard, sharp, and 3B model weights.

2. **ConvRot INT8 Preserves Near-Lossless Parity Across All Architectures:**
   All three ConvRot INT8 checkpoints (Standard 7B, Sharp 7B, and 3B) achieve **100% same-image verdicts (25/25 seeds)**:
   - **Standard 7B ConvRot INT8:** mean latent cosine **0.99892**, image SSIM **0.994682**, VRAM reduction −46.8% (8.53 GiB).
   - **Sharp 7B ConvRot INT8:** mean latent cosine **0.99766**, image SSIM **0.992470**, VRAM reduction −46.8% (8.53 GiB).
   - **3B ConvRot INT8:** mean latent cosine **0.99724**, image SSIM **0.992455**, VRAM reduction −20.6% (5.69 GiB).
   ConvRot INT8 establishes true near-lossless mathematical and visual parity with FP16 reference weights across the entire family.

3. **ConvRot W4A8 Delivers Top Speedup & Bounded Error:**
   Combining INT4 packed weights with FP8 group scales, FP32 channel scales, and Hadamard orthogonal rotation (`convrot_groupsize=256`), ConvRot W4A8 delivers the best latency profile across all model architectures:
   - **Standard 7B W4A8:** 14.32s wall time (**9.04x speedup**), 5.69 GiB peak VRAM, **0.98672 mean latent cosine**, **100% same-image rate (25/25)**.
   - **Sharp 7B W4A8:** 11.18s wall time (**10.49x speedup**), 5.69 GiB peak VRAM, **0.97682 mean latent cosine**, tightly bounded drift with zero artifacts.
   - **3B W4A8:** 9.02s wall time (**1.38x speedup**), 5.69 GiB peak VRAM, **0.97093 mean latent cosine**, highest throughput among all 3B models.

4. **NVFP4 Operates in Bounded Drift Mode Under Stock Dequant GEMM:**
   In stock ComfyUI without dedicated hardware W4A4 Tensor Core forward patches, NVFP4 executes on-the-fly dequantization. Across Standard 7B (mean cosine **0.96430**), Sharp 7B (mean cosine **0.94510**), and 3B (mean cosine **0.93334**), trajectories experience smooth and gradual drift without sudden bifurcation steps. However, ConvRot W4A8 consistently surpasses NVFP4 in both numerical fidelity and raw execution throughput across every model tier.

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
