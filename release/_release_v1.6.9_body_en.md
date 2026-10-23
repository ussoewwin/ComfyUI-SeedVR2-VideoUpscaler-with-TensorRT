<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/zhmd/v1.6.9.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# SeedVR2 Video Upscaler — Release Notes (v1.6.9)

## Overview

SeedVR2 Video Upscaler v1.6.9 introduces an end-to-end, systematic **VRAM Contamination Elimination & Deep Memory Reclamation Architecture**.

In complex ComfyUI workflows, long video or ultra-high resolution upscaling runs frequently suffer from cumulative memory degradation, leading to Out-Of-Memory (`CUDA out of memory`) crashes or generation artifacts. This stems from unevicted predecessor models in ComfyUI's model tracker, dangling tensors across temporal batches, and memory fragmentation within PyTorch's CUDA caching allocator. This release integrates ComfyUI's native model management directly with low-level PyTorch CUDA synchronization barriers at the node entry point, across every batch boundary in all four pipeline phases, within TensorRT VAE chunks, and across the core memory manager.

---

## 1. Technical Analysis of VRAM Contamination Root Causes

In the ComfyUI runtime environment, relying solely on standard Python `del` or standalone PyTorch `torch.cuda.empty_cache()` calls fails to prevent memory pool pollution due to three underlying architectural factors:

1. **Decoupled ComfyUI Model Lifecycle Tracking**:
   ComfyUI maintains its own internal model management system (`comfy.model_management`). Once upstream nodes finish executing, model weights remain pinned in VRAM for speculative caching. Without explicit eviction requests (`cleanup_models`), these inactive models continuously hold gigabytes of physical VRAM.
2. **PyTorch Caching Allocator Fragmentation**:
   Frequent allocation and deallocation of massive intermediate tensors (e.g., 8K latent grids, temporal buffers, TensorRT accumulator maps) cause severe fragmentation in the PyTorch caching pool. Without explicit `torch.cuda.synchronize()` barriers coordinated with garbage collection, these fragmented blocks cannot coalesce, preventing large tensor allocations even when total free VRAM appears sufficient.
3. **Inter-Batch Tensor Retention Across Pipeline Phases (Phases 1–4)**:
   In multi-batch video loops, intermediate computational variables left unsevered at the end of a batch loop persist into subsequent iterations, causing memory consumption to step upward with every processed batch.

---

## 2. Systematic Mitigation Architecture & Implementation

### 2.1 Pre- and Post-Execution ComfyUI Model & Cache Purge (`video_upscaler.py`)

At the absolute entry point of `SeedVR2 Video Upscaler` (prior to model loading) and in its terminal `finally` cleanup block, high-priority model eviction is enforced:

```python
# Pre-execution and Post-execution ComfyUI VRAM Pollution Purge
try:
    import comfy.model_management as mm
    mm.cleanup_models_gc()
    mm.cleanup_models()
    mm.soft_empty_cache(force=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
except Exception:
    pass
```

- **Mechanics & Operational Effects**:
  - `mm.cleanup_models_gc()`: Forces ComfyUI's internal garbage collection to evaluate resident model references.
  - `mm.cleanup_models()`: Immediately unloads and unbinds inactive models inherited from predecessor nodes.
  - `mm.soft_empty_cache(force=True)`: Flushes ComfyUI's managed soft cache pool.
  - `torch.cuda.synchronize()`: Blocks until all pending asynchronous CUDA stream deallocations finish on physical hardware.

---

### 2.2 Per-Batch Deep Memory Cleansing Across All 4 Pipeline Phases (`generation_phases.py`)

Across all four core execution phases in SeedVR2:
- **Phase 1**: VAE Encoding (`encode_all_batches`)
- **Phase 2**: DiT Latent Upscaling (`upscale_all_batches`)
- **Phase 3**: VAE Decoding (`decode_all_batches`)
- **Phase 4**: Video Post-Processing & Reassembly (`postprocess_all_batches`)

A rigorous four-tier deep purge sequence executes at the conclusion of every individual batch iteration:

```python
# Per-batch boundary deep memory reclamation sequence
import gc as _gc
_gc.collect()
if torch.cuda.is_available():
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache(force=True)
    except Exception:
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

- **Mechanics & Operational Effects**:
  Completely isolates memory states between sequential batches. Even in ultra-long video renders or massive resolution scaling, VRAM occupancy resets to its baseline level after every batch, eliminating memory leaks across batches.

---

### 2.3 TensorRT VAE & ComfyUI Memory Manager Synchronization (`trt_decoder.py` / `trt_encoder.py`)

TensorRT spatial tiling requires continuous memory exchange between dedicated TRT execution streams and the PyTorch default stream. After each spatial chunk completes, dual-channel deallocation synchronization is triggered:

```python
# trt_decoder.py & trt_encoder.py post-chunk reclamation
del source, result, weights
import gc as _gc
_gc.collect()
if torch.cuda.is_available():
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache(force=True)
    except Exception:
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

- **Mechanics & Operational Effects**:
  Severing accumulator tensor references prior to executing `torch.cuda.empty_cache()` synchronized with ComfyUI's `soft_empty_cache` guarantees memory ordering between the TensorRT execution engine and ComfyUI's memory manager.

---

### 2.4 Core Memory Manager Integration (`memory_manager.py`)

The global memory reclamation utility `clear_memory()` now natively incorporates ComfyUI soft-cache drainage:

```python
if is_cuda_available():
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache(force=True)
    except Exception:
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
```

---

## 3. Modified Modules & Scope

| Module / File | Status | Role & Modifications |
| :--- | :--- | :--- |
| `src/interfaces/video_upscaler.py` | Modified | Added pre-execution and terminal `finally` model purge (`cleanup_models`, `cleanup_models_gc`, `soft_empty_cache`) to evict upstream workflow models. |
| `src/core/generation_phases.py` | Modified | Embedded per-batch four-tier deep memory reclamation across Phase 1 (Encode), Phase 2 (Upscale), Phase 3 (Decode), and Phase 4 (Postprocess). |
| `src/core/trt_decoder.py` | Modified | Synchronized per-chunk VAE decode cleanup with `soft_empty_cache(force=True)` and CUDA barrier. |
| `src/core/trt_encoder.py` | Modified | Synchronized per-chunk VAE encode cleanup with `soft_empty_cache(force=True)` and CUDA barrier. |
| `src/optimization/memory_manager.py` | Modified | Unified ComfyUI soft-cache drainage and CUDA barrier synchronization inside `clear_memory()`. |
| `zhmd/v1.6.9.md` | Created | Complete Chinese technical release notes and specifications for v1.6.9. |
| `md/changelog.md` / `zhmd/changelogzh.md` | Modified | Synchronized changelog entries for v1.6.9. |
