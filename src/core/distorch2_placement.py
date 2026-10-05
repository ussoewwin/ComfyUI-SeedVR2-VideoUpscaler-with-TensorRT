"""
SeedVR2 DiT <-> DisTorch2 placement bridge.

Bridges SeedVR2's config-dict DiT load path onto the VERBATIM
ComfyUI-MultiGPU (comfyui-multigpu) DisTorch2 backend copied into
``src/distorch2/`` (GPL-3.0, pollockjj). The upstream analyzer
``analyze_safetensor_loading`` expects a ComfyUI ``ModelPatcher``; we feed it a
minimal adapter exposing ``_load_list()`` / ``model`` so the upstream code runs
unmodified, then physically place every module per its ``block_assignments``.

Log output is identical to upstream ("MultiGPU" logger: DisTorch2 Model Device
Allocations / Layer Distribution / Final Device/Layer Assignments).

Quantized-model note: the upstream virtual-VRAM calculator derives model size
from ``weight.numel() * weight.element_size()``, which for a QuantizedTensor is
the LOGICAL dtype size (e.g. int8 packed 7.9GB reads as fp16/bf16 15.3GB). The
per-block distribution uses real module sizes (packed). To keep both consistent
for INT8/NVFP4 DiT packs, this bridge computes the allocation string from the
PACKED sizes and hands the explicit string to the untouched upstream analyzer.
"""

import logging

import torch

from ..optimization.memory_manager import clear_memory

logger = logging.getLogger("MultiGPU")


def _is_w4a8_quant(t) -> bool:
    """__W4A8_SUPPORT__: True only for an asym_w4a8_int8 QuantizedTensor.

    W4A8 is identified by its ComfyUI layout tag (``AsymW4A8Int8Layout``), NOT by
    a width heuristic. A packed int4 (W4A8) and a packed fp4 (NVFP4) BOTH halve
    the last dim, so a `_qdata.shape[1] * 2 == weight.shape[1]` test also matches
    NVFP4 and would MERGE the two formats. This discriminator keeps W4A8 fully
    separate: INT8 (TensorWiseINT8Layout) and NVFP4 (TensorCoreNVFP4Layout) never
    match.
    """
    if getattr(t, "_layout_cls", None) == "AsymW4A8Int8Layout":
        return True
    # Fallback for a W4A8 tensor that somehow lacks its layout tag: require the
    # asym-W4A8 Params schema (per-group `scale` + per-channel `s_channel` +
    # `group_size`). No other layout carries that combination.
    p = getattr(t, "_params", None)
    if p is None:
        return False
    return all(hasattr(p, a) for a in ("scale", "s_channel", "group_size"))


def _w4a8_storage_bytes(t) -> int:
    """__W4A8_SUPPORT__: real resident bytes of a W4A8 QuantizedTensor.

    A W4A8 pack keeps its per-group fp8 scale and per-channel scale in SEPARATE
    storages under `_params` (not inside `_qdata`), so `_qdata` alone understates
    each layer by ~2.3MB. This path counts `_qdata` plus those companion tensors.
    Dedicated to W4A8; other formats never reach it.
    """
    q = t._qdata
    total = q.numel() * q.element_size()
    params = getattr(t, "_params", None)
    if params is not None:
        for attr in ("scale", "s_channel", "correction", "codebook"):
            v = getattr(params, attr, None)
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def _qt_storage_bytes(t):
    """Actual storage bytes of a tensor (QuantizedTensor-aware)."""
    q = getattr(t, "_qdata", None)
    if q is not None:
        if _is_w4a8_quant(t):
            return _w4a8_storage_bytes(t)
        return q.numel() * q.element_size()
    return t.numel() * t.element_size()


def _packed_model_bytes(model) -> int:
    """Sum of real parameter storage (int8 stays int8) over all modules."""
    total = 0
    seen = set()
    for module in model.modules():
        for p in module.parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += _qt_storage_bytes(p.data if isinstance(p, torch.nn.Parameter) else p)
        for b in module.buffers(recurse=False):
            if b is not None and b.numel() > 0:
                total += _qt_storage_bytes(b)
    return total


class FakePatcherAdapter:
    """Minimal ModelPatcher-like adapter for the upstream DisTorch2 analyzer.

    Upstream ``analyze_safetensor_loading`` uses only:
      - model_patcher._load_list()
      - model_patcher.model (named_modules for the size scan)
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.is_clip = False

    def _load_list(self, for_dynamic=False, default_device=None):
        # Mirrors comfy.model_patcher.ModelPatcher._load_list for plain modules:
        # one entry per leaf module that carries parameters, legacy 4-tuple layout.
        import comfy.model_management as mm
        loading = []
        for n, m in self.model.named_modules():
            params = {name: p for name, p in m.named_parameters(recurse=False)}
            if not params:
                continue
            module_mem = mm.module_size(m)
            loading.append((module_mem, n, m, params))
        return loading


def ensure_backend_registered() -> bool:
    """Register the upstream ModelPatcher patch (idempotent)."""
    try:
        from ..distorch2.distorch_2 import register_patched_safetensor_modelpatcher
        register_patched_safetensor_modelpatcher()
        return True
    except Exception as e:
        logger.warning(f"[MultiGPU DisTorch V2] backend registration failed: {e}")
        return False


def build_packed_allocation_string(model,
                                   compute_device: str,
                                   virtual_vram_gb: float,
                                   donor_device: str) -> str:
    """
    Build an upstream-format allocation string from PACKED model size:

        "cuda:0,<keep_frac>;cpu,<donate_frac>"

    ``virtual_vram_gb`` is the amount of the (packed) model hosted on the donor
    device; the remainder stays on the compute device. Fractions are relative to
    each device's total memory, exactly as the upstream analyzer expects.
    """
    import comfy.model_management as mm

    packed_gb = _packed_model_bytes(model) / (1024 ** 3)
    vv = max(0.0, min(float(virtual_vram_gb or 0.0), packed_gb))
    keep_gb = packed_gb - vv

    recip_total = mm.get_total_memory(torch.device(compute_device)) / (1024 ** 3)
    donor_total = mm.get_total_memory(torch.device(donor_device)) / (1024 ** 3)

    keep_frac = (keep_gb / recip_total) if recip_total > 0 else 0.0
    donor_frac = (vv / donor_total) if donor_total > 0 else 0.0

    alloc = f"{compute_device},{keep_frac:.4f};{donor_device},{donor_frac:.4f}"
    logger.info(
        f"[MultiGPU DisTorch V2] SeedVR2 packed model size: {packed_gb:.2f}GB "
        f"(int8/nvfp4 aware); keep {keep_gb:.2f}GB on {compute_device}, "
        f"host {vv:.2f}GB on {donor_device}"
    )
    return alloc


def _parse_vram_string(allocation_string: str):
    """Parse the node/CLI vram-format string into (compute, virtual_gb, donor).

    Accepted forms:
      "#compute;virtual_gb;donor"          (plain vram string)
      "compute;virtual_gb;donor"           (same, no leading '#')
      "expert#compute;virtual_gb;donor"    (expert prefix is preserved upstream)
    Returns None when the string is not a vram-format string.
    """
    if not allocation_string:
        return None
    expert, sep, rest = allocation_string.partition("#")
    if not sep:
        # no '#': vram format only if it contains ';'
        if ";" not in allocation_string:
            return None
        expert, rest = "", allocation_string
    parts = rest.split(";")
    if len(parts) < 2:
        return None
    try:
        vv = float(parts[1])
    except (TypeError, ValueError):
        return None
    compute = parts[0]
    donor = parts[2] if len(parts) >= 3 and parts[2] else "cpu"
    return (expert, compute, vv, donor)


def wrap_blocks_with_cache_release(model, debug=None, blockswap_active=False):
    """对齐 BlockSwap 的已验证机制：每个 block forward 结束后强制释放缓存。

    BlockSwap 模式在 wrap_block_forward 中每个 block 结束后调用
    clear_memory(force=True)，把 PyTorch caching allocator 持有的空闲块还给驱动，
    因此 Phase 2 的专用 VRAM 常驻在低位。DisTorch2 流式路径原本没有这一步，
    导致 Phase 2 期间专用 VRAM 顶满 16GB 并向共享内存溢出。

    仅在 BlockSwap 未启用时包装（BlockSwap 已自带 clear_memory）。
    返回包装的 block 数量。
    """
    if blockswap_active:
        return 0
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return 0
    import functools
    wrapped = 0
    for block in blocks:
        fwd = getattr(block, "forward", None)
        if fwd is None or getattr(fwd, "_distorch2_cache_release", False):
            continue

        @functools.wraps(fwd)
        def released_forward(*args, __orig=fwd, **kwargs):
            out = __orig(*args, **kwargs)
            clear_memory(debug=debug, deep=False, force=True,
                         timer_name="distorch2_block")
            return out

        released_forward._distorch2_cache_release = True
        block.forward = released_forward
        wrapped += 1

    logger.info(
        f"[MultiGPU DisTorch V2] per-block cache release wrapped on {wrapped} block(s)."
    )
    return wrapped


def apply_distorch2_placement(model: torch.nn.Module,
                              allocation_string: str,
                              debug=None,
                              virtual_vram_gb: float = 0.0,
                              donor_device: str = "cpu",
                              compute_device: str = None,
                              blockswap_active: bool = False) -> dict:
    """
    Run the VERBATIM upstream analyzer and place modules accordingly.

    When ``allocation_string`` is empty, one is derived from the PACKED model
    size (int8/nvfp4-aware) using virtual_vram_gb / donor_device.
    ``compute_device`` MUST be the intended inference device (e.g. runner
    _dit_device), NOT the current parameter device: with dit_offload_device=cpu
    the model is materialized on CPU first, so the current device lies.

    Returns the upstream result dict {"device_assignments", "block_assignments"}.
    """
    ensure_backend_registered()

    # CRITICAL: the node/CLI pass the upstream vram format
    # "#compute;virtual_gb;donor". The upstream virtual-VRAM calculator sizes
    # the model from LOGICAL dtypes (int8 packed 7.78GB reads as fp16 15.34GB),
    # which mis-sizes quotas for quantized packs (cpu ended up with 5.4% instead
    # of 100%). Recompute the final "compute,frac;donor,frac" string from the
    # PACKED sizes here; the upstream analyzer stays untouched.
    parsed = _parse_vram_string(allocation_string) if allocation_string else None
    if parsed is not None:
        expert_prefix, comp_dev, vv_gb, donor_dev = parsed
        packed_alloc = build_packed_allocation_string(
            model,
            compute_device=comp_dev,
            virtual_vram_gb=vv_gb,
            donor_device=donor_dev,
        )
        allocation_string = (expert_prefix + "#" + packed_alloc) if expert_prefix else packed_alloc
    elif not allocation_string:
        if compute_device is None:
            p = next(model.parameters(), None)
            compute_device = str(p.device) if p is not None else "cuda:0"
        allocation_string = build_packed_allocation_string(
            model,
            compute_device=compute_device,
            virtual_vram_gb=virtual_vram_gb,
            donor_device=donor_device,
        )

    from ..distorch2.distorch_2 import analyze_safetensor_loading

    adapter = FakePatcherAdapter(model)
    device_assignments = analyze_safetensor_loading(adapter, allocation_string)
    block_assignments = device_assignments["block_assignments"]

    # Physical placement: identical to upstream new_partially_load Step 4.
    moved = 0
    per_device = {}
    for name, module in model.named_modules():
        target = block_assignments.get(name)
        if target is None:
            continue
        try:
            current = next(module.parameters(recurse=False)).device
        except StopIteration:
            continue
        if str(current) != str(target):
            module.to(target)
            module.comfy_cast_weights = True  # upstream Step-4 flag: stream per forward
            moved += 1
            per_device[str(target)] = per_device.get(str(target), 0) + 1

    logger.info(
        f"[MultiGPU DisTorch V2] SeedVR2 DiT placement applied: "
        f"{moved} module(s) moved per block_assignments {per_device or '{}'}"
    )
    if debug is not None and hasattr(debug, "log"):
        try:
            debug.log(
                f"DisTorch2 placement: {moved} modules placed "
                f"(allocation='{allocation_string}')",
                category="dit", force=True,
            )
        except Exception:
            pass

    # 初始清空：丢弃放置阶段/前一个 Phase 残留的 allocator 空闲块
    try:
        clear_memory(debug=debug, deep=False, force=True,
                     timer_name="distorch2_placement")
    except Exception:
        pass

    # BlockSwap 未启用时，每个 block forward 后强制释放缓存（对齐 BlockSwap 机制）
    try:
        wrap_blocks_with_cache_release(model, debug=debug,
                                       blockswap_active=blockswap_active)
    except Exception as e:
        logger.warning(f"[MultiGPU DisTorch V2] cache-release wrap failed: {e}")

    return device_assignments
