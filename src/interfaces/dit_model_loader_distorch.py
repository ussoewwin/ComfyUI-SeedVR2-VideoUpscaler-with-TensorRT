"""
SeedVR2 DiT Model Loader Node - DisTorch2 CPU-offload variant

Same node surface as "SeedVR2 (Down)Load DiT Model" (identical base inputs and
identical SEEDVR2_DIT config dict output) plus DisTorch2 placement controls, so a
whole INT8 / NVFP4 DiT can be hosted in system RAM (host memory) and streamed to
the compute device during denoising.

DisTorch2 backend: COPIED from ComfyUI-HSWQ-Loader-and-Tools (which ports it from
ComfyUI-MultiGPU, pollockjj; GPL-3.0). The copied backend lives in
``src/distorch2/`` (distorch_2.py / wrappers.py / device_utils.py /
model_management_mgpu.py). This module wires that backend into the SeedVR2
config-dict loader path and mirrors the upstream allocation-string format
"<expert_mode_allocations>#<compute_device>;<virtual_vram_gb>;<donor_device>".
"""

import logging
from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple

from ..utils.model_registry import get_available_dit_models, DEFAULT_DIT
from ..optimization.memory_manager import get_device_list

logger = logging.getLogger("SeedVR2.DiTorch2")

_QUANTIZED_TAGS = ("int8", "nvfp4", "convrot", "fp8", "int4")


def _looks_quantized(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(tag in name for tag in _QUANTIZED_TAGS)


def build_distorch2_allocation_string(compute_device: str,
                                      virtual_vram_gb: float,
                                      donor_device: str,
                                      expert_mode_allocations: str) -> str:
    """
    Build a DisTorch2 allocation string in the upstream format:
        "<expert_mode_allocations>#<compute_device>;<virtual_vram_gb>;<donor_device>"
    or "<compute_device>" when only a block allocation is given.
    """
    # __W4A8_SUPPORT__: always emit a vram-format string so every quantized pack
    # (INT8 / NVFP4 / W4A8) is routed through the SAME packed-size placement
    # bridge in distorch2_placement. Returning "" sent the pack to the upstream
    # calculator, whose size check uses weight.numel()*element_size() (the LOGICAL
    # dtype size - for a W4A8 int4 pack that is 4x too large), so W4A8 did not
    # receive the same distorch2 treatment as INT8/NVFP4. Callers that already
    # pass a value are unaffected; only the previously-empty case changes.
    if virtual_vram_gb and virtual_vram_gb > 0:
        vram_string = f"{compute_device};{virtual_vram_gb};{donor_device}"
    elif expert_mode_allocations:
        vram_string = compute_device
    else:
        vram_string = f"{compute_device};{float(virtual_vram_gb or 0.0)};{donor_device}"
    return f"{expert_mode_allocations}#{vram_string}"


def _register_distorch2_backend():
    """Register the copied DisTorch2 ModelPatcher patch (idempotent)."""
    try:
        from ..distorch2.distorch_2 import register_patched_safetensor_modelpatcher
        register_patched_safetensor_modelpatcher()
        return True
    except Exception as e:
        logger.warning(f"[SeedVR2 DisTorch2] backend registration failed: {e}")
        return False


class SeedVR2LoadDiTModelDisTorch2(io.ComfyNode):
    """
    Configure the SeedVR2 DiT model with DisTorch2 placement.

    Same base surface as SeedVR2LoadDiTModel. Adds a virtual-VRAM budget, a donor
    device and a per-block allocation string so a quantized (INT8 / NVFP4) DiT can
    be entirely hosted in system RAM and streamed to the compute device.

    Returns the same SEEDVR2_DIT config dict (plus a ``distorch2`` block).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        dit_models = get_available_dit_models()

        return io.Schema(
            node_id="SeedVR2LoadDiTModelDisTorch2",
        display_name="SeedVR2 (Down)Load DiT Model with Distorch2",
            category="SEEDVR2",
            description=(
                "Load and configure SeedVR2 DiT (Diffusion Transformer) model for video upscaling, "
                "with DisTorch2 placement controls. Same inputs/output as SeedVR2 (Down)Load DiT Model, "
                "plus a virtual-VRAM budget, a donor device and an optional per-block allocation string. "
                "Use this to keep a whole INT8 / NVFP4 DiT in system RAM (CPU offload) and stream it to "
                "the compute device during denoising. DisTorch2 backend copied from "
                "ComfyUI-HSWQ-Loader-and-Tools (GPL-3.0). \n\n"
                "Connect to Video Upscaler node."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=dit_models,
                    default=DEFAULT_DIT,
                    tooltip=(
                        "DiT (Diffusion Transformer) model for video upscaling.\n"
                        "Models automatically download on first use.\n"
                        "Additional models can be added to the ComfyUI models folder."
                    )
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for DiT model inference (upscaling phase)"
                ),
                io.Combo.Input("attention_mode",
                    options=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3", "spargeattn"],
                    default="sdpa",
                    optional=True,
                    tooltip=(
                        "Attention computation backend:\n"
                        "? sdpa: PyTorch scaled_dot_product_attention (default, stable, always available)\n"
                        "? flash_attn_2: Flash Attention 2 (Ampere+, requires flash-attn package)\n"
                        "? flash_attn_3: Flash Attention 3 (Hopper+, requires flash-attn with FA3 support)\n"
                        "? sageattn_2: SageAttention 2 (requires sageattention package)\n"
                        "? sageattn_3: SageAttention 3 (Blackwell/RTX 50xx only, requires sageattn3 package)\n"
                        "? spargeattn: SpargeAttn-hswq block-sparse attention on SageAttention2++ kernels (requires spas_sage_hswq_attn package; topk fixed at 0.5; per-window seq_len >= 128 and headdim 64/128 required, otherwise falls back per window)\n"
                        "\n"
                        "SDPA is recommended - stable and works everywhere.\n"
                        "Flash Attention and SageAttention provide speedup through optimized CUDA kernels on compatible GPUs."
                    )
                ),
                io.Combo.Input("sparge_topk",
                    options=["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"],
                    default="0.5",
                    optional=True,
                    tooltip=(
                        "SpargeAttn topK ratio (only used when attention_mode=spargeattn):\n"
                        "? KV block keep ratio for the two-stage block-sparse filter (0.05-1.0)\n"
                        "? Higher = more blocks computed = more accurate, less acceleration\n"
                        "? 1.0 = compute all blocks (no skipping)\n"
                        "? Ignored by other attention backends\n"
                    )
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip=(
                        "Optional torch.compile optimization settings from SeedVR2 Torch Compile Settings node.\n"
                        "Provides 20-40% speedup with compatible PyTorch 2.0+ and Triton installation."
                    )
                ),
                # ---- DisTorch2 placement controls ----
                io.Boolean.Input("distorch2_enabled",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Enable DisTorch2 placement for this DiT model.\n"
                        "ON = hold a whole (INT8 / NVFP4) DiT in system RAM and stream it to the compute "
                        "device during denoising (registered via the copied DisTorch2 backend).\n"
                        "OFF = behaviour identical to SeedVR2 (Down)Load DiT Model."
                    )
                ),
                io.Float.Input("virtual_vram_gb",
                    default=4.0,
                    min=0.0,
                    max=128.0,
                    step=0.1,
                    optional=True,
                    tooltip=(
                        "DisTorch2 virtual VRAM budget (GiB) reported to the memory planner.\n"
                        "The planner treats this much host RAM as if it were VRAM, letting a model larger "
                        "than real VRAM load.\n"
                        "? 0.0: no virtual VRAM\n"
                        "Only used when distorch2_enabled=True."
                    )
                ),
                io.Combo.Input("donor_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="cpu",
                    optional=True,
                    tooltip=(
                        "DisTorch2 donor device: where the weights backing the virtual VRAM are kept.\n"
                        "? 'cpu': host RAM (default for CPU offload)\n"
                        "? 'cuda:X': another GPU acts as the donor\n"
                        "Only used when distorch2_enabled=True."
                    )
                ),
                io.String.Input("expert_mode_allocations",
                    default="",
                    optional=True,
                    tooltip=(
                        "DisTorch2 expert-mode block allocation string (advanced).\n"
                        "Comma-separated per-block device assignments, e.g. \"cpu,cpu,cuda:0\".\n"
                        "Empty = derive placement from device / virtual_vram_gb.\n"
                        "Only used when distorch2_enabled=True."
                    )
                ),
                io.Boolean.Input("eject_models",
                    default=True,
                    optional=True,
                    tooltip=(
                        "DisTorch2: mark all currently loaded models for eviction before this model loads, "
                        "freeing maximum memory for the CPU-offload stream.\n"
                        "Only used when distorch2_enabled=True."
                    )
                ),
                io.Boolean.Input("emb_repeat_nocache",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Disable the emb_repeat cache during DiT upscaling (Phase 2).\n"
                        "ON = recompute every use: saves ~1.0 GiB resident VRAM (removes the shared-memory "
                        "spill pressure), output is bit-identical (repeat_interleave is a pure copy), "
                        "costs about +0.6 s per step measured on RTX 5060 Ti.\n"
                        "OFF = keep the cache (default; identical to stock behaviour).\n"
                        "Only used when distorch2_enabled=True."
                    )
                ),
                io.Boolean.Input("norm_bf16",
                    default=False,
                    optional=True,
                    tooltip=(
                        "RMS/QK norm precision during DiT upscaling (Phase 2).\n"
                        "ON = run norm in bf16: saves significant resident VRAM in Phase 2, but output differs from the fp32 path (per-pixel PSNR ~37-39 dB vs fp32).\n"
                        "OFF = stock fp32 path (quality-priority; default).\n"
                        "Only used when distorch2_enabled=True."
                    )
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_DIT").Output(
                    tooltip="DiT model configuration containing model path, device settings, BlockSwap parameters, DisTorch2 placement, and compilation options. Connect to Video Upscaler node."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, device: str,
                     attention_mode: str = "sdpa", sparge_topk: str = "0.5",
                     torch_compile_args: Dict[str, Any] = None,
                     distorch2_enabled: bool = True, virtual_vram_gb: float = 4.0,
                     donor_device: str = "cpu", expert_mode_allocations: str = "",
                     eject_models: bool = True,
                     emb_repeat_nocache: bool = False,
                     norm_bf16: bool = False) -> io.NodeOutput:
        """
        Create a DiT model configuration for the SeedVR2 main node.

        Base fields are identical to SeedVR2LoadDiTModel. When distorch2_enabled
        is set, the copied DisTorch2 backend is registered and a ``distorch2``
        block with the placement controls is added to the returned config.
        """

        allocation_string = build_distorch2_allocation_string(
            compute_device=device,
            virtual_vram_gb=float(virtual_vram_gb or 0.0),
            donor_device=str(donor_device),
            expert_mode_allocations=str(expert_mode_allocations or ""),
        )

        backend_ready = False
        if distorch2_enabled:
            backend_ready = _register_distorch2_backend()
            logger.info(
                "[SeedVR2 DisTorch2] enabled: device=%s virtual_vram_gb=%.1f donor=%s "
                "allocation=%r backend_ready=%s",
                device, float(virtual_vram_gb or 0.0), donor_device, allocation_string, backend_ready,
            )

        distorch2 = {
            "enabled": bool(distorch2_enabled),
            "virtual_vram_gb": float(virtual_vram_gb or 0.0),
            "donor_device": str(donor_device),
            "expert_mode_allocations": str(expert_mode_allocations or ""),
            "eject_models": bool(eject_models),
            "emb_repeat_nocache": bool(emb_repeat_nocache),
            "norm_bf16": bool(norm_bf16),
            "allocation_string": allocation_string,
            "backend_ready": backend_ready,
            "preserve_quantized_storage": True,
            "model_is_quantized": _looks_quantized(model),
        }

        config = {
            "model": model,
            "device": device,
            # __W4A8_OFFLOAD__: the DiT must materialize on the DisTorch2 donor
            # device (CPU) BEFORE placement, like the non-DisTorch2 loader does via
            # its offload_device input. video_upscaler reads dit.get("offload_device");
            # exposing the donor here makes dit_offload_device the CPU, so weights go
            # to host RAM and VRAM stays low. donor_device is the node's own value.
            "offload_device": str(donor_device),
            "cache_model": False,
            "blocks_to_swap": 0,
            "swap_io_components": False,
            "attention_mode": attention_mode,
            "sparge_topk": sparge_topk,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
            "distorch2": distorch2,
        }

        return io.NodeOutput(config)
