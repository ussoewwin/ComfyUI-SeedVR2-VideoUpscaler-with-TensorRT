"""
Local DisTorch2 package for the SeedVR2 node.

Backend files (distorch_2.py / wrappers.py / device_utils.py /
model_management_mgpu.py) are VERBATIM copies of
``ComfyUI-MultiGPU`` (comfyui-multigpu, pollockjj, GPL-3.0). See the headers of
those files for the upstream notice. This __init__ reproduces the original
logger setup and device-state helpers so logging behaves exactly like upstream.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from types import MethodType

import torch
import comfy.model_management as mm

# ---- Original logger setup (name "MultiGPU", matches comfyui-multigpu) ----
MGPU_MM_LOG = False
DEBUG_LOG = False

logger = logging.getLogger("MultiGPU")
logger.propagate = False

FOCUS_LOG_LEVEL = logging.INFO + 5
logging.addLevelName(FOCUS_LOG_LEVEL, "FOCUS")

if not hasattr(logging.Logger, "focus"):
    def focus(self, message, *args, **kwargs):
        if self.isEnabledFor(FOCUS_LOG_LEVEL):
            self._log(FOCUS_LOG_LEVEL, message, args, **kwargs)

    logging.Logger.focus = focus  # type: ignore[attr-defined]

if not logger.handlers:
    log_level = logging.DEBUG if DEBUG_LOG else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)


def mgpu_mm_log_method(self, msg):
    """Add MultiGPU model management logging method to logger instance."""
    if MGPU_MM_LOG:
        self.focus(
            f"[MultiGPU Model Management] {msg}",
            extra={"mgpu_context": {"component": "model_management"}},
        )


logger.mgpu_mm_log = MethodType(mgpu_mm_log_method, logger)

# Submodule imports AFTER the logger is fully wired (they use logger.mgpu_mm_log).
from . import device_utils  # noqa: E402
from . import model_management_mgpu  # noqa: E402

# ---- Original device state helpers (matches comfyui-multigpu) ----
current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()
current_unet_offload_device = mm.unet_offload_device()


def set_current_device(device):
    global current_device
    current_device = device
    logger.debug(f"[MultiGPU Initialization] current_device set to: {device}")


def set_current_text_encoder_device(device):
    global current_text_encoder_device
    current_text_encoder_device = device
    logger.debug(f"[MultiGPU Initialization] current_text_encoder_device set to: {device}")


def set_current_unet_offload_device(device):
    global current_unet_offload_device
    current_unet_offload_device = device
    logger.debug(f"[MultiGPU Initialization] current_unet_offload_device set to: {device}")


def get_current_device():
    return current_device


def get_current_text_encoder_device():
    return current_text_encoder_device


def get_current_unet_offload_device():
    return current_unet_offload_device
