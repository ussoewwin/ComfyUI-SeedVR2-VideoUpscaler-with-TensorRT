# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Union
import functools
import math
import torch
from PIL import Image
from torchvision.transforms import functional as TVF

from ....models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)

# DiT token grid geometry (configs_*/main.yaml): VAE spatial stride 8 x patch_size 2,
# window = (4, 3, 3) for every layer, alternating regular / shifted 720p windows.
_TOKEN_STRIDE = 16
_DIT_NUM_WINDOWS = (4, 3, 3)
# Minimum healthy window extent, relative to the nominal window size. A window below
# this fraction is a degenerate "runt" sliver (too little context for stable
# softmax/RoPE attention), which surfaces as right/bottom edge smear or checkerboard
# noise after decode. Measured boundaries: healthy standard resolutions sit at
# 1280x720 -> 13/27 = 0.48, 3840x2160 -> 11/27 = 0.41, 2560x1440 -> 12/27 = 0.44;
# the degenerate case (2944x4320) is 6/17 = 0.353. 0.36 lies just above that
# degenerate ratio, so it flags the runt that produces right-edge smear while leaving
# 1280x720 / 1080p / 1440p / 4K / 4K-portrait untouched. (The previous 0.34 was
# itself <= 6/17, so the 6-token shifted edge window passed as "healthy" and the
# smear on 2944x4320 was left unfixed.)
_MIN_WINDOW_RATIO = 0.36
# Hard ceiling for the extra safe padding, per axis, in pad-factor steps
# (1 step = one pad factor = 16 px). The search range is raised to the current window
# period (see _find_safe_window_padding), so a healthy padding always exists within
# one period; this ceiling only bounds pathological aspect ratios, where a
# disproportionate pad would be worse than leaving a thin edge window.
_MAX_SAFE_PADDING_TOKENS = 64


def _window_grid(height: int, width: int):
    """
    Reproduce the exact token-grid geometry used by both DiT window generators.

    Returns (th, tw, wh, ww): th/tw = token count per axis, wh/ww = nominal
    (unclipped) window extent in tokens. ``make_720Pwindows_bysize`` and
    ``make_shifted_720Pwindows_bysize`` derive wh/ww identically and differ only in
    window placement, so this single helper keeps both paths in sync.
    """
    th, tw = height // _TOKEN_STRIDE, width // _TOKEN_STRIDE
    if th <= 0 or tw <= 0:
        return 0, 0, 0, 0
    _, nh, nw = _DIT_NUM_WINDOWS
    scale = math.sqrt((45 * 80) / (th * tw))
    wh = math.ceil(round(th * scale) / nh)
    ww = math.ceil(round(tw * scale) / nw)
    return th, tw, wh, ww


def _window_deficit(height: int, width: int) -> int:
    """
    Return how many tokens the smallest DiT attention window falls short of a
    healthy half-window minimum ``ws // 2`` (0 = every regular and shifted window
    is at least ``ws // 2`` without needing tail-slice merging).

    Both regular and shifted unmerged window layouts are checked on the exact token
    grid (pixel / 16: VAE stride 8 x patch 2). Because ``make_720Pwindows_bysize``
    and ``make_shifted_720Pwindows_bysize`` also merge any runt tail slice ``< ws // 2``
    into the preceding window at runtime, safe_padding=False is already protected
    against runt slivers; when safe_padding=True is enabled, this pads dimensions so
    every window natively satisfies ``>= ws // 2`` (i.e. ``rem in {0, ws // 2, ceil(ws / 2)}``).
    """
    th, tw, wh, ww = _window_grid(height, width)
    if th <= 0 or tw <= 0:
        return 0

    deficit = 0
    for t, ws in ((th, wh), (tw, ww)):
        if ws <= 0 or ws >= t:
            continue
        half = ws // 2
        rem = t % ws
        # Regular last window size (before merge)
        reg_last = rem if rem > 0 else ws
        # Shifted last window size (before merge): shifted starts at half = int(0.5 * ws)
        shf_rem = (t - half) % ws
        shf_last = shf_rem if shf_rem > 0 else ws
        deficit = max(deficit, half - reg_last, half - shf_last)
    return deficit


@functools.lru_cache(maxsize=256)
def _find_safe_window_padding(height: int, width: int, factor_h: int, factor_w: int):
    """
    Find the smallest extra (pad_h, pad_w), in multiples of the pad factors, such
    that no regular or shifted DiT window collapses into a runt sliver.

    The per-axis search cap is raised to the current window period (wh / ww in
    pad-factor steps): shifting an axis by a full period restores the same window
    phase, so a healthy padding always exists within one period. It is clamped down
    to _MAX_SAFE_PADDING_TOKENS (so extreme aspect ratios never receive a
    disproportionate pad) and up to at least 2 steps. Height and width are searched
    jointly because the window size depends on the total token area, not on one axis
    alone.
    """
    if _window_deficit(height, width) == 0:
        return 0, 0

    _, _, wh, ww = _window_grid(height, width)
    cap_h = max(2, min(_MAX_SAFE_PADDING_TOKENS, wh if wh else 2))
    cap_w = max(2, min(_MAX_SAFE_PADDING_TOKENS, ww if ww else 2))

    best = None  # (deficit, added_area, extra_h, extra_w)
    for sh in range(cap_h + 1):
        for sw in range(cap_w + 1):
            eh, ew = sh * factor_h, sw * factor_w
            deficit = _window_deficit(height + eh, width + ew)
            added = (height + eh) * (width + ew) - height * width
            key = (deficit, added, eh, ew)
            if best is None or key < best:
                best = key
    return best[2], best[3]


class DivisibleCrop:
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)

        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        cropped_height = height - (height % self.height_factor)
        cropped_width = width - (width % self.width_factor)

        image = TVF.center_crop(img=image, output_size=(cropped_height, cropped_width))
        return image


class DivisiblePad:
    """
    Pad image to make dimensions divisible by a factor.
    Uses edge reflection (or edge replication fallback) to avoid black-border
    bleeding and high-contrast artifacts during VAE encoding and DiT window attention.
    """
    def __init__(self, factor, safe_padding: bool = False):
        if not isinstance(factor, tuple):
            factor = (factor, factor)
        self.height_factor, self.width_factor = factor[0], factor[1]
        self.safe_padding = safe_padding

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        # Calculate base padding needed for divisible factors
        pad_height = (self.height_factor - (height % self.height_factor)) % self.height_factor
        pad_width = (self.width_factor - (width % self.width_factor)) % self.width_factor

        # Prevent DiT 720P window runt slivers (optional, enabled via safe_padding=True):
        # The DiT attends inside local windows over the patchified token grid
        # (pixel / 16: VAE stride 8 x patch 2). Layers alternate between regular
        # (720pwin_by_size_bysize) and half-shifted (720pswin_by_size_bysize) windows.
        # When a token dimension fits the window size badly, the last regular or
        # shifted window collapses to a sliver below _MIN_WINDOW_RATIO of the nominal
        # window size, which produces smear / checkerboard noise along the right edge,
        # bottom edge or bottom-right corner. The exact layout of both window types is
        # simulated and the smallest extra padding (in multiples of the pad factor)
        # that keeps every window healthy is selected.
        if self.safe_padding:
            extra_h, extra_w = _find_safe_window_padding(
                height + pad_height, width + pad_width,
                self.height_factor, self.width_factor,
            )
            pad_height += extra_h
            pad_width += extra_w

        if pad_height == 0 and pad_width == 0:
            return image

        # Pad right/bottom using edge reflection (fallback to edge replication if pad >= dim)
        # Never pad with constant 0.0 (black): a 64px-160px black band at the right/bottom
        # edge falls inside the rightmost/bottommost DiT 720P attention window and corrupts
        # attention keys/values and VAE convolutions along the entire right/bottom border!
        if isinstance(image, torch.Tensor):
            # Pad format: (left, right, top, bottom)
            padding = (0, pad_width, 0, pad_height)
            pad_mode = 'reflect' if (pad_width < width and pad_height < height) else 'replicate'
            if image.ndim == 2:
                image = torch.nn.functional.pad(
                    image.unsqueeze(0).unsqueeze(0), padding, mode=pad_mode
                ).squeeze(0).squeeze(0)
            else:
                image = torch.nn.functional.pad(image, padding, mode=pad_mode)
        elif isinstance(image, Image.Image):
            new_width = width + pad_width
            new_height = height + pad_height
            result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
            result.paste(image, (0, 0))
            if pad_width > 0:
                right_col = image.crop((width - 1, 0, width, height)).resize((pad_width, height))
                result.paste(right_col, (width, 0))
            if pad_height > 0:
                bottom_row = result.crop((0, height - 1, new_width, height)).resize((new_width, pad_height))
                result.paste(bottom_row, (0, height))
            image = result

        return image
