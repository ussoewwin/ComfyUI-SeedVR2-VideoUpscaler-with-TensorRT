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

from math import ceil
from typing import Tuple
import math

def get_window_op(name: str):
    if name == "720pwin_by_size_bysize":
        return make_720Pwindows_bysize
    if name == "720pswin_by_size_bysize":
        return make_shifted_720Pwindows_bysize
    raise ValueError(f"Unknown windowing method: {name}")


def _merge_small_tail_slices(slices: list[slice], ws: int, limit: int) -> list[slice]:
    """Ensure every spatial window slice is at least ws // 2 tokens wide/high."""
    min_extent = ws // 2
    if len(slices) >= 2 and (slices[-1].stop - slices[-1].start) < min_extent:
        slices = slices[:-2] + [slice(slices[-2].start, limit)]
    return slices


# -------------------------------- Windowing -------------------------------- #
def make_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    #cal windows under 720p
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    # Prevent window over-fragmentation into narrow strips (e.g. 256px / 16 tok) at ultra-high resolutions (4320p)
    # Clamp to nominal 720p/1080p window size (>= 27 tokens = 432px)
    ww = max(ww, 27)
    wh = max(wh, 27)
    wt = ceil(min(t, 30) / resized_nt)  # window size.
    nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)  # window size.

    t_slices = [
        slice(it * wt, min((it + 1) * wt, t))
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]
    h_slices = _merge_small_tail_slices([
        slice(ih * wh, min((ih + 1) * wh, h))
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
    ], wh, h)
    w_slices = _merge_small_tail_slices([
        slice(iw * ww, min((iw + 1) * ww, w))
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
    ], ww, w)

    return [
        (st_s, sh_s, sw_s)
        for sw_s in w_slices
        for sh_s in h_slices
        for st_s in t_slices
    ]

def make_shifted_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    #cal windows under 720p
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    # Prevent window over-fragmentation at ultra-high resolutions
    ww = max(ww, 27)
    wh = max(wh, 27)
    wt = ceil(min(t, 30) / resized_nt)  # window size.
    
    st, sh, sw = (  # shift size.
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)  # window size.
    nt, nh, nw = (  # number of window.
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )

    t_slices = [
        slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t))
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]
    h_slices = _merge_small_tail_slices([
        slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h))
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
    ], wh, h)
    w_slices = _merge_small_tail_slices([
        slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w))
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
    ], ww, w)

    return [
        (st_s, sh_s, sw_s)
        for sw_s in w_slices
        for sh_s in h_slices
        for st_s in t_slices
    ]
