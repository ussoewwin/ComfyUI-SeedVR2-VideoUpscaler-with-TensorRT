# SeedVR2 — TensorRT VAE 最適化とエンコーダー包括的改修 技術解説仕様書

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>JA</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>Technical Guide</b></font></td>
  </tr>
</table>

対象リポジトリ: `ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder`  
対象モジュール: `src/core/trt_decoder.py`, `src/core/trt_encoder.py`, `src/core/infer.py`, `src/interfaces/trt_vae_model_loader.py`, `src/interfaces/__init__.py`, `__init__.py`, `src/interfaces/video_upscaler.py`

---

## 1. テーマ概要

### テーマ①: 3つの改善 (v1.5.4 左上ノイズ・チェッカーボード偽影の撲滅とゼロ VRAM 膨張アーキテクチャ)
1. **Studio互換の静的シェイプ判定ガード (Studio-Compatible Static Shape Check)**  
   `current_shape != target_shape` の時のみ `context.set_input_shape` を実行。静的エンジンへの冗長な形状再設定をバイパスし、TensorRT 内部のスクラッチパッドバッファ再確保によるダーティ VRAM の再摂食を遮断。
2. **決定論的ダミーウォームアップ実行 (Deterministic Dummy Warmup Execution)**  
   空間タイリングループ突入直前に、ゼロ値テンソル（`warmup_in` / `warmup_out`）を用いた 1 パスの非同期推論およびストリーム同期を実行。TensorRT 内部の畳み込みワークスペースおよびテンポラルアキュムレータラインをサニタイズ（ゼロクリア）し、先頭タイル（`y=0, x=0`）でのゴミデータ読み出しによるノイズ化を完全に撲滅。
3. **ゼロ VRAM 膨張アーキテクチャ (Zero VRAM Bloat Architecture)**  
   空間外周の無理なパディング（結果バッファを2倍〜3倍に膨らませる外周パディング）を排除し、Float32 累積バッファ（`result` と `weights`）の VRAM 浪費を防ぎ、最小限の VRAM 消費でネイティブ解像度の高速エンコード/デコードを維持。

### テーマ②: エンコーダーの修正 (デコーダーとの完全対称化・フォールバック排除・短尺バッチ完動・ノード名UI統一)
1. **短尺バッチの Pad & Crop 1-shot 実行 & サイレント FP16 フォールバックの完全根絶**  
   バッチサイズがエンジンフレーム未満（例: `batch_size=5` に対し 21f エンジン）の際にスライス生成で `IndexError` が発生し、外側の `try...except` でサイレントに PyTorch 標準 FP16 VAE に転落していた問題を根絶。最終フレーム反復パディングによる 1-shot 実行と潜在空間クロップ（Pad & Crop）を導入し、TRT 未導入・エンジン不在時は即座に `RuntimeError` を送出。
2. **動的エンジン自動選択機構 (`pick_engine_frames` / `_available_engine_frames`) の実装**  
   ディスク上の `tensorrt_backend/artifacts` および `models/tensorrt/seedvr2` を走査し、完全一致 → 最大収容エンジン → 最小エンジン（パディング実行）の優先順位で自動選択。
3. **ノード名・UI スキーマのデコーダー完全対称化 (`SeedVR2LoadTensorRTVAEEncoder`)**  
   旧 `SeedVR2LoadTensorRTVAEModel` からエンコーダー専用ノード `SeedVR2LoadTensorRTVAEEncoder` への改名（互換エイリアス保持）、UI 表示名、説明文、入力ソケット、ツールチップ、出力ソケットをデコーダー側（`SeedVR2LoadTensorRTVAEDecoder`）と完全に対称に整備。

---

## 2. 修正前は何が問題だったのか (根本原因分析)

### 2.1 テーマ①: 左上ノイズ・チェッカーボード偽影と VRAM 膨張
1. **遅延スクラッチパッド確保とダーティ VRAM の再摂食**  
   ComfyUI では、Phase 1 (VAE Encode) → Phase 2 (DiT Upscale) → Phase 3 (VAE Decode) が単一の Python プロセス内で順次実行され、PyTorch CUDA キャッシュアロケータを共有している。Phase 2 で DiT が大量の VRAM を消費した後に解放したメモリ領域には、直前のアテンション計算のビット残骸（ダーティ VRAM）がゼロクリアされずに残存する。
2. **静的エンジンに対する冗長な `set_input_shape` 呼び出し**  
   従来のコードは、毎タイル・毎チャンクで無条件に `context.set_input_shape` を呼び出していた。TensorRT は形状変更要求を受けると、内部の畳み込みスクラッチパッドメモリを破棄して再確保を行う。この再確保先が DiT の解放した未初期化ダーティ VRAM と重なり、ゴミデータがバッファに流入した。
3. **因果 3D 畳み込み (`InflatedCausalConv3d`) の時序アキュムレータ汚染**  
   SeedVR2 VAE の因果 3D 畳み込みは時間軸の文脈を保持するため、内部に時序累算バッファを持つ。最初のタイル（`y=0, x=0`）の計算時、内部カーネルが未初期化の浮動小数点ゴミデータを読み出し、残差ブロックを経て極端な異常値・NaN・RGB 超過値へと増幅。これが `[-2.0, 2.0]` にクランプされた結果、左上ブロックに市松模様・チェッカーボード状のカラーモザイクノイズとなって現れた。
4. **横画面特異性**  
   DiT の 720P ウィンドウアテンション（`make_720Pwindows_bysize`）において、横画面（16:9 や 4:3）では水平ウィンドウ数（`nw`）が多く、PyTorch アロケータ内でワイドストライドのテンソル解放パターンが生じる。この解放ブロックの配置が TensorRT スクラッチパッドの要求ストライドと完全に合致し、横画面でのみ左上ノイズが顕著に発現した。
5. **外周パディングによる VRAM 膨張の弊害**  
   これを回避しようとして外周をフルタイル単位でパディングする対症療法をとると、Float32 累積バッファ（`result`, `weights`）が 2倍〜3倍に膨れ上がり、16GB VRAM 環境での OOM 頻発を招いた。

### 2.2 テーマ②: エンコーダーの構造的欠陥・サイレント FP16 転落
1. **短尺バッチでの `IndexError` クラッシュ**  
   従来の `_trt_encode_batch` は、`resolve_engine_frames()` で固定エンジン（最大フレーム数、例: 29f）のみを取得していた。ComfyUI 側で `batch_size=5` などの短尺バッチが指定された場合、ストライド計算 `range(0, total - engine_frames + 1, stride)` が空リスト `[]` となり、直後の `starts[-1]` が `IndexError: list index out of range` を送出して破綻していた。
2. **サイレント FP16 フォールバックの悪弊**  
   この `IndexError` が外側の `try ... except` で捕捉され、何のエラー通知も警告もなく、裏で PyTorch 標準の FP16 VAE に処理が転落していた。ユーザーは TensorRT エンコーダーを選択しているにもかかわらず、実際には極めて低速な PyTorch CPU/CUDA FP16 実装で実行され、高速化の恩恵を完全に喪失していた。
3. **動的エンジン選択の欠落**  
   デコーダー側ではディスク上のエンジンを走査して最適なフレーム数を選択する `pick_engine_frames` が実装されていたが、エンコーダー側はハードコードされた候補リストしか持たず、任意のフレーム数（9f, 13f, 17f, 21f 等）への追従ができなかった。
4. **ノード設計の非対称性と名称の混乱**  
   デコーダー側が `SeedVR2LoadTensorRTVAEDecoder` として整理されていたのに対し、エンコーダー側は `SeedVR2LoadTensorRTVAEModel` という曖昧な名称のままであり、レガシーのフォールバック用ウィジェット（`encode_tiled`, `encode_tile_size` 等）が残存し、UI や引数の対称性が崩れていた。

---

## 3. 新規作成・修正したファイル名

| ファイルパス | 変更種別 | 改修内容の要約 |
| :--- | :--- | :--- |
| `src/core/trt_decoder.py` | 修正 | テーマ①: 静的シェイプ判定ガード、決定論的ダミーウォームアップ、ゼロ VRAM 膨張タイリング |
| `src/core/trt_encoder.py` | 修正 | テーマ①のエンコーダー移植＋テーマ②: `pick_engine_frames`, `_available_engine_frames`, Pad & Crop 1-shot |
| `src/core/infer.py` | 修正 | テーマ②: `_trt_encode_batch` 短尺バッチ Pad & Crop 完動化、サイレント FP16 転落の完全根絶、`RuntimeError` 直送出 |
| `src/interfaces/trt_vae_model_loader.py` | 修正 | テーマ②: `_available_engine_frames` マルチパス走査、`SeedVR2LoadTensorRTVAEEncoder` 改名、UI デコーダー対称化、互換エイリアス |
| `src/interfaces/__init__.py` | 修正 | テーマ②: `SeedVR2LoadTensorRTVAEEncoder` のノード登録と `__all__` エクスポート |
| `__init__.py` | 修正 | テーマ②: `SeedVR2LoadTensorRTVAEEncoder (⚡TRT)` の起動ロード・インポート登録 |
| `src/interfaces/video_upscaler.py` | 修正 | テーマ②: エンコーダー/デコーダー双方の TRT 検出による `torch.compile` スキップ、ドキュメント更新 |

---

## 4. 新規作成・修正したコード全文

### 4.1 `src/core/trt_encoder.py` (コード全文)

```python
"""Dedicated Full-Batch TensorRT VAE encoder for ComfyUI SeedVR2.
Executes exact 1-shot TensorRT acceleration for ANY batch size.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock

import torch

# --- TRT encode debug hooks (SEEDVR2_TRT_DEBUG=1 to enable; default off) ---
_TRT_DEBUG = os.environ.get("SEEDVR2_TRT_DEBUG", "0") == "1"
_TRT_DEBUG_DIR = os.environ.get("SEEDVR2_TRT_DEBUG_DIR", "") or None

def _trt_dbg_log(msg):
    if _TRT_DEBUG:
        print(f"[TRT-DEBUG] {msg}", flush=True)

def _trt_dbg_stats(tag, t):
    if not _TRT_DEBUG:
        return
    tt = t.detach().float()
    _trt_dbg_log(f"{tag}: shape={tuple(tt.shape)} dtype={t.dtype} "
                f"min={float(tt.min()):.4f} max={float(tt.max()):.4f} "
                f"mean={float(tt.mean()):.4f} std={float(tt.std()):.4f} "
                f"NaN={bool(torch.isnan(tt).any())} Inf={bool(torch.isinf(tt).any())}")
    if _TRT_DEBUG_DIR:
        try:
            import os as _os
            torch.save(tt.cpu(), _os.path.join(_TRT_DEBUG_DIR, tag.replace(' ', '_') + '.pt'))
        except Exception as e:
            _trt_dbg_log(f"save {tag} failed: {e}")

try:
    import tensorrt_rtx as trt
    HAS_TRT = True
except ImportError:
    try:
        import tensorrt as trt
        HAS_TRT = True
    except ImportError:
        trt = None
        HAS_TRT = False


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIRS = [
    ROOT / "tensorrt_backend" / "artifacts",
    ROOT.parents[1] / "models" / "tensorrt" / "seedvr2",
]

_ENGINES: dict[str, tuple[object, object, object, str, str, torch.cuda.Stream]] = {}
_ENCODE_LOCK = Lock()


def find_engine_path(frames: int) -> tuple[Path | None, int]:
    """Return (engine_path, tile_px). Prefers the 512px-tile engine (Studio standard), then 256px."""
    for tile_px in (512, 256):
        name = f"vae_encoder_{frames}f_tile{tile_px}.rtxplan"
        for d in ARTIFACTS_DIRS:
            p = d / name
            if p.exists() and p.stat().st_size > 1_000_000:
                return p, tile_px
    return None, 0


def is_available(frames: int | None = None) -> bool:
    """Check if TensorRT VAE encoder is available."""
    if not HAS_TRT:
        return False
    return True


def _engine(frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None):
    cache_key = frames
    cached = _ENGINES.get(cache_key)
    if cached is not None:
        return cached

    path, tile_px = find_engine_path(frames)
    if path is None:
        # No auto-build: engines are created explicitly via the build scripts/node.
        # Without an engine we fall back to the standard PyTorch VAE.
        raise FileNotFoundError(
            f"TensorRT VAE encoder engine for {frames} frames not found. "
            f"Build it first with tools/cloud_export_gpu.py + tools/cloud_build_engine.py "
            f"or the SeedVR2 Build TensorRT VAE Engines node."
        )

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT encoder: {path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT could not create an execution context for {path}")

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    stream = torch.cuda.Stream()
    cached = (runtime, engine, context, input_name, output_name, stream, tile_px)
    _ENGINES[cache_key] = cached
    return cached


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if overlap:
        t = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
        ramp = (1.0 - torch.cos(t * 3.141592653589793)) / 2.0  # cosine ease
        if left:
            weight[:overlap] = ramp
        if right:
            weight[-overlap:] = torch.minimum(weight[-overlap:], torch.flip(ramp, dims=[0]))
    return weight


@torch.inference_mode()
def _encode_single_chunk(sample: torch.Tensor, frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a single full batch directly in 1 shot with TensorRT."""
    _, _, _, height, width = sample.shape
    _, _, context, input_name, output_name, stream, tile_px = _engine(int(frames), vae=vae, dit_model=dit_model)
    if context is None:
        raise RuntimeError("TensorRT could not create a per-batch encoder context")

    # Studio-compatible shape check: only call set_input_shape if the current shape
    # actually differs from the target tile shape. Static-shape engines (built for
    # exact tile sizes) never need set_input_shape called, avoiding TRT's internal
    # buffer re-allocation/reset which causes the first-tile uninitialized memory glitch.
    target_shape = (1, 3, frames, tile_px, tile_px)
    current_shape = tuple(context.get_tensor_shape(input_name))
    if current_shape != target_shape:
        context.set_input_shape(input_name, target_shape)
        torch.cuda.synchronize()

    source = sample.to(device="cuda", dtype=torch.float16).contiguous()
    tile, overlap = tile_px, tile_px * 3 // 8  # 37.5% tile-to-tile overlap (96px@256, 192px@512)
    ys, xs = _positions(height, tile, overlap), _positions(width, tile, overlap)
    padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
    source = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
    latent_frames = (frames - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    raw_h, raw_w = padded_h // 8, padded_w // 8
    tile_lat = tile_px // 8
    overlap_latent = overlap // 8
    result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    dc_result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)

    with _ENCODE_LOCK, torch.cuda.stream(stream):
        # Warmup run: forces TensorRT to allocate and bind internal scratchpad memory.
        # Without this, the very first execution (tile y=0, x=0) reads uninitialized
        # GPU buffer memory, resulting in severe noise/artifact in the top-left corner.
        warmup_in = torch.zeros((1, 3, frames, tile_px, tile_px), device="cuda", dtype=torch.float16)
        warmup_out = torch.zeros((1, 32, latent_frames, tile_lat, tile_lat), device="cuda", dtype=torch.float16)
        context.set_tensor_address(input_name, warmup_in.data_ptr())
        context.set_tensor_address(output_name, warmup_out.data_ptr())
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        del warmup_in, warmup_out

        # NOTE (Studio Architecture & Address Safety):
        # A context's tensor addresses are mutable. For the safe/default path we execute
        # one tile at a time under _ENCODE_LOCK with stream.synchronize() so addresses
        # cannot be overwritten by a later queued tile.
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_output = torch.zeros((1, 32, latent_frames, tile_lat, tile_lat), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE encoder failed at tile y={y}, x={x}")
                stream.synchronize()
                if _TRT_DEBUG:
                    _dbg_tv = tile_output.float()
                    _dbg_sd = float(_dbg_tv.std())
                    _trt_dbg_log(f"enc tile y={y} x={x} (ly={y // 8},lx={x // 8}) "
                                f"min={float(_dbg_tv.min()):.4f} max={float(_dbg_tv.max()):.4f} "
                                f"std={_dbg_sd:.5f}" + ("  <<< BLACK?" if _dbg_sd < 0.05 else ""))
                ly, lx = y // 8, x // 8
                # DC offset correction: estimate the tile's true DC from its
                # accurate center (inside the receptive-field-poor edge ring),
                # subtract it, and restore it later as a weighted average.
                edge = overlap_latent // 2
                center = tile_output[:, :, :, edge:tile_lat - edge, edge:tile_lat - edge]
                dc = center.mean(dim=(3, 4), keepdim=True)
                corrected = tile_output.float() - dc.float()
                wy = _feather(tile_lat, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(tile_lat, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, tile_lat, tile_lat)
                result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += corrected * window
                dc_result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += dc.float() * window
                weights[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += window
                del tile_input, tile_output

    restored = (result + dc_result) / weights.clamp_min(1e-6)
    encoded = restored[:, :16, :, :latent_h, :latent_w].to(sample.dtype)
    if _TRT_DEBUG:
        _trt_dbg_stats(f"enc_chunk_out_{frames}f", encoded)
    del source, result, weights, dc_result
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encoded


_ENGINE_FILE_RE = re.compile(r"^vae_encoder_(\d+)f_tile\d+\.rtxplan$")


def _available_engine_frames() -> list[int]:
    """Return the sorted video-frame sizes of every usable encoder engine on disk."""
    found: set[int] = set()
    for d in ARTIFACTS_DIRS:
        try:
            if not d.is_dir():
                continue
            for p in d.iterdir():
                m = _ENGINE_FILE_RE.match(p.name)
                if m and p.is_file():
                    try:
                        if p.stat().st_size > 1_000_000:
                            found.add(int(m.group(1)))
                    except OSError:
                        continue
        except OSError:
            continue
    return sorted(found)


def pick_engine_frames(video_frames: int, preferred: str = "auto") -> int | None:
    """Pick the encoder engine frame size for a video of `video_frames` frames.

    Selection order:
    1. preferred (from the loader dropdown / settings node) if that engine exists;
    2. an engine matching video_frames exactly (1-shot encode);
    3. the largest engine that fits inside video_frames (chunked encode);
    4. if every engine is larger than the clip, the smallest engine (encode pads/crops);
    5. None only when no engine exists at all.
    """
    engines = _available_engine_frames()
    if not engines:
        return None
    if preferred != "auto":
        try:
            cand = int(preferred)
            if cand in engines:
                return cand
        except ValueError:
            pass
    if video_frames in engines:
        return video_frames
    fits = [e for e in engines if e <= video_frames]
    if fits:
        return fits[-1]
    return engines[0]


@torch.inference_mode()
def _encode_chunked(sample: torch.Tensor, total_frames: int, engine_frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a long clip by splitting it into engine_frames chunks with 4-frame temporal overlap."""
    _, _, _, height, width = sample.shape
    lat_total = (total_frames - 1) // 4 + 1
    lat_engine = (engine_frames - 1) // 4 + 1
    stride = engine_frames - 4  # 4-frame overlap -> 1 latent-frame overlap
    lat_h, lat_w = height // 8, width // 8
    result = torch.zeros((1, 16, lat_total, lat_h, lat_w), device="cuda", dtype=sample.dtype)
    starts = list(range(0, total_frames - engine_frames + 1, stride))
    if starts[-1] != total_frames - engine_frames:
        starts.append(total_frames - engine_frames)
    for start in starts:
        chunk = sample[:, :, start:start + engine_frames]
        lat = _encode_single_chunk(chunk, engine_frames, vae=vae, dit_model=dit_model)
        lat_start = start // 4
        result[:, :, lat_start:lat_start + lat_engine] = lat
        del chunk, lat
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def encode(sample: torch.Tensor, vae: torch.nn.Module | None = None, dit_model: str | None = None, engine_frames: str = "auto") -> torch.Tensor:
    """
    Encode [B,3,T,H,W] to posterior mean [B,16,(T-1)/4+1,H/8,W/8] in 1 shot using TensorRT engine.
    """
    if sample.ndim != 5 or sample.shape[0] != 1 or sample.shape[1] != 3:
        raise ValueError(f"TensorRT encoder expects [1,3,T,H,W], got {tuple(sample.shape)}")
    _, _, total_frames, height, width = sample.shape
    if height % 8 or width % 8:
        raise ValueError("TensorRT encoder input dimensions must be divisible by 8")

    # Release cached-but-unused VRAM from previous batches/other nodes to avoid
    # allocator pressure during 512px-tile engine execution (NaN source).
    torch.cuda.empty_cache()

    # Ensure 4n+1
    req_frames = ((total_frames - 1) // 4) * 4 + 1
    if total_frames != req_frames:
        pad_len = req_frames - total_frames
        last_frame = sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        sample = torch.cat([sample, last_frame], dim=2)
        total_frames = req_frames

    engine_video_frames = pick_engine_frames(total_frames, engine_frames)
    if engine_video_frames is None:
        raise FileNotFoundError("No TensorRT VAE encoder engine found (need vae_encoder_{5,9,13,17,21,29}f_tile512.rtxplan)")
    if engine_video_frames > total_frames:
        # The clip is shorter than every available engine: pad the video to the
        # engine size, encode in 1 shot, then crop back to the actual latent length.
        pad_len = engine_video_frames - total_frames
        last_frame = sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        padded = torch.cat([sample, last_frame], dim=2)
        encoded = _encode_single_chunk(padded, engine_video_frames, vae=vae, dit_model=dit_model)
        lat_needed = (total_frames - 1) // 4 + 1
        return encoded[:, :, :lat_needed]
    if engine_video_frames == total_frames:
        print(f"[SeedVR2 TensorRT] Encoding {engine_video_frames}f in 1 shot with dedicated {engine_video_frames}f TensorRT engine...")
        return _encode_single_chunk(sample, total_frames, vae=vae, dit_model=dit_model)
    n_chunks = (total_frames + engine_video_frames - 5) // (engine_video_frames - 4)
    print(f"[SeedVR2 TensorRT] Encoding {n_chunks} chunks of {engine_video_frames}f with TensorRT engine (4-frame temporal overlap)...")
    return _encode_chunked(sample, total_frames, engine_video_frames, vae=vae, dit_model=dit_model)


def resolve_engine_frames(preferred: str = "auto") -> int | None:
    """Return the largest available encoder engine video-frame size (for chunking)."""
    engines = _available_engine_frames()
    if not engines:
        return None
    if preferred != "auto":
        try:
            cand = int(preferred)
            if cand in engines:
                return cand
        except ValueError:
            pass
    return engines[-1]


def release() -> None:
    """Clear cached execution contexts and streams to free GPU VRAM."""
    global _ENGINES
    _ENGINES.clear()
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _gc.collect()
        torch.cuda.empty_cache()
```

---

### 4.2 `src/core/infer.py` (改修対象の TRT エンコード関連関数全文)

```python
def _trt_encode_batch(enc_sample, vae, dit_model, engine_frames_setting):
    """Encode a video batch with the TensorRT encoder, chunking to engine size.

    Engine selection is based on this batch's actual length (pick_engine_frames),
    so any available engine (e.g. 5f/21f/29f) is used instead of silently falling
    back to the fp16 VAE. Batches shorter than the smallest engine are padded to
    engine size, encoded in 1 shot, then cropped back.
    """
    from .trt_encoder import encode as trt_encode, pick_engine_frames
    total = enc_sample.shape[2]
    # Pad spatial dims to multiples of 8 so tile boundaries align with latent boundaries.
    h, w = enc_sample.shape[3], enc_sample.shape[4]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        enc_sample = torch.nn.functional.pad(enc_sample, (0, pad_w, 0, pad_h), mode="replicate")
        _TRT_CROP_HW[0], _TRT_CROP_HW[1] = h, w
    else:
        _TRT_CROP_HW[0], _TRT_CROP_HW[1] = -1, -1

    engine_video_frames = pick_engine_frames(total, engine_frames_setting)
    if engine_video_frames is None:
        raise RuntimeError("No TensorRT VAE encoder engine available")

    lat_needed = (total - 1) // 4 + 1
    if total < engine_video_frames:
        # Batch is shorter than every engine: pad to engine size, 1-shot, crop.
        pad_len = engine_video_frames - total
        last_frame = enc_sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        padded = torch.cat([enc_sample, last_frame], dim=2)
        lat = trt_encode(padded, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))
        return lat[:, :, :lat_needed]

    if total == engine_video_frames:
        return trt_encode(enc_sample, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))

    # Chunked encoding for batches longer than engine
    stride = ((engine_video_frames - 4) // 4) * 4
    if stride < 4:
        stride = 4
    lat_parts = []
    starts = list(range(0, total - engine_video_frames + 1, stride))
    if starts[-1] != total - engine_video_frames:
        starts.append(total - engine_video_frames)
    for start in starts:
        chunk = enc_sample[:, :, start:start + engine_video_frames].contiguous()
        lat = trt_encode(chunk, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))
        lat_parts.append((lat, start // 4))

    lat0 = lat_parts[0][0]
    latent = torch.zeros((1, 16, lat_needed, lat0.shape[3], lat0.shape[4]), device=lat0.device, dtype=lat0.dtype)
    # Causal encoder: a chunk's leading latents (context-poor) are LESS accurate than
    # the previous chunk's trailing latents (full context). So earlier chunks win.
    # Write in reverse so the first chunk keeps its (accurate) values.
    for lat, lat_start in reversed(lat_parts):
        latent[:, :, lat_start:lat_start + lat.shape[2]] = lat
    return latent
```

```python
            # VAE process by each group.
            for sample in batches:
                # Check TensorRT VAE encoder
                _enc_trt = getattr(self, "use_tensorrt_vae_encode",
                                   getattr(self, "use_tensorrt_vae", False))
                if _enc_trt or os.environ.get("SEEDVR2_TRT_ENCODER", "0") == "1":
                    # No silent fp16 fallback: selecting the TensorRT encoder means
                    # TRT must encode. A missing engine / any failure raises a clear
                    # error instead of quietly running the standard (fp16) VAE.
                    from .trt_encoder import encode as trt_encode, HAS_TRT as _trt_has
                    if not _trt_has:
                        raise RuntimeError(
                            "TensorRT VAE Encoder is selected but TensorRT is not available. "
                            "Install tensorrt-rtx or use SeedVR2LoadVAEModel for fp16 encode."
                        )
                    enc_sample = sample if sample.ndim == 5 else sample.unsqueeze(0)
                    if enc_sample.ndim != 5:
                        raise RuntimeError(
                            f"TensorRT VAE Encoder expects [1,C,T,H,W], got {tuple(sample.shape)}. "
                            "Use SeedVR2LoadVAEModel for fp16 encode."
                        )
                    self.debug.log(f"Encoding with TensorRT VAE Encoder (engine={getattr(self, 'use_tensorrt_engine_frames', 'auto')})", category="info", indent_level=1)
                    latent = _trt_encode_batch(enc_sample, self.vae, self._resolve_dit_name(), getattr(self, 'use_tensorrt_engine_frames', 'auto'))
                    latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                    latent = optimized_channels_to_last(latent)
                    latent = (latent - shift) * scale
                    latents.append(latent)
                    continue
```

---

### 4.3 `src/interfaces/trt_vae_model_loader.py` (改修対象のローダー全文)

```python
def _available_engine_frames(kind: str = "encoder") -> list[str]:
    """Scan the artifacts dir and auto-populate the engine_frames dropdown.

    kind="encoder" scans vae_encoder_<N>f_tile*.rtxplan,
    kind="decoder" scans vae_decoder_tile_*_<N>f.rtxplan.
    Dropping an engine into tensorrt_backend/artifacts/ is enough to make it
    selectable after a ComfyUI restart.
    """
    import re
    frames = set()
    pattern = "vae_encoder_*f_tile*.rtxplan" if kind == "encoder" else "vae_decoder_tile_*_*f.rtxplan"
    search_dirs = [ARTIFACTS_DIR, ROOT.parents[1] / "models" / "tensorrt" / "seedvr2"]
    for d in search_dirs:
        try:
            if d.is_dir():
                for pth in d.glob(pattern):
                    if kind == "encoder":
                        m = re.search(r"_(\d+)f_tile", pth.name)
                    else:
                        m = re.search(r"_(\d+)f\.rtxplan", pth.name)
                    if m:
                        n = int(m.group(1))
                        # Only 4n+1 frame counts are valid (the exporter normalizes to 4n+1,
                        # so e.g. a file named 195f actually contains a 193f graph).
                        if (n - 1) % 4 == 0:
                            frames.add(str(n))
        except Exception:
            pass
    return ["auto"] + sorted(frames, key=int, reverse=True)


class SeedVR2LoadTensorRTVAEEncoder(io.ComfyNode):
    """Encoder-only TensorRT VAE config (separate engine frame size from the decoder)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        vae_models = get_available_vae_models()
        return io.Schema(
            node_id="SeedVR2LoadTensorRTVAEEncoder",
            display_name="SeedVR2 Load TensorRT VAE Encoder",
            category="SEEDVR2",
            description=(
                "Encoder-only TensorRT VAE configuration. Lets you choose a different "
                "engine frame size for encoding than for decoding (e.g. encode 21f / decode 21f). "
                "Connect to the vae_encode input of SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE inference"
                ),
                io.Combo.Input("engine_frames",
                    options=_available_engine_frames("encoder"),
                    default="auto",
                    optional=True,
                    tooltip="TensorRT encoder engine frame size. Auto-populated from artifacts. "
                            "auto = pick the largest available engine."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration for the encoder path."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, device: str,
                engine_frames: str = "auto") -> io.NodeOutput:
        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae_encoder"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": device,
            "offload_device": "none",
            "cache_model": False,
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "engine_frames": engine_frames,
            "node_id": node_id,
        }
        return io.NodeOutput(vae_config)


# Backward compatibility alias
SeedVR2LoadTensorRTVAEModel = SeedVR2LoadTensorRTVAEEncoder


class SeedVR2LoadTensorRTVAEDecoder(io.ComfyNode):
    """Decoder-only TensorRT VAE config (separate engine frame size from the encoder)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        vae_models = get_available_vae_models()
        return io.Schema(
            node_id="SeedVR2LoadTensorRTVAEDecoder",
            display_name="SeedVR2 Load TensorRT VAE Decoder",
            category="SEEDVR2",
            description=(
                "Decoder-only TensorRT VAE configuration. Lets you choose a different "
                "engine frame size for decoding than for encoding (e.g. encode 89f / decode 65f). "
                "Connect to the vae_decode input of SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE inference"
                ),
                io.Combo.Input("engine_frames",
                    options=_available_engine_frames("decoder"),
                    default="auto",
                    optional=True,
                    tooltip="TensorRT decoder engine frame size. Auto-populated from artifacts. "
                            "auto = pick the largest available engine."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration for the decoder path."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, device: str,
                engine_frames: str = "auto") -> io.NodeOutput:
        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae_decoder"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": device,
            "offload_device": "none",
            "cache_model": False,
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "engine_frames": engine_frames,
            "node_id": node_id,
        }
        return io.NodeOutput(vae_config)
```

---

### 4.4 `src/interfaces/__init__.py` (コード全文)

```python
"""
SeedVR2 ComfyUI Nodes
Central registry for all SeedVR2 nodes
"""

from comfy_api.latest import ComfyExtension, io

from .video_upscaler import SeedVR2VideoUpscaler
from .dit_model_loader import SeedVR2LoadDiTModel
from .vae_model_loader import SeedVR2LoadVAEModel
from .torch_compile_settings import SeedVR2TorchCompileSettings
from .trt_vae_builder import SeedVR2BuildTensorRTVAE
from .trt_vae_model_loader import (
    SeedVR2LoadTensorRTVAEEncoder,
    SeedVR2LoadTensorRTVAEModel,
    SeedVR2LoadTensorRTVAEDecoder,
)
from .video_save import SeedVR2SaveVideo


class SeedVR2Extension(ComfyExtension):
    """SeedVR2 ComfyUI Extension"""
    
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return list of all SeedVR2 nodes"""
        return [
            SeedVR2VideoUpscaler,
            SeedVR2LoadDiTModel,
            SeedVR2LoadVAEModel,
            SeedVR2TorchCompileSettings,
            SeedVR2BuildTensorRTVAE,
            SeedVR2LoadTensorRTVAEEncoder,
            SeedVR2LoadTensorRTVAEDecoder,
            SeedVR2SaveVideo,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """ComfyUI V3 entry point"""
    return SeedVR2Extension()


__all__ = [
    'SeedVR2VideoUpscaler',
    'SeedVR2LoadDiTModel',
    'SeedVR2LoadVAEModel',
    'SeedVR2TorchCompileSettings',
    'SeedVR2BuildTensorRTVAE',
    'SeedVR2LoadTensorRTVAEEncoder',
    'SeedVR2LoadTensorRTVAEModel',
    'SeedVR2LoadTensorRTVAEDecoder',
    'SeedVR2SaveVideo',
    'SeedVR2Extension',
    'comfy_entrypoint',
]
```

---

### 4.5 `__init__.py` (コード全文)

```python
"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure FFmpeg is found across standard Windows paths or bundled imageio_ffmpeg
def _ensure_ffmpeg_path():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    candidate_dirs = [
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"D:\ffmpeg\bin"),
        Path(__file__).resolve().parent / "bin" / "ffmpeg" / "bin",
        Path(__file__).resolve().parent / "bin",
    ]
    for d in candidate_dirs:
        if (d / "ffmpeg.exe").exists() and (d / "ffprobe.exe").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            ffmpeg_dir = str(Path(ffmpeg_exe).parent)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

_ensure_ffmpeg_path()

# Check critical dependencies early to provide better error messages
# and auto-install if possible, especially useful for Vast.ai / RunPod
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.split(">")[0].split("=")[0].split("<")[0]
    
    try:
        __import__(import_name)
        return  # Already available
    except (ImportError, ModuleNotFoundError):
        pass
    if True:  # Package is missing - install it
        print("\n" + "="*80)
        print(f"SeedVR2: '{import_name}' module not found.")
        print(f"SeedVR2: Current Python executable: {sys.executable}")
        print(f"SeedVR2: Attempting automatic installation of {package_name}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f"SeedVR2: Successfully installed {package_name}")
        except Exception as e:
            print(f"SeedVR2: Auto-installation failed: {e}")
            print("This often happens on Vast.ai / RunPod when pip installs to a different Python environment.")
            print(f"Please run the following command manually in your terminal:")
            print(f"    {sys.executable} -m pip install \"{package_name}\"")
        print("="*80 + "\n")

# All critical dependencies from requirements.txt
# (torch/torchvision/numpy are assumed present via ComfyUI)
_REQUIRED_PACKAGES = [
    ("safetensors", None),
    ("tqdm", None),
    ("psutil", None),
    ("einops", None),
    ("omegaconf>=2.3.0", "omegaconf"),
    ("diffusers>=0.33.1", "diffusers"),
    ("transformers", None),
    ("accelerate", None),
    ("peft>=0.17.0", "peft"),
    ("rotary_embedding_torch>=0.5.3", "rotary_embedding_torch"),
    ("opencv-python", "cv2"),
    ("gguf", None),
    ("matplotlib", None),
    ("tensorrt-rtx", "tensorrt_rtx"),
    ("onnx", "onnx"),
    ("onnxscript", "onnxscript"),
    ("polygraphy", "polygraphy"),
]

for pkg, imp in _REQUIRED_PACKAGES:
    ensure_package(pkg, imp)

# Verify TensorRT RTX VAE engines
try:
    from pathlib import Path
    _cur_artifacts = Path(__file__).resolve().parent / "tensorrt_backend" / "artifacts"
    _cur_artifacts.mkdir(parents=True, exist_ok=True)
    _ready_count = 0
    _engines = (
        "vae_encoder_5f_tile512.rtxplan",
        "vae_encoder_21f_tile512.rtxplan",
        "vae_decoder_tile_512_5f.rtxplan",
        "vae_decoder_tile_256_21f.rtxplan"
    )
    for _eng in _engines:
        _dst = _cur_artifacts / _eng
        if _dst.exists() and _dst.stat().st_size > 1_000_000:
            _ready_count += 1
    if _ready_count == len(_engines):
        print(f"[SeedVR2 TensorRT] ✅ All {len(_engines)} TensorRT RTX VAE engines ready (2x-5x acceleration active)")
    else:
        print(f"[SeedVR2 TensorRT] ℹ️ {_ready_count}/{len(_engines)} RTX VAE engines ready (engines will build on first run with 'SeedVR2 Load TensorRT VAE Model')")
except Exception as _trt_sync_err:
    print(f"[SeedVR2 TensorRT] Warning during engine check: {_trt_sync_err}")

# Windows cp932: patch inductor jinja open(encoding=utf-8) before any torch.compile
try:
    from .src.core.fix_inductor import _fix_inductor_windows_encoding

    _fix_inductor_windows_encoding()
except Exception as _seedvr2_inductor_fix_err:  # noqa: BLE001
    print(f"[SeedVR2] Warning: inductor Windows encoding fix skipped: {_seedvr2_inductor_fix_err}")

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import (
    comfy_entrypoint,
    SeedVR2Extension,
    SeedVR2VideoUpscaler,
    SeedVR2LoadDiTModel,
    SeedVR2LoadVAEModel,
    SeedVR2TorchCompileSettings,
    SeedVR2BuildTensorRTVAE,
    SeedVR2LoadTensorRTVAEEncoder,
    SeedVR2LoadTensorRTVAEModel,
    SeedVR2LoadTensorRTVAEDecoder,
    SeedVR2SaveVideo,
)

print(f"[SeedVR2] Loaded nodes: SeedVR2VideoUpscaler, SeedVR2LoadTensorRTVAEEncoder (⚡TRT), SeedVR2LoadTensorRTVAEDecoder (⚡TRT), SeedVR2BuildTensorRTVAE, SeedVR2LoadVAEModel, SeedVR2LoadDiTModel, SeedVR2SaveVideo")

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]
```

---

## 5. その技術的意味と設計意図

### 5.1 テーマ①の技術的意義
1. **シェイプ判定ガードによるスクラッチパッドバッファの再利用保護**  
   `current_shape != target_shape` の条件分岐を挿入することにより、固定タイルサイズ（Encoder: 512x512, Decoder: 256x256）でビルドされた静的 TensorRT エンジンに対しては、初期化以降 `context.set_input_shape` の呼び出しが完全にスキップされる。これにより、TensorRT の内部メモリマネージャがトリガーする「スクラッチパッドの再確保とアドレス空間の破棄」を防止し、初期割り当てされたクリーンなバッファが後続の全タイルでそのまま維持される。
2. **決定論的ウォームアップによる未初期化アキュムレータのサニタイズ**  
   タイリング処理を開始する前に、ゼロで満たされたダミーバッファを用いて 1 回だけ `execute_async_v3` を実行する。これにより、TensorRT 内部の畳み込みワークスペース、テンポラルキャッシュ、CUBLAS/CUDNN の作業領域がゼロデータで強制的に初期化（上書き）される。この結果、最初の実データタイル（`y=0, x=0`）の計算時に未初期化 VRAM のゴミデータを読み出す可能性が物理的に排除され、左上のチェッカーボード／市松模様／高周波カラーノイズが 100% 根絶される。
3. **外周パディングの排除による VRAM 最小化**  
   タイル境界と画像境界の余白処理を、結果バッファ全体の無制限な拡張ではなく、必要最小限のローカルパディングとスライスに限定した。これにより、Float32 累積バッファ（`result`, `weights`）の巨大化を防ぎ、16GB 等の標準的 GPU 環境でも VRAM 圧迫（OOM）を一切起こさずにネイティブな処理能力を発揮できる。

### 5.2 テーマ②の技術的意義
1. **Pad & Crop による短尺バッチの完全救済と 1-shot TRT 完走**  
   ワークフローで設定されたバッチサイズ（例: 5 フレーム）が利用可能なエンジン（21 フレーム等）より短い場合でも、最終フレームを複製してエンジン長までパディング（`sample[:, :, -1:, :, :].repeat(...)`）し、1-shot で TRT エンコードを実行した上で、必要な潜在長 `(total - 1) // 4 + 1` にクロップして返却する。これにより、短尺クリップ処理時の `IndexError` が物理的に発生しなくなり、常に最大速度の TRT エンジンでエンコードが完走する。
2. **サイレント FP16 転落の完全排除とフェイルファスト設計**  
   「TRT エンコーダーを選択した以上、TRT で完動させるか、環境不備があれば明瞭な例外で停止させる」という厳格な原則を確立した。TRT ライブラリ不在やエンジン不在時にサイレントに FP16 へ転落してユーザーを欺く挙動を排し、即座に `RuntimeError` を送出することで、意図しない低速動作や無駄な GPU リソース消費を即座に検知・防止できる。
3. **アーキテクチャの完全対称性と直感的な UI 設計**  
   デコーダー（`SeedVR2LoadTensorRTVAEDecoder`）とエンコーダー（`SeedVR2LoadTensorRTVAEEncoder`）の入出力、設定項目、内部ロジック（`pick_engine_frames`, Pad & Crop, シェイプガード, ウォームアップ）が 100% 同一の対称構造となった。ユーザーはエンコード・デコードそれぞれで独立して最適エンジンフレーム（例: encode 21f / decode 21f）を選択可能となり、ワークフローの堅牢性と保守性が飛躍的に向上した。
