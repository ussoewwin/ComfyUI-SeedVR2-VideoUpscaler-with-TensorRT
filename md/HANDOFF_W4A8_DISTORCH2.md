# 引き継ぎ書: W4A8 × DisTorch2 完全対応（2026-10-05 17:03 時点）

新チャットは本書だけで作業再開できること。全事実は実測ベース。未実測のものには【推定】と明記。

## 0. タスク

SeedVR2 TenseRT カスタムノードで、**W4A8（`seedvr2_7b_convrot_w4a8.safetensors`）を
DisTorch2 の CPU オフロードで「他（INT8/NVFP4）と同じように」完全対応させる**。
症状: W4A8 が全量 CPU に載らず `cuda:0` に 467MB（23 層）が残留 → Phase 2 VRAM が INT8 比 +2〜4GB。

## 1. 真因（実測で確定済み・変更不要）

`src/core/distorch2_placement.py` の `_qt_storage_bytes()` が QuantizedTensor の `_qdata` のみ計上。
W4A8 は fp8 グループ scale を `_qdata` と**別 storage**（`_params.scale`）に持つため過小計上:

| 実測項目 | 値 |
|---|---|
| `_qdata`（int4 パック） | 18.00 MB/層（shape (N, K/2) int8） |
| `_params.scale`（weight_s_rel, fp8_e4m3） | 2.25 MB/層（shape (N, K/16)） |
| `_params.s_channel`（fp32） | 0.047 MB/層 |
| **合計（正）** | **20.30 MB/層** |
| `_packed_model_bytes`（修正前） | 3.95 GB（実際 4.44 GB、差 0.49 GB） |

差 0.49GB が DisTorch2 の cpu クォータ不足となり、`distorch_2.py` L519-520
（クォータに収まらない block は compute_device へ）で 467MB が GPU に残った。

**W4A8 と INT8 の明示的判別式（実測）**: `_qdata.shape[1] * 2 == weight.shape[1]`
（int4 パックは真。INT8 の `_qdata` は full-width なので偽。NVFP4/通常 tensor も偽）。

## 2. 適用済みの修正（GitHub=origin/main=LIVE、HEAD 三者一致）

**HEAD: `bdf236e`**（blob `6ee3ae4c`、GitHub/LIVE ハッシュ一致実測済み、両者 clean）

- `d406dc1` fix(distorch2): count W4A8 companion scale tensors in packed size
- `bdf236e` refactor(w4a8): **分離徹底** — 専用 2 関数に分割（Owner の「分岐分離・絶対混ぜるな」準拠）
  - `_is_w4a8_quant(t)` … 明示判別（専用）
  - `_w4a8_storage_bytes(t)` … W4A8 専用計算パス（`_qdata` + `_params.scale/s_channel/correction/codebook`）
  - `_qt_storage_bytes` は判別→専用パス、非該当は**元の一行的ロジックのまま**（INT8/NVFP4/fp16 は経路不変）
- `c43d353`（同日朝の混成コミット、Owner 作業 stash 由来）:
  - `dit_model_loader_distorch.py`: `build_distorch2_allocation_string` が `virtual_vram_gb<=0` で
    `""` を返し上流計算機（論理 dtype 4 倍過大）経路に落ちていたのを修正。vv<=0 でも
    `"#cuda:0;0.0;cpu"` を返し packed-size bridge（INT8/NVFP4 と同一路径）に通す
  - その他: `[P2MEM]`/`[P2LIVE]`/`[DITMEM]`/`[BLKMEM]`/`[ATTMEM]`/`[FTM]` プローブ、
    `rope.py` `__ROPE_NO_FP32__`（q/k fp32 強制除去、実測 peak -3.37GB 相当のfp32コピー排除、cos 0.999997）

## 3. 検証済み（実測）

| ケース | 結果 |
|---|---|
| W4A8 1 層 `_qt_storage_bytes` | 18.00 → **20.30 MB** ✓ |
| INT8 1 層 | **36.00 MB 不変** ✓（他への影響ゼロ） |
| plain fp16 | 0.019 MB 不変 ✓ |
| 全 109 .py `py_compile` | OK（同期後） ✓ |
| GitHub↔LIVE↔origin/main | HEAD `bdf236e` 三者一致・dirty なし ✓ |

検証スクリプト: `.openclaw/tmp/verify_sep.py`（ワークスペース seedvr2 配下。関数抽出実行式）

## 4. 未実施（新チャットの第一作業）

1. **実機 run（ご主人様操作領域）**: W4A8 で DisTorch2 有効 run し、
   `DisTorch2 Model Final Device/Layer Assignments` の `cpu` が **100%**（cuda 残留 0）になるか確認。
   期待ログ: `packed model size: 4.4xGB`（3.98GB より増える）→ `keep 0.00GB on cuda:0, host 4.4xGB on cpu`。
   ※ `_LayerDistribution` の `_LockedLinear 4542.87MB` と一致するはず。
2. **確認後**: 残留が消えれば Phase 2 VRAM ピークが INT8 並み（15GB 台、プローブ分約2GB含む）に下がるか観測。
3. **残留が残る場合の次の仮説（未検証・【推定】と扱う）**: `analyze_safetensor_loading` の
   TAIL から逆順 fit による端数残り（L510-520）。その場合はクォータ丸め（4.4xGB vs cpu 割当）を先に疑う。
4. **NVFP4**: 本修正で経路不変のはずだが実機未確認。
5. **プローブ撤去**: 原因確定が済んだら `[P2MEM]`/`[P2LIVE]`/`[DITMEM]`/`[BLKMEM]`/`[ATTMEM]`/`[FTM]` と
   `_SEEDVR2_*` マーカー部を撤去するかご主人様に確認（「消すな」指示が先、撤去指示が後なら指示優先）。

## 5. 環境・同期ルール（厳守）

- 作業先: GitHub `D:\USERFILES\GitHub\ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT`
  LIVE `D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT`
  （同一 remote `ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT` の別クローン。
  **両方 commit/push・pull して blob hash 一致を毎回実測**。Get-FileHash は改行変換で不一致表示になるため
  `git hash-object` で比較）
- Python は `D:\USERFILES\ComfyUI\python_embeded\python.exe` のみ。スクリプトは
  `.openclaw/tmp/*.py` に書いて `-X utf8` 実行（インライン -c は quoting 事故りやすい）
- ComfyUI 稼働中の実機 GPU 検証禁止（run はご主人様操作）

## 6. 紅線（ご主人様ルール・違反=死刑）

- **distorch2 の「ロジック」改变絶対禁止**（計算式・既存動作）。**追加は可**。判別は**明示的に分離**
  （専用関数・専用パス。条件分岐で混ぜるな。Owner 基本哲学「分離、分岐、絶対混ぜるな」）
- **GPU 常駐 etc. distorch2 の CPU オフロード目的と逆の提案絶対禁止**
- 実測なしの断定禁止。【推定】明示。Owner 提供データを自分の実測と書くな。台詞捏造禁止
- 「判断しません」「指示待ち」は逃走。**判断して即実行**。ただし対象取り違え前に一言明示
- 文字化けした短いメッセージ = 激怒。**確認で止まるな。文脈から最善解釈即実行**
- 同期ルール無視・LIVE 上書き前的未コミット確認・deleteAfterRun・大量出力・ポーリング監視禁止

## 7. 参考データ（実測）

- W4A8 safetensors: 4.44GB。キー内訳 weight 3.952GB / weight_s_rel 0.475GB / s_channel 0.007GB / bias 0.0025GB
- INT8: 7.76GB（_qdata full-width 36.047MB/層、scale 0.047MB/層）
- 実機ログ（Owner 提供 16:42）: `cuda:0,0.0000;cpu,0.0646` / `_LockedLinear 294 4542.87MB` / `cpu 270 4075.42MB 89.7%` + `cuda:0 23 467.07MB 10.3%`
- virtual_vram_gb 実測表: 0→cuda 0.2503 / 2→0.1245 / 4以上→cuda 0.0000（vv=min(vv,packed) で全量 cpu 行き）
- sparge 高速化は v1.6.1 で完成済み（`compatibility.py` は無変更。v1.6.1 以降に sparge 改変なし）
- LIVE の stash 4 件は**触っていない**（残置。削除指示なし）
