<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/blob/main/md/changelog.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

# 更新日志

## v1.6.5 — 2026-10-10
- **摘要：** TensorRT VAE 回退路径彻底清除与引擎构建中间 ONNX 自动删除：
  - **清除 SeedVR2 Video Upscaler 的 PyTorch VAE 意外回退：** 修复了当仅连接单个 VAE 输入端（`vae_decode` 或 `vae_encode`）时，另一端因空配置而意外脱落回退到标准 PyTorch FP16 VAE、导致在高分辨率（如 4208px）下发生严重 CUDA 显存溢出（OOM）的问题。现已实现单端连接时自动双向同步配置；且只要任一端指定了 TensorRT，Encode 与 Decode 均强制锁定为 TensorRT 执行并同步 Runner 状态标识，彻底杜绝静默回退到 FP16 路径。
  - **构建引擎后自动清理中间 ONNX：** 在 `SeedVR2 Build TensorRT VAE Engines` 节点以及加载器引擎编译逻辑中，当 TensorRT `.rtxplan` 引擎成功生成后，立即自动删除临时生成的中间 `.onnx` 文件及关联的 `.onnx.data`，避免每次构建残留数 GB 的冗余磁盘占用。
  - **节点显示名称精简：** 移除节点注册时的历史版本号后缀，规范统一为 `SeedVR2 Video Upscaler`。
- **技术详情：** 参见 [v1.6.5 发行说明](v1.6.5.md) 获取完整说明

## v1.6.4 — 2026-10-10
- **摘要：** 更新 ConvRot INT8 与 w4a8 DiT 模型的自动下载仓库：
  - **ConvRot INT8 与 w4a8 模型注册表迁移：** 将所有 3B / 7B / 7B sharp ConvRot INT8 及非对称 w4a8 INT8 DiT 模型的自动下载源仓库更新为 [`ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8`](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8)。
  - **直链根目录文件解析：** 远程下载文件名直接对齐根目录资源（`seedvr2_3b_convrot_int8.safetensors`、`seedvr2_7b_convrot_int8.safetensors`、`seedvr2_7b_sharp_convrot_int8.safetensors` 及其对应的 w4a8 文件），保留 SHA256 完整性校验，移除多余的子目录层级。
- **技术详情：** 参见 [v1.6.4 发行说明](v1.6.4.md) 获取完整说明

## v1.6.3 — 2026-10-09
- **摘要：** `triton-windows` 动态解析与解除版本固定：
  - **动态 Triton 解析：** 从 `requirements-windows-cu132.txt` 中移除了硬编码版本绑定（`triton-windows==3.5.1.post24`）。安装流水线与运行时环境现会自动从 PyPI 解析并安装最新的 `triton-windows` 构建版本，自适应更新的 PyTorch 运行时与 CUDA 环境，无需手动维护或锁定版本。
- **技术详情：** 参见 [v1.6.3 发行说明](v1.6.3.md) 获取完整说明

## v1.6.2 — 2026-10-08
- **摘要：** norm bf16 模式现已登陆标准（传统）DiT 加载器：
  - **SeedVR2 (Down)Load DiT Model 新增 norm_bf16：** 此前仅 DisTorch2 加载器（v1.6.0）提供的
    `norm_bf16` 开关，现已同样实装到标准 `SeedVR2 (Down)Load DiT Model` 节点。
    关闭（默认）= 传统 fp32 norm 路径（质量优先，行为不变）。
    开启 = Phase 2 的 RMS/QK norm 以 bf16 执行，显著节省常驻显存，
    代价是 bf16 舍入（逐像素 PSNR 约 37–39 dB vs fp32）。
  - 两个加载器共用同一开关语义与同一 DiT 侧 bf16 norm 路径，无论使用哪个加载器节点，行为一致。
- **技术详情：** 参见 [v1.6.2 发行说明](v1.6.2.md) 获取完整说明

## v1.6.1 — 2026-10-05
- **摘要：** v1.6.0 以来的 SpargeAttn 提速工作 —— spargeattn 后端现已完整跑通其每次调用的路径，
  去除了此前使其慢于 SageAttention2 的固定开销：
  - **Sage2++ fp16 累加路径：** 修复 ragged-window 路径，并将 spargeattn 路由至
    SageAttention2++ 的 fp16 累加内核（进化形态），不再回退到较慢的 f32 累加内核。
  - **等长窗口批处理（零拷贝）：** 将连续等长窗口合并进 stock API 调用，
    去除逐窗口的 Python/启动开销，且不改变结果。
  - **去除每次调用的 D2H 同步：** 移除 spargeattn 快速路径中每次调用的主机同步
    （plan-cache 以 shape/numel 为键），快速路径实现零 device-to-host 同步。
  - **变长（varlen）委托：** spargeattn 的 varlen 处理现委托给 SpargeAttn-hswq 库新增的
    `varlen` 入口（>= 1.2.1）。等长窗口保持单次零拷贝批处理启动；混合长度窗口在库内
    按相同长度分组（每种长度一次启动，无 padding 浪费，无逐序列启动）。
    从未违反内核的定长前提。
  - **备注：** 曾尝试实验性的持久化 Triton 缓存并已回退；spargeattn 路径保持经实测确认的正确状态
    （无持久化缓存计装）。
- **技术详情：** 参见 [v1.6.1 发行说明](v1.6.1.md) 获取完整说明

## v1.6.0 — 2026-10-03
- **摘要：** 新增 DisTorch2 DiT 加载器节点与 Phase 2 显存控制：
  - **SeedVR2 (Down)Load DiT Model with Distorch2：** 新节点，将整个（量化）DiT 常驻于系统内存，
    并在去噪时按需流式传输到计算设备，使用内置的 DisTorch2 后端
    （ComfyUI-MultiGPU（pollockjj）的逐字副本，GPL-3.0；详见 Credits/License 节）。
    新增虚拟 VRAM 预算、donor 设备、逐块分配字符串与模型卸载控制，
    输出类型与标准加载器同为 `SEEDVR2_DIT`。
  - **norm_bf16 开关：** 新增逐节点开关，用于 Phase 2 的 RMS/QK norm 精度。
    关闭（默认）= 传统 fp32 路径，输出逐位相同。
    开启 = bf16 norm 路径，节省常驻显存，代价是 bf16 舍入（逐像素 PSNR 约 37–39 dB vs fp32）。
  - **emb_repeat_nocache 开关：** 禁用 Phase 2 的 emb_repeat 缓存（每次重算；
    节省常驻显存；输出逐位相同）。
  - **模型注册表：** 移除 fp8 DiT 条目（3B / 7B / 7B sharp）；
    默认 DiT 现为 `seedvr2_7b_int8_convrot.safetensors`。
- **技术详情：** 参见 [v1.6.0 发行说明](v1.6.0.md) 获取完整说明

## v1.5.8 — 2026-10-02
- **摘要：** TensorRT VAE 权重精度优化（解码器 + 编码器）：
  - **仅权重 FP16 转换：** TRT VAE Decoder/Encoder 非引擎回退路径现将引擎权重转换为 FP16，
    `result` 累积缓冲区保持 FP32 并保留 4D 输出契约，在不影响输出精度的前提下削减权重常驻显存。
  - **验证一致性：** 在 RTX 5060 Ti 上确认多次运行稳定 — 解码器 PSNR 84–85 dB，
    编码器 PSNR 72.4 dB（对比 FP32 基准）。
- **技术详情：** 参见 [v1.5.8 发行说明](v1.5.8.md) 获取完整说明

## v1.5.7 — 2026-09-30
- **摘要：** DiT 执行显存瞬时峰值抑制与激活中间张量显存稳定性优化：
  - **SDPA 输出缓冲区预分配：** 在 `pytorch_varlen_attention` 中彻底废除 Python 列表切片暂存与最终 `torch.cat()` 拼接机制，改为单一输出缓冲区预分配并在切片视图上就地写入，杜绝注意力输出 2 倍显存瞬时膨胀与冗余 CPU-GPU 同步。
  - **SwiGLU MLP 分块前向执行：** 在 `SwiGLUMLP` 中针对超过 8,192 标记的长序列引入标记维度分块计算，将 `gate`、`up`、`hidden` 瞬时并存显存限制在单个块内，在保持计算结果完全位精确一致的前提下，将每层变换器块激活峰值从约 2.0 GB 压缩至约 330 MB。
  - **`torch.index_select` 高效张量收集：** 将 Swin 窗口注意力中通过高级花式索引（`[tgt_idx]`）进行的文本标记复制改为专用 CUDA 算子 `torch.index_select`，杜绝暗中复制造成的显存开销。
  - **Euler 采样条件张量就地复用：** 在采样循环外单次预分配 33 通道条件张量缓冲区，全采样步数中仅在切片视图上就地写入当前潜空间，彻底消除每步动态分配造成的垃圾回收与显存碎片化。
- **技术详情：** 请参阅 [v1.5.7 发行说明](v1.5.7.md) 获取完整说明

## v1.5.6 — 2026-09-29
- **摘要：** 下载策略修正与模型注册表更新：
  - **仅限加载器选择触发的模型下载：** 移除全部强制自动下载——模型仅在选择于加载器节点且文件缺失时下载；安装/更新时不再预下载默认模型，TensorRT VAE 引擎构建也不再附带拉取默认 DiT（[#2](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/issues/2)）。
  - **新增 6 个 ConvRot INT8 / NVFP4 DiT 模型：** `seedvr2_3b_int8_convrot`、`seedvr2_3b_nvfp4`、`seedvr2_7b_int8_convrot`、`seedvr2_7b_nvfp4`、`seedvr2_7b_sharp_int8_convrot`、`seedvr2_7b_sharp_nvfp4`（托管于 `Comfy-Org/SeedVR2`，SHA256 锁定）已加入模型注册表与下载列表。
  - **移除 GGUF 条目：** 从模型注册表中删除 GGUF（Q4_K_M / Q8_0）DiT 条目。
- **技术详情：** 请参阅 [v1.5.6 发行说明](v1.5.6.md) 获取完整说明

## v1.5.5 — 2026-09-28
- **摘要：** 生产环境稳定性改进与 TensorRT VAE 编码器全面重构：
  - **FFmpeg CFR 时间戳整流：** 彻底根除视频合并时的音画不同步及播放末端卡死。
  - **ExecutionContext 地址安全：** 引入互斥锁与流同步，杜绝显存覆盖与黑块损坏。
  - **确定性显存深度清理：** 全流程应用三阶段内存回收，消除显存碎片堆积。
  - **TensorRT VAE 编码器重构：** 移植 v1.5.4 伪影与显存修复基盘、短批次 Pad & Crop 极速单次执行、彻底移除静默降级，并引入对称的 `SeedVR2LoadTensorRTVAEEncoder` 节点。
- **技术详情：** 请参阅 [v1.5.5 发行说明](v1.5.5.md) 获取完整说明

## v1.5.4 — 2026-09-27
- **摘要：** 彻底修复横屏视频下 TensorRT VAE 解码器左上角马赛克/棋盘格伪影问题，且零显存膨胀：
  - **对齐 Studio 的静态形状判定：** 仅在当前形状与目标分块形状不符时才调用 `context.set_input_shape`。静态形状引擎完全跳过冗余重构，防止 TRT 内部暂存区缓冲区重新分配并抓取前序脏显存。
  - **确定性 Dummy 热身空跑：** 在进入空间分块循环前，使用全零张量执行一次单次 Dummy 推理。强制 TensorRT 清洗所有内部卷积工作区和时序累加器状态，彻底杜绝首个分块（`y=0, x=0`）读取未初始化内存。
  - **零显存膨胀架构：** 坚决摒弃会导致 float32 累加缓冲区（`result` 与 `weights`）显存激增 2~3 倍的外周 Padding 方案，保持原生分辨率最高解码速度与最小显存开销。
- **技术详情：** 请参阅 [v1.5.4 发行说明](v1.5.4.md) 获取完整说明

## v1.5.3 — 2026-09-09
- **摘要：** TensorRT VAE 编码器启用未成功；FP16 VAE 编码保持不变：
  - **TensorRT VAE 编码器：** `SeedVR2LoadTensorRTVAEModel` 在启用尝试期间注册，因左上角分块伪影在 256px 或 512px 分块尺寸下均无法解决而被再次移除。`SeedVR2LoadTensorRTVAEDecoder`（仅解码 TRT）与 `SeedVR2BuildTensorRTVAE` 仍可用。此外还评估了 FP16 编码路径的批量一次性变体并已回退（其在 FP16 上同样复现了模糊），逐帧循环仍是 FP16 编码实现。
- **技术详情：** 参见 [v1.5.3 发行说明](v1.5.3.md) 获取完整说明

## v1.5.2 — 2026-09-08

- **摘要：** 修复 TensorRT VAE 解码器中的黑块（空白 tile）回归问题，修正引擎选择逻辑：
  - **引擎选择重构：** `pick_engine_frames` 现在扫描实际存在的引擎文件（如 25f / 29f / 41f / 61f），取代硬编码的 `(video_frames, 29, 21, 5)` 列表，使已下载的引擎真正被使用，不再静默回退到 PyTorch VAE。
  - **不再静默回退：** `resolve_engine_frames` 返回磁盘上最大的引擎；短于最小引擎的片段会先填充、单次解码后再裁剪，而不是回退。
  - **按批次选择引擎：** `_trt_decode_batch` 根据实际批次长度选择引擎（短批次自动填充 + 裁剪）。
- **技术详情：** 参见 [v1.5.2 发行说明](v1.5.2.md) 获取完整说明

## v1.5.1 — 2026-09-05

- **摘要：** 安装程序与运行时稳定性全面改进：
  - **全自动零干预安装：** 修复 `install.py` 无法自动装全依赖的问题，无需手动运行批处理文件即可在目标 Python 环境中自动完成 `requirements.txt` 完整安装。
  - **统一注意力机制与 SDPA 标准：** 彻底废除 `install.py` 与 `scripts/install.ps1` 中脆弱的 FlashAttention 2 / SageAttention 2 外部 wheel 强制下载与安装逻辑；未安装自定义注意力加速库时，统一安全回退至 PyTorch 原生 SDPA（`attention_mode: sdpa`）（[#1](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/issues/1)）。
  - **全链路 FFmpeg 路径自动解析：** 在节点初始化（`__init__.py`）、安装器（`install.py`）、环境验证（`scripts/verify_install.py`）及 CLI 中全面引入多候选路径扫描与 `imageio_ffmpeg` 自动兜底机制，彻底杜绝视频合成与导出时的 PATH 缺失异常。
  - **解码器引擎规范明示：** 在文档中明确规定构建 TensorRT VAE 解码器引擎（`kind: decoder`）时必须使用 `tile_size: 256`，彻底杜绝推理时的空间维度不匹配问题。
  - **全面支持 64-bit 随机种子：** 将 seed 控件范围拓展至完整 64 位（`0..0xffffffffffffffff`），对齐 ComfyUI 核心节点（KSampler），并彻底移除冗余的 NumPy 随机种子依赖（[PR #635](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/pull/635)）。
- **技术详情：** 请参阅 [v1.5.1 发行说明](v1.5.1.md) 获取完整说明

## v1.5 — 2026-09-03

- **摘要：** 新增 TensorRT VAE 解码器支持与专用加载节点（`SeedVR2LoadTensorRTVAEDecoder`），支持多 Tile 引擎（256px/512px，4n+1 帧长规格）、执行上下文缓存复用、编码/解码独立解耦配置，以及引擎缺失时自动安全回退至 PyTorch VAE。
- **技术详情：** 请参阅 [v1.5 发行说明](v1.5.md) 获取完整说明

## v1.4 — 2026-07-31

- **摘要：** 在 `MODEL_REGISTRY` 中登记 3B HSWQ INT8 ConvRot 与 NVFP4 DiT 权重包（与 7B 相同的原生显存路径）。
- **技术详情：** 请参阅 [v1.4 发行说明](v1.4.md) 获取完整说明

## v1.3 — 2026-07-28

- **摘要：** Windows 上 torch.compile / inductor 运行时改进：在 win32 上启用并行 inductor 编译；在每个阶段首个 batch 之后关闭 compile worker 以释放 CUDA 上下文；运行期间启用 `cudnn.benchmark`；以及更均匀的 VAE 时序切片，以减少编译形状变体。
- **技术详情：** 请参阅 [v1.3 发行说明](v1.3.md) 获取完整说明

## v1.2 — 2026-07-28

- **摘要：** 通过构建时 `comfy.ops.mixed_precision_ops` 为 SeedVR2 DiT 增加原生 NVFP4 加载；并修复 Windows / inductor，使 FP16 VAE 的 `torch.compile` 不再因 cp932 解码或 `aten.bmm` 的 fallback+decomp 断言而失败。v1.1 的 INT8 路径仍然可用。
- **技术详情：** 请参阅 [v1.2 发行说明](v1.2.md) 获取完整说明

## v1.1 — 2026-07-27

- **摘要：** 通过构建时 `comfy.ops.mixed_precision_ops` 为 SeedVR2 DiT 增加原生 INT8 加载（`int8_tensorwise` + `comfy_quant` / `weight_scale`），使 INT8 权重包在 `load_state_dict` 过程中保持量化，而不是展开为完整 FP16（降低显存）。仅限 DiT；VAE 仍为 FP16。
- **技术详情：** 请参阅 [v1.1 发行说明](v1.1.md) 获取完整说明

## v1.0 — 2026-04-05

- **摘要：** 节点加载时将缺失的 SeedVR2 依赖自动安装到当前 ComfyUI 所用 Python（`sys.executable`），以解决 Vast.ai / RunPod 等云模板上终端 `pip` 与 ComfyUI 虚拟环境不一致导致的 `ModuleNotFoundError`（例如 `diffusers`、`rotary_embedding_torch`）。
- **技术详情：** 请参阅 [v1.0 发行说明](v1.0.md) 获取完整说明
