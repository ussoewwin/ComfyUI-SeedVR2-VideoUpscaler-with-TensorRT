# ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../README.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) 的 ComfyUI 官方发布版本，支持高质量视频和图像放大。

本仓库是官方仓库（[https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)）的 Fork，基于 Apache 2.0 许可证创建。独立实现了 ConvRot INT8 与 NVFP4 量化模型支持，以及显存（VRAM）节省功能。

[![SeedVR2 v2.5 Deep Dive Tutorial](https://img.youtube.com/vi/MBtWYXq_r60/maxresdefault.jpg)](https://youtu.be/MBtWYXq_r60)

## 工作流与节点示例

### 完整工作流概览（TensorRT VAE 与量化模型）

- 工作流 JSON：[`example_workflows/SeedVR2_tensorrt.json`](../example_workflows/SeedVR2_tensorrt.json)

![Usage Example - Full Workflow](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/main/docs/usage_01.png)

### TensorRT VAE 编码器与解码器节点

![Usage Example - TensorRT VAE Encoder & Decoder Nodes](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/main/docs/usage_02.png)

**`SeedVR2 Load TensorRT VAE Encoder`** 与 **`SeedVR2 Load TensorRT VAE Decoder`** 节点提供两个选择器：**`engine_frames`**（引擎帧长规格；`auto` = 使用可用的最大引擎）与 **`engine_tile`**（空间分块；`auto` / `256` / `512`）。`auto` 保持默认偏好（编码器优先 512px，解码器优先 256px、512px 作为旧版回退）；选择 `256` 或 `512` 时将严格限定为该分块的引擎——若对应分块的引擎不存在，节点会直接报错，而不会静默回退到其他分块。

### TensorRT VAE 引擎构建节点

- 工作流 JSON：[`example_workflows/Tensor Build.json`](../example_workflows/Tensor%20Build.json)

![TensorRT VAE Engine Builder Node](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/main/docs/build.png)

**`SeedVR2 Build TensorRT VAE Engines`** 节点允许用户在 ComfyUI 中直接按需显式构建专用的 TensorRT RTX VAE 引擎（`.rtxplan`）。该节点在后台调用 GPU 追踪（`tools/cloud_export_gpu.py`）与 TensorRT 编译（`tools/cloud_build_engine.py`）。

构建完成的引擎会自动保存至 `tensorrt_backend/artifacts/` 目录中。重启 ComfyUI 后，TensorRT VAE 解码器加载节点的 `engine_frames` 下拉菜单中将自动出现新构建的帧长规格。

#### 节点参数与设置详解

- **`model`**：原始 PyTorch VAE 模型权重（例如 `ema_vae_fp16.safetensors`）。
- **`frames`**：引擎目标帧长。自动规范化为所需的 **4n+1** 格式（例如 `5`、`21`、`29`、`61`、`89`、`101`、`185`、`205` 等）。
- **`tile_size`**：空间分块尺寸（`256` 或 `512`）：
  - **`256`**：空间分块较小；编译与运行时显存占用更低。适合在 16GB–24GB 显存显卡上构建长帧序列（60f–185f+）引擎。
  - **`512`**：空间分块更大、细节还原更优，但编译时显存占用较高。编码器与解码器引擎均支持。
- **`kind`**：选择构建的引擎类型：
  - **`both`**：同时构建编码器与解码器引擎。
  - **`decoder`**：仅构建 VAE 解码器引擎（推荐用于 Phase 3 解码加速）。
  - **`encoder`**：仅构建 VAE 编码器引擎。
- **`workspace_gb`**：TensorRT 编译时的最大工作区显存上限（单位：GB，默认 `8.0`–`16.0` GB）。
- **`min_ws`**：启用（`True`）后，通过二分搜索查找可成功构建的最小工作区尺寸，进一步降低运行时显存占用（编译时间略有增加）。
- **`force_rebuild`**：启用（`True`）后，即使引擎文件已存在也会强制重新构建并覆盖。
- **输出（`STRING`）**：输出构建状态、生成引擎文件名、文件大小及总耗时。可连接 `Show Text` 节点实时查看。

#### 使用与构建步骤

1. 在工作流中添加 **`SeedVR2 Build TensorRT VAE Engines`** 节点。
2. 配置所需的帧长（`frames`）、空间尺寸 `tile_size` 以及 `kind`（如 `decoder`）。
3. 点击 **Queue Prompt** 启动构建，系统将在后台自动完成 ONNX 导出与 TensorRT 引擎构建。
4. 构建完成后重启 ComfyUI，**`SeedVR2 Load TensorRT VAE Decoder`** 节点的 `engine_frames` 下拉列表中将显示新构建的帧数，即可开启极速解码。


### SeedVR2 (Down)Load DiT Model（标准 / 传统加载器）节点

![SeedVR2 (Down)Load DiT Model Node](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/main/docs/legacy_dit.png)

**`SeedVR2 (Down)Load DiT Model`** 节点是标准 DiT 加载器。它将 DiT 检查点（FP16 或 ConvRot INT8 / NVFP4 量化）加载到计算设备上，并可通过 BlockSwap（`blocks_to_swap` / `swap_io_components`）把部分模型转移到卸载设备，适合低显存系统。其输出类型（`SEEDVR2_DIT`）与 DisTorch2 加载器完全一致，可同样连接到 **`SeedVR2 Video Upscaler`** 节点。

#### 节点参数与设置

- **`model`**：DiT 检查点（如 `seedvr2_7b_int8_convrot.safetensors`）。支持量化（INT8 / NVFP4）与 FP16 检查点。
- **`device`**：DiT 推理的计算设备。
- **`blocks_to_swap`**：推理时在设备之间交换的 transformer 块数量（0 = 禁用）。需要设置 `offload_device` 且与 `device` 不同。
- **`swap_io_components`**：将输入/输出嵌入与归一化层卸载到卸载设备。需要设置 `offload_device` 且与 `device` 不同。
- **`offload_device`**：DiT 非活跃处理时的承载设备（`none` / `cpu` / 其他 GPU）。BlockSwap 的前提条件。
- **`cache_model`**：在工作流多次运行之间将 DiT 保留在 `offload_device` 上（需要设置 `offload_device`）。
- **`eject_models`**：加载本 DiT 前卸载其他常驻模型以释放 VRAM（建议开启）。
- **`attention_mode`** / **`sparge_topk`**：注意力后端与 SpargeAttn KV 保留比例（与 DisTorch2 加载器相同）。
- **`torch_compile_args`**：可选的 `torch.compile` 设置，来自 SeedVR2 Torch Compile Settings 节点。
- **`norm_bf16`**：Phase 2 的 RMS/QK norm 精度。关闭 = 传统 fp32 路径（质量优先）。开启 = bf16 norm 路径（节省常驻 VRAM；输出与 fp32 路径不同）。

### SeedVR2 (Down)Load DiT Model with Distorch2 节点

![SeedVR2 (Down)Load DiT Model with Distorch2 节点](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT/main/docs/distorch2.png)

**`SeedVR2 (Down)Load DiT Model with Distorch2`** 节点将整个（量化）DiT 常驻于系统内存，并在去噪时按需流式传输到计算设备，使用内置的 DisTorch2 后端（来自 `ComfyUI-MultiGPU` / `pollockjj`，GPL-3.0）。其 DiT 输出类型（`SEEDVR2_DIT`）与标准加载器完全一致，因此可同样连接到 **`SeedVR2 Video Upscaler`** 节点。

#### 节点参数与设置

- **`model`**：DiT 检查点（如 `seedvr2_7b_int8_convrot.safetensors`）。支持量化（INT8 / NVFP4）与 FP16 检查点。
- **`device`**：DiT 推理的计算设备。
- **`attention_mode`** / **`sparge_topk`**：注意力后端与 SpargeAttn KV 保留比例（与标准加载器相同）。
- **`distorch2_enabled`**：启用 DisTorch2 放置。关闭时行为等同标准加载器。
- **`virtual_vram_gb`**：在计算设备上为流式传输预留的虚拟 VRAM 预算（GB）。`0` = 整个模型保留在 donor（默认放置）。
- **`donor_device`**：物理承载打包 DiT 权重的设备（通常为 `cpu` = 系统内存）。
- **`expert_mode_allocations`**：高级逐块设备分配字符串，如 `"cpu,cpu,cuda:0"`。留空 = 由 `device` / `virtual_vram_gb` 推导放置。
- **`eject_models`**：放置前卸载其他常驻模型以释放 VRAM（建议开启）。
- **`emb_repeat_nocache`**：禁用 Phase 2 的 `emb_repeat` 缓存。开启则每次重算（节省常驻 VRAM；输出逐位相同）。
- **`norm_bf16`**：Phase 2 的 RMS/QK norm 精度。关闭 = 传统 fp32 路径（质量优先）。开启 = bf16 norm 路径（节省常驻 VRAM；输出与 fp32 路径不同）。

节点会在控制台报告最终放置结果（`[MultiGPU DisTorch V2] ... Final Allocation String` 及逐设备的层分布表）。
## 文档

详细说明请参阅官方仓库：

https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler

### 基准测试结果

- [SeedVR2 7B 量化基准测试结果](../benchmark/benchmark%20result.md)

### 本 Fork 技术指南（中文）

- [3B INT8 / NVFP4 模型支持](SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md)
- [INT8 原生推理](SEEDVR2_INT8_NATIVE_OPS_GUIDE.md)
- [NVFP4 与 torch.compile](SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md)
- [速度 / 显存余量](SEEDVR2_SPEED_VRAM_HEADROOM.md)
- [Windows 并行编译修复](SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md)
- [云环境依赖错误](vastai_dependency_guide.md)

## 更新日志

- [changelogzh.md](changelogzh.md)

## 🙏 致谢

本 ComfyUI 实现由 **[NumZ](https://github.com/numz)** 与 **[AInVFX](https://www.youtube.com/@AInVFX)**（Adrien Toupet）协作完成，基于 ByteDance Seed Team 的原始 [SeedVR2](https://github.com/ByteDance-Seed/SeedVR)。

特别感谢社区贡献者 [naxci1](https://github.com/naxci1)、[thehhmdb](https://github.com/thehhmdb)、[s-cerevisiae](https://github.com/s-cerevisiae)、[benjaminherb](https://github.com/benjaminherb)、[cmeka](https://github.com/cmeka)、[FurkanGozukara](https://github.com/FurkanGozukara)、[JohnAlcatraz](https://github.com/JohnAlcatraz)、[lihaoyun6](https://github.com/lihaoyun6)、[Luchuanzhao](https://github.com/Luchuanzhao)、[Luke2642](https://github.com/Luke2642)、[proxyid](https://github.com/proxyid)、[q5sys](https://github.com/q5sys) 以及许多其他人，在官方仓库（[https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)）中的改进、错误修复与测试。

### TensorRT VAE 后端

本仓库中的 TensorRT VAE 编码/解码引擎受到 [VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio)（Apache 2.0）的启发。我曾考虑将 DiT 移植到 TensorRT，但由于困难重重而放弃，转而通过创建支持 ConvRot INT8/NVFP4 量化模型的 ComfyUI 节点来提升性能。不过，将 VAE 编码/解码移植到 TensorRT 的构想正是来自该项目——没有这项工作，这一方案根本不会诞生。向原作者致以诚挚的敬意与感谢。

### DisTorch2 后端

**`SeedVR2 (Down)Load DiT Model with Distorch2`** 节点使用的 DisTorch2 后端，是
[ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)（作者 **pollockjj**，亦以
`comfyui-multigpu` 分发）的逐字副本。被复制的文件位于 `src/distorch2/`
（`distorch_2.py`、`wrappers.py`、`device_utils.py`、`model_management_mgpu.py`），
与上游源码逐字节一致；`src/core/distorch2_placement.py` 是本仓库自有的桥接层，
负责将 SeedVR2 DiT 接入该后端。DisTorch2 实现的所有功劳归于上游作者；
原始项目见 [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)。

## 📜 许可协议

本仓库的代码以 **Apache 2.0** 许可协议发布，详见
[LICENSE](../LICENSE) 文件。

TensorRT VAE 后端受
[VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio)
启发，该项目同样以 Apache 2.0 许可协议发布。依照 Apache 2.0 要求保留署名与版权声明。

**DisTorch2 后端（GPL-3.0 声明）。** `src/distorch2/` 下的 DisTorch2 后端是
[ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) 的逐字副本，后者以
**GNU General Public License v3.0（GPL-3.0）** 发布，而本仓库其余部分为
**Apache 2.0**。GPL-3.0 是 copyleft（传染性）许可：GPL-3.0 的义务附着于这些被复制的文件
——其源码在此提供，上游版权与许可声明在各复制文件中完整保留，
任何再分发或修改 `src/distorch2/` 的一方均须遵守 GPL-3.0。
