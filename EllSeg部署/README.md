# EllSeg 独立部署

位置：`C:\Users\16087\Desktop\AR美瞳\EllSeg部署`

这是可独立启动的 EllSeg 测试程序，使用作者现成的 **all.git_ok** 权重，不重新训练、不更换原有美瞳 App 的模型。

## 双击使用

双击 **启动EllSeg.lnk** 或 **run.cmd**，在启动窗口选择：

- **人像 / 摄像头**：先用 MediaPipe FaceMesh 裁出双眼，再分别运行 EllSeg。
- **单眼近景**：输入本来就是单眼的图片/视频，直接运行 EllSeg，不使用 FaceMesh。

再选择图片、离线视频或 USB 相机。**run_camera.cmd** 直接打开编号 0 的摄像头。相机使用前请关闭占用同一设备的其他软件。

界面左侧是原画面与椭圆，右侧是各眼原始裁剪、分割/椭圆叠加和标签图：

- 绿色：虹膜区域、虹膜椭圆。
- 橙色：瞳孔区域、瞳孔椭圆。
- 黑色：背景类。它不是单独的眼皮/睫毛分割类别。
- `plausible`：仅通过基本输出合理性检查，不是准确率或置信度承诺。

**Q / Esc** 退出；**空格** 暂停/继续；**S** 保存当前帧到 outputs。默认不上传、不录制摄像头、不保存用户画面；--save 或 S 为主动保存操作。

## 命令行示例

在此文件夹打开 PowerShell：

```powershell
# 图片：整张人像，自动裁双眼
.\run.cmd --image "D:\照片\人像.jpg" --mode face

# 图片：单眼近景
.\run.cmd --image ".\test_assets\eye.png" --mode eye

# 离线视频：手机竖屏方向自动处理
.\run.cmd --video "..\微信视频2026-09-21_015131_959.mp4" --mode face

# 无窗口，处理并保存整段视频和逐帧椭圆数据
.\run.cmd --video "D:\视频\眼睛.mp4" --mode eye --headless --save --output outputs/my-video

# USB 相机，其他设备可尝试编号 1、2
.\run_camera.cmd --camera 1

# 明确选择 CPU 或 GPU
.\run.cmd --image test_assets/eye.png --mode eye --device cpu
.\run.cmd --image test_assets/eye.png --mode eye --device dml
```

参数 `--frames 12` 限制处理帧数，默认 0 为处理到结束。`--report outputs/report.json` 保存运行统计。headless 摄像头必须指定帧数，避免无限运行。

本程序离线视频是**边推理边诊断预览**，速度取决于处理性能，不是原美瞳 App 的预处理缓存播放器。--save 生成的 MP4 使用输入的平均 FPS，可通过普通播放器播放；不保留音轨，变帧率源视频导出为固定平均帧率。

## 输出

图片/S 快照：

```text
outputs/<时间>/
  result.png           原画面上的椭圆诊断
  preview.png          完整程序界面预览
  right-eye.png        眼部模型输出叠加（单眼模式为 eye-eye.png）
  right-labels.png     标签值 0/1/2，不是彩色显示图
  left-eye.png
  left-labels.png
  ellipses.json        原始图像坐标中的虹膜/瞳孔椭圆及裁剪变换
```

视频 --save：`ellseg-result.mp4` + `ellipses.jsonl`。每行 JSON 对应一帧。椭圆参数为 `[中心x, 中心y, 半长轴, 半短轴, 旋转弧度]`，SVD 统一轴顺序，角度模 π 等价；半轴不是直径。错误预测仍可能通过基础合理性检查，需要人工查看。

## 模型和适配

- 源码：https://github.com/RSKothari/EllSeg
- 固定提交：`8f1ea13336fa9c662403bd5823ac13cc6a6dd632`
- 网络：DenseElNet / RITnet_v3，**2,608,965** 参数。
- 权重：`weights/all.git_ok`，约 10.45 MB。
- SHA-256：`0b8d3f4c2c936de3950150d302f3608679874317862942e1ec53b21c33a03a1a`
- ONNX：`weights/ellseg.onnx`，FP32、opset 17，输入 `[1,1,240,320]`。
- 输入：单眼灰度图，按宽度缩放并居中裁剪/填充到 320×240，再做图内均值/标准差归一化。
- 输出：三类分割 logits + 10 个椭圆回归参数。

推理遵循作者 `evaluate_ellseg.py`：瞳孔中心来自类别 2 logits 的空间 softmax，虹膜中心来自背景类别 0 logits 的相反数，温度 4；两者轴长和角度使用回归头。作者的变量命名与实际运行索引存在不一致，本部署按推理入口的索引：虹膜 regression[2:5]，瞳孔 regression[7:10]。没有随意交换输出。

`vendor/ellseg_model.py` 提取作者实际使用的网络块，仅将弃用的 np.int 改为 int、移除训练依赖并增加推理入口。全部 119 个 state_dict 键严格匹配。预训练权重使用 `weights_only=True` 加载。原文件放在 upstream_reference；MIT 许可证和来源哈希在 vendor。

灰度统计标准差低于 2 的平坦画面被标为无效并隐藏分割。防止缺失/纯黑输入被模型当成眼睛；其他输入上的 plausible 判定仍不保证真实眼部精度。

**EllSeg 估计的是完整椭圆，包括可能被眼皮遮住的部分。**它不是眼睑遮罩、不是注视点估计，也不是现成的美瞳材质渲染器。模型面向近眼追踪数据，普通自拍、深色瞳孔、眼镜反光和远距离场景需重新评估。此部署验证了运行与数值一致性，尚未证明它优于 LightIrisNet。

## 环境

本机已可用，启动器读取 `runtime-python.txt` 指向：

`D:\ARLensRuntime\venv\Scripts\python.exe`

复用已有环境中的 PyTorch、MediaPipe、OpenCV 和 ONNX Runtime DirectML。默认 `auto` 在 ONNX 与 DirectML 可用时选择 GPU，本机已选中 **NVIDIA GeForce GTX 1060 5GB**；手动 `--device cpu` 可回退。没有安装/更换系统驱动或改动主 App。

代码和权重都在当前文件夹内，运行不导入父目录的美瞳程序。若换电脑或要单独的 Python 环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -RuntimeRoot D:/EllSegRuntime
```

该脚本创建独立环境并更新本目录 runtime-python.txt；默认安装 CPU PyTorch 用于校验和导出，GPU 通过 DirectML 运行 ONNX，不依赖完整 CUDA 包。不要用作者的旧 Linux conda requirements.txt 安装 Windows 环境。

重新导出：

```powershell
& 'D:/ARLensRuntime/venv/Scripts/python.exe' export_onnx.py
& 'D:/ARLensRuntime/venv/Scripts/python.exe' test_deployment.py
```

prepare_vendor.py 仅在重新从父项目 research/EllSeg 构建 vendor 时需要，不是运行时依赖。导出警告中的 InstanceNorm train=True 表示此层始终使用当前输入统计量，不表示在训练权重；网络已经 eval()。

## 本次验证

- **8 项部署检查通过**：严格权重加载、网络运算与原始代码对齐、坐标往返、非等比椭圆变换、空间 softmax、CPU/DirectML 对照、低对比输入、JSON 可序列化。
- 普通单眼图片 CPU/GPU logits 最大误差约 `4.8e-5`；暗化图的分割差异约 1 个像素 / 76,800 像素。纯空白输入在归一化层有更明显数值差异，两端均作为无效输入隐藏，未把它伪装成高精度眼睛。
- 图片人像：检出双眼并生成输出；单眼模式：生成分割及椭圆。
- 项目内 HEVC 竖屏视频测试前 12 帧：24 次双眼推理，保存为 1080×1920、30 FPS、12 帧，JSON 帧号及记录一一对应。
- USB 相机入口已提供；当前未占用主 App 的摄像头进行真机验收，使用前请关闭其他相机程序。

报告：outputs/gpu-parity.json、outputs/smoke-photo-report.json、outputs/smoke-eye-report.json、outputs/smoke-video-report.json。示例输出仅用于验收，不作为准确率评估数据。

## 文件导航

- ellseg_app.py：选择文件、相机、自动裁眼、可视化及保存。
- ellseg_core.py：预处理、模型加载、分割与椭圆解码、坐标变换。
- vendor/ellseg_model.py：网络定义。
- weights/：原始及 ONNX 权重和校验信息。
- test_deployment.py：部署回归。
- test_assets/eye.png：由 scikit-image 的 NASA astronaut 公共样例裁出的单眼，仅供运行测试。

代码按作者 MIT 许可保留署名；权重和各训练数据集的具体使用条件应分别核对，代码许可不自动代替数据授权。
