# AR 美瞳试戴

[English](README_EN.md) | 中文

**© 王建中原创**

桌面端美瞳 AR 试戴原型：前置/USB 摄像头、本地图片与离线视频三种输入；FaceMesh 定位双眼 → 虹膜/瞳孔分割与跟踪 → 美瞳花纹渲染（含眼睑遮挡与瞳孔保护）。可选 EllSeg 完整虹膜椭圆几何。

> 本仓库为效果验证与工程原型，不提供医学适配、处方或眼健康判断。

## 功能概览

- **摄像头试戴**：Full HD webcam 实时预览，款式切换、强度/直径调节、原眼对比
- **图片试戴**：本地 JPG/PNG 等，适合快速对比款式
- **离线视频试戴**：整段预处理后按源 FPS 播放缓存结果（本机处理，不上传）
- **材质管线**：导入商品图、校准中心/内外圈、生成可渲染透明花纹
- **几何模型**：默认 EllSeg（完整椭圆估计）+ 可见区裁剪；可回退 FaceMesh

## 快速开始（Windows）

建议 Python 3.11。首次配置：

```powershell
.\setup_camera.ps1
# 可选：.\setup_camera.ps1 -Cuda
```

启动入口：

| 入口 | 用途 |
|---|---|
| `run_usb_camera.cmd` / `run_camera.cmd` | USB 摄像头实时试戴 |
| `run_image.cmd` | 本地图片试戴 |
| `run_video.cmd` | 离线视频试戴 |

常用参数示例：

```powershell
.\run_usb_camera.cmd --geometry ellseg
.\run_image.cmd --geometry facemesh
.\run_video.cmd --video "your.mp4"
```

依赖见 `requirements-camera.txt`。Windows 默认优先 ONNX + DirectML；失败可回退 CPU / PyTorch。更细的操作说明见 [README_USB_CAMERA.md](README_USB_CAMERA.md)。

## 目录结构

```text
├── camera_app.py / image_app.py / video_app.py   # 三种输入入口
├── eye_pipeline.py / lens_renderer.py            # 跟踪与花纹合成
├── ellseg_adapter.py                             # EllSeg 接入与门槛
├── models/                                       # LightIrisNet ONNX（本仓已带）
├── 处理后美瞳/                                   # 三款示例材质
├── test_assets/                                  # 界面/效果样例图
├── EllSeg部署/                                   # 独立 EllSeg 部署（含上游引用）
└── *.md                                          # 产品/渲染/对齐方案文档
```

## 示例材质

`处理后美瞳/` 含三款可直接导入的示例：灰棕放射纹、粉棕花瓣纹、深棕渐变纹（`lens_rgba.png` + `material.json`）。完整素材库体积较大，未纳入本仓；可用 `lens_import.py` 自行导入。

## 模型与第三方说明

- **LightIrisNet ONNX**：`models/lightiris-2x160x256.onnx`，用于可见虹膜/瞳孔概率。
- **EllSeg**：独立目录 `EllSeg部署/`，权重与推理封装来自上游 [RSKothari/EllSeg](https://github.com/RSKothari/EllSeg)（MIT）。上游 README、License 与 `source.json` 已保留；本仓应用层适配与试戴产品逻辑为王建中原创，**请勿将上游内容视为本人原创**。
- MediaPipe FaceMesh、OpenCV、ONNX Runtime 等为依赖库。

## 未纳入本仓的内容

为控制体积与避免超限，以下未推送（本地仍可保留）：

- `.venv/`、`research/`（上游完整克隆）、`materials/`（大批量校准素材）
- `logs/`、`output/`、微信示例视频等运行产物
- 体积过大的临时超清素材图

## 相关文档

- [AR美瞳试戴产品与技术方案.md](AR美瞳试戴产品与技术方案.md)
- [AR美瞳APP技术开发方案.md](AR美瞳APP技术开发方案.md)
- [美瞳花纹渲染实施方案.md](美瞳花纹渲染实施方案.md)
- [瞳孔对齐与低延迟跟踪修正方案.md](瞳孔对齐与低延迟跟踪修正方案.md)
- [README_USB_CAMERA.md](README_USB_CAMERA.md)

---

© 王建中原创
