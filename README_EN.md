# AR Contact-Lens Try-On

English | [中文](README.md)

**© Original work by Jianzhong Wang**

A desktop AR contact-lens try-on prototype for webcam / local image / offline video. Pipeline: FaceMesh eye localization → iris/pupil segmentation & tracking → lens pattern rendering with eyelid occlusion and pupil protection. Optional EllSeg full-iris ellipse geometry.

> Demo / engineering prototype only. Not for medical fit, prescription, or eye-health claims.

## Features

- **Webcam try-on**: Full HD live preview, style switching, intensity/diameter controls, original-eye compare
- **Image try-on**: Local JPG/PNG for quick comparisons
- **Offline video**: Preprocess the whole clip, then play cached results at source FPS (local only)
- **Material pipeline**: Import product images, calibrate rings, build renderable transparent patterns
- **Geometry modes**: EllSeg by default (full ellipse + visible-region crop); FaceMesh fallback

## Quick start (Windows)

Python 3.11 recommended. First-time setup:

```powershell
.\setup_camera.ps1
# optional: .\setup_camera.ps1 -Cuda
```

Launchers:

| Entry | Purpose |
|---|---|
| `run_usb_camera.cmd` / `run_camera.cmd` | Live USB webcam try-on |
| `run_image.cmd` | Local image try-on |
| `run_video.cmd` | Offline video try-on |

Examples:

```powershell
.\run_usb_camera.cmd --geometry ellseg
.\run_image.cmd --geometry facemesh
.\run_video.cmd --video "your.mp4"
```

See `requirements-camera.txt`. On Windows, ONNX + DirectML is preferred; CPU / PyTorch fallback is available. Detailed ops notes: [README_USB_CAMERA.md](README_USB_CAMERA.md).

## Layout

```text
├── camera_app.py / image_app.py / video_app.py   # input modes
├── eye_pipeline.py / lens_renderer.py            # tracking & compose
├── ellseg_adapter.py                             # EllSeg bridge
├── models/                                       # LightIrisNet ONNX (included)
├── 处理后美瞳/                                   # three sample materials
├── test_assets/                                  # UI / result samples
├── EllSeg部署/                                   # standalone EllSeg deploy + upstream refs
└── *.md                                          # product / render / tracking docs
```

## Sample materials

`处理后美瞳/` ships three importable samples (gray-brown radial, pink-brown petal, deep-brown gradient). The full materials library is large and omitted; use `lens_import.py` to add your own.

## Models & third-party

- **LightIrisNet ONNX**: `models/lightiris-2x160x256.onnx` for visible iris/pupil maps.
- **EllSeg**: under `EllSeg部署/`, from upstream [RSKothari/EllSeg](https://github.com/RSKothari/EllSeg) (MIT). Upstream README, License, and `source.json` are kept. App-level try-on logic here is original work by Jianzhong Wang — **do not treat upstream assets as original**.
- MediaPipe FaceMesh, OpenCV, ONNX Runtime, etc. are dependencies.

## Not included in this repo

Omitted to keep size under limits:

- `.venv/`, `research/` (full upstream clones), `materials/` (bulk calibration packs)
- `logs/`, `output/`, WeChat sample video and other runtime artifacts
- Oversized temporary ultra-HD stock images

## Docs

- [AR美瞳试戴产品与技术方案.md](AR美瞳试戴产品与技术方案.md)
- [AR美瞳APP技术开发方案.md](AR美瞳APP技术开发方案.md)
- [美瞳花纹渲染实施方案.md](美瞳花纹渲染实施方案.md)
- [瞳孔对齐与低延迟跟踪修正方案.md](瞳孔对齐与低延迟跟踪修正方案.md)
- [README_USB_CAMERA.md](README_USB_CAMERA.md)

---

© Original work by Jianzhong Wang
