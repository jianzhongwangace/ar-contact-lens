# 眼部识别：USB 摄像头和本地图片

流程：FaceMesh 定位双眼 → LightIrisNet 现成权重分割虹膜/瞳孔 → 用眼睑轮廓裁剪可见区域。

## 双击使用

| 入口 | 用途 |
|---|---|
| run_usb_camera.cmd | 打开 Full HD webcam，编号 0，1920×1080、MJPEG |
| run_image.cmd | 打开图片选择框，支持 JPG、PNG、BMP、WebP、TIFF |
| run_video.cmd | 打开离线视频页面，支持 MP4、MOV、MKV、AVI 等可解码格式 |

当前环境：D:\ARLensRuntime\venv；下载缓存：D:\ARLensRuntime\pip-cache。代码和模型仍在桌面项目目录。启动器通过 runtime-python.txt 找到 D 盘 Python，无需手动激活环境。

默认 --device auto 优先使用 ONNX + DirectML，运行于本机 GTX 1060 5GB；初始化失败会输出警告并回退 PyTorch。--device dml 强制 GPU，失败报错；--device cpu 强制 CPU。

## 操作

### EllSeg 已接入试戴（默认启用）

现在摄像头、图片和离线视频入口默认 `--geometry ellseg`：

1. FaceMesh 定位双眼、眼睑和虹膜参考点。
2. LightIrisNet 继续输出可见虹膜、瞳孔概率，保留现有统一瞳孔中心和低延迟跟踪。
3. EllSeg 推理完整虹膜椭圆，并检查尺寸、中心、眼睑范围、与可见虹膜的重合程度。
4. 检查通过时采用完整椭圆的半轴与方向；眼睑可见区域独立裁剪，绝不将眼皮后方补全区域直接画出来。检查不通过时回退 FaceMesh 原几何。
5. 同一套材质、强度、着色直径、稳定滤波和眼皮/瞳孔保护继续生效。

界面底部新增 **G 几何模型** 按钮，或按 G 切换到原方案。右侧显示 `EllSeg` 或 `FaceMesh fallback`；粉色椭圆线表示采用的完整虹膜估计，青色仍为眼睑。切换会清空旧跟踪历史。照片立即重算；离线视频会按所选模型重新预处理。

```powershell
.\run_usb_camera.cmd --geometry ellseg
.\run_image.cmd --geometry ellseg
.\run_video.cmd --geometry ellseg
# 原方案对照
.\run_usb_camera.cmd --geometry facemesh
```

EllSeg 使用 `EllSeg部署/weights/ellseg.onnx` 及该目录的实现，通过隔离的模块命名空间加载，不与主 App 的 vendor 混淆。独立 EllSeg 程序仍可运行。此版本为两模型串联，计算量会增加；首帧测试中 EllSeg 双眼约额外 62 ms，不应假定能保持之前单模型的实时帧率。离线视频在预处理完成后播放缓存，不在播放时再次运行 AI。

视频缓存键现在包含模型模式、关键算法文件/模型清单版本、材质像素和尺寸/强度；EllSeg 与原方案不会复用同一缓存。完成标记写入后才允许命中缓存，避免使用中断的处理结果。左右眼最终图与右侧模型诊断逐帧缓存，显示对应帧的模型来源。模型文件改动需同步其校验清单。

验证：原有 343 帧视频有 686 个逐眼判定，其中 557 个采用 EllSeg，129 个因可见证据、大小或中心不一致回退。该统计不是准确率，不证明 EllSeg 对每个人、每个角度都更好。眼镜/手指遮挡仍需实拍检查，FaceMesh 眼睑并非完整遮挡分割。

新增代码：ellseg_adapter.py（适配和接受门槛），EyePipeline.set_geometry()（切换），LensTracker 中的 geometry_ellipse 分支（完整形状与遮挡分离）。`test_ellseg_integration.py` 验证旋转、闭眼、眼皮外不绘制、回退和缓存隔离。87 项主 App 回归通过。报告：logs/ellseg-integration-video.json。

### 固定比例与双眼效果放大

三个输入模式统一为 **1920×1080（16:9）逻辑画布**。窗口按屏幕可用空间等比例显示，当前常用显示尺寸为 1440×810；窗口大小固定，不再因横屏、竖屏或切换照片而改变比例。图片和视频内容保持自身宽高比，空余区域留黑。

左侧新增左右眼“最终佩戴效果”放大区，中间是完整画面，右侧保留分割诊断。左侧取自与原分辨率保存相同的合成画面，不显示绿色/橙色诊断掩膜；放大不会恢复原视频中不存在的细节。O 原眼对比、M 镜像以及强度/尺寸调节同步影响完整画面和局部效果。眼睛丢失时清空局部区；闭眼或定位不可靠时显示当前原眼而不保留旧花纹。

界面布局及缩放点击坐标在 app_layout.py；最终效果取样与放大在 eye_pipeline.py 的 final_eye_view() 和 render()。离线视频的播放进度区固定在底部，其他模式为其预留空间，确保全程保持 16:9。

### 放大着色直径（覆盖可见眼白）

底部“直径 − / +”和 90%–140% 滑杆控制花纹外圈的真实显示范围；100% 约对应当前检测到的虹膜，130% 表示显示直径放大约 30%。提供“自然 100%”与“放大 130%”快捷按钮，设置随当前素材配置保存。新素材默认使用 130%。

大于 100% 时，着色范围可延伸到原虹膜外的可见眼白。渲染将稳定的着色区域向外放大，单独保留当前眼睑和真实瞳孔遮罩；中央透明区采用环形映射，避免跟着外圈一起放大。白色眼白不参与原虹膜的光照基准，避免外环被过度提亮。

这是相对视觉尺寸，不是眼睛或实际镜片的毫米测量。闭眼和跟踪失效仍隐藏效果；眼镜/睫毛等遮挡效果仍取决于现有识别质量。摄像头、照片、离线视频与原分辨率保存共用同一个放大渲染器。实现位于 lens_renderer.py 的 compose_lens()，范围在 lens_settings.py，验证在 test_lens_enlargement.py。

### 离线视频（新增）

摄像头和图片页面底部新增 **导入离线视频** 按钮，也可按 V。选择本机视频后打开离线处理页面，父页面暂停 AI 处理，关闭视频页后返回。无需上传服务器，也无需开启摄像头；可直接双击 **离线视频试戴.lnk** 或 run_video.cmd。

- 支持 1080×1920 竖屏，也接受其他尺寸，不强行拉伸到固定分辨率。
- 读取 MP4/MOV 的旋转信息。例如项目里的微信视频实际编码为 1920×1080，显示旋转后为 1080×1920。
- 先完整顺序处理每一帧并生成 D:\ARLensRuntime\video-cache 下的本地结果视频，再按源视频 FPS 播放结果；处理阶段会等待，播放阶段不会受 AI 推理耗时影响。
- 显示当前时间、总时长、尺寸和处理状态；点击进度条定位，重置人脸/虹膜跟踪后继续。
- 空格或按钮播放/暂停，R 从头播放，O 原眼/试戴对比，S 保存当前帧，Q/Esc 返回；播放结束后保留最后一帧，可以重播。
- 美瞳、强度、相对尺寸与父页面共用，暂停时仍可调节并立即查看结果。
- 当前为无声效果预览，不播放原视频音轨；结果视频已经在本地缓存中，S 保存当前播放帧图片。

```powershell
.\run_video.cmd
.\run_video.cmd --video "D:\视频\人像.mp4"
# 不显示窗口，顺序处理全片并记录检测/性能数据（不保存视频帧）
.\run_video.cmd --video "微信视频2026-09-21_015131_959.mp4" --headless --report logs/offline-video.json
```

本次对项目内 11.43 秒、30 FPS 的 HEVC 微信视频完成全片预处理：343/343 帧写入结果视频，结果文件保持 1080×1920、30 FPS、343 帧；处理约需 28 秒，之后按约 11.43 秒正常速度播放。这里的计数不代表边缘分割准确率。报告在 logs/preprocess-then-play-validation.json。

对应代码：video_app.py 的 VideoReader 负责解码及旋转，process_video_to_file() 负责完整逐帧处理和本地编码，ProcessedVideoSession 负责按源 FPS 播放缓存结果，play_video 负责界面；test_video.py 验证原始读取状态，处理/缓存验证见 logs/preprocess-then-play-validation.json。

摄像头和图片窗口底部均有 **导入美瞳图片** 按钮：点击后选择本机 PNG、JPG、WebP 等素材，显示文件名、尺寸及缩略图；透明 PNG 的透明通道会保留。U 键也可打开选择框，移除按钮清空当前素材。取消或选择损坏文件会保留之前的素材。两种模式共用最近一次成功导入的本机文件路径。

导入后会自动生成透明环形材质并应用试戴；原图不会被覆盖。当前三个示例有按内容哈希匹配的边界和反光处理参数，其他亮底正面图使用自动检测，必要时点击“校准素材”。校准窗口可调整中心、横纵半径、方向、中央透明区和反光修补开关，在深浅背景下检查后应用。

底部有强度与相对尺寸按钮；默认强度 70%、尺寸 100%，着色直径范围 90%–140% 是视觉比例而非毫米测量。U 导入，C 校准，+/- 调强度，O 原眼对比。处理耗时集中在导入时，之后复用缓存。视频按原图坐标平滑虹膜几何，闭眼/丢失跟踪时清除该眼效果。

三款处理后的素材位于“处理后美瞳”文件夹，各有 lens_rgba.png、material.json、校准图和深浅背景预览。商品图透明度、反光遮住的图案无法从单张照片准确恢复，本程序用保守透明度估计及其他角度纹理做近似修补；第三款原图反光范围大，仍可能有局部色阶/纹理接缝，可关闭修补或替换成品牌原始花纹。不能将这些近似结果宣称为实物显色复现。

图片模式 S 保存原分辨率试戴结果；摄像头 S 同时保存诊断预览与原始方向的试戴 result.png。保存使用与预览相同的材质、强度、尺寸及当前帧的定位。O 只切换预览，S 始终保存当前款式的试戴结果。

```powershell
# 重新批处理三张示例
& 'D:/ARLensRuntime/venv/Scripts/python.exe' process_examples.py
# 指定人像和美瞳素材，后台处理并保存
.\run_image.cmd --image "D:\照片\人像.jpg" --lens "下载美瞳超清图片.png" --headless --save
# 素材与渲染专项检查
& 'D:/ARLensRuntime/venv/Scripts/python.exe' test_lens_rendering.py
```

- Q / Esc：退出并释放相机。
- O：切换原图/识别叠加。
- M：切换预览镜像。图片默认不镜像。
- S：主动保存到本机 output 目录。
- L：仅图片模式，选择另一张图片。

绿色为虹膜，橙色为瞳孔，青色为眼睑。右侧依次显示眼部输入、原始分割和裁剪后的结果。图片模式保存原分辨率 result.png、诊断预览 diagnostic.png 和两眼局部掩膜；result.png 始终按照片本身方向保存标注，不受预览镜像和 O 键影响。

默认不录制、不保存摄像头画面、不上传照片。正常推理不需要联网。图片支持中文路径、手机 EXIF 方向校正，换图失败会保留上一张结果。

导入素材后，大画面显示花纹试戴，右侧仍显示识别诊断。未导入素材时，大画面显示彩色分割标注。素材显色需要与实物参考校准。

## 命令行

在项目目录 PowerShell 运行：

```powershell
.\run_usb_camera.cmd
.\run_usb_camera.cmd --device cpu
.\run_camera.cmd --list-cameras
.\run_camera.cmd --camera 0 --backend msmf
.\run_image.cmd --image "D:\照片\人像.jpg"
.\run_image.cmd --image "D:\照片\人像.jpg" --headless --save
.\run_usb_camera.cmd --seconds 30 --report logs/camera-test.json
```

run_camera.cmd 默认尝试编号 0–3、1280×720；run_usb_camera.cmd 使用本机已验证的编号 0、1080p。摄像头编号可能随设备变化。

## 已完成的优化

1. 同一权重导出为 FP32 ONNX，用 DirectML 调用 GTX 1060，无需完整 CUDA 包。
2. 按 DXGI 枚举选择 NVIDIA 硬件，避免误选软件渲染设备。
3. 双眼批量推理；单眼输入复制后丢弃多余输出，保持固定 batch=2。
4. 预览先缩放，仅在眼部小区域映射掩膜。推理和照片保存保留原始坐标。
5. 采集线程只保留最新帧；分割结果显示在对应原帧上，不用旧坐标覆盖新画面。
6. 最后设置 MJPEG 格式。本机相机的 YUY2 在 1080p 仅支持 5 FPS；更改分辨率可能重置格式。日志显示实际 FOURCC 和实收帧率。

固定公开测试图、1080p 画面、每眼 256×160，3 次预热后采样 20 次的耗时中位数：

| 环节 | 优化前 CPU | 优化后 DirectML |
|---|---:|---:|
| 双眼分割和局部后处理 | 183.8 ms | 16.8 ms |
| 全部识别处理 | 195.0 ms | 26.9 ms |
| 预览合成 | 66.4 ms | 6.3 ms |
| 合计 | 260.5 ms | 33.8 ms |

约 7.7 倍计算加速。这是固定输入基准，不等于真实相机在所有场景下都达到 30 FPS。报告见 logs/optimization-before.json、logs/optimization-after.json；实拍见 logs/camera-optimized-mjpg.json。无眼部的帧不执行分割，不能用这些帧计算虹膜推理速度。

## 模型来源与精度验证

LightIrisNet：https://github.com/naveengv7/LightIrisNet

- 固定提交：034c807796a881173519c9984cae341dbb2e632d。
- 权重：research/LightIrisNet/model/best_mobilenetv3.pt。
- SHA-256：71119c33a8a95a83ff6315260cda62eed5a26ad97c7b3934630d7579a32e27f1。
- MobileNetV3-Large，完整源结构 4,229,858 参数。
- Python 严格加载权重；ONNX 核对源权重和导出文件 SHA-256。
- 只导出虹膜、瞳孔主分支；省去未使用辅助输出，没有训练、量化或缩小默认眼部输入。
- vendor 保留模型结构、来源记录和 MIT 许可证。上游训练/测试入口有缩进问题，本项目仅提取模型定义。

转换测试包括原图、明暗变化、翻转、单眼、空白共 6 组输入，FP32 概率最大差约 0.0000042，在这些样本上 0.5 阈值掩膜一致。性能剖析确认计算节点由 DmlExecutionProvider 执行。见 logs/acceleration-validation.json；这是数值一致性验证，不是标注准确率评估。

FaceMesh 使用 MediaPipe 0.10.21 legacy FaceMesh（refine_landmarks=True），478 个关键点。它是 FaceMesh with Attention + 虹膜点，不是 Tasks FaceMesh-V2 模型包。

## 限制

### 瞳孔对齐与低延迟跟随（当前版本）

- **外圈中心与透明孔中心分开**：FaceMesh 虹膜几何决定外圈；通过 LightIrisNet 瞳孔概率图提取可信轮廓并拟合椭圆中心，驱动中央透明孔。花纹内外环采用连续映射；放大改变外圈，真实瞳孔保护使用同一个内部锚点。
- **位置滤波独立于尺寸与旋转**：中心改为按当前帧时间戳更新的 alpha-beta 位置/速度观测器，取消中心的无条件三帧中值和硬死区。它估计当前帧状态，不把未来位置画到当前帧。单次大异常暂缓，后续连续运动或虹膜/瞳孔共同位移确认后快速跟随。尺寸、角度和左侧放大取景框继续强防抖。
- **同帧同坐标**：瞳孔观测先转换为原始图像坐标再滤波，避免“滤波虹膜中心 + 另一次滤波瞳孔偏移”叠加延迟。透明孔与瞳孔保护共享最终中心；当前眼睑仍直接裁剪。摄像头传入采集线程读取时间戳（不是传感器曝光时间）；离线视频使用媒体时间戳。
- **质量判定与回退**：检查瞳孔中心、面积、轮廓覆盖、椭圆残差和眼睑接触程度。反光造成内部孔洞时使用外轮廓拟合；瞳孔不可靠时可短时保持相对偏移，过期退回虹膜中心。闭眼、丢脸、换静态照片和重播均不复用失效状态。内孔/外圈不满足嵌套约束时取消该次渲染，避免纹理翻折。

按 **D** 开关右侧对齐诊断：青色十字为原始虹膜中心，橙色圆为当前模型瞳孔观测，绿色十字为最终透明孔中心。error 是两者在源图像上的像素距离；q 是由轮廓拟合得到的质量分数，不是医学测量或真实准确率。关闭诊断不影响渲染；诊断只画在右侧，不写入保存的试戴图。

当前已切换为**统一渲染中心**：可靠的瞳孔位置作为外圈和中央透明孔共同使用的中心；虹膜只提供半轴与朝向。内外圈不再用偏心环形 UV 连接，因此不会因两个圆心不同被拉成一边粗、一边细。pupil_anchor、lens_pose[:2] 在渲染前保持一致；放大只改变外圈尺寸，瞳孔保护不放大。

验证：17 项对齐/低延迟检查覆盖小位移收敛、运动后停止、快速连续运动、单帧异常、静止噪声、偏心瞳孔、反光内部孔洞、图片隔离、同心 UV、放大后瞳孔保护和闭眼清理。原有视频的模型瞳孔参考对齐检查以当前实现重新生成；报告中的误差是模型输出接线检查，不是人工真值，不能据此宣称真实瞳孔准确率。统一中心视频检查见 logs/unified-center-video-validation.json。

代码：center_tracking.py（中心观测器、同心环形 UV），lens_renderer.py（瞳孔拟合、状态融合、统一中心渲染），test_pupil_alignment.py 与 test_unified_center.py（专项测试），diagnose_pupil_alignment.py（同输入对照）。

### 静止防抖（上一版记录）

上一版摄像头和离线视频使用的自适应稳定：三次观测的中值剔除单帧异常；位置、尺寸、旋转分别设置小幅波动区间；静止时抑制噪声，持续移动时提高跟随速度。算法根据帧时间差工作，图片模式不使用历史平滑。

虹膜可见边界在统一的虹膜坐标中稳定，然后与瞳孔保护一起映射到平滑后的镜片位置，避免“花纹已经平滑、遮罩还在跳”。明显的新遮挡和当前眼睑仍立即裁剪，闭眼/丢失人脸清空历史。左侧局部放大的取景框单独防抖，不改变送给 AI 的眼部裁剪图。

固定真实视频第 30 帧重复运行 90 次，去掉前 20 次预热：右眼中心标准差从 0.239 px 降至 0.058 px，左眼从 0.135 px 降至 0.015 px，约降低 76% / 89%。这是冻结画面测试，不是实际摄像头静止程度或全部场景的保证。见 logs/stability-comparison.json。移动响应测试确认 8 px 的持续位移在两帧后误差小于 0.8 px；快速移动仍可能有短暂的滤波延迟。

验证脚本：test_lens_stability.py；对照脚本：diagnose_stability.py。防抖实现位于 lens_renderer.py 的 stabilize_pose()、stabilize_support() 和 EyeViewTracker。

### 轻微转动闪断与花纹缺口修正

跟踪现在将“完整镜片几何”与“可靠的可见边界”分开。原始二值虹膜图不再直接裁切花纹；使用虹膜概率图中靠近当前几何的有效区域生成连续边界，修补内部孔洞与裂缝，同时限制外侧覆盖，不把完整椭圆无条件画到眼白或皮肤上。

短暂分割不足时，可在几何连续的前提下跟随当前 FaceMesh 关键点过渡最多 220 ms，并在后段淡出；不是固定上一帧坐标。闭眼、整脸丢失、大幅位置跳变立即隐藏。静态图片不借用上一张的历史。

瞳孔保护使用独立瞳孔概率图，检查中心、面积和形状；过大或偏离的预测不会整块裁掉花纹。取消原先整块膨胀，改为经过检查的椭圆和窄过渡。瞳孔预测短时异常只允许复用有限时长的可靠参数。

原有 343 帧离线视频同输入对照：686 个逐眼样本，修正前显示 621 个，修正后 619 个；显示/隐藏状态切换从 36 次降到 10 次。计数只表示状态连续性，不代表边缘标注准确率。见 logs/continuity-comparison.json 与部分帧对比 logs/continuity-before-after.png。

眼镜反光、镜框遮挡和较大侧视仍可能影响两个模型。不能只放宽阈值或延长保持时间来强行维持效果；持续无可靠信息时仍会取消对应眼睛的花纹。自然尺寸保持原有边界约束；放大模式将着色边界向外扩展，但眼睑和瞳孔保护始终保持原位。

仅支持单人；远距离、模糊、侧视和眼镜反光会影响效果。眼睑开合是关键点比例判断，--close-threshold 0.08 可调整。眼睑多边形不是独立睫毛/手指分割。非零掩膜不能证明位置正确，需要观察贴合程度。

默认 ONNX 每眼 256×160。其他尺寸需 export_onnx.py 生成对应模型，否则 auto 使用 PyTorch。GPU 首帧可能较慢。

## 维护、重装和测试

### 转头时的鲁棒跟踪

实时跟踪在每个眼睛上同时检查 FaceMesh 的眼部关键点、LightIrisNet 的瞳孔概率和 EllSeg 的当前眼部图像。LightIrisNet 短暂漏分割时，若 EllSeg 的当前分割仍与独立瞳孔和眼睑一致，可以继续生成镜片；处理间隔较长时还要通过当前眼部纹理的前后向光流核对。光流只在已有可靠轨迹的 450ms 内使用，并且不会刷新可靠时间，因此持续没有新证据会自动隐藏。闭眼、丢脸、眼睛可见区域过小或位置突变会立即清除。

这套逻辑减少转头过程中的短暂闪断，但不能从不存在的像素恢复被眼皮、镜框或强反光完全遮挡的虹膜。侧视时如果完整椭圆与当前可见证据不一致，程序会隐藏该眼睛，避免把花纹画到眼白或皮肤上。`logs/head-turn-comparison.json` 是同一输入的连续性统计，不是人工标注准确率。

```powershell
# 默认安装到 D:/ARLensRuntime，优先 DirectML
powershell -ExecutionPolicy Bypass -File .\setup_camera.ps1

# 自定义运行环境位置；-Cuda 可另外安装 CUDA PyTorch
powershell -ExecutionPolicy Bypass -File .\setup_camera.ps1 -RuntimeRoot D:/ARLensRuntime

& 'D:/ARLensRuntime/venv/Scripts/python.exe' test_pipeline.py
& 'D:/ARLensRuntime/venv/Scripts/python.exe' validate_acceleration.py
& 'D:/ARLensRuntime/venv/Scripts/python.exe' benchmark_pipeline.py --device dml --report logs/benchmark-new.json
& 'D:/ARLensRuntime/venv/Scripts/python.exe' probe_camera.py
```

当前 D 盘环境用 system-site-packages 复用本机已有 CPU PyTorch；固定版本的 MediaPipe、NumPy、OpenCV、DirectML 安装在 D 盘。全局 Python 仍在 C 盘。新电脑安装脚本创建独立环境，旧项目 .venv 保留作为回退。

旧版 MediaPipe 从中文路径加载模型可能失败：兼容模块会在必要时复制到本机 ASCII 缓存路径。当前 D 盘环境不需要该处理。

核心文件：camera_app.py（摄像头）、image_app.py（图片）、eye_pipeline.py（识别/遮罩）、onnx_predictor.py（GPU）、gpu_info.py（硬件选择）、export_onnx.py（模型导出）。
