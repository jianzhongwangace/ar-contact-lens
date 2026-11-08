# AR 美瞳 App 技术开发方案

日期：2026-09-20  
范围：iOS + Android 独立 App；首版单人前置摄像头实时试戴、款式切换、拍照与同图对比。本文是实施设计，尚未实现代码或进行性能实测。

## 1. 技术决策

首版采用原生双端：iOS 使用 Swift、AVFoundation、Metal；Android 使用 Kotlin、CameraX、OpenGL ES 3.0。以 MediaPipe Face Landmarker 作为人脸与眼部关键点基线，配合自研局部材质渲染。后端使用 FastAPI、PostgreSQL 与对象存储/CDN，只服务商品、材质和配置，不参与实时视频处理。

选择原生的原因是首版大部分功能围绕摄像头、GPU、视频缓冲与设备生命周期展开，原生便于控制延迟、内存和热负载。代价是维护两套界面与平台接入代码；通过统一数据协议、素材规范和算法测试向量降低差异。

不以 Flutter、Unity 或 WebView 承载首版核心视频链路。若团队已有 Flutter 产品，可保留其业务界面，并接入原生试戴页面；跨平台桥接只传控制命令与低频状态，不传逐帧视频。Flutter 官方文档明确指出 Platform Views 存在合成性能取舍，不能假定嵌入原生视图没有成本。

暂定设备覆盖：iOS 15+、Android 10+ 且支持 OpenGL ES 3.0。这是产品支持基线，不是各 SDK 的最低版本结论；立项阶段核对所锁定 SDK 的部署要求和目标用户机型后确定。

## 2. 总体架构

```text
客户端
├─ Presentation：试戴、款式列表、对比、相册保存、权限提示
├─ TryOnSession：状态机、款式切换、质量控制、生命周期
├─ CameraSource：采集、时间戳、方向、曝光状态、缓冲管理
├─ FaceTracker：关键点、眼睑轮廓、人脸有效性
├─ EyeEstimator：虹膜几何估计、局部分割扩展、眼球朝向估计
├─ TemporalFilter：滤波、失踪重置、结果过期处理
├─ LensRenderer：GPU 材质变形、遮挡、颜色融合、最终合成
├─ CaptureService：冻结同帧结果、离屏渲染、对比、导出
├─ AssetRepository：款式数据、下载校验、本机缓存、版本回滚
└─ Diagnostics：耗时、掉帧、错误码；不记录人脸影像

服务端
├─ 商品/款式只读 API
├─ 管理后台与受保护的发布 API
├─ PostgreSQL：SKU、材质版本、发布记录
└─ 对象存储 + CDN：材质、商品图、模型与配置资源
```

App 内置一套可用模型和少量示例材质，让第一次启动和弱网场景也能验证基础功能；线上材质下载成功后原子替换缓存。正式商用前确认模型与示例素材的使用许可。

## 3. 各平台实现

| 层 | iOS | Android |
|---|---|---|
| 界面 | SwiftUI，试戴渲染视图使用 MTKView 封装 | Jetpack Compose，试戴视图封装原生 GL Surface |
| 采集 | AVCaptureSession + AVCaptureVideoDataOutput | CameraX Preview + ImageAnalysis |
| 数据 | CMSampleBuffer / CVPixelBuffer | Preview SurfaceTexture；分析输入来自 ImageProxy |
| 关键点 | MediaPipeTasksVision，LIVE_STREAM | MediaPipe Tasks Vision，LIVE_STREAM |
| GPU | Metal、CVMetalTextureCache | OpenGL ES 3.0，外部相机纹理及 FBO |
| 导出 | 离屏 Metal 渲染、Photos 写入接口 | FBO 导出、MediaStore 写入 |
| 本机缓存 | 文件缓存 + SQLite/轻量索引 | 文件缓存 + Room/轻量索引 |
| 诊断 | Instruments、Metal 调试、OS 日志 | Perfetto、Android Studio Profiler、GPU 工具 |

Android 初始验证允许有限的 YUV/RGB 转换以兼容推理接口，并对拷贝耗时单独计时。不能把“预览通过 GPU”宣称为全链路零拷贝。是否启用 GPU delegate 由实测决定：它可能与渲染竞争 GPU。

iOS 可直接通过 MPImage 的像素缓冲或采样缓冲入口构建推理输入，但仍需验证具体路径是否内部转换。关闭不必要的采集输出。

## 4. 每帧处理和线程模型

### 4.1 数据流

```text
相机采集 t0
  ├─ GPU 预览帧 ────────────────────────┐
  └─ 推理输入 → 关键点 → 眼部几何 → 滤波 ─┤
                                         ↓
                              按时间戳关联与有效性检查
                                         ↓
                              GPU 美瞳合成 → 屏幕
```

线程职责：

- UI 主线程：按钮、页面、授权与少量状态，不运行推理和像素转换。
- 相机回调队列：保留必要输入、记录时间戳、快速释放回调。
- 推理工作队列：同一跟踪实例的创建/调用/释放遵循其线程及 delegate 要求；限制在途帧数。
- 渲染队列：GPU 纹理更新、眼部网格和着色器参数提交。
- 文件队列：材质下载、校验、照片编码及保存。

只保留最新待处理帧，禁止无界队列。Android 使用 ImageAnalysis 的 KEEP_ONLY_LATEST 策略并正确关闭 ImageProxy；iOS 丢弃迟到视频帧。推理输入若独立拷贝，可在拷贝后释放相机缓冲；若异步引用原始缓冲，必须保证引用生命周期覆盖推理完成。具体以 SDK 的输入所有权约定为准。

### 4.2 时间同步

每帧携带单调递增 timestamp、frameId、方向和裁剪矩阵。异步推理回调按原始输入时间戳索引；帧、结果和缓冲均有明确所有权。

Android 的 Preview 与 ImageAnalysis 分辨率、裁剪和坐标可能不同，应绑定共同 ViewPort 并显式计算变换。两个输出时间戳不可直接假定属于同一时间基，需使用相机元数据/平台时间源验证映射；SDK 需要的毫秒时间戳与相机时间戳分别保存并建立映射。

提供两种策略并实测选定默认值：

1. 严格匹配：显示与关键点相同的输入帧，贴合更可靠，但会增加画面延迟，需要少量有界帧缓冲。
2. 低延迟预览：显示最新相机帧，用短时间外推估计眼部位置；结果过旧或运动过快时取消外推并淡出效果。

首版优先实现严格匹配的正确性，再根据真机延迟决定是否增加低延迟模式。对结果设置可配置过期阈值，例如从 80–100 ms 起调优；这些数值是待测初值。严禁无限复用最后一次关键点。

Android 严格匹配模式直接将分析输入的同一帧上传为渲染纹理，先接受并测量必要转换成本；独立 Preview SurfaceTexture 路径主要用于后续低延迟预览，不能仅凭相近回调时间就当作与分析输入完全相同的一帧。iOS 严格匹配模式保留对应 CVPixelBuffer，并通过有界缓冲与 GPU 完成回调管理释放。

## 5. 眼部识别与跟踪算法

### 5.1 基线模型

官方示例的 Face Landmarker 输出每张脸 478 个关键点，可选表情系数与人脸变换矩阵，支持异步摄像头输入。首版 numFaces = 1，关闭不用的输出。单脸配置不保证检测出画面中所有其他人；多人提示若需要可靠实现，增加独立多脸检测，或以目标丢失/跳变检测避免效果换人。

建立适配层提取左右眼眼睑轮廓、虹膜中心/周边点。关键点索引使用固定模型版本对应的官方拓扑，不把未经核对的常量散落在业务代码中。

模型没有提供的逐眼置信度不能伪造为模型输出；可根据眼部尺寸、形状一致性、可见区域、时间变化和人脸级结果构建产品自身的质量评分。

### 5.2 几何估计

每只眼输出：

```text
EyeState
  frameId, sourceTimestampNs
  centerPx
  irisAxesPx, irisAngleRad
  eyelidPolygonPx
  openness
  qualityScore              # 产品派生质量评分
  visible, stale
  irisMask? / pupilMask? / occlusionMask?
```

少量虹膜点只能支持近似几何，侧看、睫毛遮挡和深色瞳孔场景可能不可靠。首版使用有限角度范围内的椭圆/局部网格；脸部朝向矩阵不能替代眼球自身的注视方向。

虹膜可见区域被遮挡时，不用被截断的轮廓直接缩小整片镜片，应结合稳定历史状态和几何约束，或降低质量评分。

### 5.3 稳定与失踪恢复

- 对中心、尺度和旋转做 One Euro Filter 等自适应滤波候选，分别标定平滑度。
- 旋转角在滤波前做周期展开，避免跨边界跳转。
- 静止时减抖，快速移动时降低滤波滞后；重新识别、换脸、方向变化后重置历史状态。
- 两只眼独立处理闭眼状态；使用眼睑开合比例与时间滞回，不靠单阈值频繁闪烁。
- 检测丢失或结果过期：短时间淡出，停止投射材质；恢复稳定后淡入。

### 5.4 何时引入眼部分割

第一阶段先测试几何眼睑遮罩。若出现明显的睫毛覆盖、瞳孔不自然或虹膜边缘不准，再引入局部眼部分割模型。

输入可从每只眼 128×128 或 192×192 裁剪起做实验，输出虹膜、瞳孔和遮挡区域。尺寸是实验候选，不是已验证配置。模型可用 LiteRT 等端侧运行时，需分别验证 iOS/Android 算子与 delegate；不为一个自定义模型无理由叠加多个推理框架。

如需训练，要另立数据授权、标注、划分与评估任务，按身份划分训练集和测试集。自研分割不包含在下文基础工期中。也可用商业 SDK 替代这部分，但必须先做统一条件下的效果与性能对比。

## 6. 坐标与 GPU 渲染

### 6.1 坐标统一

定义四个明确坐标系：相机原始像素、推理输入、渲染纹理、屏幕预览。全部通过显式矩阵连接，不在不同模块各自翻转坐标。

```text
p_screen = T_fit × T_mirror × T_orientation × T_modelToCamera × p_model
```

这是齐次坐标变换的约定示意；实现中按实际裁剪/缩放构建矩阵并验证乘法顺序。明确纹理 UV 原点、左右眼命名（用户自身左右）和前置镜像规则。

推理使用非镜像、方向正确的图像，镜像在预览合成阶段处理。iOS 官方文档指出 Face Landmarker 不支持 mirrored UIImage orientation，不能直接依赖该枚举完成前置镜像。

旋转、横竖屏、刘海屏裁剪、预览宽高比和导出尺寸都用同一套坐标契约测试。

### 6.2 渲染通道

1. 相机颜色转换：根据输入元数据处理 YUV 范围、色彩矩阵和方向；首版统一输出 SDR sRGB，避免混用 HDR 数据。
2. 眼部掩膜：生成眼睑可见区域，在需要时叠加虹膜/瞳孔/遮挡分割。
3. 材质映射：依据眼部几何把镜片 UV 投影到局部网格；限制角度过大时的畸变。
4. 光照与颜色融合：结合原画面亮度、材质颜色及透明度，尽量保留自然高光。
5. 合成：输出到屏幕目标；拍照时使用相同逻辑输出到离屏目标。

线性颜色空间中的基础合成关系：

```text
A = textureAlpha × visibleEyeMask × occlusionVisibility × trackingFade
C_out = C_lensAdjusted × A + C_camera × (1 − A)
```

该式只描述基础透明合成，不是最终真实感算法。C_lensAdjusted 还需要依据真实商品参考标定纹理强度、明暗响应和颜色。材质若使用预乘 alpha，进入混合前统一约定，不能重复乘 alpha。

镜片中央透明区由商品材质决定；用户瞳孔边界是另一概念，不能将两者混为一谈。需保留真实瞳孔时，可以引入经过验证的分割约束，但不能用人为画上的固定黑圆替代真实瞳孔。

### 6.3 尺寸标定

同时存储镜片总直径和着色直径，视觉参数使用独立的 calibratedScale。普通单目自拍没有可靠毫米标尺，不能从 SKU 的毫米参数直接推算用户眼睛上的精确尺寸。

首版基于相对虹膜尺寸和真实佩戴参考标定；放大环可能覆盖部分可见巩膜，不能简单将所有效果硬裁剪在原生虹膜内部。是否扩展、扩展多少要按真实款式校准，并始终受眼睑与遮挡区域约束。

## 7. 材质包与发布协议

材质以完整、可版本化的包交付，而不是由客户端临时为商品照片抠图。示例结构：

```text
lens-package/
  manifest.json
  lens-rgba.png
  thumbnail.webp
  optional-mask.png
```

建议制作源文件从 1024×1024 RGBA 起步；交付尺寸按画质与纹理内存测试确定。一张该尺寸 RGBA8 纹理解码后约 4 MiB，不等于网络压缩文件大小，因此不应一次将全部款式上传 GPU。

示例清单（参数仅用于说明协议，不代表真实商品）：

```json
{
  "schemaVersion": 1,
  "materialId": "demo-brown-v1",
  "skuId": "demo-brown",
  "materialVersion": 1,
  "minRendererVersion": 1,
  "colorSpace": "sRGB",
  "alphaMode": "straight",
  "texture": {
    "path": "lens-rgba.png",
    "sha256": "REPLACE_WITH_RELEASE_CHECKSUM"
  },
  "calibration": {
    "profileId": "demo-profile-v1",
    "relativeScale": 1.05,
    "opacityGain": 0.65
  }
}
```

商品真实规格独立存入 SKU 元数据，避免视觉调参修改商品事实。禁止材质包携带任意可执行脚本或远程 shader；只接受已知参数和支持的资源类型。

发布流程：上传 → 校验尺寸/哈希/配置 → 人工视觉审核 → 生成不可变版本 → 小流量启用 → 正式上架。下载使用 HTTPS；校验失败不替换旧版本。对象路径包含版本，避免 CDN 缓存串款。仅 SHA-256 不能证明发布方身份，若需要离线验证可信来源，对清单另做签名。

缓存当前款、相邻少量款式和缩略图，采用字节预算限制的 LRU；纹理缓存与磁盘缓存分开计量。下载、解码在后台完成，在帧边界切换 GPU 资源，切款失败继续显示原款。

## 8. App 模块接口

保持双端接口语义一致，不强求首版共享二进制代码。

```text
CameraSource.start(config) / stop()
CameraSource.onFrame(frame)

FaceTracker.processAsync(frame)
FaceTracker.onResult(frameId, timestamp, landmarks)

EyeEstimator.estimate(faceResult, frameMetadata) -> EyePairState
TemporalFilter.update(eyePairState) -> StableEyePairState

LensRenderer.setMaterial(materialHandle)
LensRenderer.render(frameHandle, eyePairState, renderConfig)

TryOnSession.selectSku(skuId)
TryOnSession.setOriginalVisible(visible)
TryOnSession.capture() -> CaptureResult
TryOnSession.onStateChanged(state)
```

状态机：Idle → PermissionRequired → Loading → Searching → Tracking；质量不足进入 Degraded，后台进入 Paused，不可恢复错误进入 Error。闭眼为逐眼渲染状态，不必使整个会话退出 Tracking。

核心错误码：CAMERA_DENIED、CAMERA_UNAVAILABLE、MODEL_LOAD_FAILED、MATERIAL_INCOMPATIBLE、DOWNLOAD_FAILED、SAVE_DENIED、DEVICE_UNSUPPORTED。错误码与用户文案分离。

## 9. 拍照、同图对比和保存

点击拍照时冻结一组互相匹配的数据：原始帧、眼部状态、坐标变换、材质版本与渲染参数。不能抓取某一时刻 UI 截图，再拼接另一时刻的跟踪结果。

首版直接导出匹配的视频帧；如需高分辨率静态拍照，需要对新的静态图重新检测，不能沿用旧视频坐标。

对比页在同一冻结底图上重新渲染原眼和各款式。照片渲染可以使用更高质量的同源处理，但必须避免与实时预览的风格明显不同。

默认保存镜像效果与用户预览一致，并在设置中明确规则。导出时去除不必要位置元数据。原图和临时结果只在会话或明确的本机保存范围内存在，设置临时文件清理策略。

iOS 按需请求照片写入权限；Android 使用 MediaStore 的受支持保存路径，不为写入自己生成的图片申请不必要的整个图库读取权限。分享由用户主动调用系统分享面板。

## 10. 服务端与数据模型

### 10.1 部署

初期使用一个 FastAPI 服务、一个 PostgreSQL 实例与对象存储/CDN，按需容器化部署。不要为首版引入微服务、消息队列或 GPU 云集群。管理后台可用 Vue 3 + TypeScript 或团队已有后台系统。

### 10.2 核心表

| 表 | 主要字段 |
|---|---|
| lens_sku | id、brand、name、color、lens_diameter_mm、graphic_diameter_mm、thumbnail、status |
| material_version | id、sku_id、version、manifest_url、hash、renderer_min_version、status |
| release_channel | platform、app_version_range、material_version、rollout_rule |
| calibration_profile | id、parameters、reference_set_id、review_status |
| admin_audit_log | actor、action、target、before/after_version、timestamp |

### 10.3 API

| 方法与路径 | 功能 |
|---|---|
| GET /v1/catalog?cursor=… | 分页款式列表、缩略图和材质版本摘要 |
| GET /v1/lenses/{skuId} | 商品详情与当前可用材质引用 |
| GET /v1/materials/{materialId}/manifest | 材质清单、下载地址与校验信息 |
| GET /v1/app-config?platform=…&version=… | 兼容性、开关与最低可用配置 |
| POST /v1/events/batch | 可选的匿名聚合使用/性能事件 |
| POST /admin/v1/materials | 受保护的素材登记与上传流程 |
| POST /admin/v1/releases | 经审核的上架/回滚操作 |

只读 API 使用 ETag / Cache-Control；客户端离线仍能使用已下载、兼容的材质。公共接口限流，管理接口独立认证和角色权限，发布操作写审计日志。客户端不能包含对象存储永久写入密钥。

首版无需用户账号，收藏可只在本机保存 SKU ID。服务端没有默认照片上传接口。

## 11. 性能目标与降级

下列是测试目标及初始预算，不是 SDK 性能承诺。

- 主流目标机型预览 30 fps；同帧推理模式需能持续供应接近 30 fps 的结果，否则明确评估分辨率、delegate 或跟踪预测策略。
- 720p 视频作为起始配置，预处理分辨率和眼部裁剪独立调整；模型固定输入尺寸应遵守模型要求，降低相机分辨率不等于自动减少全部推理计算。
- 30 fps 对应约 33.3 ms 帧周期。单个推理阶段可先以 10–20 ms、局部处理 2–5 ms、GPU 合成 3–6 ms 做剖析预算，但不同阶段可能流水并行或竞争资源，不能直接相加后声称已达标。
- 用户感知的端到端跟随延迟先以 P95 < 100 ms 为目标，使用外部高速录像等方式测量；同步记录帧龄与 camera-to-render 时间。
- 连续使用 10–15 分钟观察热状态、持续帧率和内存趋势；短时跑分不能代表持续试戴。

降级次序：降低非必要高分辨率输出 → 减少昂贵材质处理 → 降低采集/推理频率并验证帧龄 → 关闭可选分割 → 最后提供照片试戴或提示设备不支持。

降级后若遮挡或贴合明显不可靠，应暂停实时效果，不能以低帧率持续展示错位镜片。GPU/模型崩溃时隔离会话并支持重启；进入后台释放或暂停相机，恢复时重置历史状态。

## 12. 测试与验收

### 自动化与离线回归

- 坐标测试：方向、镜像、裁剪、UV 与导出坐标使用固定测试向量。
- 时序测试：乱序回调、帧丢失、过期结果、暂停恢复，验证不使用已释放缓冲和旧状态。
- 状态测试：拒绝授权、相机占用、资源下载失败、材质不兼容、保存失败。
- 渲染对照：固定输入帧和标注眼部状态，比对双端输出；允许经过定义的色差/像素容差。
- 资源测试：损坏文件、错误哈希、过大纹理、回滚、缓存淘汰和下载中断。

### 真机验证

至少 6–10 台覆盖高、中、低性能档的 iPhone/Android，最终由目标用户机型确定。测试正常室内光、低光、逆光、不同基础瞳色、轻度侧看、眨眼、眼镜反光、刘海/手遮挡、快速移动和前后台切换。

正常光照和约定正面范围内，虹膜中心偏差以真实标注的虹膜直径归一化，暂以 P95 ≤ 5% 为候选门槛；另测静止抖动、闭眼残留、失踪恢复时间。所有指标都要记录测试集、标注口径与无效帧定义。

真实感使用经授权的同人、同款、同光照实物佩戴参考，按基础瞳色分组做盲评。没有这类参考时只能验收交互和稳定性，不能证明商品颜色真实。

## 13. 开发目录建议

```text
ar-lens/
  apps/
    ios/                    # Swift App 与 Metal shader
    android/                # Kotlin App 与 GLSL shader
  services/
    api/                    # FastAPI
    admin/                  # 管理后台
  contracts/
    api/                    # OpenAPI
    materials/              # JSON Schema 与示例清单
    tracking/               # 坐标、时间戳和状态约定
  assets/
    demo/                   # 已授权演示材质
  tests/
    fixtures/               # 授权、脱敏的回归输入
    reference-renders/      # 基准输出
  docs/
    architecture.md
    calibration-guide.md
    device-matrix.md
```

共享几何算法若确有大量重复、且已稳定，再提取 C++ 核心与 Swift/Kotlin 绑定；首版不为共享而提前增加跨语言内存管理复杂度。

## 14. 实施顺序与交付物

### 第 1–2 周：双端技术验证

完成相机 → 异步关键点 → 双眼定位 → 3 款材质 → 眨眼遮罩；在至少 3 台真机测性能。交付安装包、演示录像、耗时剖析和 SDK 选型结论。

阶段门槛：核心场景贴合稳定、闭眼无持续贴图、可连续运行。若不满足，先决定增加分割或切换商业 SDK，不直接开发大量外围业务。

### 第 3–5 周：渲染质量与应用流程

完成滤波、颜色融合、镜像/方向统一、款式切换、素材缓存、同帧拍照和同图对比；与素材人员校准首批款式。

### 第 6–8 周：服务与完整测试版

完成后台发布、版本控制、异常处理、诊断、权限与本机保存；完成代表机型测试和关键回归自动化。

### 第 9–12 周：质量优化与小范围发布

修复热降频、兼容性与真实感问题；完成内部/外部测试版本、支持机型范围、回滚流程及商店发布准备。素材准备和商店审核耗时单独跟踪。

双端首版建议规划 8–12 周，配置 iOS 工程师 1 名、Android 工程师 1 名、视觉/图形工程师 1 名；后端、设计、素材与测试按阶段投入。假设使用成熟模型、10–20 款材质、无支付和账号体系。新训练分割模型、同时开发电商闭环或只有一名工程师时需重新排期。

当前工作环境为 Windows，可进行 Android、后端与文档开发；iOS 编译、签名、真机调试和发布需要 macOS + Xcode（本机、远程 Mac 或具备相应能力的 CI）及真机/签名条件。

## 15. 官方参考与待确认项

以下页面已在本次方案编写时核对。实际实施时锁定 SDK、模型和依赖版本，并记录其许可证；本文不声称已验证任何特定组合的帧率或商业授权。

- Android Face Landmarker：https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/android
- iOS Face Landmarker：https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/ios
- Flutter 原生视图性能取舍：https://docs.flutter.dev/platform-integration/android/platform-views

开工前需要确认：优先验证 Android 还是 iOS、最低机型、素材来源和授权、是否已有技术团队、是否允许商业 SDK、是否需要商品购买入口。以上不会阻止基础相机与跟踪原型，但会影响最终采购与排期。
