# VLA、Jetson与ROS端到端性能

## VLA Pipeline

```text
Camera → Decode → Resize/Normalize → H2D
→ Vision Encoder → Projector → LLM/VLM → Action Head
→ Action Decode → ROS Publish → Controller
```

Demo：`09_vla_e2e_bad_good.py`，每阶段输出Mean、Median/P50/P90/P95/P99/Min/Max/StdDev，并支持serial与producer/consumer。

```bash
python3 src/python/09_vla_e2e_bad_good.py --frames 200 --mode serial
python3 src/python/09_vla_e2e_bad_good.py --frames 200 --mode pipeline
nsys profile -t cuda,nvtx,osrt python3 src/python/09_vla_e2e_bad_good.py --frames 100
```

producer/consumer不保证降低单帧E2E：若producer快于consumer，queue wait会增加。必须同时报告pipeline throughput、queue wait和单帧E2E。单stage省5ms也不一定胜过CPU/GPU重叠。

## 阶段指标

| 阶段 | 指标 |
|---|---|
| Camera | FPS、capture latency、frame drop |
| Decode | latency、CPU/GPU/VPU占用 |
| Preprocess | resize/normalize latency、copy |
| H2D | bytes、duration、overlap |
| Vision/LLM | operator、kernel、GEMM、attention、KV |
| ROS | publish→callback、serialization、queue/drop |
| Controller | period、jitter、deadline miss |

GPU Util低可能不是GPU程序差，而是Camera、Decode或CPU Preprocess喂不饱GPU。

## Jetson

```bash
tegrastats --interval 1000
jtop
nsys profile -t cuda,nvtx,osrt ./application
ncu --set full ./application
```

关注CPU、GR3D、EMC、RAM、SWAP、power、temperature和clock。EMC代表共享外部内存系统活动，CPU、GPU、ISP和视频编解码单元会争用带宽。EMC高但GR3D不满仍可能Memory Bound。

启动50ms、运行20分钟70ms可能来自thermal throttling、power limit或clock drop。长期服务应把P99、功耗、温度、频率、RSS和drop放在同一时间轴。

`nvpmodel`与`jetson_clocks`需要权限并改变平台状态，本实验室不自动执行。

## ROS1/ROS2

```bash
rostopic hz /camera/image
rostopic bw /camera/image
ros2 topic hz /camera/image
ros2 topic bw /camera/image
ros2 trace
```

关注Serialization、Deserialization、Copy、DDS、Executor、Callback、Queue和Drop。sensor_msgs/Image与PointCloud2尤其容易受大消息copy限制。

Shared Memory Transport、Loaned Message和Zero-Copy不是同义词：共享内存仍可能序列化/copy；loaned message是生命周期与缓冲所有权机制；是否真正zero-copy取决于RMW/DDS、消息类型和进程布局。

端到端Tracing应让frame_id和monotonic timestamp贯穿Camera Publish→Transport→Callback→Inference NVTX→Action Publish→Control Callback。

## 统一时间戳与Frame ID

端到端Tracing的基础不是工具，而是可关联性。每帧应携带frame_id、capture monotonic timestamp、每阶段start/end、queue enqueue/dequeue、action_id和controller apply time。不同机器还需时钟同步并记录误差。

## Stage Time与Queue Time

```text
E2E = service time之和 + queue wait - overlap
```

只统计函数执行时间会遗漏排队；简单相加又会重复计算并发阶段。最终E2E必须由capture到controller的同一frame直接测量。

## 丢帧策略

实时系统不能无限排队。常见策略：

- 处理所有帧：完整但延迟会累积；
- latest-frame：丢旧帧保持新鲜度；
- bounded queue：在容量处显式drop；
- adaptive rate：根据负载降低camera/inference频率。

必须报告drop count、frame age和控制语义，不能只报告FPS。

## Camera/Decode真实工具链

- V4L2：capture buffer、timestamp、frame drop；
- NVDEC/Jetson decoder：硬件surface与decode latency；
- Rockchip MPP/RGA：硬件decode/resize；
- OpenCV CPU resize：容易成为CPU feeding瓶颈；
- CUDA preprocess：可融合resize、颜色转换、normalize和layout。

从CPU decode切到硬件decode后，应检查是否引入新的surface copy或颜色格式转换。

## Jetson热稳态实验

建议至少运行20–30分钟，并每秒记录：

```text
timestamp, E2E_P50/P99, FPS, drop
CPU, GR3D, EMC, RAM, SWAP
power rails, temperature, CPU/GPU/EMC clocks
```

将延迟上升时间点与温度和clock drop对齐，才能区分thermal throttling与代码状态增长。

## EMC竞争案例

Camera/ISP写入帧、decoder读写surface、CPU preprocess扫描像素、GPU inference读取权重/KV，会共享DRAM。若这些阶段并行，虽然pipeline利用率提高，也可能因EMC争用导致各阶段变慢。优化应比较E2E，而不是假设并行必然线性收益。

## ROS2 Executor与Callback

单线程executor中，一个耗时image callback可阻塞control callback。多线程executor需要callback group和线程安全设计，否则可能转化为锁竞争。应记录publish、DDS write/read、take、callback ready/start/end和action apply。

## Shared Memory层次

- Shared Memory Transport：数据通过共享内存区域传递，但可能仍序列化；
- Loaned Message：中间件借出buffer，减少用户分配/copy；
- Zero-Copy：端到端没有数据复制，是更强条件。

必须用trace或地址/生命周期验证，不要仅凭启用shared memory声称zero-copy。

## VLA故障诊断案例

```text
症状：GPU Util 35%，E2E P99高
Stage Timer：normalize P99 12ms
nsys：GPU每帧前空洞8–13ms
perf：CPU resize/normalize热点
优化：预分配+pinned buffer+CUDA fused preprocess
复验：GPU空洞下降、H2D与vision重叠、P99下降、图像数值一致
```

## 长时间稳定性

同时监控RSS、GPU allocated/reserved、queue depth、open file count、线程数、frame age、temperature和power。周期性抖动可能来自日志flush、模型cache整理、DDS discovery、thermal governor或内存回收。

## 练习

1. 给mock pipeline增加queue wait和frame age。
2. 实现latest-frame策略并比较drop与P99。
3. 用NVTX为每个frame和stage编码。
4. 在Jetson上对齐tegrastats与nsys时间线。

## 已准备的目标机代码

### V4L2

```bash
./src/build/34_v4l2_capture_benchmark --device /dev/video0 --width 1280 --height 720 --format YUYV --frames 1000 --buffers 4
```

输出frame interval分布、sequence gap、FPS和MB/s；用sleep-ms可模拟下游阻塞。

### ARM、Jetson、GStreamer与FFmpeg如何选型

#### 结论先行

更准确的判断不是“ARM架构更常使用GStreamer”，而是“Jetson、Rockchip等嵌入式实时多媒体平台更常使用GStreamer”。ARM只是CPU指令集；真正决定软件栈的是厂商多媒体SDK、摄像头接口、硬件codec、buffer机制以及应用是否要求实时流式处理。

| 平台/任务 | 常见选择 | 主要原因 |
|---|---|---|
| Jetson CSI/USB Camera实时推理 | GStreamer/DeepStream | JetPack提供Argus、V4L2、NVMM和NVIDIA插件，方便组织实时pipeline |
| Rockchip Camera/视频推理 | GStreamer或MPP/RGA厂商API | 便于连接V4L2、RKMPP、RGA与DMA-BUF，但插件随厂商镜像变化 |
| x86服务器离线转码/数据预处理 | FFmpeg | codec覆盖广、CLI成熟、批处理与自动化方便 |
| NVIDIA dGPU视频分析服务 | FFmpeg或GStreamer | FFmpeg适合能力验证和批处理，GStreamer适合长驻实时服务 |
| 普通ARM Linux/ARM服务器 | 取决于任务 | ARM本身不决定框架；无摄像头或硬件媒体需求时未必使用GStreamer |
| 自定义C++媒体服务 | LibAV或GStreamer API | LibAV控制直接；GStreamer擅长插件、队列、时钟和状态管理 |

#### 为什么Jetson上GStreamer更常见

Jetson的典型实时路径是：

```text
CSI/USB Camera
  → Argus/V4L2 Source
  → NVIDIA硬件Decode/Convert
  → NVMM/DMA-BUF类Buffer
  → CUDA/TensorRT/DeepStream Inference
  → Display/Encode/ROS
```

常见原因包括：

- `nvarguscamerasrc`可连接Jetson CSI Camera与Argus；
- NVIDIA提供硬件decoder、encoder、converter等GStreamer element；
- caps negotiation可以描述分辨率、帧率、像素格式和memory type；
- `queue`能把capture、decode、preprocess和consumer拆到不同线程，便于构建producer/consumer流水线；
- `appsink`与`appsrc`允许把媒体流接入自定义C++/Python推理程序；
- GStreamer具有clock、timestamp、buffer pool、backpressure、drop和状态机，适合长驻实时服务；
- NVIDIA DeepStream建立在GStreamer之上，因此多路视频分析通常沿用这一生态。

目标机必须查询真实插件，不能假定不同JetPack版本完全相同：

```bash
gst-launch-1.0 --version
gst-inspect-1.0 nvarguscamerasrc
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvvidconv
gst-inspect-1.0 appsink
gst-inspect-1.0 | grep -Ei "nvidia|nvargus|nvv4l2|nvvidconv"
```

一个概念性的CSI Camera命令如下，caps与element应以目标机`gst-inspect-1.0`结果为准：

```bash
gst-launch-1.0 -e \
  nvarguscamerasrc num-buffers=300 \
  ! "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1" \
  ! nvvidconv \
  ! fakesink sync=false
```

`sync=false`用于测最大离线/管线吞吐；做真实30 FPS时钟与延迟实验时不能机械照搬，应明确是否按pipeline clock同步。

#### GStreamer与FFmpeg不是替代关系

| 需求 | 优先入口 | 后续工具 |
|---|---|---|
| 查询codec、分辨率、pix_fmt | `ffprobe` | FFmpeg/LibAV |
| 快速验证软件/硬件decode | FFmpeg CLI | `-benchmark`、系统监控 |
| Camera实时采集与多阶段pipeline | GStreamer | tracer、GST_DEBUG、nsys |
| 批量离线转码/生成测试输入 | FFmpeg CLI | shell/Python自动化 |
| 嵌入式C++精细控制 | LibAV或GStreamer API | 自定义stage timer/NVTX |
| Jetson多路视频AI服务 | GStreamer/DeepStream | tegrastats、nsys |

推荐组合流程：

```text
ffprobe/FFmpeg
  → 确认码流、codec、pix_fmt和纯decode基线
GStreamer
  → 构建Camera→Decode→Queue→Preprocess实时流水线
NVTX + Nsight Systems
  → 检查Decode→Preprocess→Inference重叠和GPU空洞
tegrastats/jtop
  → 检查NVDEC、GR3D、EMC、CPU、功耗、温度和频率
```

#### 使用NVMM仍不等于端到端零拷贝

以下路径中，即使前半段使用Jetson硬件decoder和NVMM，进入普通CPU `cv::Mat`后仍可能发生映射或复制：

```text
Decoder Surface/NVMM
  → CPU BGR/RGB cv::Mat
  → PyTorch CPU Tensor
  → H2D
  → CUDA Tensor
```

真正需要证明的是buffer从decoder到推理输入的memory domain变化：

```text
Decoder Surface
  → NVMM/DMA-BUF
  → CUDA可访问Buffer
  → TensorRT/推理输入
```

NVMM、DMA-BUF、shared memory、pinned host memory和CUDA device memory不是同义词。即使底层能共享buffer，也可能因为像素格式转换、stride/layout不兼容、CPU mapping或框架导入接口缺失而复制。因此不能只因pipeline字符串包含`memory:NVMM`就宣称zero-copy。

验证时应同时查看：

- GStreamer caps协商和实际buffer memory type；
- 是否出现CPU map、颜色转换、`hwdownload`或H2D；
- nsys中的Memcpy、CUDA API、GPU bubble和NVTX stage；
- CPU、GR3D、NVDEC与EMC占用；
- capture interval、frame age、drop、queue depth、E2E P50/P99。

#### 选择原则

- Camera/VLA长驻实时服务：通常以GStreamer为主线，用FFmpeg做码流检查与基线。
- 离线数据预处理、转码和快速实验：通常先用FFmpeg。
- 需要逐帧访问、嵌入已有C++调度器：比较LibAV和GStreamer API的集成成本。
- 需要多路流、队列、动态重连、时钟与backpressure：GStreamer通常更自然。
- 最终选择必须由目标镜像的插件能力、correctness、E2E P99、CPU/GPU/EMC、功耗和稳定性A/B决定，而不是只按ARM或x86标签决定。

### NVDEC/MPP/RGA与OpenCV

```bash
python3 src/python/11_media_pipeline_target_lab.py --source video.mp4 --frames 500
python3 src/python/11_media_pipeline_target_lab.py --pipeline 'TARGET_GSTREAMER_PIPELINE ! appsink' --frames 500
```

具体GStreamer element依JetPack、NVIDIA dGPU或Rockchip镜像插件，通过参数注入，避免硬编码不存在的接口。

### Jetson长时间热稳态

```bash
./src/scripts/monitor_jetson_long_run.sh 1200 1000 ./your_vla_application
```

脚本并行采集tegrastats并转换CSV，不调用nvpmodel或jetson_clocks改变平台状态。

### ROS2

目标包位于`src/integrations/ros2_vla_profiling`，提供mock camera和VLA consumer。Image Header贯穿capture timestamp/frame id，并输出queue age、work和E2E CSV；配合ros2 trace观察publish/take/callback/executor。

## FFmpeg/LibAV媒体流水线实验

### 这组实验回答什么问题

GStreamer适合插件化实时流水线，FFmpeg CLI适合快速验证codec、硬件后端与转码路径，LibAV API适合把demux/decode/scale直接嵌入C++服务。三者不是互斥关系：工程中可先用FFmpeg CLI确认能力与性能，再在LibAV或GStreamer应用中复现相同数据路径。

当前提供两层实验：

- `src/build/35_ffmpeg_decode_benchmark`：直接使用libavformat、libavcodec和libswscale，输出逐帧decode+scale统计、整段wall吞吐和checksum；
- `src/python/12_ffmpeg_target_lab.py`：调用ffprobe与ffmpeg CLI，对软件、CUDA/NVDEC或自定义硬件路径重复warmup与A/B，保存JSON。

### “4228 FPS”和“2088 FPS”到底是什么

当前机器曾对一个短小的640×360 MPEG-4合成文件处理100帧，得到：

| 路径 | wall吞吐 | 反推平均处理时间 | 正确解释 |
|---|---:|---:|---|
| C++ LibAV进程内路径 | 约4228 frame/s | 约0.237 ms/frame | 同一进程连续demux/decode/scale的离线吞吐 |
| FFmpeg CLI软件路径 | 约2088 frame/s | 约0.479 ms/frame | 包含FFmpeg进程与filter初始化等成本的离线吞吐 |

计算方式是：

```text
throughput_fps = processed_frames / wall_seconds
inverse_throughput_ms = 1000 / throughput_fps
```

这里的FPS不是摄像头帧率，也不是端到端VLA FPS，更不表示每一帧的真实P99延迟。离线解码器可以在没有实时节拍限制时尽快处理已有文件，因此30 FPS视频也可能以数千frame/s被处理。反推的`1000/FPS`只是单位工作平均成本；当存在多线程、流水并行或batch时，它不等于单帧响应延迟。

这两个数字也不能直接用来断言“C++比FFmpeg快两倍”，原因包括：

- CLI包含新进程、协议、demux、filter graph的初始化与销毁，短输入会放大固定成本；
- 两条路径统计边界不同，C++逐帧字段主要覆盖receive/scale，wall吞吐才覆盖完整循环；
- 输入是低分辨率、易解码的合成素材，不能代表H.264/H.265 1080p/4K或真实相机；
- 输出像素格式、resize算法、线程数和硬件surface是否下载到CPU都会改变结果；
- 100帧样本太短，不代表热稳态、功耗、降频和尾延迟。

因此报告中应把它们写为“本机、该输入、该命令下的离线吞吐观测值”，而不是产品性能结论。

### 三类能力查询命令怎么读

先查询当前FFmpeg二进制实际编译进了什么：

```bash
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -decoders 2>/dev/null | grep -E "cuvid|nvdec|v4l2|rkmpp"
ffmpeg -hide_banner -filters 2>/dev/null | grep -E "scale_cuda|scale_npp|rga"
```

`ffmpeg -hwaccels`列出FFmpeg认识的硬件加速框架，例如`cuda`、`vaapi`、`vdpau`、`drm`。它只表示该FFmpeg构建具有相应接口，不保证驱动、设备节点、codec和运行权限全部可用。

`ffmpeg -decoders`列出解码器；管道后的`grep`只是从长列表中筛选常见硬件关键字：

| 关键字 | 常见平台/含义 | 注意事项 |
|---|---|---|
| `cuvid` | NVIDIA CUVID/NVDEC包装解码器，如`h264_cuvid` | 常见于NVIDIA dGPU构建；仍需驱动与支持的codec |
| `nvdec` | NVIDIA硬解相关命名 | 不同FFmpeg版本未必直接以该字符串出现在decoder名中 |
| `v4l2` | Linux V4L2 M2M硬件codec，如`h264_v4l2m2m` | 需要正确的codec设备和驱动，不等于普通摄像头采集 |
| `rkmpp` | Rockchip MPP硬件解码 | 通常需要厂商补丁/定制FFmpeg与MPP运行库 |

`ffmpeg -filters`列出filter；这里寻找的是硬件resize/颜色转换能力：

| Filter | 执行位置 | 典型用途 |
|---|---|---|
| `scale_cuda` | CUDA设备 | GPU surface上的resize/部分格式转换 |
| `scale_npp` | NVIDIA NPP | NPP实现的GPU resize/格式处理，取决于构建选项 |
| 名称含`rga`的filter | Rockchip RGA | Rockchip上的resize、crop、色彩转换；名字依厂商构建而异 |

命令没有输出，只能说明“当前FFmpeg列表里没有匹配该正则的名字”，不能直接说明机器没有硬件加速。还要检查完整列表、FFmpeg构建参数、厂商文档、驱动和设备：

```bash
ffmpeg -hide_banner -buildconf
ffmpeg -hide_banner -decoders | less
ffmpeg -hide_banner -filters | less
ls -l /dev/video* /dev/dri/* 2>/dev/null
```

### 硬解码不等于端到端零拷贝

必须画清帧所在的memory domain：

```text
压缩码流 → 硬件Decoder Surface → GPU/RGA resize → 推理输入
                              ↓ hwdownload
                           CPU内存 → CPU resize → H2D → GPU推理
```

第一条路径有机会保持device resident；第二条虽然使用硬解码，却把surface下载到CPU，随后又H2D上传，copy与同步可能抵消硬解收益。Shared Memory、DMA-BUF、NVMM、CUDA device memory也不是同一个概念，能否少拷贝取决于插件、像素格式、allocator以及推理框架能否导入该buffer。

因此要同时记录：decode latency、resize latency、hwdownload/H2D、CPU占用、GPU/codec利用率、wall吞吐、P50/P99和E2E，而不能只看decoder名称。

### 可复制的实验步骤

先探测输入和软件基线：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,pix_fmt,avg_frame_rate \
  -of json input.mp4

python3 src/python/12_ffmpeg_target_lab.py input.mp4 \
  --mode software --frames 500 --warmup 1 --iterations 5 \
  --width 224 --height 224 --json software.json

./src/build/35_ffmpeg_decode_benchmark input.mp4 500 224 224
```

NVIDIA目标机在查询确认`cuda`和`scale_cuda`可用后再运行：

```bash
python3 src/python/12_ffmpeg_target_lab.py input.mp4 \
  --mode cuda --frames 500 --warmup 1 --iterations 5 \
  --width 224 --height 224 --json cuda.json
```

厂商后端名称不统一，使用custom模式注入目标机已经验证过的参数：

```bash
python3 src/python/12_ffmpeg_target_lab.py input.mp4 \
  --mode custom --frames 500 \
  --extra-input -hwaccel rkmpp \
  --extra-filter "TARGET_RGA_FILTER=224:224" \
  --json rkmpp.json
```

上面的`TARGET_RGA_FILTER`是占位符，必须替换为目标镜像`ffmpeg -filters`实际列出的filter，不能照抄。统一入口会先打印能力：

```bash
./src/scripts/run_ffmpeg_target.sh input.mp4 software
./src/scripts/run_ffmpeg_target.sh input.mp4 cuda
```

### 如何做公平的Software vs Hardware A/B

比较前固定：输入文件、codec、分辨率、帧数、目标尺寸、输出像素格式、resize算法、线程策略、warmup次数、同步点和统计边界。输入帧数必须不少于`--frames`，否则按请求帧数计算吞吐会造成高估。长视频还应分别报告冷启动和稳态。

至少保存以下结果：

| 指标 | 用途 | 常见误判 |
|---|---|---|
| wall throughput | 整体处理能力 | 当成单帧latency或camera FPS |
| per-frame P50/P95/P99 | 尾延迟与抖动 | 只看mean掩盖卡顿 |
| CPU utilization | 软件decode/preprocess压力 | CPU低也可能在同步等待 |
| decode engine利用率 | 硬解单元是否工作 | GPU Util不能替代codec engine指标 |
| copy/H2D duration | surface下载和传输成本 | 只比较decode engine时间 |
| output checksum/帧数 | correctness与完整性 | 更快其实是丢帧或少处理 |
| power/temperature/clock | Jetson长期稳态 | 用冷机短跑推断20分钟性能 |

### 结果诊断

- 软件decode时CPU高且decode耗时占主导，硬解后CPU下降、E2E下降：硬解有效。
- 硬解engine工作但E2E不降，同时出现hwdownload、CPU scale和H2D：memory-domain往返是候选根因。
- GPU Util低且GPU timeline有长空洞，空洞前是decode/preprocess：GPU被媒体前端饿住，不应先优化kernel。
- 硬解吞吐高但P99差：检查队列、buffer数量、线程调度、阻塞式同步、动态分辨率和热降频。
- FFmpeg单独很快但VLA E2E慢：继续用NVTX/nsys检查decode输出到tensor、normalize、layout转换、H2D和模型之间的边界。

Jetson FFmpeg后端随JetPack镜像构建而变化，Rockchip RKMPP/RGA也取决于厂商FFmpeg、MPP/RGA版本和编译选项。本实验代码故意支持参数注入，不假定所有平台使用同一个decoder/filter名称；这些硬件路径需要在对应目标机上验证。
