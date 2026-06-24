# VLA Profiler 待做功能清单（Roadmap）

记录后续可添加的功能。结合使用场景：π0.5 / SmolVLA、chunk action、ROS 实时约束、
~705M 模型、机器人端部署，以及 MIT 6.5940 模型压缩课程主题。

每个功能项统一用三段式说明：**为什么补充** / **如何补充** / **补充后的好处**。

图例：`[ ]` 待做 · `[~]` 进行中 · `[x]` 已完成 · ⭐ 推荐优先

---

## 已完成（现状）

- [x] 模块级参数统计（vision/language/fusion/action 拆分，按真实 dtype 算大小）
- [x] 多后端 MACs：`fvcore → torchprofile → thop → hook` 自动回退 + 交叉校验
- [x] 理论延迟 + CUDA Event 实测（mean / p50 / p99 / efficiency）
- [x] Roofline（arithmetic intensity / ridge / compute-vs-memory）+ 双图绘制
- [x] VLA 专用：chunk rollout 摊销、KV cache 带宽、多相机、ROS 实时性耦合
- [x] Kernel 级 profiling（torch.profiler）+ Chrome/Perfetto trace + ncu/nsys 命令
- [x] 文本 + markdown 报告
- [x] 内置 ~705M 合成 SmolVLA 模型 + CLI

> **当前局限**：工具只能「诊断现状」（这个模型现在多快、瓶颈在哪），
> 还不能「预测优化收益」（量化/剪枝后会怎样）、「跑真实模型」、「换硬件验证」。
> 下面的功能就是补上这三块缺口。

---

## A. 压缩 / 优化收益（课程核心）

### A1. 量化 / 剪枝 What-if 收益模拟器 ⭐
- **为什么补充**：现在工具只告诉你「memory-bound、efficiency 3.6%」，但不告诉你
  「量化到 int8 能省多少、该剪哪个模块」。优化前如果不能先估收益，就只能盲目试错，
  跑一轮真实量化/剪枝/重训要几天。
- **如何补充**：不真正改模型，只改参数——量化把 dtype_bytes 改小后重算字节数 →
  重跑 Roofline → 得新延迟/带宽；剪枝按稀疏度缩放各模块 MACs/params 后重算。
  新增 `whatif.py`（`simulate_quantization` / `simulate_pruning`），逐模块扫描排序。
- **补充后的好处**：几秒钟就能回答「int8 vs fp8 哪个值得做」「剪哪个模块收益最高」，
  把「跑几天验证」变成「先算后做」。直接对应课程目标与 lecture-02 误区 #7。

### A2. 剪枝稀疏度 sweep 曲线
- **为什么补充**：剪枝不是越多越好——memory-bound 模型剪到某个点后 MACs 再降也不省延迟。
  需要看到「收益拐点」在哪，否则会过度剪枝伤精度却无速度回报。
- **如何补充**：复用 A1 的剪枝模拟，扫描 0%→90% 稀疏度，画各模块
  MACs / 理论延迟 / 内存曲线（复用 `plot.py`）。
- **补充后的好处**：一张图看清「压到多少开始无收益」，指导精度-速度权衡。

### A3. 能耗模型（mJ / 推理 + 续航）⭐
- **为什么补充**：机器人/AR 端真正的约束常是功耗和续航，不是延迟。当前工具完全没有
  能耗维度，而你 lecture-02 里已经备齐了 pJ 级数据（memory wall）。
- **如何补充**：用 INT8 MAC 0.2pJ、SRAM 读 10pJ、DRAM 读更高的系数，拆 compute 能耗 +
  memory 能耗，按精度对比；给定电池容量算续航。新增 `energy.py`（`estimate_energy_mj`）。
- **补充后的好处**：能回答「这个 VLA 在机器人上跑一次耗多少 mJ、电池能撑多久」，
  并量化「量化省的是计算能耗还是内存搬运能耗」（边缘端后者占大头）。

---

## B. 部署 / 真实化（脱离合成模型）

### B1. 真实 SmolVLA / LeRobot 加载器 ⭐
- **为什么补充**：现在只能 profile 内置合成模型，结论无法直接套到线上 policy；
  而当前环境就是 `lerobot_ghr`，手边就有真实模型，不接太可惜。
- **如何补充**：写 adapter 封装 LeRobot policy 的 forward 差异（observation dict → tuple），
  从 config 自动推断 KV cache 几何，用 `SplitConfig.overrides` 适配真实命名
  （siglip / gemma / expert）。新增 `models/lerobot_adapter.py` + CLI `--model lerobot`。
- **补充后的好处**：工具从「demo 玩具」变成「线上模型体检仪」，所有指标都是你真实
  policy 的数据。

### B2. ONNX / TensorRT 导出 + 实测 + op fallback 检测
- **为什么补充**：PyTorch 里跑得快不代表部署引擎里也快——某个 op 不被支持会悄悄
  fallback 到 CPU，延迟翻几倍且无报错（lecture-02 提的经典坑）。
- **如何补充**：导出 ONNX，用 onnxruntime / TensorRT 实测对比 PyTorch，逐 op 检查是否
  fallback 到 CPU provider。新增 `export_bench.py`（onnx/onnxruntime requirements 已列）。
- **补充后的好处**：上线前就能抓出「导出成功但实际跑得慢」的算子，避免线上事故；
  对应 lecture-02 面试 Q1。

### B3. 多硬件可行性矩阵 ⭐
- **为什么补充**：VLA 要从 A100 训练迁到 Jetson 等机器人端，「放不放得下、跑不跑得动」
  得逐个硬件验证，手动算很费劲。
- **如何补充**：复用已有 GPU 预设表 + A1/A3 的延迟/能耗模型，一次性算
  A100 / Orin / Jetson Nano / RTX4090 上的理论延迟、能耗、峰值内存是否放得下，
  输出硬件 × 指标 的可行性表（✅/❌ + 数值）。
- **补充后的好处**：一张表完成部署选型决策，回答「这模型能不能上某块板子」。

### B4. 推理峰值内存分析
- **为什么补充**：当前 model size 只算权重，但推理 OOM 往往是激活值 + workspace 撑爆
  （lecture-02 误区 #6：权重 100MB 但激活值 800MB）。
- **如何补充**：forward hook 累计中间 tensor 字节峰值 + 权重 + cuDNN workspace 估计。
- **补充后的好处**：能在部署前判断「峰值内存会不会超目标设备的 80% 安全线」，
  对应落地 Checklist。

---

## C. 诊断 / 工程化

### C1. 逐模块 latency 实测
- **为什么补充**：现在 kernel 分解只给 `addmm`、`bmm` 这种算子名，看不出「是 vision
  还是 fusion 慢」，定位瓶颈还要人工对应。
- **如何补充**：hook + CUDA Event 测每个顶层模块（vision/language/fusion/action）真实耗时。
- **补充后的好处**：直接看到「fusion 占了 60% 时间」，瓶颈定位从算子级回到业务模块级。

### C2. Sweep 扫描 + 曲线（batch / 分辨率 / chunk-steps）
- **为什么补充**：「chunk action 为什么 jitter」「batch 开多大最优」这类问题靠单次 profile
  答不了，需要扫一组配置看趋势。
- **如何补充**：自动跑多组配置，输出 latency / throughput / efficiency 曲线（复用绘图）。
- **补充后的好处**：用曲线回答 jitter 来源与最优配置；对应 lecture-02 面试 Q2。

### C3. NVTX 模块标注
- **为什么补充**：nsys 时间线默认全是底层 kernel，看不出对应哪个模块，排查费劲。
- **如何补充**：给 vision/language/fusion/action 打 `torch.cuda.nvtx.range`。
- **补充后的好处**：nsys 时间线按模块着色，一眼看清各模块时间分布。

### C4. JSON 导出 + 回归对比
- **为什么补充**：优化是迭代的，需要量化「这次改动到底变快没、精度退化没」，
  纯文本报告没法自动 diff。
- **如何补充**：结果存 JSON，支持两次 profile 的 diff。参考同仓库
  `edge_ai_compression_deployment/reports` 风格。
- **补充后的好处**：建立优化前后回归监控，每次改动都有可对比的量化证据。

### C5. 配置文件驱动（YAML）
- **为什么补充**：实验参数一多，命令行又长又易错，难以复现。
- **如何补充**：用 YAML 描述硬件 + VLA 参数 + 实验组，参考
  `edge_ai_compression_deployment/configs/config.yaml`。
- **补充后的好处**：实验可复现、可版本化，一条命令跑一整组配置。

### C6. pytest 测试套件
- **为什么补充**：工具一旦被信任用于决策，自身正确性必须有保障，否则错误结论比没工具更糟。
- **如何补充**：覆盖各模块 + 后端一致性 + 合成模型 smoke test。
- **补充后的好处**：重构/升级不怕回归，保证 profiler 自身可靠。

---

## 建议实施顺序

1. **A1 量化/剪枝 What-if** + **A3 能耗模型** —— 课程核心、收益直接
2. **B1 真实 SmolVLA 加载器** —— 让工具脱离合成模型真正可用
3. **B3 多硬件可行性矩阵** + **B4 峰值内存** —— 部署选型
4. **C1 逐模块 latency** + **C2 sweep** —— jitter / 瓶颈诊断
5. **B2 ONNX/TensorRT**、**C3-C6 工程化** —— 收尾
