# 17 · 机器人 / VLA 部署：把编译器接到真实机器人

> 本笔记把前面所有模块（方言/融合/量化/lowering/内存/运行时/profiling）落到机器人与 VLA
> （Vision-Language-Action）部署场景，说明每项编译优化"为什么对机器人重要"。

---

## 1. 为什么机器人推理是"延迟优先"的编译问题

机器人是**实时闭环**系统：传感 → 推理 → 动作，必须在固定周期内完成（典型控制环 10–50 Hz，
即每步 20–100 ms 预算）。与云端"吞吐优先"不同，机器人是"**单帧延迟 + 确定性**优先"：
- 延迟必须 < 控制周期，否则控制发散；
- 延迟必须**低抖动**（无 GC、无运行期 malloc、无动态 shape 重编译），否则周期不稳。

这正是本项目所有优化的目标：融合降延迟、量化降延迟、内存规划保证零运行期分配、静态 shape 保证可预测。

## 2. VLA 部署（Vision-Language-Action）

VLA 策略网络（如 RT-2、OpenVLA、π0）结构 ≈ 视觉编码器(ViT) + 语言模型(Transformer) + 动作头。
部署挑战：模型大（数亿~数十亿参数）、含大量 attention、要在边缘算力（Jetson Orin / Ascend）上跑到
控制频率。映射到本项目：

- **视觉主干 Conv+BN+ReLU 融合**（Module 05）：直接压低视觉编码延迟。
- **`edge.attention` 一等算子**（Module 03）：为整体融合成 FlashAttention 风格 kernel 预留，避免
  MHA 拆开后的访存爆炸——这是 VLA 延迟的大头。
- **量化**（Module 07）：KV cache / 线性层 INT8/混合精度，换吞吐与显存。
- **静态化 + 内存规划**（Module 04/09）：固定输入分辨率/序列长后做 shape 固化 + arena 规划，零分配。

## 3. Policy Inference（策略推理）

- 动作可能是**连续向量**（关节力矩）或**离散 token**（动作分块 action chunking）。
- action chunking（一次推理出 H 步动作）摊薄推理开销：把 1/控制周期的推理频率降到 1/(H·周期)，
  是延迟与算力的关键权衡。编译器侧需支持把"chunk 维"静态化以利内存规划。

## 4. Action Latency Optimization（动作延迟优化）

端到端动作延迟 = 传感拷贝 + 预处理 + 推理 + 后处理 + 下发。编译器能优化的部分：
- 预处理融进首层（常量折叠相机归一化的 scale/mean，Module 05）。
- 推理：融合 + 量化 + kernel 选择（Module 05/07/10）。
- 用 Profiler（Module 12）做延迟分解，定位是视觉、attention 还是动作头最耗时，针对性优化。

## 5. Multi-Camera Pipeline（多相机管线）

机器人常有多路相机（手眼 + 全局）。编译/运行时策略：
- 多路图像可 **batch** 成一个张量走同一视觉编码器（提高 GPU/NPU 利用率），或分流并行。
- 布局选择（`Layout` 枚举）让相机数据以硬件偏好格式（NHWC）进卷积，减少 reformat。
- 多路中间张量在 arena 内**跨相机复用**（Module 09），压低峰值内存。

## 6. Scheduling Optimization（调度优化）

- 计算与拷贝重叠：H2D 拷贝下一帧时，GPU/NPU 算当前帧（多 stream）——本项目 `OperatorScheduler`
  预留异步扩展点（Module 11）。
- 视觉编码（重）与动作头（轻）可流水化；多相机路可并行。
- 实时性优先时，宁可同步执行以保证**确定性**，也不盲目追异步吞吐——需按机器人需求权衡。

## 7. TensorRT 部署路径（GPU 机器人，如 Jetson）

典型生产路径：PyTorch → ONNX → `trtexec`/TensorRT API 构建 INT8/FP16 engine → 在 Jetson 上用
`IExecutionContext.enqueueV3` 跑。本项目的 EdgeDialect→优化→lowering 对应 TensorRT builder 的优化阶段；
真实机器人项目里常用 TensorRT 做后端，而本项目可作为"前置图优化 + 量化决策"的可控层（再导出给 TensorRT）。

## 8. 编译器集成笔记（如何把本项目接进机器人栈）

1. 前端：PyTorch → ONNX → 导入为 EdgeDialect（或 Tosa→Edge）。
2. 图优化：`edge-shape-inference` → `edge-fuse-conv-bn-relu` → 量化（Module 07）。
3. 后端二选一：
   - **CPU/通用**：`edge-lower-to-llvm`（Module 10）→ JIT/AOT，配 `edge-run` 风格运行时。
   - **GPU/NPU**：导出到 TensorRT / Ascend om（用本项目做前置优化与量化决策）。
4. 部署：静态 shape + arena（Module 09）+ 同步确定性执行，接进 ROS2 / 实时控制节点。
5. 上线前：`edge-statistics`（算力体检）+ `edge-memplan`（峰值内存）+ `edge-run` Profiler（延迟分解）
   三件套做"部署前体检"，确认落在控制环预算内。

## 9. 关键 Trade-off（机器人特有）

- 吞吐 vs 延迟：机器人选延迟；不要为吞吐盲目加 batch（除非多相机天然 batch）。
- 异步 vs 确定性：实时控制偏好确定性同步执行。
- 精度 vs 速度：量化要在动作精度（安全性！）与延迟间谨慎权衡，机器人对精度退化更敏感。

## 10. 一句话总结

机器人/VLA 部署把编译器的价值具体化为"**在固定延迟预算内、以确定性方式、用受限算力跑完策略网络**"。
本项目的融合、量化、内存规划、静态化、profiling 全部服务于这个目标——这也是 AI 编译器在
机器人方向最硬核的价值点。
