# 14_heterogeneous_scheduling - 异构调度

## 工业背景：LLM 推理/训练的调度核心技术

当模型大到单卡放不下，或者需要高吞吐并发推理时，异构调度是核心竞争力。本模块借鉴 **vLLM**、**DeepSpeed** 和 **Megatron-LM** 三个工业级项目的设计模式，实现完整的异构调度系统。

### 借鉴的设计模式

| 来源 | 核心组件 | 设计模式 |
|------|---------|---------|
| **vLLM** | BlockManager, Scheduler | PagedAttention KV cache 块管理、Continuous Batching、Preemption |
| **DeepSpeed** | ZEROStageManager | ZeRO-1/2/3 分片策略、Memory-DX 优化 |
| **Megatron-LM** | PPScheduler, TPScheduler | 1F1B 流水线调度、TP Column/Row Parallel 通信模式 |
| **FlexFlow/Alpa** | WorkloadBalancer | 异构 GPU 容量感知负载均衡 |

---

## 目录结构

```
14_heterogeneous_scheduling/
├── README.md                  ← 本文件
├── __init__.py                ← 空文件，使模块可导入
├── scheduler.py               ← 核心调度器实现（750+ 行）
│   ├── BlockManager           — PagedAttention KV cache 块管理器
│   ├── Scheduler              — Continuous Batching 调度器
│   ├── MemoryPlanner          — 显存感知调度策略
│   ├── PPScheduler            — 1F1B 流水线并行调度器
│   ├── TPScheduler            — 张量并行调度器
│   ├── ZEROStageManager       — ZeRO 分片策略管理器
│   ├── WorkloadBalancer       — 异构 GPU 负载均衡
│   ├── NCCLCommManager        — NCCL 通信管理
│   ├── HybridScheduler        — 混合并行（TP+PP+DP）调度器
│   └── SimulationEngine       — 端到端推理模拟引擎
├── test_scheduler.py          — pytest 测试（500+ 行，30+ 测试用例）
└── scheduler_demo.py          — 交互式演示脚本（300+ 行）
```

---

## 核心组件一览

### 1. BlockManager（借鉴 vLLM）
- O(1) 物理块分配/回收
- 引用计数：支持 prefix caching（相同 prompt 前缀共享 KV cache）
- 逻辑块表映射（logical → physical block）

### 2. Scheduler（借鉴 vLLM）
- Continuous batching：waiting → running → finished
- Preemption：显存不足时换出低优先级序列
- Swap-in/Swap-out：支持 KV cache 换入换出

### 3. MemoryPlanner（借鉴 vLLM + DeepSpeed）
- Watermark 策略：防止 KV cache 耗尽
- 动态 block 数量和大小调整
- 基于显存容量估算最大并发序列数

### 4. PPScheduler（借鉴 Megatron-LM）
- 1F1B（one-forward-one-backward）调度
- Bubble ratio 公式: `(PP - 1) / (PP - 1 + M)`
- 每 rank 的 warmup/steady/cooldown 阶段跟踪

### 5. TPScheduler（借鉴 Megatron-LM）
- ColumnParallel / RowParallel 权重切分
- 每层 AllReduce 次数: 4（forward 2 + backward 2）
- 通信代价精确估算

### 6. ZEROStageManager（借鉴 DeepSpeed）
- ZeRO-1/2/3 四级分片策略
- 内存公式: Standard = N×(2+K)Ψ, ZeRO-3 = (K+3)Ψ
- 各阶段通信开销分析

### 7. WorkloadBalancer（借鉴 FlexFlow/Alpa）
- 按显存容量分配（capacity-based）
- 按计算速度分配（speed-based）
- 混合分配（hybrid）
- 不均衡度评分（CV = std/mean）

---

## 快速开始

```bash
# 运行所有演示
python scheduler_demo.py --demo all

# 查看可用的演示
python scheduler_demo.py --list

# 运行特定演示
python scheduler_demo.py --demo block        # BlockManager prefix caching
python scheduler_demo.py --demo batch        # Continuous batching 流程
python scheduler_demo.py --demo pp           # Pipeline parallelism 1F1B
python scheduler_demo.py --demo zero         # ZeRO 内存节省分析
python scheduler_demo.py --demo balancer     # 异构 GPU 负载均衡
python scheduler_demo.py --demo throughput   # 吞吐量对比
python scheduler_demo.py --demo hybrid       # 混合并行策略

# 运行测试
pytest test_scheduler.py -v

# 运行特定测试类
pytest test_scheduler.py::TestBlockManager -v
pytest test_scheduler.py::TestPPScheduler -v
pytest test_scheduler.py::TestZEROStageManager -v
pytest test_scheduler.py::TestSimulationEngine -v
```

---

## 并行策略速查

| 并行策略 | 全称 | 切分对象 | 通信模式 | 每步通信量 | 适合场景 |
|----------|------|----------|----------|-----------|----------|
| **DP** | Data Parallel | 数据 batch | AllReduce（梯度） | 模型参数 ×2 | 训练、batch 推理 |
| **TP** | Tensor Parallel | 模型权重 | AllReduce（每层 4 次） | high | 超大模型单层放不下 |
| **PP** | Pipeline Parallel | 模型层 | P2P send/recv | low | 深模型跨 GPU 分布 |
| **ZeRO-1** | Optimizer offload | Optimizer States | AllGather | low | 训练优化器显存 |
| **ZeRO-2** | + Gradient offload | Gradients | ReduceScatter | medium | 中等规模训练 |
| **ZeRO-3** | + Parameter offload | Parameters | AllGather 每层 | high | 超大模型训练 |

---

## Pipeline Bubble 公式

```
bubble_ratio = (PP_size - 1) / (PP_size - 1 + num_microbatches)

例如:
  PP=4, M=8:   bubble = 3/11 = 27.3%
  PP=4, M=32:  bubble = 3/35 = 8.6%
  PP=4, M=128: bubble = 3/131 = 2.3%

结论: M >> PP 时 bubble 可忽略
```

---

## 测试覆盖

```bash
pytest test_scheduler.py -v --tb=short
```

30+ 测试用例覆盖:
- BlockManager: 分配/释放、引用计数、prefix caching、reset
- Sequence: 生命周期、逻辑块数、完成判断
- Scheduler: 调度逻辑、preemption、队列满处理
- MemoryPlanner: watermark 策略、动态调整、并发估算
- PPScheduler: 1F1B 正确性、bubble ratio、rank 一致性
- TPScheduler: 权重切分、通信量估算
- ZEROStageManager: 各阶段内存节省、通信开销
- WorkloadBalancer: 容量/速度/混合分配、不均衡度
- HybridScheduler: 混合并行组合、通信开销
- SimulationEngine: 1000 请求压测、端到端生命周期

---

## 相关模块

- [07_cuda_streams_async](../07_cuda_streams_async/) - CUDA Stream 和异步执行
- [08_memory_management](../08_memory_management/) - GPU 内存管理
- [13_kernel_profile](../13_kernel_profile/) - 性能分析与诊断
