# Lecture 19: 分布式训练 I — 数据并行、ZeRO 与模型并行

## 1. 本讲核心问题

当模型大到单张 GPU 装不下（或数据多到单 GPU 训练太慢），就需要分布式训练。本讲回答：

1. **数据并行（DDP）如何工作？** AllReduce 通信、梯度同步与反向传播的重叠
2. **ZeRO（零冗余优化器）如何消除内存冗余？** ZeRO-1/2/3 的递进优化策略，以及 Adam 优化器状态为什么占用 12 bytes/参数
3. **模型并行如何切分模型？** 流水线并行（GPipe）和张量并行（Megatron-LM）

## 2. 通俗解释

**数据并行的直觉**：你有 8 个工人（GPU），每人拿一份不同的数据，各自独立算梯度。然后所有人把梯度汇集起来取平均，用平均梯度更新模型。这样模型每次能看到 8 倍的数据（batch size = 8 × per-GPU batch size）。问题是：每张 GPU 上都要存一份**完整的模型副本**——如果模型本身就要占 40GB，单卡根本装不下。

**ZeRO 的直觉**：8 个工人不需要每人手里拿一本完整的公司章程。ZeRO-1 把优化器状态（Adam 的 m 和 v）分给 8 个人，每人只存 1/8；ZeRO-2 连梯度也分了；ZeRO-3 连模型参数本身也分了——只在需要计算时临时通信获取。这就像图书馆的分布式书架：每人只保管一部分书，需要时互相借阅。

**Adam 优化器为什么占用 12 bytes/参数**（关键！）：每个参数需要 3 样东西：
1. 参数本身（FP16 = 2 bytes）
2. 一阶动量 $m$（FP32 = 4 bytes）
3. 二阶动量 $v$（FP32 = 4 bytes）
4. 额外：FP32 参数副本（用于数值精度 = 4 bytes）
总计 = 2 + 4 + 4 + 4 = 14 bytes（混合精度训练），或全 FP32 则为 4 + 4 + 4 = 12 bytes（纯 FP32）。以一个 7B 模型为例：7B × 12 = 84 GB，仅优化器状态就远超单卡显存。

**流水线并行的直觉**：把模型切成几段（如 4 段），每段放在不同的 GPU 上。数据像流水线一样流过：GPU0 处理完传给 GPU1，GPU1 传给 GPU2……这就像工厂流水线——每个人只做一部分工序。问题是：有"气泡"（bubble）——上游处理时下游在空等。

**张量并行的直觉**：不按层切，而是把**单层内部**的矩阵乘法切分。比如一个 4096×4096 的矩阵乘法，切成 4 个 1024×4096 的矩阵分别在 4 张 GPU 上算，然后汇总。这就像把一张大拼图分给 4 个人同时拼——粒度更细，但通信更多。

## 3. 关键公式

**AllReduce 通信量**（Ring AllReduce，最常用的实现）：
$$
T_{\text{comm}} = 2(N-1) \cdot \frac{K}{B}
$$
其中 $N$ 为 GPU 数，$K$ 为梯度总字节数，$B$ 为网络带宽（NVLink: 600 GB/s, InfiniBand: 200 GB/s）

**梯度同步（DDP 核心）**：
$$
\mathbf{g}_{\text{avg}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{g}_i
$$
每张 GPU 上的模型副本在 optimizer.step() 之前获得相同的平均梯度

**Adam 优化器状态内存分解**：
$$
M_{\text{Adam}} = M_{\text{params}} + M_{\text{gradients}} + M_{\text{optimizer}}
$$
$$
M_{\text{optimizer}}^{\text{Adam}} = \underbrace{|\theta| \times 4}_{\text{FP32 参数副本}} + \underbrace{|\theta| \times 4}_{\text{一阶动量 } m} + \underbrace{|\theta| \times 4}_{\text{二阶动量 } v} = 12|\theta|
$$

如果使用混合精度（FP16 参数 + FP32 优化器）：
$$
M_{\text{total}} = \underbrace{2|\theta|}_{\text{FP16 参数}} + \underbrace{2|\theta|}_{\text{FP16 梯度}} + \underbrace{12|\theta|}_{\text{FP32 优化器}} = 16|\theta|
$$

**ZeRO-1 内存节省**（只分片优化器状态）：
$$
M_{\text{optimizer per GPU}} = \frac{12|\theta|}{N}
$$

**ZeRO-3 内存节省**（分片参数 + 梯度 + 优化器）：
$$
M_{\text{model per GPU}} \approx \frac{16|\theta|}{N}
$$

**流水线并行气泡率**（Bubble Ratio）：
$$
\text{Bubble} = \frac{p-1}{m}
$$
其中 $p$ 为流水线段数（GPU 数），$m$ 为 microbatch 数量。$m \gg p$ 时气泡可忽略。

**Megatron-LM 张量并行通信量**：
$$
T_{\text{TP}} \propto B \times S \times H \quad \text{(每次前向/反向)}
$$
其中 $B$ 为 batch size，$S$ 为序列长度，$H$ 为隐藏维度。张量并行的通信频率高（每层都需要通信），适合高带宽互联（NVLink/NVSwitch）。

## 4. 公式背后的直觉

- **AllReduce 的 $2(N-1)$ 因子**：Ring AllReduce 分为两步：Scatter-Reduce（每张卡把 $\frac{K}{N}$ 数据传给邻居，共 $N-1$ 次传输）和 AllGather（同样 $N-1$ 次）。总传输量为 $2(N-1) \cdot \frac{K}{N}$，当 $N$ 较大时趋近于 $\frac{2K}{N}$。**关键在于**：传输量与 GPU 数量无关（趋近于常数 $2K$），所以 Ring AllReduce 具有良好的扩展性。

- **"12 bytes per parameter" 这个数字为什么重要**：想象你有一个 10B 参数的模型，FP16 推理只需 20GB 显存。但训练时 Adam 优化器需要 120GB 仅用于优化器状态，加上参数、梯度、激活值，轻松超过 200GB。**这解释了为什么大模型训练必须用分布式**——不是计算不够，是内存不够。

- **ZeRO 的三级递进**：ZeRO-1 解决"优化器状态冗余"（最明显的冗余，因为所有 GPU 存着完全相同的 $m$ 和 $v$），ZeRO-2 解决"梯度冗余"（AllReduce 之后所有 GPU 的梯度都相同），ZeRO-3 解决"参数冗余"（所有 GPU 存着相同的模型权重）。只有 ZeRO-3 真正实现了模型大小的近线性扩展。

- **流水线气泡的含义**：前向传播时，数据从 GPU0 流向 GPU3，GPU1/2/3 需要等待前序 GPU 完成。反向传播也是类似的等待。$\frac{p-1}{m}$ 的气泡率意味着：microbatch 越多，等待时间的占比越小。GPipe 用大量 microbatch 来摊销气泡，但会增加激活内存。

- **张量并行的通信/计算比**：张量并行中每次前向都需要 AllReduce（矩阵乘法的列/行并行后汇总），通信量 = $4 \times B \times S \times H$（前向两次 AllReduce，反向两次）。当 $B \times S \times H$ 很大时，通信成为瓶颈。这就是为什么张量并行一般在单机内（NVLink 高速互联）使用。

## 5. 工业界用途

| 技术 | 代表框架 | 典型规模 | 适用场景 |
|------|---------|---------|---------|
| DDP (数据并行) | PyTorch DDP | 8 GPU, 模型 < 10B | CV 模型训练、小规模 LLM 微调 |
| ZeRO-1/2 | DeepSpeed ZeRO-2 | 64 GPU, 模型 < 30B | 中等规模 LLM 训练 |
| ZeRO-3 (FSDP) | PyTorch FSDP, DeepSpeed ZeRO-3 | 256+ GPU, 模型 70B-175B | 大模型全参数训练 |
| GPipe (流水线) | DeepSpeed Pipeline | 16-64 GPU, 跨节点 | 极深模型（如 100+ 层 Transformer） |
| Megatron-LM (张量) | Megatron-LM, TransformerEngine | 8 GPU 单机内 | 单层计算密集型（如超大 FFN） |

**具体实践**：
- **LLaMA 系列（Meta）**：使用 FSDP（PyTorch Fully Sharded Data Parallel）在 2000+ A100 上训练
- **GPT-3（OpenAI）**：使用 Megatron-LM（张量并行 + 流水线并行 + 数据并行）
- **BLOOM（BigScience）**：使用 DeepSpeed ZeRO-3 在 384 张 A100 上训练 176B 模型
- **PyTorch 生态**：`torch.nn.parallel.DistributedDataParallel`（DDP） + `torch.distributed.fsdp.FullyShardedDataParallel`（FSDP）

## 6. PyTorch 实现思路

```python
# ====================== DDP (Data Parallel) ======================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_training_example(model, dataloader, rank, world_size):
    setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])  # 自动处理梯度同步

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for data, target in dataloader:
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # DDP 在 backward() 中自动触发 AllReduce 梯度同步
        # 通过 hook 注册在 parameter.grad 上
        optimizer.step()

# ====================== FSDP (ZeRO-3) ======================
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)

def fsdp_training_example(model, rank, world_size):
    setup(rank, world_size)

    # FSDP 配置：分片策略 + 混合精度
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,  # ZeRO-3
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        "cpu_offload": CPUOffload(offload_params=False),
    }

    model = FSDP(model, **fsdp_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for data, target in dataloader:
        # forward: FSDP 自动 all-gather 收集分片参数
        output = model(data)
        loss = F.cross_entropy(output, target)
        # backward: FSDP 自动 reduce-scatter 聚合分片梯度
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# ====================== Pipeline Parallelism ======================
# 使用 torch.distributed.pipelining (PyTorch 2.0+)
from torch.distributed.pipelining import pipeline, ScheduleGPipe

def pipeline_example(model, microbatch_size, chunks=4):
    # 将模型切分为 chunks 个阶段
    stage_modules = [model.embed, model.layers[:8], model.layers[8:16],
                     model.layers[16:24], model.layers[24:], model.lm_head]
    pipe = pipeline(stage_modules, microbatch_size)
    schedule = ScheduleGPipe(pipe, chunks=chunks)  # chunks = microbatch 数
    # schedule 自动处理前向/反向的交错调度

# ====================== Tensor Parallelism ======================
# Megatron-LM 风格：将 Linear 层切分
class ColumnParallelLinear(torch.nn.Module):
    """沿输出维度切分：Y = XW, W=[W1|W2|...|Wn]"""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.out_per_gpu = out_features // world_size
        self.weight = nn.Parameter(torch.randn(in_features, self.out_per_gpu))
        # 每张 GPU 只存 out_features/world_size 列

    def forward(self, x):
        # 输入 x 在所有 GPU 上相同（由前一层 RowParallelLinear 保证）
        local_out = F.linear(x, self.weight)
        # AllGather 收集所有 GPU 的输出（拼接）
        return dist.all_gather(local_out)  # 沿最后一维拼接

class RowParallelLinear(torch.nn.Module):
    """沿输入维度切分：Y = XW, X=[X1|X2|...|Xn]"""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_per_gpu = in_features // world_size
        self.weight = nn.Parameter(torch.randn(self.in_per_gpu, out_features))

    def forward(self, x):
        # x 的输入维度已被前一层沿列切分，每张 GPU 只处理一部分
        local_out = F.linear(x, self.weight)
        # AllReduce 求和所有 GPU 的部分结果
        dist.all_reduce(local_out)  # SUM reduction
        return local_out

# ====================== 内存分析 ======================
def print_memory_breakdown(model, optimizer):
    total_params = sum(p.numel() for p in model.parameters())
    param_mem = total_params * 2  # FP16
    grad_mem = total_params * 2   # FP16
    optimizer_mem = total_params * 12  # Adam FP32 states
    total = param_mem + grad_mem + optimizer_mem

    print(f"Parameters: {total_params/1e9:.1f}B")
    print(f"Model (FP16): {param_mem/1e9:.2f} GB")
    print(f"Gradients (FP16): {grad_mem/1e9:.2f} GB")
    print(f"Optimizer (FP32): {optimizer_mem/1e9:.2f} GB")
    print(f"Total (no activations): {total/1e9:.2f} GB")
    # 注意：激活值内存未计算，在长序列训练中可达总内存的 30-50%
```

## 7. TinyML / Edge AI 部署意义

分布式训练似乎与 TinyML 的"小"背道而驰，但有几个重要连接点：

- **大模型 → 小模型的训练管道**：端侧部署的 TinyML 模型通常是通过在云端用分布式训练大模型，然后蒸馏/剪枝/量化得到小模型。分布式训练是实现"大模型教师"的前提。
- **联邦学习**：本质上是一种特殊的数据并行——数据分布在设备上，梯度在服务器聚合（FedAvg）。理解 DDP 的 AllReduce 是理解联邦学习聚合机制的基础。
- **ZeRO 思想对边缘内存的启发**：ZeRO 的分片策略启发了模型在边缘设备上的内存优化——例如将模型参数的加载与计算流水线化，在有限 RAM 的设备上运行稍大的模型。
- **通信压缩（下一讲）**：分布式训练的梯度压缩技术（1-Bit SGD, Deep Gradient Compression）直接适用于联邦学习中的上行带宽节省——边缘设备上传压缩梯度到服务器。
- **混合精度训练**：FP16/BF16 训练的工程经验直接指导了边缘端 INT8 推理的量化和校准策略设计。

## 8. 常见误区

1. **"DDP 就是 DataParallel 的改进版"** — 不准确。`DataParallel`（DP）是单进程多线程，使用 Python GIL 存在严重瓶颈；`DistributedDataParallel`（DDP）是多进程，每个 GPU 独立运行一个 Python 进程，通过 NCCL 通信，性能远超 DP。**永远不要在生产环境中使用 DP**。

2. **"ZeRO 是 DeepSpeed 独有的"** — 不准确。PyTorch 的 `FullyShardedDataParallel`（FSDP）实现了相同的思想（分片参数/梯度/优化器）。算法核心是公开的（ZeRO 论文），DeepSpeed 和 FSDP 都是其工程实现。

3. **"通信和计算完全重叠就可以隐藏延迟"** — DDP 的梯度 AllReduce 确实可以在 backward() 过程中异步启动（通过 `grad_hook`），但只有梯度通信与后续层的反向计算重叠，最后一层的梯度同步无法隐藏。此外 AllReduce 本身也有延迟，当梯度很大时通信时间 > 计算时间。

4. **"流水线并行总是高效的"** — 冷启动（warm-up）和清空（flush）阶段存在严重的气泡（bubble），特别是 pipeline stage 数量较多时。Gpipe 的 bubble 比例为 $\frac{p-1}{m}$，需要 $m \gg p$ 才能高效。1F1B（one-forward-one-backward）调度可以减少峰值激活内存，但不减少气泡。

5. **"混合精度训练只要把参数转成 FP16 就行"** — 需要三个关键组件：(1) FP32 优化器状态的主副本 (2) 损失缩放（loss scaling）防止梯度下溢 (3) 特定算子保持 FP32（如 softmax, layer norm）。缺少任何一个都会导致训练不稳定。

6. **"FSDP 比 DDP 慢很多"** — FSDP 在前向和反向时需要 AllGather 参数，引入了额外通信。但这是**内存换通信**的权衡——模型大到 DDP 跑不了时，FSDP 是唯一选择。而且 FSDP 的通信可以与计算重叠。

## 9. 面试问题

**Q1: 为什么 Adam 优化器每个参数占用 12 bytes？（高频面试题）**
A: Adam 需要存储三个 FP32 量：(1) 一阶动量估计 $m$（4 bytes）(2) 二阶动量估计 $v$（4 bytes）(3) FP32 参数主副本（4 bytes，用于混合精度训练的数值稳定性）= 12 bytes per parameter。如果是全 FP32，参数本身也是 4 bytes，总计 16 bytes。对于 10B 参数模型，仅优化器就需要 120GB 显存。

**Q2: ZeRO-1、ZeRO-2、ZeRO-3 的区别？**
A: ZeRO-1 只分片**优化器状态**（$m$, $v$），内存节省为 $\frac{1}{N}$（$N$ 为 GPU 数）。ZeRO-2 额外分片**梯度**，进一步节省。ZeRO-3 分片**参数本身**，前向时需要 AllGather 获取参数，反向后释放。ZeRO-3 实现内存的完全线性扩展，但通信量最大。

**Q3: DDP 中的梯度同步是在什么时候发生的？**
A: 在 `loss.backward()` 执行期间，DDP 通过注册在每个 `parameter.grad` 上的 hook 自动触发异步 AllReduce。梯度一旦计算完成，通信就立即开始，与后续层的反向计算重叠。所有梯度同步完成后，`optimizer.step()` 才能执行。

**Q4: 流水线并行和张量并行的区别和适用场景？**
A: 流水线并行按**层**切分（GPU0 放前几层，GPU1 放中间几层），通信量小（只传激活值），但存在气泡。适合跨节点部署（低带宽）。张量并行按**矩阵维度**切分（切分单层的权重矩阵），所有 GPU 同时工作无气泡，但每层都需要 AllReduce，通信量大。适合单机内高带宽互联（NVLink/NVSwitch）。

**Q5: FSDP 的 `reshard_after_forward` 参数有什么作用？**
A: `reshard_after_forward=True`（默认）：前向完成后立即释放 AllGather 收集的参数，在反向需要时再次 AllGather。节省峰值内存，但增加通信。`False`：保留参数直到反向完成，减少通信但增加内存。通常在内存紧张时设为 True。

## 10. 本讲总结

分布式训练是现代大模型开发的基石。核心矛盾是：模型参数×12 bytes 的优化器内存需求远超单卡显存。解决路径有三条：

1. **数据并行**（DDP）：多卡存完整副本，AllReduce 同步梯度。内存不节省，但吞吐线性扩展。适用于模型能在单卡装下的场景。

2. **ZeRO/FSDP**：分片优化器状态（ZeRO-1）→ 梯度（ZeRO-2）→ 参数（ZeRO-3），实现内存的近线性扩展。这是当前大模型训练的标准方案。

3. **模型并行**：流水线并行（按层切分，适合跨节点）+ 张量并行（按矩阵维度切分，适合机内高速互联）。Megatron-LM 是 3D 并行的代表（数据+流水线+张量）。

Adam 的 12 bytes/参数是理解内存瓶颈的钥匙。将 10B 模型从 FP16 推理的 20GB 膨胀到训练的 160GB（含激活值），正是驱动 ZeRO 等技术的根本动力。下一讲将讨论更复杂的混合并行策略和通信压缩技术。
