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

**真实案例分析**：

**案例 1：Meta 用 FSDP 训练 LLaMA-65B — 从 DDP 到 FSDP 的跨越**
Meta 在训练 LLaMA-65B 时面临严峻挑战：如果用 DDP，每张 A100（80GB）需要存储完整模型副本（65B × 2 bytes = 130GB FP16），加上优化器状态（65B × 12 bytes = 780GB FP32），单卡根本无法运行。Meta 切换到 FSDP（ZeRO-3 等效），使用 256 张 A100 GPU，将参数、梯度和优化器状态均匀分片到所有 GPU。最终实现了 **MFU（Model FLOPs Utilization）57%**——这意味着超过一半的 GPU 算力真正用于矩阵乘法运算，而非通信等待或内存搬运。作为对比，同样规模下用 DDP 的 MFU 仅为 35%，差距来自 DDP 无法重叠 AllReduce 与计算。关键优化技术包括：`reshard_after_forward=True`（前向后立即释放 AllGather 参数）、`limit_all_gathers=True`（限制同时进行的 AllGather 数量防止网络拥塞）、梯度累积（累积 4 个 micro-batch 后再同步以减少通信频率）。

**案例 2：字节跳动用 Deep Gradient Compression 训练推荐模型 — 600× 通信压缩**
字节跳动的推荐模型参数量达数万亿，训练集群跨多个数据中心，网络带宽成为瓶颈。传统的 AllReduce 同步梯度，每个 step 需要传输数 TB 的梯度数据，在 10Gbps 以太网上通信时间占训练总时间的 60% 以上。字节采用 Deep Gradient Compression（DGC）技术：每个 worker 只传输梯度中绝对值最大的 0.1%（Top-k 稀疏化），未传输的小梯度通过 momentum correction 和残差累积机制在后续 step 补传。具体做法是——每个参数的梯度有一个 `residual` 缓冲区，未传输的小梯度不断累积，一旦累积到超过阈值就被纳入下一轮的 Top-k。此外，`momentum correction` 确保被延迟的梯度不会改变优化方向：对于未传输的参数，其动量值乘以衰减因子 $\beta$（而非重新计算），从而等效于"这些参数在当前 step 没有梯度更新"。最终效果：**通信量减少 600×，训练速度与未压缩方案持平，模型 AUC 指标无统计显著下降**。这个方案要求 warm-up 阶段（前 5-10 个 epoch）使用全量梯度通信，因为初始阶段梯度方向不稳定，压缩会严重损害收敛——这是一个血的教训。

**案例 3：Google 用 TPU Pod 训练 PaLM 540B — 3D 并行 + 跨数据中心**
Google 的 PaLM 540B 模型使用 6144 块 TPU v4 芯片，部署在两个 TPU Pod 上（每个 Pod 4096 芯片，实际使用了跨 Pod 连接）。并行策略：数据并行跨 Pod（每个 Pod 独立处理不同 batch），张量并行在单个 TPU 芯片内的 MXU 单元间完成（利用 TPU 的 2D 脉动阵列自动并行），流水线并行按 Transformer 层分割。由于 TPU v4 的互联带宽（ICI, Inter-Chip Interconnect）极高（~90 GB/s per chip），张量并行的通信开销被大幅降低。最终的 MFU 达到 46.2%，在 540B 参数规模下实现了 10 个月稳定训练，期间仅发生 3 次重大故障（均自动恢复）。Google 的经验是：**不要试图在训练启动前就找到完美配置**——训练过程中 GPU/TPU 可能会发生内存碎片化、ECC 错误率上升等软故障，需要有自动 checkpoint + 动态梯度累积调整的能力。

**案例 4：NVIDIA 用 Megatron-LM 训练 MT-NLG 530B — 3D 并行的工程极限**
NVIDIA 和 Microsoft 联合训练的 Megatron-Turing NLG 530B 是 3D 混合并行的经典案例。使用 280 个 DGX A100 节点（共 2240 张 A100 GPU），部署在 NVIDIA Selene 超级计算机上。并行维度分配：张量并行 8 路（单节点内，NVLink 600 GB/s），流水线并行 16 段（跨节点，InfiniBand HDR 200 GB/s），数据并行 35 路。这是一次对工程极限的挑战：流水线气泡率约为 $\frac{16-1}{m}$，需要 $m$ 至少为 160 才能将气泡控制在 10% 以下。NVIDIA 的解决方案是 1F1B（one-forward-one-backward）调度 + 交错流水线（interleaved pipeline）：每个 GPU 不是只负责一个 stage，而是负责多个非连续的 stage 块，从而在每个 GPU 上交替执行不同 micro-batch 的前向和反向，进一步压榨流水线的利用率。训练全程中，InfiniBand 网络利用率保持在 85% 以上，GPU 总可用率达到 99.2%（按 3 个月的训练周期计）。

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

### 生产环境 P0 级故障实录

7. **"FSDP ZeRO-3 AllGather 延迟在跨节点时雪崩"** — 这是很多团队踩过的最惨痛的坑。FSDP 在单机内（NVLink）的 AllGather 延迟约 10-50 μs，但跨节点（InfiniBand/RoCE）的 AllGather 延迟飙升至 200-500 μs。当模型有 100+ 层时，每层都需要 AllGather 参数，跨节点的延迟累积导致每个 step 的额外开销达到 **2-5 秒**。如果不使用 `prefetch`（预取下一层的参数），GPU 会频繁 stall 等待参数，吞吐量直接**腰斩 50%**。正确的做法是：启用 FSDP 的 `forward_prefetch`（在前向计算当前层时异步预取下一层的 AllGather 参数），并将 `backward_prefetch` 设为 `BACKWARD_PRE` 或 `BACKWARD_POST`。更重要的是——如果模型层数少（< 20 层）但每层计算量大（如 ViT 的大矩阵乘法），通信的相对开销较小，可以把 `limit_all_gathers` 设为 `False` 提高并发；但如果层数多（100+ 层），必须设为 `True` 限制同时进行的 AllGather 数量，否则网络拥塞反噬吞吐。

8. **"流水线并行 micro-batch 太少 → GPU 利用率只有 30%"** — 流水线并行中的气泡比例 $\frac{p-1}{m}$ 如果 $m$（micro-batch 数）太少，气泡会吞噬大部分 GPU 时间。一个真实的 P0 案例：某团队用 8 段流水线并行训练一个 96 层 Transformer，最初只设置了 8 个 micro-batch（每个 micro-batch 32 samples），气泡率高达 $\frac{8-1}{8} = 87.5\%$，GPU 利用率实际上不到 30%。后来调整为 64 个 micro-batch，气泡率降至 $\frac{7}{64} \approx 10.9\%$，GPU 利用率提升到 75%+。**经验法则：micro-batch 数量至少为 pipeline stage 数的 4-8 倍。**如果总 batch size 受限，可以使用梯度累积来弥补——每个 micro-batch 更小，但数量更多。

9. **"梯度压缩在 warm-up 阶段使用 → 模型不收敛"** — 梯度压缩（DGC/1-Bit SGD）依赖于"梯度方向稳定"的假设。在训练初始的 warm-up 阶段（前 3-10 个 epoch），模型参数在剧烈变化，梯度方向高度不稳定。如果此时就开始压缩梯度（特别是 1-Bit SGD 只保留了符号信息），被丢弃的小梯度信息永远无法恢复，模型很难甚至不可能收敛。已有的 P0 事故教训：字节跳动的推荐模型训练中，如果在 **warm-up 的前 5 个 epoch 使用了 DGC**，AUC 会从 0.82 直接掉到 0.75 且无法恢复。正确的做法是 **(1) warm-up 阶段使用全量梯度通信 (2) 逐步增加压缩率（从 100% 梯度 → 1% → 0.1%）(3) 在 loss 平台期再切换到最激进的压缩率**。

10. **"ZeRO-3 的 CPU Offload → 训练速度暴跌 10×"** — 当模型大到 GPU 内存即使分片后也装不下时，FSDP/ZeRO-3 提供了 CPU Offload 选项（将优化器状态或参数卸载到 CPU 内存）。但 CPU 与 GPU 的通信带宽仅为 PCIe 的 32 GB/s（双向），对比 GPU 内存带宽（HBM 2 TB/s+），差了 **60 倍以上**。一个 65B 参数模型的 ZeRO-3 CPU offload 训练中，每个 step 需要从 CPU 读取 ~30GB 参数，仅通信就耗时 ~1 秒，导致 step 时间从 ~500ms（纯 GPU）膨胀到 ~5 秒。**CPU Offload 是最后的救命稻草，不是性能优化手段**——应该先尝试梯度检查点（Gradient Checkpointing）、Flash Attention、混合精度等其他内存优化手段。只有在前述手段都用尽后，才启用 CPU Offload。

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

**Q6: 如何推导和计算 MFU（Model FLOPs Utilization）？为什么 LLaMA-65B 用 FSDP 能达到 57% MFU 而 DDP 只能到 35%？**（高频面试题，FAANG 级别）

A: MFU 的定义是：
$$\text{MFU} = \frac{\text{实际吞吐量 (tokens/s) × 理论 FLOPs/token}}{\text{GPU 峰值 FLOPS × GPU 数量}}$$

对于 Transformer 模型，每 token 的理论 FLOPs 可通过以下公式估算（仅考虑矩阵乘法）：
$$\text{FLOPs/token} \approx 2 \times (6 \times N_{\text{params}})$$

（因子 6 来自：前向 2× 矩阵乘法 + 反向 4× 矩阵乘法；因子 2 来自一次乘加算 2 FLOPs）

LLaMA-65B 用 DDP 只能到 35% MFU 的核心原因有三个（按影响从大到小排序）：

**第一，AllReduce 通信无法完全重叠**。DDP 的梯度 AllReduce 虽然在 backward 的 hook 中异步触发，但 (1) 最后一层的梯度同步无法与任何计算重叠——backward 完成后所有梯度必须同步完才能执行 optimizer.step()；(2) 在 256 张 GPU 上，Ring AllReduce 的延迟为 $2(N-1) \times K/B \approx$ 数百毫秒，而单层反向计算可能只需几十毫秒。即使有重叠，`通信时间 > 计算时间` 时 GPU 就会 stall 等待；(3) 多机跨节点时，InfiniBand 带宽（200 GB/s）远低于 NVLink 带宽（600 GB/s），跨节点通信延迟进一步加剧 stalling。

**第二，内存墙效应**。DDP 每张 GPU 存储完整的 65B × 16 bytes ≈ 1TB 的内存需求（含激活值），远超 A100 80GB。必须依赖极小的 micro-batch（如 batch_size=1 per GPU），导致 GPU 的 Tensor Core 利用率极低——batch size 越小，矩阵乘法越趋向 memory-bound 而非 compute-bound。

**第三，梯度累积引入的 bubble**。为模拟大 batch size，DDP 需要做梯度累积（如 16 步累积一次），但累积期间的 micro-batch 无法受益于 AllReduce 重叠设计，且 optimizer.step() 的批量参数更新也引入额外延迟。

FSDP 为什么能到 57%：(1) 参数分片使每卡内存需求降低到 1TB/256 ≈ 4GB，可以使用更大的 micro-batch，Tensor Core 利用率提升；(2) FSDP 的 AllGather（前向）和 Reduce-Scatter（反向）通信量虽比 DDP 的 AllReduce 略多，但 FSDP 通信粒度更细（per-layer），与计算的交错重叠更充分；(3) `forward_prefetch` 和 `limit_all_gathers` 等工程优化进一步压缩了通信 stall。

**Q7: ZeRO-3/FSDP 中 `forward_prefetch` 和 `backward_prefetch` 的通信调度策略是什么？如果模型有 100 层 Transformer，prefetch 如何减少通信 stall？**（系统设计面试）

A: FSDP 的 prefetch 机制是通信与计算重叠的核心技术，需要深入理解其状态机：

**1. 无 prefetch 时的执行流程（最差情况）**：
每层经历：`AllGather(params) → wait → forward() → AllGather(next_params) → wait → forward() → ...`。GPU 在 `wait` 期间完全空闲。

**2. `forward_prefetch=True`（默认）的工作机制**：
在前向传播计算第 $i$ 层时，FSDP 的 autograd hook 异步发起第 $i+1$ 层参数的 AllGather（使用单独的 CUDA stream）。如果第 $i$ 层的计算时间 ≥ 第 $i+1$ 层 AllGather 的通信时间，那么当计算到第 $i+1$ 层时参数已经就绪，实现了 **100% 通信隐藏**。

**3. `backward_prefetch` 有两个选项**：
- `BACKWARD_PRE`：在反向传播计算第 $i+1$ 层梯度时，预取第 $i$ 层的参数（因为反向传播是自顶向下的，第 $i$ 层在计算图中的位置在第 $i+1$ 层之后）。这要求 AllGather 时间 ≤ 单层反向传播时间。
- `BACKWARD_POST`：在反向传播计算完第 $i$ 层后，立即预取第 $i-1$ 层参数。这在计算时间不可预测时更安全，但重叠效率略低。

**4. 100 层 Transformer 的实际策略**：
对于 100 层模型，每层计算量相对较小（与 ViT 的单层大矩阵乘法不同），AllGather 通信时间可能大于单层计算时间。这意味着即使有 prefetch，GPU 仍然可能 stall。此时的关键配置是 `limit_all_gathers=True`——限制同时进行的 AllGather 请求数量（通常限制在 1-2 个），防止多个 AllGather 请求在网络上互相竞争导致整体吞吐下降。更进一步，可以将相邻的 2-4 层组成一个 FSDP unit（`auto_wrap_policy` 的 `transformer_auto_wrap_policy` 中将 `min_num_params` 设置得较大），从而减少 AllGather 的频率（代价是降低分片粒度，内存占用微增）。

**Q8: 如果你在训练过程中发现 GPU 利用率经常从 90% 跌落到 30% 又恢复，应该从哪些维度排查？**（综合系统诊断面试）

A: 这是一种典型的通信瓶颈导致的"锯齿状"利用率模式。排查维度按优先级排序：

**第一优先级——网络层**：(1) 检查 NVLink/InfiniBand 的带宽利用率（`nvidia-smi nvlink -g 0` 查看 NVLink 错误计数，`ib_read_bw` 测试 InfiniBand 有效带宽）；(2) 检查是否有 ECC 错误导致链路降级（NVLink 降速到 x8 → 带宽减半）；(3) 多机训练时检查是否有某台机器网络链路不稳定（该机器的梯度到达延迟导致全局 AllReduce 等待）。

**第二优先级——调度层**：(1) 检查 FSDP 的 `limit_all_gathers` 配置是否合理——如果不加限制，100+ 层的模型可能同时触发数十个 AllGather 请求，网络拥塞导致延迟飙升；(2) 检查 `reshard_after_forward` 和 prefetch 的组合——如果 `reshard_after_forward=True` 但未启用 prefetch，前向的反向都需要重新 AllGather，每次 AllGather 都会引入 stall；(3) PyTorch Profiler 查看 `cudaLaunchKernel` 和 `ncclAllGather` 的时间线——找出空白的 GPU 时间槽是哪个通信操作造成的。

**第三优先级——负载均衡层**：(1) 流水线并行中不同 stage 的负载不均衡——某个 GPU 的层数/计算量比其他 GPU 多，成为瓶颈；(2) 数据并行中数据加载不均匀——某个 worker 的 dataloader 慢导致其他 worker 等待；(3) 检查 GC（垃圾回收）或 CPU-GPU 同步（如 `.item()`, `.cpu()` 调用）是否在关键路径上阻塞。

**面试加分回答**：使用 `torch.distributed.monitored_barrier` 和 PyTorch Profiler 的分布式 tracing 功能（`torch.profiler.tensorboard_trace_handler`），可以在 TensorBoard 中看到所有 GPU 的 timeline 并排显示，直接定位到"谁在等谁"。在大规模训练中，"先定位再优化"比"盲调参数"重要 100 倍。

**Q9: 为什么 Megatron-LM 的 3D 并行中，TP（张量并行）通信量最大却被放在最内层（单机内）？设计背后的硬件拓扑逻辑是什么？**（系统架构面试）

A: 这是 3D 并行设计的核心 trade-off，理解这个就理解了大规模分布式训练的硬件拓扑设计哲学。

**通信量与通信频率的维度分析**：

| 并行策略 | 单次通信量 | 通信频率 | 总通信量（per step） | 对带宽要求 |
|---------|-----------|---------|---------------------|-----------|
| TP（张量并行） | $O(B \cdot S \cdot H)$ | 每层 2 次 AllReduce（前向）+ 2 次（反向）= 4×$L$ 次 | **极高** | **极高（>400 GB/s 才能不瓶颈）** |
| PP（流水线并行） | $O(B \cdot S \cdot H)$ | 每层 1 次 P2P 发送激活值（前向）+ 1 次 P2P 发送梯度（反向） | 中等 | 中（>50 GB/s 足够） |
| DP（数据并行） | $O(\vert\theta\vert)$ | 每 step 1 次 AllReduce | **高**（梯度总量大） | 中（>100 GB/s） |

**TP 放在最内层的核心原因不是通信量小，而是通信频率极高**：
1. TP 的 AllReduce 每层都发生（对于 96 层 Transformer，每 step 384 次 AllReduce！），每次 AllReduce 延迟即使只有 10 μs，累积延迟也达 3.8ms。如果在 InfiniBand（延迟 2-5 μs）上运行，总延迟为 384 × 50 μs ≈ 19ms，可能超过单层计算时间。
2. NVLink + NVSwitch 的环形延迟极低（~1 μs + wire time），且 8 GPU 间实现了全互联（all-to-all），AllReduce 的 Ring 通信可以经过最短路径。
3. **拓扑约束**：PP 放在跨节点层是因为 P2P 通信的发送/接收之间可以 pipeline（流水线本身的设计就考虑了通信延迟），而 TP 的 AllReduce 要求**所有 GPU 严格同步**——任何延迟都会阻塞整个 TP 组。

**实际设计决策树**：
- 第一步：确定 TP 大小 = 单节点 GPU 数（如 8）——因为 NVLink 是唯一能支撑 TP 通信需求的互联
- 第二步：确定 PP 大小 = 模型层数 / 单 GPU 可容纳层数（如 96 层 / 12 层每 GPU = 8）
- 第三步：DP 大小 = 总 GPU 数 / (TP × PP)，DP 跨剩余的所有节点

**面试加分点**：如果能提到 Megatron-LM 的 "Interleaved Pipeline" 概念（每个 GPU 管理多个非连续的 stage 块而非一个连续段），以及它与 TP + DP 的交互（interleaved 模式下 TP 通信频率不变但 PP 通信频率翻倍），说明你真正理解了 3D 并行的工程复杂度。

## 10. 本讲总结

分布式训练是现代大模型开发的基石。核心矛盾是：模型参数×12 bytes 的优化器内存需求远超单卡显存。解决路径有三条：

1. **数据并行**（DDP）：多卡存完整副本，AllReduce 同步梯度。内存不节省，但吞吐线性扩展。适用于模型能在单卡装下的场景。

2. **ZeRO/FSDP**：分片优化器状态（ZeRO-1）→ 梯度（ZeRO-2）→ 参数（ZeRO-3），实现内存的近线性扩展。这是当前大模型训练的标准方案。

3. **模型并行**：流水线并行（按层切分，适合跨节点）+ 张量并行（按矩阵维度切分，适合机内高速互联）。Megatron-LM 是 3D 并行的代表（数据+流水线+张量）。

Adam 的 12 bytes/参数是理解内存瓶颈的钥匙。将 10B 模型从 FP16 推理的 20GB 膨胀到训练的 160GB（含激活值），正是驱动 ZeRO 等技术的根本动力。下一讲将讨论更复杂的混合并行策略和通信压缩技术。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| FSDP 跨节点时必须开启 forward_prefetch 并设置 limit_all_gathers | 100+ 层模型跨 InfiniBand：无 prefetch 时每层 AllGather 延迟 200-500μs → 累积 2-5s per step，吞吐腰斩 50% | 跨节点训练速度比单机还慢，分布式投入完全无效 |
| 流水线并行 micro-batch 数量必须 ≥ pipeline stage 数的 4-8 倍 | 某团队 8 段 PP + 8 micro-batch：气泡率 87.5%，GPU 利用率 < 30%；切换到 64 micro-batch → 气泡率 10.9%，利用率 75%+ | 8 张 GPU 的流水线并行吞吐还不如 2 张 GPU 的数据并行 |
| 梯度压缩（DGC/1-Bit SGD）在 warm-up 阶段禁用，必须逐步递增压缩率 | 字节推荐模型教训：warm-up 前 5 epoch 使用 DGC → AUC 从 0.82 跌到 0.75 且无法恢复；正确做法是 0%→1%→0.1% 渐进 | 模型永久不收敛，必须回滚重训——浪费数天甚至数周的算力 |
| 3D 并行的进程网格配置必须与 GPU 拓扑对齐——TP 组必须在同一 NVLink 域内 | TP 组跨两节点 → NVLink 不可用 → 每层 AllReduce 延迟从 ~10μs 飙升到 ~500μs，96 层 × 4 次 × 500μs = 192ms per step 额外开销 | 训练慢 3-5x，GPU 大部分时间在等通信而非计算 |
| ZeRO-3 CPU Offload 是最后手段而非性能优化，应该先尝试 Gradient Checkpointing + Flash Attention | 65B 模型 CPU offload 后每个 step 需从 CPU 读取 ~30GB 参数 → 仅通信耗时 ~1s，step 时间从 500ms→5s（慢 10x） | 盲目启用 CPU offload 导致训练时间从 3 天膨胀到 30 天 |
| 大模型训练前必须做内存预算分析：parameters + gradients + optimizer + activations | 7B 模型 FP16 推理仅需 14GB，训练需求 = 14 + 14 + 84 + ~20（activations）= ~132GB——已超单 A100 80GB | 不上 ZeRO-3 就无法启动训练，白等数小时排队才发现 OOM |
| 永远不要在生产中使用 nn.DataParallel，必须用 DistributedDataParallel | DP 是单进程多线程（Python GIL 瓶颈），DDP 是多进程（NCCL 通信）；DDP 比 DP 快 2-5x | 训练速度比预期慢数倍，GPU 利用率低下，项目延期 |
