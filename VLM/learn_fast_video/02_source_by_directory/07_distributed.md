# distributed —— 分布式层

> 模块作用：分布式初始化、通信组管理、序列并行、FSDP、张量并行。是多 GPU 推理/训练的基础设施。

## 1. 模块结构

```
distributed/
├── parallel_state.py          # GroupCoordinator + 组管理（1233 行）
├── communication_op.py        # SP/TP 通信操作（158 行）
├── utils.py                   # padding/shard 工具（274 行）
└── device_communicators/
    ├── base_device_communicator.py  # 基类 + Autograd 函数（277 行）
    ├── cuda_communicator.py         # CUDA（PyNccl）
    ├── npu_communicator.py          # NPU（PyHccl）
    ├── cpu_communicator.py          # CPU（共享内存）
    ├── pynccl.py / pynccl_wrapper.py# NCCL 纯 Python 封装
    └── pyhccl.py / pyhccl_wrapper.py# HCCL 封装
```

（TP 线性层在 `fastvideo/layers/linear.py`，见 [`10_configs.md`](10_configs.md) 相邻的 layers 说明）

## 2. GroupCoordinator（parallel_state.py）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/distributed/parallel_state.py
关键类：GroupCoordinator (L117)
```

模块级全局变量存各组：`_WORLD`, `_TP`, `_SP`, `_DP`, `_NODE`。

`GroupCoordinator` 封装 PyTorch ProcessGroup，同时管理 CPU 组（gloo）和设备组（NCCL），暴露原语：
- `all_reduce`（L285）、`all_gather`（L314）、`shard`（L323，forward=slice/backward=all-gather）、`all_to_all_4D`（L343）、`broadcast`、`barrier`（用 CPU 组避免 NCCL barrier 问题）。

## 3. 初始化

```
maybe_init_distributed_environment_and_model_parallel(tp_size, sp_size)   # L893
  → init_distributed_environment()   # L723 创建 _WORLD, _NODE
  → initialize_model_parallel()      # L789 创建 _TP, _SP, _DP
  → 设置 torch device
```

分组（world_size=8, sp_size=4）：
| 组 | 方式 | 结果 |
|----|------|------|
| TP | 连续 | `[[0,1,2,3],[4,5,6,7]]` |
| SP | 连续 | `[[0,1,2,3],[4,5,6,7]]` |
| DP | 交错 | `[[0,4],[1,5],[2,6],[3,7]]` |

幂等：已初始化则校验一致后返回。调用位置：`worker/gpu_worker.py:69`（推理）、`train/entrypoint/train.py`（训练）。

## 4. 序列并行（SP）

```
源码位置：communication_op.py + device_communicators/base_device_communicator.py
```

`sequence_model_parallel_all_to_all_4D`（communication_op.py L28）→ `GroupCoordinator.all_to_all_4D` → `DistributedAutograd.AllToAll4D`（base L123）。

两种模式（attention 前后）：
```
前 (scatter=2, gather=1): [bs, shard_seq, hn, hd] → [bs, seq, shard_hn, hd]
后 (scatter=1, gather=2): [bs, seq, shard_hn, hd] → [bs, shard_seq, hn, hd]
```

`sequence_model_parallel_shard`（communication_op.py L64）：padding 对齐 → slice（forward）/ all-gather（backward）。

## 5. FSDP

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/loader/fsdp_load.py
```

- `maybe_load_fsdp_model`（L100）：meta 设备建模型 → `MixedPrecisionPolicy`（param_dtype=bf16, reduce_dtype=fp32）→ `DeviceMesh(replicate, shard)` → `shard_model` → 从 safetensors `distribute_tensor` 加载。
- `shard_model`（L219）：反向遍历模块树，对满足 `_fsdp_shard_conditions` 的模块 `fully_shard`（FSDP2）。
- `load_model_from_full_model_state_dict`（L312）：`distribute_tensor(full, mesh, placements)` 分发。

```python
device_mesh = init_device_mesh("cuda",
    mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
    mesh_dim_names=("replicate", "shard"))
```

MacOS 上禁用 FSDP。可用 `FASTVIDEO_FSDP2_AUTOWRAP` 按参数量自动分片。

## 6. TP 线性层（fastvideo/layers/linear.py）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/layers/linear.py（1066 行）
```

| 类 | 行号 | 切分 |
|----|------|------|
| `ReplicatedLinear` | L208 | 不切（复制），DiT/VAE 热路径 |
| `ColumnParallelLinear` | L344 | 输出维切分 |
| `RowParallelLinear` | L922 | 输入维切分 + all-reduce |
| `MergedColumnParallelLinear` | L478 | 多输出打包（MLP gate_up） |
| `QKVParallelLinear` | L674 | QKV 融合，支持 GQA |

## 7. 设备通信器

```
DeviceCommunicatorBase (base L196)
├── CudaCommunicator → PyNcclCommunicator
├── NpuCommunicator → PyHcclCommunicator
└── CpuCommunicator → 共享内存
```

所有原语 autograd-aware（`AllReduce`/`AllGather`/`Slice`/`AllToAll4D`）。

**PyNcclCommunicator**（pynccl.py）：纯 Python + ctypes 封装 NCCL，绕过 PyTorch（CUDA graph 兼容 + 版本灵活）。

## 8. 推理 vs 训练并行

| 维度 | 推理默认 | 训练 |
|------|---------|------|
| FSDP | 可选 | 必开（`training_mode=True`） |
| SP | `sp_size=num_gpus` | YAML 指定 |
| HSDP | `replicate=1, shard=num_gpus` | 常 `replicate=8, shard=1` |

## 9. 源码阅读重点
1. `parallel_state.py` 的 `initialize_model_parallel` 分组逻辑。
2. `base_device_communicator.py` 的 `AllToAll4D`（SP 精华）。
3. `fsdp_load.py` 的 `maybe_load_fsdp_model`。

## 10. 相关笔记
- 序列并行深入：[`04_knowledge_expansion/07_sequence_parallelism.md`](../04_knowledge_expansion/07_sequence_parallelism.md)
- FSDP 深入：[`04_knowledge_expansion/08_fsdp_and_distributed_training.md`](../04_knowledge_expansion/08_fsdp_and_distributed_training.md)
