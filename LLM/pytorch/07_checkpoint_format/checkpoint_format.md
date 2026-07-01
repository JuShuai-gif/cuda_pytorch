# Checkpoint 格式、state_dict 与 SafeTensors 边界

> 序列化: `torch/serialization.py`
> Module state: `torch/nn/modules/module.py`
> Optimizer state: `torch/optim/optimizer.py`

## 0. 一句话总览

`torch.save` 默认使用 Python pickle + zip 容器格式保存对象。推荐只保存 `state_dict`（不保存完整模型）以避免安全风险和兼容性问题。LLM 场景中，tied weights、sharded checkpoint、dtype/device 转换是常见的坑点。SafeTensors 提供了一种只保存 tensor 数据、不包含任意 Python 对象的格式，适合安全分发和零开销加载。

## 1. 最小例子

```python
import torch

model = torch.nn.Linear(4, 4)
optim = torch.optim.AdamW(model.parameters())

ckpt = {
    "model": model.state_dict(),
    "optimizer": optim.state_dict(),
    "step": 10,
}

torch.save(ckpt, "train.pt")

# 加载
ckpt = torch.load("train.pt", weights_only=True)
model.load_state_dict(ckpt["model"])
optim.load_state_dict(ckpt["optimizer"])
```

SafeTensors 用法:

```python
from safetensors.torch import save_file, load_file

# 保存 state_dict 为 safetensors
save_file(model.state_dict(), "model.safetensors")

# 加载
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
```

## 1.5 实战例子

### 1.5.1 恢复 LLM Tied Weights Checkpoint

许多 LLM（如 GPT-2、LLaMA）使用 tied embedding（embedding 和 lm_head 共享权重），加载时需特殊处理：

```python
import torch

class LLMWithTiedWeights(torch.nn.Module):
    def __init__(self, vocab_size=32000, hidden=4096):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)
        # 权重绑定: lm_head 复用 embed 的权重
        self.lm_head.weight = self.embed.weight

    def forward(self, x):
        return self.lm_head(self.embed(x))

model = LLMWithTiedWeights()

# 保存 state_dict - 两个 key 指向同一 tensor
sd = model.state_dict()
print("embed.weight is lm_head.weight:",
      sd["embed.weight"] is sd["lm_head.weight"])
# 默认: False (state_dict 会复制数据)

# 正确的保存方式: 保存完整 state_dict
torch.save({"model": model.state_dict()}, "llm_ckpt.pt")

# 加载时: 需要重建 tied weights 关系
ckpt = torch.load("llm_ckpt.pt", weights_only=True)
model.load_state_dict(ckpt["model"], strict=False)
# 重新建立绑定
model.lm_head.weight = model.embed.weight
print("Tied weights restored:", model.lm_head.weight is model.embed.weight)
```

### 1.5.2 FSDP Sharded Checkpoint 的保存与加载

FSDP 训练的 checkpoint 分布在多个 rank 上，需要用分布式 checkpoint API：

```python
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    FileSystemReader, FileSystemWriter, load_state_dict, save_state_dict
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

dist.init_process_group("nccl")
model = FSDP(torch.nn.Linear(1024, 1024).cuda())

# 保存: 每个 rank 只保存自己的 shard
state_dict = model.state_dict()
writer = FileSystemWriter("/tmp/fsdp_ckpt")
save_state_dict(state_dict, writer)

# 加载: 自动根据当前 world_size 重新分片
model.load_state_dict(
    torch.distributed.checkpoint.load_state_dict(
        model.state_dict(),
        FileSystemReader("/tmp/fsdp_ckpt")
    )
)
# 注意: 加载时 world_size 必须与保存时一致, 否则 shard 不匹配
# 如果 world_size 变化, 需要先 consolidate 为单文件
```

### 1.5.3 SafeTensors + JSON 元数据双文件方案

在 HuggingFace 模型分发中，常用 safetensors 存 tensor + JSON 存元数据的组合：

```python
import json
from safetensors.torch import save_file, load_file
import torch

model = torch.nn.Linear(4, 4)
optim = torch.optim.AdamW(model.parameters())

# 保存: tensor 走 safetensors, 非 tensor 走 JSON
save_file(model.state_dict(), "model.safetensors")

metadata = {
    "optimizer_state": {
        "step": 10,
        # Adam 的 exp_avg/exp_avg_sq 是 tensor, 也存 safetensors
    },
    "config": {
        "arch": "Linear",
        "in_features": 4,
        "out_features": 4,
    },
    "training_info": {
        "loss": 0.023,
        "epoch": 5,
    }
}

# 将优化器 tensor state 也存 safetensors
optim_sd = optim.state_dict()
optim_tensors = {}
for param_id, param_state in optim_sd["state"].items():
    for k, v in param_state.items():
        if torch.is_tensor(v):
            optim_tensors[f"optim/{param_id}/{k}"] = v

save_file(optim_tensors, "optim.safetensors")

# 非 tensor 元数据存 JSON
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Files created:")
print("  model.safetensors  - 模型权重")
print("  optim.safetensors  - 优化器 tensor state")
print("  metadata.json      - 训练元数据")
```

## 2. 从 Python API 到源码的调用链

```
torch.save(ckpt, "train.pt")
  -> torch/serialization.py: save()
  -> zipfile.ZipFile 包装
  -> pickle.dump() 序列化对象
  -> 写入 zip 中的 data.pkl

torch.load("train.pt")
  -> torch/serialization.py: load()
  -> zipfile.ZipFile 打开
  -> pickle.load() 反序列化
  -> 安全检查 (weights_only=True 时限制 global 类型)

model.state_dict()
  -> torch/nn/modules/module.py: Module.state_dict()
  -> 遍历所有 Parameter 和 persistent buffer
  -> 返回 OrderedDict[str, Tensor]

optim.state_dict()
  -> torch/optim/optimizer.py: Optimizer.state_dict()
  -> 按 param_groups 和 state (per-parameter) 组织
  -> state 键是 parameter id (内存地址)
```

## 3. 核心源码文件

```
torch/serialization.py                           # torch.save / torch.load
torch/nn/modules/module.py                       # Module.state_dict / load_state_dict
torch/optim/optimizer.py                         # Optimizer.state_dict / load_state_dict
torch/distributed/checkpoint/                    # 分布式 checkpoint (PiPPy/FSDP)
torch/distributed/_shard/checkpointer/           # Sharded checkpoint
torch/distributed/checkpoint/planner.py          # 分片规划器
```

## 4. 关键机制源码解读

### 4.1 torch.save 是 pickle/zip 语义

PyTorch 的默认保存格式是 **zip 文件中包含 pickle 数据**：

```
train.pt (zip file)
  ├── archive/data.pkl     # pickle 序列化后的对象
  ├── archive/data/<index> # tensor 数据（可选，大 tensor 单独存储）
  └── archive/version      # 版本号
```

`torch.save` 对 tensors 有特殊处理：大 tensors 会以原始二进制格式写入 zip 条目（mmap 友好），而非嵌入 pickle 流，使加载更快、支持 mmap。

**安全风险**：pickle 可以执行任意 Python 代码。恶意 checkpoint 可以在加载时执行系统命令。

**长期兼容风险**：pickle 依赖于类的定义路径。重构后类路径变了，旧 checkpoint 就无法加载。

```python
# 安全加载（PyTorch >= 2.0 推荐）
ckpt = torch.load("train.pt", weights_only=True)
```

### 4.2 为什么推荐只保存 state_dict

| 方式 | 优点 | 缺点 |
|------|------|------|
| `torch.save(model)` | 简单 | 不安全、类路径绑定、大文件 |
| `torch.save(model.state_dict())` | 安全、跨版本兼容 | 不保存模型结构 |

保存 `state_dict` 后，可以自由改变模型类定义（加层、改超参），只要 tensor 名称和形状匹配，就能加载。

### 4.3 Module.state_dict() 的边界

```python
# torch/nn/modules/module.py (简化)
def state_dict(self, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.data
    for name, buf in self._buffers.items():
        if buf is not None and name not in self._non_persistent_buffers_set:
            destination[prefix + name] = buf
    for name, child in self._children.items():
        child.state_dict(destination, prefix + name + '.', keep_vars)
    return destination
```

- **只保存 `_parameters` 和 `_buffers`**，不保存 `_modules` 结构本身
- **non_persistent buffer**（如 BatchNorm 的 `num_batches_tracked` 默认持久化）
- 注册 buffer 时 `persistent=False` 可排除

### 4.4 Optimizer.state_dict() 的 per-parameter state

```python
# torch/optim/optimizer.py (简化)
def state_dict(self):
    return {
        'state': {
            id(p): {
                'step': state['step'],
                'exp_avg': state['exp_avg'],
                'exp_avg_sq': state['exp_avg_sq'],
            }
            for p, state in self.state.items()
        },
        'param_groups': self.param_groups,
    }
```

Optimizer state 以 **parameter id（内存地址）** 为 key。当模型结构变化时，parameter id 改变，optimizer state 无法恢复。

### 4.5 SafeTensors 的边界

SafeTensors 是一种**只存 tensor 数据**的格式：

| 特性 | torch.save | SafeTensors |
|------|-----------|-------------|
| 序列化方式 | pickle | 纯 tensor 数据 |
| 可执行任意代码 | 是 | 否 |
| 零拷贝加载 | 有限支持 | 支持（mmap） |
| 存储元数据 | 任意 Python 对象 | 仅 tensor shape/dtype |
| 文件大小 | 类似（tensor 数据同样） | 类似 |
| 格式规范 | 内部实现 | 开放规范 |

SafeTensors 不适合保存 optimizer state 中的标量（如 step 计数器）等非 tensor 数据，需要另配一个 JSON 元数据文件。

### 4.6 LLM 场景特殊处理

**Tied weights**: 权重绑定的 embedding/lm_head 在 `state_dict` 中是同一个 tensor 的两个条目。

```python
# 保存时两者内容相同
{"model.embedding.weight": ..., "model.lm_head.weight": ...}
# 加载时需特殊处理
```

**Sharded checkpoint**: FSDP/DeepSpeed 等并行策略下，每个 rank 只保存自己的 shard。`torch/distributed/checkpoint` 提供了全局 planner 来合并 shards。

**dtype/device 转换**: `state_dict` 中 tensor 有固定的 dtype 和 device。加载时需注意：

```python
# 加载到不同 device/dtype
state_dict = torch.load("ckpt.pt", map_location="cuda:0", weights_only=True)
for k, v in state_dict.items():
    state_dict[k] = v.to(dtype=torch.bfloat16)
model.load_state_dict(state_dict)
```

## 5. 和已有笔记的连接

```
serialization/      — torch.save/torch.load 序列化机制
module/             — Module.state_dict 的组织方式
optimizer/          — Optimizer.state_dict 的 per-parameter state
checkpoint/         — Activation checkpoint（不同于持久化 checkpoint）
distributed_techniques/ — FSDP 等分布式 checkpoint
```

## 6. 常见坑点

- **`torch.load(weights_only=True)` 可能和旧版 checkpoint 不兼容**，因为限制了可以反序列化的类型。
- **Optimizer state 按 parameter id 保存**，模型结构变化后无法恢复 optimizer state。
- **`torch.save(model)` 保存了模型类定义路径**，重构代码后加载失败。
- **SafeTensors 不支持保存非 tensor 对象**（如 Adam 的 step 计数器）。
- **Tied weights 加载时需要 `strict=False`** 或手动处理重复条目。
- **Sharded checkpoint 加载时必须使用正确的 world_size**，否则 shard 不匹配。
- **torch.save 对大 tensor 自动使用 zip 内独立存储**，小 tensor 嵌入 pickle 流。

## 7. 阅读源码时建议搜索的关键词

```bash
# torch.save 主逻辑
rg -n "def save" torch/serialization.py

# torch.load 主逻辑
rg -n "def load" torch/serialization.py

# Module.state_dict
rg -n "def state_dict" torch/nn/modules/module.py

# Optimizer.state_dict
rg -n "def state_dict" torch/optim/optimizer.py

# weights_only 安全检查
rg -n "weights_only|Unpickler" torch/serialization.py

# 分布式 checkpoint
rg -n "load_state_dict|save_state_dict" torch/distributed/checkpoint/
```
