# torch.save/load 序列化源码分析

> 源码: `torch/serialization.py` (2252 行) — `save`, `load`, `StreamingFile`
> C++ 后端: `torch/csrc/jit/serialization/` — Pickler/Unpickler

## 0. 一句话总览

`torch.save(obj, path)` 使用 Python pickle + zipfile 格式存储。`torch.load(path)` 反序列化。`weights_only=True` 安全模式只允许加载 tensor 和原始类型，防止 pickle 代码执行。`mmap=True` 通过 `torch.UntypedStorage.from_buffer` 实现零拷贝大模型加载。

---

## 一、`torch.save` 内部流程

### 1.1 文件格式

```
.pt / .pth 文件 = zip archive:
  ├─ archive/
  │   ├─ data.pkl           ← pickle 序列化的对象图
  │   ├─ version             ← PyTorch 版本号
  │   └─ data/               ← 独立存储的大 tensor (可选)
  │       ├─ 0
  │       ├─ 1
  │       └─ ...
```

### 1.2 `torch.save` 核心参数

| 参数 | 含义 |
|------|------|
| `_use_new_zipfile_serialization` | True=zip 格式 (推荐), False=legacy |
| `pickle_module` | 默认 `pickle`, 可以换成 `dill` |
| `pickle_protocol` | pickle 协议版本 (2/3/4/5), 最高=5 |

---

## 二、`torch.load` 核心参数

| 参数 | 含义 |
|------|------|
| `weights_only` | True=只 load tensor/dict/list/基本类型 (安全), False=完整 pickle (危险) |
| `mmap` | True=mmap 文件 → tensor 零拷贝加载 |
| `map_location` | 指定加载到的 device (如 `"cpu"`, `"cuda:0"`) |

### 2.1 `weights_only=True` 为什么安全

正常 `pickle.load()` 可以执行任意 Python 代码 (RCE 风险)。
`weights_only=True` 使用 `Unpickler(find_class=...)` 限制了可反序列化的类型 → 只能恢复 tensor, dict, list, int, float, str 等 → 没有代码执行风险。

### 2.2 `mmap=True` 原理

```python
# serialization.py (简化):
with open(path, 'rb') as f:
    # 内存映射文件 (mmap)
    storage = torch.UntypedStorage.from_file(f.name, shared=True, size=...)
    # 创建 tensor 但 data_ptr 直接指向 mmap 区域
    tensor = torch.tensor([], dtype=...).set_(storage, offset, size, stride)
```

零拷贝: tensor 的数据直接映射到文件, 不分配额外显存。适合加载大 checkpoint 但只使用部分 tensor。

---

## 三、关键常见场景

### 3.1 保存 checkpoint

```python
torch.save({
    "model": model.state_dict(),
    "opt": optimizer.state_dict(),
    "sched": scheduler.state_dict(),
    "epoch": epoch,
    "best_loss": best_loss,
}, "checkpoint.pt")
```

### 3.2 加载并迁移到指定 device

```python
ckpt = torch.load("model.pt", map_location="cuda:1", weights_only=True)
model.load_state_dict(ckpt["model"])
```

### 3.3 跨版本 load 的安全姿势

```python
ckpt = torch.load("model.pt", weights_only=True, mmap=True)
# mmap: 避免提前加载所有 tensor → 内存友好
# 然后逐个检查 key 并迁移
```

---

## 四、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `torch.save` | `torch/serialization.py` | — |
| `torch.load` | `torch/serialization.py` | — |
| `_load` (内部) | `torch/serialization.py` | — |
| `StreamingFile` | `torch/serialization.py` | — |
| Pickler (C++) | `torch/csrc/jit/serialization/pickle.cpp` | — |
| `UntypedStorage.from_file` (mmap) | `c10/core/Storage.h` | — |

---

## 五、实战常见坑点

### 1. 旧 torch 版本 save 的文件在新版本 load 失败
**原因**: pickle 协议 / tensor 格式 / ATen op schema 变化。
**解决**: `weights_only=True` 可以跨版本手动恢复 key; 或用 `map_location="cpu"` 先迁移。

### 2. load 时 OOM (尽管 batch_size 很小)
**原因**: checkpoint 中包含 optimizer state (可能 3× 参数量) → load 全部 tensor 到内存 → OOM。
**解决**: `mmap=True` 延迟加载; 或者只 load model state_dict 不 load optimizer。

### 3. `weights_only=False` 加载第三方 checkpoint → RCE
**原因**: pickle 可以执行 `os.system("rm -rf /")` 等任意 Python 代码。
**解决**: 永远用 `weights_only=True`; 如果是自己的 checkpoint 且包含非 tensor 状态, 确保来源可信。

### 4. mmap tensor 修改后未 persist
mmap tensor 直接映射文件，修改后会写回文件。但 Python 退出时 mmap 区域可能不 flush。
**解决**: 如果修改了 mmap tensor 且需要 persist, 显式调用 `tensor.storage()._flush()` 或 `del tensor` 强制 flush。
