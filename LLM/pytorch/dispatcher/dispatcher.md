# PyTorch Dispatcher 调度系统源码分析

> Python 端: `/home/ghr/code/pytorch/torch/library.py` (1905 行)
> C++ 核心: `aten/src/ATen/core/dispatch/Dispatcher.h` (948 行), `OperatorEntry.h` (336 行)
> Dispatch Key 枚举: `c10/core/DispatchKey.h` (730 行)

## 0. 一句话总览

PyTorch 的调度器是一个**全局单例的分发表**：每个算子维护从 `DispatchKey` → `KernelFunction` 的映射。当你调用 `torch.add(a, b)`，调度器从 tensor 参数提取 `DispatchKeySet`，查表找到最高优先级的 kernel，调用它。

---

## 一、三大核心概念

### 1.1 DispatchKey — 算子的"分路标记"

定义在 `c10/core/DispatchKey.h:136`，是一个 `uint16_t` 枚举，每个 operator 可以有多个 kernel，按 dispatch key 区分：

```
DispatchKey 分类:
  ├─ 后端 Key:   CPU, CUDA, HIP, XLA, MPS, Meta, ...
  ├─ 功能 Key:   Autograd, Autocast, Tracer, FuncTorchBatched, ...
  ├─ 别名 Key:   CompositeImplicitAutograd, CompositeExplicitAutograd, Autograd
  └─ 特殊 Key:   Python, BackendSelect, ADInplaceOrView, ...
```

### 1.2 DispatchKeySet — 多个 key 的位掩码

`c10/core/DispatchKeySet.h` 定义了 64 位位掩码。每个 tensor 持有自己的 `key_set()`，调度器对参数做位或（OR）得到最终的 key set，然后选取**最高优先级**的 key。

### 1.3 OperatorEntry — 每个算子的分发表

`aten/src/ATen/core/dispatch/OperatorEntry.h:232`:

```cpp
class OperatorEntry {
    std::array<KernelFunction, num_runtime_entries> dispatchTable_;  // 计算后的分发表
    ska::flat_hash_map<DispatchKey, std::list<AnnotatedKernel>> kernels_;  // 所有注册的 kernel
    DispatchKeyExtractor dispatchKeyExtractor_;  // 从参数提取 DispatchKeySet
};
```

`dispatchTable_` 是**预热态**的分发表。`dispatchTable_[idx]` 存储了该 runtime dispatch key 对应的 kernel，O(1) 查表。

---

## 二、完整调度流程（单次算子调用）

### Step 1: 从 tensor 参数提取 DispatchKeySet

```cpp
// DispatchKeyExtractor.h:186
template <class... Args>
DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const {
    auto ks = detail::multi_dispatch_key_set(args...);  // 所有 tensor 的 key_set() 的 OR
    return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
}
```

### Step 2: 合并 TLS (Thread Local State)

```cpp
// DispatchKeyExtractor.h:24
inline DispatchKeySet computeDispatchKeySet(DispatchKeySet ks, DispatchKeySet key_mask) {
    c10::impl::LocalDispatchKeySet local = c10::impl::tls_local_dispatch_key_set();
    // 包含的 key (如 Autocast) 加入，排除的 key (如关闭 Autograd) 移除
    return (((ks | local.included_) - local.excluded_) & key_mask);
}
```

这就是 `torch.no_grad()`、`torch.autocast()` 等上下文如何工作的 — 它们修改 TLS，从而影响 dispatch。

### Step 3: O(1) 查分发表

```cpp
// OperatorEntry.h:182
const KernelFunction& lookup(DispatchKeySet ks) const {
    const auto idx = ks.getDispatchTableIndexForDispatchKeySet();  // 最高优先级 key 的索引
    return dispatchTable_[idx];  // O(1) 返回 kernel
}
```

`getDispatchTableIndexForDispatchKeySet()` 从 key set 中找到**最高优先级**的 key，返回它在 `dispatchTable_` 中的索引。

### Step 4: 调用 kernel

```cpp
// Dispatcher.h:773
template <class Return, class... Args>
Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
    auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
        .template getDispatchKeySetUnboxed<Args...>(args...);
    const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}
```

---

## 三、Kernel 选择优先级算法

`OperatorEntry.cpp:352` `computeDispatchTableEntryWithDebug()`:

对于给定的 DispatchKey，按以下优先级选择 kernel：

```
1. 直接注册: kernels_[key] 非空 → 直接使用
2. CompositeExplicitAutogradNonFunctional 别名
3. CompositeExplicitAutograd 别名
4. CompositeImplicitAutograd 别名 (仅当无后端 kernel 时)
5. Autograd 别名
6. FuncTorchBatchedDecomposition 别名
7. 后端回退: dispatcher.backendFallbackKernels_[idx]
8. 缺失: 抛出 "no kernel for key" 错误
```

**关键**: `CompositeExplicitAutograd` kernel 对所有后端 key 都可见（作为 fallback），而 `CompositeImplicitAutograd` 只在没有后端特化 kernel 时才生效。

---

## 四、Python 端注册 API

### 4.1 `torch.library.define` — 声明算子

```python
# library.py:272
def define(self, schema, alias_analysis="", *, tags=()):
    """注册新算子 schema"""
    result = self.m.define(schema, alias_analysis, tuple(tags))
    #        ^^^^^^^^^^^^ 调用 C++ Dispatcher::registerDef()
```

### 4.2 `torch.library.impl` — 注册 kernel

```python
# library.py:438
def impl(self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=True):
    """为指定 dispatch key 注册 kernel 实现"""
    self.m.impl(name, dispatch_key, fn, with_keyset)
    #    ^^^^^^^^^ 调用 C++ Dispatcher::registerImpl()
```

### 4.3 装饰器 API

```python
@torch.library.impl("mylib::sin", "CPU")
def sin_cpu(x):
    return x.sin()

@torch.library.impl("mylib::sin", "CUDA")
def sin_cuda(x):
    return x.sin()  # CUDA kernel
```

`_device_type_to_key` (`library.py:954`) 将 `"cpu"` → `"CPU"`, `"cuda"` → `"CUDA"`。

---

## 五、TLS 与上下文管理器

`torch.no_grad()` / `torch.enable_grad()` 本质是操作 `Autograd` dispatch key 的排除/包含：

```python
# no_grad 排除 Autograd key → 所有算子跳过 autograd 包装
# enable_grad 包含 Autograd key（默认）
# autocast 包含 AutocastCUDA key → 算子经过 autocast policy 层
```

C++ 端通过 `c10::impl::tls_set_dispatch_key_excluded()` 修改 TLS。

---

## 六、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `Library` 类 | `torch/library.py` | 212 |
| `Library.define` | `torch/library.py` | 272 |
| `Library.impl` | `torch/library.py` | 438 |
| `torch.library.impl` 装饰器 | `torch/library.py` | 766 |
| `_device_type_to_key` | `torch/library.py` | 954 |
| `Dispatcher` 单例 | `aten/src/ATen/core/dispatch/Dispatcher.h` | 110 |
| `Dispatcher::call` | `aten/src/ATen/core/dispatch/Dispatcher.h` | 773 |
| `OperatorEntry` | `aten/src/ATen/core/dispatch/OperatorEntry.h` | 232 |
| `OperatorEntry::lookup` | `aten/src/ATen/core/dispatch/OperatorEntry.h` | 182 |
| `computeDispatchTableEntry` | `aten/src/ATen/core/dispatch/OperatorEntry.cpp` | 352 |
| `DispatchKey` 枚举 | `c10/core/DispatchKey.h` | 136 |
| `DispatchKeySet` | `c10/core/DispatchKeySet.h` | — |
| `DispatchKeyExtractor` | `aten/src/ATen/core/dispatch/DispatchKeyExtractor.h` | — |
| `computeDispatchKeySet` | `aten/src/ATen/core/dispatch/DispatchKeyExtractor.h` | 24 |

---

## 七、可借鉴的工程技巧

1. **查表替代 if-else 链**: 用 `dispatchTable_[idx]` O(1) 替代 `if key==A elif key==B...`，新增 key 只需更新表。

2. **TLS 做特性开关**: `torch.no_grad()` / `torch.autocast()` 通过改 TLS 实现零热路径开销的开关（不散落 if 判断）。

3. **位掩码做集合运算**: `DispatchKeySet` 用 64 位掩码，合并/过滤都是位运算，极快。

4. **Fallback 链**: `CompositeExplicitAutograd` → `CompositeImplicitAutograd` → `Autograd` → backend fallback，优先级链保证有兜底。

5. **Python/C++ 分层**: Library API 在 Python 侧做校验和 sugar，C++ 侧做性能关键路径。
