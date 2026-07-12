# c10::intrusive_ptr：侵入式引用计数

> 源码: `c10/util/intrusive_ptr.h`, `c10/core/TensorImpl.h`, `c10/core/StorageImpl.h`

## 0. 一句话总览

PyTorch 不使用 `std::shared_ptr`，而是用自定义的 `c10::intrusive_ptr`。它的引用计数直接嵌入目标对象内部（侵入式），避免了 shared_ptr 的双重分配（control block + object），性能更高，且支持从裸指针安全转换。

## 1. 最小例子

```python
import torch

# Tensor 内部使用 intrusive_ptr 管理 TensorImpl 生命周期
x = torch.randn(3)
y = x  # intrausive_ptr 的引用计数 +1（Python refcount 也 +1）

# Storage 也是 intrusive_ptr 管理
print(f"x use count: {x.storage()._use_count()}")

# 每次 detach/view 不会拷贝数据，只是增加引用计数
v = x.view(-1)
print(f"view use count: {v.storage()._use_count()}")  # 仍然是 1（shared storage）
```

## 2. 实战例子

### 2.1 引用计数观测

```python
import torch

# TensorImpl 和 Storage 都用 intrusive_ptr
x = torch.randn(1024)

# 观察 Storage 引用计数
s = x.storage()
print(f"Initial: use_count={s._use_count()}")

y = x.view(2, 512)
print(f"After view: use_count={s._use_count()} (shared storage, no copy)")

del x
print(f"After del x: use_count={s._use_count()} (still held by y)")

del y
# storage 引用计数归零 -> StorageImpl 析构 -> CUDA memory free
```

### 2.2 与 shared_ptr 的对比

```python
# PyTorch 选择 intrusive_ptr 的原因:
# 1. sizeof(intrusive_ptr<T>) == sizeof(T*)  (shared_ptr 是 2x 指针大小)
# 2. 不需要额外分配 control block
# 3. 可以从裸指针安全构造（refcount 在对象内部）
# 4. 支持弱引用 (WeakIntrusivePtr)

# 在 Python 端:
# Tensor lifecycle = Python refcount(CPython) + intrusive_ptr count(C++)
# Refcount 机制: _use_count() 查看 C++ 侧引用计数
```

### 2.3 内存泄漏排查

```python
import torch

# 循环引用导致内存泄漏?
# PyTorch 的 intrusive_ptr 不处理循环引用
# 但 Python GC 会处理 Python 层的循环引用

class LeakTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)

model = LeakTest()
x = torch.randn(128, 1024)

# 循环引用: output 持有 autograd graph -> 引用模型参数
# Python GC 会断开这个循环
# 但 C++ intrusive_ptr 的循环引用需要用 weak_ptr
```

## 3. 核心源码文件

```
c10/util/intrusive_ptr.h              # intrusive_ptr 实现
c10/util/WeakIntrusivePtr.h           # 弱引用
c10/core/TensorImpl.h                 # 使用 intrusive_ptr 管理的目标
c10/core/StorageImpl.h                # Storage 也使用 intrusive_ptr
```

## 4. 关键机制源码解读

### 4.1 侵入式引用计数的结构

```cpp
// c10/util/intrusive_ptr.h (简化)
template <typename T>
class intrusive_ptr {
    T* target_;
public:
    // 构造时 refcount++
    intrusive_ptr(T* target) : target_(target) {
        if (target_) intrusive_ptr_add_ref(target_);
    }
    // 析构时 refcount--
    ~intrusive_ptr() {
        if (target_) intrusive_ptr_release(target_);
    }
};

// TensorImpl 继承自 intrusive_ptr_target
class TensorImpl : public c10::intrusive_ptr_target {
    // refcount_ 成员在这个基类里
    mutable std::atomic<size_t> refcount_;
};
```

### 4.2 与 shared_ptr 的关键区别

| 特性 | shared_ptr | intrusive_ptr |
|------|-----------|---------------|
| 引用计数位置 | 外部 control block | 嵌入对象内部 |
| sizeof | 2 × sizeof(T*) | sizeof(T*) |
| 内存分配 | 对象 + control block | 仅对象 |
| 裸指针构造 | 危险(多重 control block) | 安全(refcount 在对象内) |

## 5. 和已有笔记的连接

```
01_tensor/        — Tensor 内部用 intrusive_ptr 管理 TensorImpl
29_memory_allocator/ — allocator 与 refcount 协作释放内存
40_design_patterns/  — intrusive_ptr 是 RAII + reference counting 的结合
```

## 6. 搜索关键词

```bash
rg -n "class intrusive_ptr_target" c10/util/intrusive_ptr.h
rg -n "intrusive_ptr_add_ref" c10/util/intrusive_ptr.h
rg -n "use_count" c10/util/intrusive_ptr.h
```
