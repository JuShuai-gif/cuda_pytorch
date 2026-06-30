# Tensor 内存模型与视图机制源码分析

> Python 端: `/home/ghr/code/pytorch/torch/_tensor.py` (1725 行); `class Tensor(torch._C.TensorBase)` (:102)
> C++ 核心: `aten/src/ATen/core/TensorBase.h` (TensorBase 类), `c10/core/TensorImpl.h` (底层实现)
> 算子定义: `aten/src/ATen/native/native_functions.yaml`

## 0. 一句话总览

PyTorch Tensor = `TensorImpl`（存储 metadata: sizes, strides, storage, autograd_meta） + `Storage`（实际数据块）。`view`/`reshape` 只改 metadata 不改数据，`contiguous` 拷贝数据使 stride 与 size 对齐。

---

## 一、Tensor 的三层结构

```
Python Tensor (torch._tensor.py:102)
  └─ torch._C.TensorBase  (pybind11 绑定)
       └─ c10::TensorImpl  (c10/core/TensorImpl.h)
            ├─ sizes_and_strides_  (c10/core/impl/SizesAndStrides.h)
            │    ├─ sizes_[]        — 每个维度的元素个数
            │    └─ strides_[]      — 每个维度的步幅（元素数，非字节数）
            ├─ storage_            — c10::Storage (引用计数的数据块)
            └─ autograd_meta_      — 梯度、grad_fn、requires_grad
```

**关键认识**: Tensor 只是一个 **metadata 壳** + **共享数据块的指针**。`view`、`slice`、`transpose` 都不拷贝数据，只创建新的 TensorImpl metadata 指向同一个 `Storage`。

---

## 二、`view` vs `reshape` vs `contiguous`

### 2.1 `view` — 零拷贝，要求满足 stride 约束

C++ 声明 (`native_functions.yaml:8167`):

```yaml
- func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
```

实现路径: `aten/src/ATen/native/TensorShape.cpp` → `_reshape_alias`。

**规则**: `view` 要求新 shape 的 strides 与原始 strides 兼容（即存在一组合理的 strides 使给定的 shape 能与原始 storage 对齐）。如果 `self.is_contiguous()` → 任意 shape 都可以。如果非 contiguous（如 transpose 后），只有特定 shape 可以。

### 2.2 `reshape` — 先尝试 view，失败则拷贝

声明 (`native_functions.yaml:4989`):

```yaml
- func: reshape(Tensor self, SymInt[] shape) -> Tensor
  dispatch:
    CompositeImplicitAutograd: reshape_symint
```

实现 (`aten/src/ATen/native/Reshape.cpp`):

```cpp
// reshape_symint 逻辑:
try:
    return self.view(shape)    // 尝试零拷贝
catch:
    return self.clone().view(shape)  // 拷贝后再 view
```

`reshape` = `view` 的安全版本：能 view 就零拷贝，不行就自动拷贝。

### 2.3 `contiguous` — 强制内存连续

C++ 声明 (`aten/src/ATen/core/TensorBase.h:135`):

```cpp
TensorBase contiguous(MemoryFormat memory_format=MemoryFormat::Contiguous) const {
    if (is_contiguous_or_false(memory_format)) {
        return *this;                          // 已经 contiguous，零成本
    } else {
        return __dispatch_contiguous(memory_format);  // 拷贝重排
    }
}
```

**判断 contiguous 的条件** (`TensorImpl.h`):

```
stride[i] == stride[i+1] * size[i+1]   (for all i)
```

即步幅从最后一维开始依次等于下一维的元素数：对于 shape `[3, 4, 5]`，contiguous strides 是 `[20, 5, 1]`。

---

## 三、`stride` 与 `is_contiguous` 的计算

### 3.1 `stride(dim)` (`TensorBase.h:190-195`)

```cpp
int64_t stride(int64_t dim) const {
    const auto strides = this->strides();          // 返回 IntArrayRef
    return strides[c10::maybe_wrap_dim(dim, ...)]; // 支持负索引
}
```

`strides()` 返回 `impl_->strides()`，后者直接从 `TensorImpl` 的 `SizesAndStrides` 读取。

### 3.2 `is_contiguous` (`TensorBase.h:268-270`)

```cpp
bool is_contiguous(at::MemoryFormat memory_format=...) const {
    return impl_->is_contiguous(memory_format);
}
```

底层 `TensorImpl::is_contiguous()` 遍历 strides 验证连续性。

---

## 四、`as_strided` — 任意 stride 视图的底层原语

```yaml
# native_functions.yaml:930
- func: as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
```

`as_strided` 允许你**自定义 size + stride + storage_offset**，创建的视图可以与原始 tensor 的布局完全不同。它是 `view`、`transpose`、`slice` 等操作在 kernel 层面的基础。

**安全限制**: 在 `torch._subclasses.fake_tensor.py` 和 `torch._subclasses.meta_utils.py` 中，`as_strided` 被特别处理，因为它可能创建**越界访问**（out-of-bounds）的视图。

---

## 五、`storage()` 与 `data_ptr()`

### 5.1 `storage()` (`TensorBase.h:369-371`)

```cpp
const Storage& storage() const {
    return impl_->storage();
}
```

返回 `c10::Storage` — 持有实际数据块的引用计数指针。**多个 tensor 可以共享同一个 storage**。

### 5.2 `data_ptr()` (`TensorBase.h:590-604`)

```cpp
const void* const_data_ptr() const {
    return this->unsafeGetTensorImpl()->data();
}
void* mutable_data_ptr() const {
    return this->unsafeGetTensorImpl()->mutable_data();
}
```

返回 storage 中**实际数据的原始指针**。注意：对于有 `storage_offset` 的视图，`data_ptr()` 指向 storage 中偏移后的位置。

---

## 六、`set_` — 原地替换 storage (`native_functions.yaml:8012-8061`)

```yaml
- func: set_.source_Storage_storage_offset(Tensor(a!) self, Storage source,
      SymInt storage_offset, SymInt[] size, SymInt[] stride) -> Tensor(a!)
```

`set_` 允许你**原地替换一个 tensor 的内部存储**，常用于:
- 从 checkpoint 恢复参数
- `_apply` 中的权重替换（旧版用 `param.data = new`）

---

## 七、`requires_grad_()` 和 autograd 元数据

### 7.1 `requires_grad_()` (`variable.cpp:599-606`)

```cpp
void VariableHooks::requires_grad_(const at::TensorBase& self, bool _requires_grad) const {
    TORCH_CHECK(
        self.is_leaf() || !_requires_grad,
        "only leaf Tensor can set requires_grad=True"
    );
    self.set_requires_grad(_requires_grad);
}
```

**规则**:
- 叶子 tensor (`is_leaf() == True`) → 可以设置 `requires_grad=True` 或 `False`
- 非叶子 tensor → 只能设置 `requires_grad=False`（`True` 会报错）
- 只有浮点/复数 dtype 支持 requires_grad (`variable.h:293`)

### 7.2 `detach()` (`TensorBody.h:758`)

返回共享 storage 的新 tensor，但 `autograd_meta_` 为 `nullptr`。等价于从 autograd 图中**切断**。

### 7.3 `retain_grad()` (`variable.cpp:531-573`)

非叶子 tensor 默认不保留 `.grad`。`retain_grad()` 在 grad_fn 上注册 hook，在 backward 时将梯度拷贝到 `.grad`。

### 7.4 `register_hook()` (`_tensor.py:655-703` + `TensorBase.h:875`)

```python
def register_hook(self, hook):
    # hook(grad) -> Tensor or None
    # 在 backward 计算完此 tensor 的梯度后触发
    if self._backward_hooks is None:
        self._backward_hooks = OrderedDict()
        if self.grad_fn is not None:
            self.grad_fn._register_hook_dict(self)
    handle = RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle
```

---

## 八、`to()` 方法 (`native_functions.yaml:7829-7847`)

```yaml
- func: to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None,
      Layout? layout=None, Device? device=None, bool? pin_memory=None,
      bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
```

`tensor.to(device)` / `tensor.to(dtype)` 本质上调用 `_to_copy` kernel。
- `non_blocking=True`: 异步 host→device 拷贝
- `copy=False` 且目标与当前相同 → 返回自身
- 支持 `memory_format` 参数指定 contiguous/channels_last 等

---

## 九、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `Tensor` 类定义 | `torch/_tensor.py` | 102 |
| `TensorBase` 类 | `aten/src/ATen/core/TensorBase.h` | — |
| `TensorImpl` (底层) | `c10/core/TensorImpl.h` | — |
| `view` 声明 | `native_functions.yaml` | 8167 |
| `reshape` 声明 | `native_functions.yaml` | 4989 |
| `contiguous` | `aten/src/ATen/core/TensorBase.h` | 135 |
| `stride(dim)` | `aten/src/ATen/core/TensorBase.h` | 190 |
| `is_contiguous` | `aten/src/ATen/core/TensorBase.h` | 268 |
| `storage()` | `aten/src/ATen/core/TensorBase.h` | 369 |
| `data_ptr()` | `aten/src/ATen/core/TensorBase.h` | 590 |
| `as_strided` | `native_functions.yaml` | 930 |
| `set_` | `native_functions.yaml` | 8012 |
| `requires_grad_` | `torch/csrc/autograd/variable.cpp` | 599 |
| `detach` | `aten/src/ATen/templates/TensorBody.h` | 758 |
| `register_hook` | `torch/_tensor.py` | 655 |
| `to` 声明 | `native_functions.yaml` | 7829 |
| `SizesAndStrides` | `c10/core/impl/SizesAndStrides.h` | — |

---

## 十、可借鉴的工程技巧

1. **metadata vs data 分离**: Tensor 只存 metadata + 共享指针，`view` 零开销。类比：写自己的容器类时用 view 概念避免不必要拷贝。

2. **先检测再拷贝** (`contiguous:135`): `if (is_contiguous) return *this` — 已知满足条件时跳过工作，热路径零成本。

3. **引用计数共享数据**: 多个 tensor 共享 `Storage`，引用计数控制生命周期。类比：多个视图共享底层数据。

4. **安全 view 与快速 view**: `reshape` = `try view else clone+view`，给用户安全默认，高级用户用 `view` 或 `as_strided`。

5. **out-of-bounds 防护**: `as_strided` 被 FakeTensor/Meta 子系统特殊处理，防止创建越界访问。类比：对外 API 做安全检查，内部 kernel 信任调用者。

---

## 十一、实战常见坑点

### 1. view 后 in-place 导致原 tensor 静默被修改
**现象**: 训练到一半 loss 突然炸了; debug 发现某个参数在不该变的时候变了。
**原因**: `view` / `transpose` / `slice` 共享 storage。对 view 做 `+=`, `mul_` 等 in-place 操作会直接修改原 tensor 的数据。
```python
a = torch.tensor([1.,2.,3.])
b = a.view(3, 1)
b += 1  # a 也被改了!
print(a)  # tensor([2.,3.,4.]) — 意料之外
```
**解决**: 需要独立的 copy 时手动 `.clone()`; in-place 前检查 `tensor.is_leaf`。

### 2. reshape 悄悄拷贝导致梯度断裂
**现象**: loss.backward() 报 "element 0 of tensors does not require grad and does not have a grad_fn"。
**原因**: `reshape` 对 non-contiguous tensor 会拷贝（而非 view），拷贝出的新 tensor 与计算图断开。
```python
x = torch.randn(10, requires_grad=True)
x_t = x.t()          # non-contiguous view
y = x_t.reshape(10)  # 拷贝了! y 是叶子 tensor
y.sum().backward()
print(x.grad)  # None! 梯度断了
```
**解决**: 对需要梯度的 tensor，用 `.contiguous().view()` 或确保 reshape 走 view 路径。

### 3. contiguous() 不总是拷贝
**现象**: 期望 `.contiguous()` 返回新 tensor 做 in-place，结果还是改了原 tensor。
**原因**: `.contiguous()` 在 tensor 已经 contiguous 时返回 `self`（零开销优化）。
```python
x = torch.randn(3, 4)
y = x.contiguous()
y is x  # True! y 就是 x 自身
```
**解决**: 需要强制拷贝用 `.clone()`，不要依赖 `.contiguous()` 做拷贝。

### 4. sparse tensor 与 dense 混用的隐式转换
**现象**: 稀疏模型突然 OOM。
**原因**: sparse + dense 做运算时 PyTorch 可能隐式 densify → 显存暴涨。
**排查**:
```python
print(tensor.is_sparse, tensor._nnz(), tensor.numel())
# 如果 _nnz() 很大但 is_sparse 是 False → 已经 densified
```
**解决**: 全程保持 sparse layout，或显式控制 `t.to_dense()` 的时机。

### 5. 跨设备 tensor 比较静默失败
**现象**: `torch.allclose(cpu_tensor, cuda_tensor)` 返回 True, 但实际值不同。
**原因**: `allclose` 在比较前隐式转换了 dtype/device, 可能导致精度截断。CPU fp32 vs GPU fp16 比较时尤其危险。
**解决**: 显式统一 device 和 dtype:
```python
assert torch.allclose(cpu_t.to(device="cuda", dtype=torch.float32),
                       cuda_t.to(dtype=torch.float32))
```

