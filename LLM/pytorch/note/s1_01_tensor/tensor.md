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


---

## 十二、Python ↔ C++ 调用链路(gdb 实测)

> 配套脚本:`trace_empty.py` + `trace_empty.gdb`(本目录)。
> 复现:`conda activate torch_env && gdb -q -batch -x trace_empty.gdb --args python trace_empty.py`

### 12.1 完整对象模型(比 §1 更细,补上 Python 那两层)

```
torch.Tensor            (torch/_tensor.py:102)   纯 Python 类,加方法/语法糖
  └ torch._C.TensorBase (python_variable.cpp:3626, tp_name="torch._C.TensorBase")
        │  每个实例其实是一个 C++ PyObject:
        ▼
     struct THPVariable {          (python_variable.h:17)
         PyObject_HEAD             ← Python 对象头(引用计数/类型)
         at::Tensor cdata;         ← ★真正的 C++ Tensor 存这里
         PyObject* backward_hooks; ← register_hook 挂的钩子
     }
        │  cdata 是 at::Tensor
        ▼
     at::Tensor : TensorBase       只有一个指针成员:
         c10::intrusive_ptr<TensorImpl> impl_;   (TensorBase.h:928)
        ▼
     c10::TensorImpl               元数据 sizes/strides/dtype/device/key_set
        ▼
     c10::Storage → StorageImpl → DataPtr + Allocator   真正的内存
```

一句话:Python 的 `torch.Tensor` 实例 = C++ 的 `THPVariable`(PyObject),它的 `cdata` 字段持有 `at::Tensor`;`at::Tensor` 只是 `intrusive_ptr<TensorImpl>` 的壳。

### 12.2 两个方向的桥

| 方向 | 函数 | 作用 | 位置 |
|------|------|------|------|
| C++ → Python | `THPVariable_Wrap(tensor)` | `at::Tensor` 包成 Python 对象 | python_variable.cpp:451 |
| Python → C++ | `THPVariable_Unpack(obj)` | 从 Python 对象取出 `at::Tensor& cdata` | python_variable.h |

### 12.3 创建链路:`torch.empty(2,3)`(gdb 实测栈)

```
Python: t = torch.empty(2, 3)
   │  _PyEval_EvalFrameDefault → call_function → cfunction_call   (CPython)
   │
   ▼ 【① 进入 C++ 绑定】(torchgen 生成)
   THPVariable_empty(self, args, kwargs)   python_torch_functions_2.cpp:3354
       - PythonArgParser 解析 (2,3)/dtype/device
       - 调 at::empty(...) 造出临时 at::Tensor
         (at::empty → dispatcher → at::native::empty_cpu
          → Allocator 分配内存 → 建 StorageImpl → 建 TensorImpl)
   │
   ▼ 【② 把结果包回 Python】       python_torch_functions_2.cpp:3379
   torch::autograd::utils::wrap(tensor)     wrap_outputs.h:74
   │
   ▼
   THPVariable_Wrap(var)                    python_variable.cpp:451
       - new THPVariable(PyObject),把 at::Tensor 存进 cdata
       - 返回 PyObject* → Python 得到 torch.Tensor 实例
```

实测要点:
1. **`THPVariable_empty` 是入口**:`import torch` 不走它,只有 Python 显式 `torch.empty()` 才触发(内部代码用 C++ `at::empty`)。
2. **返回值走 move 重载** `THPVariable_Wrap(at::TensorBase&&)`:`empty` 返回临时对象,用移动语义避免拷贝。
3. **`utils::wrap`** 是统一封装:所有返回 Tensor 的算子绑定都经它调 `THPVariable_Wrap`。

### 12.4 访问链路:`t.dim()` / `t.data_ptr()`(反方向,不建新对象)

```
Python: t.dim()
   │
   ▼ 方法表(torchgen 生成)  python_variable_methods.cpp
   THPVariable_dim(self)
   │
   ▼ 取出 C++ Tensor
   auto& self_ = THPVariable_Unpack(self);   // 拿到 cdata
   │
   ▼ 调 C++ 方法(纯读元数据)
   self_.dim() → impl_->dim()                c10/core/TensorImpl.h
```

`t.shape`/`t.dtype`/`t.device` 是 property,同样 `Unpack → 读 TensorImpl 字段`;
`t.data_ptr()` → `cdata.data_ptr()` → `impl_->data()` → `storage_.data() + storage_offset_*itemsize`(见 §5.2)。

---

## 十三、view 视图链路:如何共享同一个 Storage(gdb 实测)

> 配套脚本:`trace_view.py` + `trace_view.gdb`(本目录)。
> `python trace_view.py` 证明共享;`gdb -q -batch -x trace_view.gdb --args python trace_view.py` 看 C++ 链路。

### 13.1 Python 层实测:view 零拷贝、共享内存

```
base.data_ptr : 97676040256512
v.data_ptr    : 97676040256512   (== base:同一块内存)
same storage  : True
base.shape/stride: (12,) (1,)
v.shape/stride   : (3, 4) (4, 1)   ← 只有元数据变了
after v[0,0]=999 -> base[0] = 999.0  ← 改 view 就是改 base
```

结论:`view` 不拷贝数据——新 Tensor 与原 Tensor **共享同一个 Storage**,只是 `sizes/strides/storage_offset` 不同。

### 13.2 C++ 调用链(gdb 实测栈)

```
Python: base.view(3, 4)
   │
   ▼ 【① Python 方法绑定】(torchgen 生成)
   THPVariable_view(self, args)     python_variable_methods.cpp:16012
       - Unpack 取出 cdata,解析 size=(3,4)
       - 调 self_.view({3,4})
   │
   ▼ 【② dispatcher 分发】
   at::view → Dispatcher::redispatch → wrapper_CPU__view
   │
   ▼ 【③ C++ 实现】
   at::native::view(self, size)     TensorShape.cpp:4396
       → view_impl(self, size)
   │
   ▼ 【④ 建共享 Storage 的新 TensorImpl】★核心
   alias_with_sizes_and_strides(self, sizes, strides)   TensorShape.cpp
       self_ = make_tensor<TensorImpl>(
                   TensorImpl::VIEW,
                   Storage(self.storage()),   ← ★复用原 Storage(引用计数 +1)
                   self.key_set(), self.dtype());
       self_->set_storage_offset(self.storage_offset());
       self_->set_sizes_and_strides(sizes, strides);   ← 只改元数据
       return self_;
```

### 13.3 关键源码点

| 步骤 | 位置 | 作用 |
|------|------|------|
| Python 绑定 | `python_variable_methods.cpp` `THPVariable_view` | 取 cdata、解析 size |
| C++ 入口 | `TensorShape.cpp:4396` `at::native::view` | → view_impl |
| 共享 storage | `alias_with_sizes_and_strides` | `Storage(self.storage())` 复用内存 + 新 sizes/strides |
| 约束检查 | `computeStride`(view_impl 内) | stride 不满足则 view 失败(需 reshape/contiguous,见 §2) |

### 13.4 与 §2 / §12 的关系

- 印证 §1「Tensor = 元数据壳 + 共享数据指针」:`view` 是这句话最直接的体现。
- `TensorImpl::VIEW` 这个构造标记告诉 autograd:这是视图,反向要走 view 的特殊逻辑(与 §7 的 version_counter、functionalization 相关)。
- `reshape`(§2.2)= 能 view 就 view(走本节链路),否则 `contiguous()` 拷贝后再 view。

---

## 十四、`torch.empty(2,3)` 逐层实现与内存分配(源码逐行)

> §12.3 给的是 gdb 实测调用栈;本节把**每一层的真实源码**摊开,并补上 §12
> 一笔带过的**内存分配**细节。以默认 CPU、float32 为例。
> VSCode 逐层单步:见文末断点清单,或直接用 `trace_empty.gdb`。

### 14.1 三层对象职责(内存视角)

```
Tensor / TensorBase        句柄,内部就一个 intrusive_ptr
  └─ TensorImpl            元数据:sizes[2,3] strides[3,1] dtype device requires_grad
     │                                                    (c10/core/TensorImpl.h)
     └─ StorageImpl        管一块连续内存:size_bytes + allocator + data_ptr
        │                                                 (c10/core/StorageImpl.h)
        └─ DataPtr → void*  堆上真正的字节(带 deleter)
```

**元数据(TensorImpl)与数据(StorageImpl)分离** → view/reshape 零拷贝(见 §13);
内存生命周期靠 `intrusive_ptr` **引用计数**,StorageImpl 归零时自动调 deleter 释放,无 GC。

### 14.2 七层调用,逐层真实代码

**① Python**
```python
t = torch.empty(2, 3)                 # trace_empty.py:15
```

**② C++ 绑定入口 — 参数解析**(torchgen 生成)
```cpp
// python_torch_functions_2.cpp:3354
static PythonArgParser parser({"empty(SymIntArrayRef size, *, ... ScalarType? dtype=None, ...)"});
auto _r = parser.parse(nullptr, args, kwargs, parsed_args);  // PyObject* -> C++ 类型
const auto options = TensorOptions().dtype(_r.scalartypeOptional(3)).device(...)...;
auto dispatch_empty = [](c10::SymIntArrayRef size, at::TensorOptions options, ...) {
    pybind11::gil_scoped_release no_gil;                     // 释放 GIL,进入纯 C++
    return torch::empty_symint(size, options, memory_format);
};
return wrap(dispatch_empty(_r.symintlist(0), options, ...)); // wrap = 第⑦步
```

**③ Dispatcher 分发** — `at::empty` 按 `DispatchKey::CPU` 路由到 `empty_cpu`(生成的路由表,单步会穿过大量模板,直接在 `empty_cpu` 下断点)。

**④ CPU 后端实现**
```cpp
// aten/src/ATen/native/TensorFactories.cpp:263
Tensor empty_cpu(IntArrayRef size, ...) {
  Tensor result = at::detail::empty_cpu(size, dtype_opt, ...);
  if (deterministicAlgorithms() && ...) fill_empty_deterministic_(result);
  return result;
}
// aten/src/ATen/EmptyTensor.cpp:272
TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory, ...) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);   // 默认 CPU 分配器
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}
```

**⑤ 建对象 + 触发分配**
```cpp
// aten/src/ATen/EmptyTensor.cpp:179 (_empty_generic)
auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize()); // 2*3*4 = 24
auto storage_impl = c10::make_intrusive<StorageImpl>(   // ★构造时立即分配内存
    c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, /*resizeable=*/true);
auto tensor = detail::make_tensor_base<TensorImpl>(std::move(storage_impl), ks, dtype);
tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size); // 写 sizes=[2,3] strides=[3,1]
return tensor;
```

**⑥ 内存分配**(本节重点,§12 未展开)
```cpp
// c10/core/StorageImpl.h:78  —— 委托构造,在初始化列表就 allocate
StorageImpl(use_byte_size_t, const SymInt& size_bytes, Allocator* allocator, bool resizable)
    : StorageImpl(use_byte_size_t(), size_bytes,
          allocator->allocate(size_bytes.as_int_unchecked()),  // ← 分配 24 字节
          allocator, resizable) {}

// c10/core/CPUAllocator.cpp:20  DefaultCPUAllocator::allocate
at::DataPtr allocate(size_t nbytes) override {
    void* data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(CPU)};     // DataPtr 绑定释放函数
}

// c10/core/impl/alloc_cpu.cpp:92
void* alloc_cpu(size_t nbytes) {
    if (nbytes == 0) return nullptr;
    void* data = nullptr;
    posix_memalign(&data, gAlignment, nbytes);   // Linux: 64 字节对齐(SIMD/cache line)
    // debug build: memset_junk(data, nbytes)    // 填垃圾值 → empty 内容随机
    return data;
}
```

**⑦ C++ Tensor 包回 Python 对象**
```cpp
// torch/csrc/autograd/python_variable.cpp:451
PyObject* THPVariable_Wrap(at::TensorBase&& var) {            // move 重载(临时对象)
  return THPVariable_WrapWithType(std::move(var), std::nullopt); // 建 THPVariable 壳
}
// 返回 Python,赋给 t
```

### 14.3 内存关键结论

- **立即分配、非惰性**:`StorageImpl` 一构造就 `allocate`,`t` 拿到手数据内存已就位。
- **24 字节 = 2*3*float32(4B)**;`posix_memalign` 对齐到 `gAlignment=64`。
- **`empty` 不清零**,内容是内存旧值;`zeros` 在⑤之后多一步 `fill_(0)`。
- **释放靠引用计数**:`DataPtr` 存 deleter(`ReportAndDelete → free_cpu`),StorageImpl 引用计数归零时自动释放。
- **CUDA 路径**:③选到 `empty_cuda`,allocator 换成 `CUDACachingAllocator`(显存池缓存,避免频繁 `cudaMalloc`),其余结构一致。

### 14.4 VSCode / gdb 逐层断点清单(按调用顺序)

```
torch::autograd::THPVariable_empty        ②  入口/参数解析
at::native::empty_cpu                      ④  CPU 后端
at::detail::empty_generic                  ⑤  建 TensorImpl+StorageImpl(函数名 _empty_generic)
c10::alloc_cpu                             ⑥  裸内存分配
THPVariable_Wrap(at::TensorBase&&)         ⑦  包回 Python
```
已写入 `.vscode/launch.json` 的 "(gdb) Attach to Python" 配置(`setupCommands`),
attach 后依次 `continue` 即可逐层跟下来。混合调试步骤见 §12 与 launch.json。
