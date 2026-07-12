# PyTorch 中的 C++ 设计模式

> 源码路径: `c10/core/`, `c10/util/`, `aten/src/ATen/core/`, `torch/csrc/`

## 0. 一句话总览

PyTorch 的 C++ 核心代码大量运用 RAII、Singleton、CRTP、Type Erasure、Observer、PIMPL 等经典设计模式。理解这些模式是读懂 Dispatcher、TensorImpl、IValue、Allocator 等核心组件的钥匙。

## 1. 最小例子

```python
import torch

# RAII: torch.no_grad() 是 RAII 的 Python 体现
with torch.no_grad():
    x = torch.randn(3, requires_grad=True)
    y = x + 1
    print(y.requires_grad)  # False

# 等价 C++ 端:
# AutoDispatchBelowAutograd guard;  // RAII: 构造时 set, 析构时 restore

# Singleton: Dispatcher 是全局单例
d1 = torch._C._dispatch_keys(torch.randn(3))
d2 = torch._C._dispatch_keys(torch.randn(3))
# 背后是 Dispatcher::singleton()
```

## 2. 实战例子

### 2.1 RAII: DeviceGuard / StreamGuard / NoGradGuard

```python
import torch

# RAII = Resource Acquisition Is Initialization
# 构造时获取资源，析构时释放

# torch.cuda.device() -> CUDAGuard (RAII)
with torch.cuda.device(0):
    x = torch.randn(3, device="cuda")
    with torch.cuda.device(1):
        y = torch.randn(3, device="cuda")
    # CUDAGuard析构: 自动恢复 device 0

# StreamGuard: 切换 CUDA stream 并自动恢复
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    z = x * 2
# StreamGuard析构: 恢复默认 stream

# torch.no_grad() -> AutoDispatchBelowAutograd
with torch.no_grad():
    w = x.clone()
# AutoDispatchBelowAutograd析构: 恢复 Autograd key
```

### 2.2 Singleton: Dispatcher 全局单例

```python
import torch

# Dispatcher 是全局唯一实例
# c10/core/dispatch/Dispatcher.h:
#   static Dispatcher& singleton() { static Dispatcher s; return s; }

# 所有 op 注册都进同一个 singleton
lib_a = torch.library.Library("demo_a", "DEF")
lib_a.define("op_a(Tensor x) -> Tensor")

lib_b = torch.library.Library("demo_b", "DEF")
lib_b.define("op_b(Tensor x) -> Tensor")

# 两个 op 在同一个 Dispatcher 中
dump_a = torch._C._dispatch_dump_table("demo_a::op_a")
dump_b = torch._C._dispatch_dump_table("demo_b::op_b")
print("Both ops live in same Dispatcher singleton")
```

### 2.3 Observer: Autograd Hook 模式

```python
import torch

# Autograd hook = Observer 模式
# Engine 遍历 Node graph, 每个 Node 通知其 subscriber

x = torch.randn(3, requires_grad=True)

# 注册 observer
hook_data = []
def my_hook(grad):
    hook_data.append(grad.clone())
    print(f"  Hook fired: grad={grad}")
    return grad * 2  # 可以修改梯度

x.register_hook(my_hook)

y = x * 2
y.sum().backward()

# DDP Reducer 就是用同样的 hook 机制
# reducer.cpp: variable.register_hook(autograd_hook)
```

## 3. 从 Python API 到源码的调用链

```
with torch.no_grad():
  -> torch/autograd/grad_mode.py: __enter__
  -> _C._set_grad_enabled(False)
  -> AutoDispatchBelowAutograd (RAII)
  -> Dispatcher::setExcluded(DispatchKey::Autograd)

with torch.cuda.device(0):
  -> CUDAGuard(DeviceIndex(0))  (RAII)
  -> cudaSetDevice(0)
  -> ~CUDAGuard() -> cudaSetDevice(prev_device)

x.register_hook(fn):
  -> TensorImpl::add_hook(fn)
  -> autograd_meta_->hooks_.push_back(fn)
  -> backward 时 AutogradEngine 调用 hooks
```

## 4. 核心源码文件

```
c10/cuda/CUDAGuard.h              # RAII: DeviceGuard
c10/cuda/CUDAStream.h             # RAII: StreamGuard
c10/util/Singleton.h              # CRTP Singleton helper
c10/core/dispatch/Dispatcher.h    # Singleton: 全局调度器
c10/core/TensorImpl.h             # PIMPL: 数据隐藏
c10/util/intrusive_ptr.h          # 侵入式引用计数
c10/core/impl/InlineDeviceGuard.h # RAII variant
aten/src/ATen/core/IValue.h       # Type Erasure (tagged union)
torch/csrc/autograd/engine.h      # Observer: Hook 通知
```

## 5. 关键机制源码解读

### 5.1 RAII 模式：构造/析构管理资源

```cpp
// c10/cuda/CUDAGuard.h
class CUDAGuard {
    c10::DeviceIndex prev_device_;
public:
    CUDAGuard(c10::Device device) {
        prev_device_ = c10::cuda::GetDevice();     // 保存当前
        c10::cuda::SetDevice(device.index());      // 设置新值
    }
    ~CUDAGuard() {
        c10::cuda::SetDevice(prev_device_);        // 自动恢复
    }
};
```

PyTorch 中 RAII 的应用：
- `CUDAGuard` / `CUDAStreamGuard` — device/stream 切换
- `AutoDispatchBelowAutograd` — torch.no_grad()
- `AutoNonVariableTypeMode` — 跳过 autograd 包装
- `c10::OptionalDeviceGuard` — 可选的 device guard
- `at::AutoGradMode` — 控制 grad 启用

### 5.2 Singleton: Meyers' Singleton + CRTP

```cpp
// c10/core/dispatch/Dispatcher.h
class Dispatcher {
    static Dispatcher& singleton() {
        static Dispatcher s;   // Meyers' Singleton (C++11 线程安全)
        return s;
    }
    // 全局唯一的分发表
    ska::flat_hash_map<OperatorName, OperatorHandle> op_registry_;
};
```

### 5.3 PIMPL: 隐藏实现细节

```cpp
// c10/core/TensorImpl.h
class TensorImpl {
    // 对外暴露的接口
    IntArrayRef sizes() const;
    IntArrayRef strides() const;
private:
    // 内部实现 → PIMPL 隐藏
    struct Caffe2TensorImpl;  // 旧的实现, 被 TensorImpl 包裹
    c10::Storage storage_;
    c10::impl::SizesAndStrides sizes_and_strides_;
    // ...
};
```

### 5.4 Type Erasure: IValue 的 tagged union

```cpp
// aten/src/ATen/core/IValue.h
class IValue {
    // 可以存储任意类型: Tensor, int, double, string, list, dict ...
    union Payload {
        Tensor t;
        int64_t i;
        double d;
        c10::intrusive_ptr<Object> obj;
        // ...
    };
    // Type tag 标识当前存的是什么类型
    Tag tag_;
};
```

## 6. 和已有笔记的连接

```
14_dispatcher/    — Dispatcher 是 Singleton + Registry 模式
02_device_copy/   — DeviceGuard/StreamGuard 是 RAII 模式
12_autograd/      — Autograd Hook 是 Observer 模式
04_module/        — Module 的 register_buffer 是 Registry 模式
36_ddp_reducer/   — DDP reducer 用 Hook (Observer) 监听梯度
43_intrusive_ptr/ — c10::intrusive_ptr 引用计数
44_ivalue_type/   — IValue type erasure
```

## 7. 常见坑点

- RAII guard 嵌套时注意顺序，析构与构造反向
- Singleton 的初始化时机：首​​次调用时 lazy-init，多线程安全靠 C++11 static 局部变量
- Hook 中修改梯度要返回新 tensor，返回 None 会清除梯度
- IValue 存自定义类型需要注册到类型系统

## 8. 阅读源码时建议搜索的关键词

```bash
rg -n "class.*Guard" c10/cuda/
rg -n "static.*singleton" c10/core/dispatch/Dispatcher.h
rg -n "class IValue" aten/src/ATen/core/IValue.h
rg -n "intrusive_ptr" c10/util/intrusive_ptr.h
rg -n "register_hook" torch/csrc/autograd/
```
