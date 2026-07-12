### 理解 PyTorch 中的自动微分（Autograd）

这一节是进入 AOTAutograd 部分之前的前置知识。

Torch Autograd 是通过自动微分驱动神经网络训练的引擎。

假设我们要训练一个 10 层模型，它发生在两个阶段：
1. **前向传播** - 输入流经所有层，在最后一层产生对输出的预测。
2. **反向传播** - 将预测输出与真实标签比较，计算误差和梯度，然后按相反顺序传播。

PyTorch 使用**动态图**（也称为 eager 模式）为你处理这一切。这意味着它逐行运行代码，计算图可以根据输入动态变化。另一种理解方式是：图的构建和计算是同步发生的。

相比之下，**静态图**在计算发生之前就已构建完成。它只捕获一个永不改变的固定图。

二者各有优劣，但我们就此打住，进入实际概念。

PyTorch 的计算图只有两种元素：**数据（张量）**和**运算**。

运算可以是加、减、乘、除、平方根、指数、三角函数和其他可微运算等。

数据是实际的输入值。它分为**叶节点**和**非叶节点**。叶节点是用户创建的节点，不依赖于其他节点。

二者的区别在于，反向传播后，非叶节点的梯度会被释放，只有叶节点的梯度被保留，从而节省内存。如果你想保留非叶节点的梯度，可以使用 `retain_grad()`。

## 目录结构

```
12_autograd/
├── 01_backward.py      基础 backward 示例
├── 02_autograd.py      autograd 用法
├── 03_second_order.py  二阶导数示例
├── test3.py            测试脚本
├── mini_pro/           🔥 从零实现自动微分引擎
│   ├── mini_autograd.py  对标 PyTorch 的 Value + backward + 拓扑排序
│   ├── graph_trace.py    追踪 grad_fn / next_functions 节点图
│   ├── engine_sim.py     模拟 Engine::execute() 依赖计数
│   └── higher_order.py   高阶导数 & WGAN-GP 梯度惩罚
└── README.md
```


---

# [合并自 no_grad]

# torch.no_grad / enable_grad / inference_mode 源码分析

> 源码: `torch/autograd/grad_mode.py` (489 行) — no_grad / enable_grad / set_grad_enabled / inference_mode
> C++ 关键: `torch/csrc/autograd/python_variable.cpp` — `THPAutograd_set_grad_enabled`
> TLS 操作: `c10/core/impl/LocalDispatchKeySet.h` — 线程局部 DispatchKeySet 的 include/exclude

## 0. 一句话总览

`@torch.no_grad()` = 从当前线程的 TLS DispatchKeySet 中**排除 `Autograd` key**。排除后，所有算子的 dispatch 路径都跳过 autograd 包装层 → 不建图、不保存中间值、forward 不用 `torch.enable_grad()` 包裹。本质是一次 DispatchKeySet 位运算，零热路径开销。

---

## 一、三种 API 的区别

| API | 作用域 | 可嵌套恢复? | 对 Factory 函数的影响 |
|-----|--------|------------|---------------------|
| `@torch.no_grad()` | 装饰函数 / `with` 上下文 | 可被 `enable_grad` 覆盖 | Factory 仍可创建 `requires_grad=True` 的 tensor |
| `torch.set_grad_enabled(False)` | 函数调用 (不自动恢复) | 手动恢复 | 同上 |
| `torch.inference_mode()` | 装饰函数 / `with` 上下文 | 不可被 `enable_grad` 覆盖! | 更强 — 连 Autograd key 的 fallback 也禁用 |

---

## 二、`no_grad` 源码分析 (`grad_mode.py:22`)

```python
# grad_mode.py:22
class no_grad(_NoParamDecoratorContextManager):
    def __init__(self):
        self.prev = False

    def __enter__(self):                         # :81
        self.prev = torch.is_grad_enabled()       # 保存当前状态
        torch.set_grad_enabled(False)              # 设置新状态

    def __exit__(self, exc_type, exc_value, traceback):  # :85
        torch.set_grad_enabled(self.prev)          # 恢复旧状态
```

`_NoParamDecoratorContextManager` 是 `no_grad` 既可以当 `with no_grad():` 又可以当 `@no_grad()` 用的原因 — 它实现了 `__call__` 将自身作为装饰器。

### 2.1 `torch.set_grad_enabled(False)` 到底做了什么

```python
# Python 端:
def set_grad_enabled(mode: bool) -> None:
    torch._C._set_grad_enabled(mode)

# C++ 端 (python_variable.cpp):
void set_grad_enabled(bool enabled) {
    // 本质操作:
    c10::impl::tls_set_dispatch_key_excluded(
        DispatchKey::Autograd, !enabled  // enabled=False → exclude=True
    );
}
```

**核心**: `no_grad()` → 在 TLS DispatchKeySet 中标记 `Autograd` key 为**排除**。后续所有 operator 的 `DispatchKeyExtractor` 取 key_set 时，Autograd key 不会出现在最终 key set 中 → 调度器跳过 autograd 包装层 → 算子直接调用 backend kernel，不再创建 `grad_fn`。

### 2.2 为什么 `@no_grad()` 可以装饰函数

```python
# _NoParamDecoratorContextManager 内部伪代码:
class _NoParamDecoratorContextManager:
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:        # self = no_grad() 实例
                return func(*args, **kwargs)
        return wrapper
```

所以 `@torch.no_grad()` 和 `with torch.no_grad():` 完全等价，只是作用域不同。

---

## 三、`enable_grad` — 在 no_grad 内部局部恢复梯度 (`:89`)

```python
# grad_mode.py:89
class enable_grad(_NoParamDecoratorContextManager):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(True)   # 强制包含 Autograd key

    def __exit__(self, ...):
        torch._C._set_grad_enabled(self.prev)
```

用途: 在一个大的 `no_grad` 上下文中，让某一段仍然追踪梯度。

```python
@torch.no_grad()
def evaluate(model, x):
    features = model.backbone(x)  # 不追踪梯度
    with torch.enable_grad():
        logits = model.head(features)  # 这段追踪梯度
    return logits
```

---

## 四、`inference_mode` — 比 no_grad 更强 (`grad_mode.py`)

```python
class inference_mode(_DecoratorContextManager):
    def __init__(self, mode=True):
        self.mode = mode

    def __enter__(self):
        if self.mode:
            # 1. 禁用 Autograd (同 no_grad)
            self.prev_grad_mode = torch.is_grad_enabled()
            torch._C._set_grad_enabled(False)
            # 2. 额外: 设置 inference 模式
            #    Autograd key 不仅被 exclude, 连 fallback 也被禁用
            #    意味着所有 view 操作、in-place 操作也跳过 autograd 版本检查
            torch._C._enter_inference_mode()

    def __exit__(self, ...):
        torch._C._exit_inference_mode()
        torch._C._set_grad_enabled(self.prev_grad_mode)
```

### 4.1 `no_grad` vs `inference_mode` 核心区别

| | no_grad | inference_mode |
|---|---|---|
| Autograd key | 排除 | 排除 |
| View 操作的 version counter | 正常递增 | 不递增 (更快) |
| 能被 `enable_grad` 恢复 | 能 | **不能** |
| 适用场景 | 训练中的 eval 阶段 | 纯推理 (production inference) |
| 性能 | 好 | 更好 (省 version counter 检查) |

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `no_grad` 类 | `torch/autograd/grad_mode.py` | 22 |
| `no_grad.__enter__` | `torch/autograd/grad_mode.py` | 81 |
| `enable_grad` 类 | `torch/autograd/grad_mode.py` | 89 |
| `set_grad_enabled` 函数 | `torch/autograd/grad_mode.py` | 144 |
| `inference_mode` 类 | `torch/autograd/grad_mode.py` | — |
| C++ `_set_grad_enabled` | `torch/csrc/autograd/python_variable.cpp` | — |
| TLS `tls_set_dispatch_key_excluded` | `c10/core/impl/LocalDispatchKeySet.h` | — |
| `_NoParamDecoratorContextManager` | `torch/utils/_contextlib.py` | — |

---

## 六、实战常见坑点

### 1. no_grad 内创建的 tensor 仍然可以 requires_grad=True
Factory 函数 (`torch.randn(..., requires_grad=True)`) 不受 `no_grad` 影响 — 它显式接受 `requires_grad` 参数。`no_grad` 只影响**通过计算图推导**的 `requires_grad`。

### 2. no_grad 内的操作不会出现在 backward 图中
即使输入 `requires_grad=True`，`no_grad` 内的所有 intermediate tensor 都 `requires_grad=False` → 反向传播时这段计算被「跳过」。

### 3. `@no_grad()` 装饰的函数的返回值没有 grad_fn
如果后续需要在返回值上做 backward，必须在 `no_grad` **外面**做。

### 4. inference_mode 内不能调用 enable_grad
会抛出 `RuntimeError`。如果需要局部恢复梯度，应该用 `no_grad` 而不是 `inference_mode`。

### 5. no_grad 是 thread-local 的
不同线程可以独立设置。一个线程的 `no_grad` 不影响另一个线程的梯度追踪。
