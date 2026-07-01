# PTQ & QAT 量化部署实践源码分析

> 源码路径: `torch/ao/quantization/` — PTQ/QAT eager mode 和 FX graph mode
> Observer: `observer.py:150,440,587` — MinMaxObserver / MovingAverageMinMaxObserver
> FakeQuantize: `fake_quantize.py:128,228` — QAT 核心
> 融合: `fuse_modules.py:129` — `fuse_modules()`
> convert: `quantize.py` — `convert()` / 模块替换

## 0. 一句话总览

量化部署 = **校准** (observer 统计 min/max) → **量化** (将浮点权重/激活映射到 INT8) → **部署** (用 INT8 kernel 推理)。PTQ 无需训练，QAT 在训练中模拟量化误差。

---

## 一、Observer 源码分析 (`observer.py`)

### 1.1 继承链

```
ObserverBase (:150)                    ← ABC, 定义 forward/observer_enabled
  └─ UniformQuantizationObserverBase   ← 添加 _calculate_qparams (:349)
       └─ MinMaxObserver (:440)        ← running min/max
            └─ MovingAverageMinMaxObserver (:587) ← EMA
```

### 1.2 `ObserverBase.forward` — 数据不中断的关键设计 (:172)

```python
# observer.py:172
def forward(self, x):
    if self.observer_enabled[0] == 1:
        self.activation_post_process(x.detach())  # 只统计，不返回统计值
    return x  # 原样返回输入！
```

**关键**: `forward` 做统计后用 `detach()` 切断梯度，然后**原样返回**。这是 Observer 模式的核心 — 数据流完全不中断，只是旁路记录统计量。

### 1.3 `MinMaxObserver.forward` (:558) — 实际统计逻辑

```python
# observer.py:558
def forward(self, x_orig):
    x = x_orig.detach()  # 不追踪梯度
    min_val, max_val = torch.aminmax(x)
    self.min_val = torch.min(self.min_val, min_val)  # 与历史值合并
    self.max_val = torch.max(self.max_val, max_val)
    return x_orig
```

`min_val`/`max_val` 是 `register_buffer` — 存在 `_buffers` 中，会参与 `state_dict()` 序列化，但不参与 `parameters()` 遍历。

### 1.4 `MovingAverageMinMaxObserver.forward` (:668) — EMA 更新

```python
# observer.py:668
def forward(self, x_orig):
    x = x_orig.detach()
    min_val, max_val = torch.aminmax(x)
    # 指数移动平均：self.min_val = a * min_val + (1-a) * self.min_val
    averaging_constant = self.averaging_constant  # 默认 0.01
    self.min_val = self.min_val + averaging_constant * (min_val - self.min_val)
    self.max_val = self.max_val + averaging_constant * (max_val - self.max_val)
    return x_orig
```

EMA 的优势: 对异常 batch（如全零输入）有缓冲，不会使 scale 剧烈变化。

### 1.5 `_calculate_qparams` — scale/zero_point 计算 (:349)

```python
# observer.py:349
def _calculate_qparams(self, min_val, max_val):
    # 对称量化
    if qscheme == torch.per_tensor_symmetric:
        scale = 2 * max(abs(min_val), abs(max_val)) / (quant_max - quant_min)
        zero_point = 0 if dtype == torch.qint8 else 128

    # 非对称量化
    else:  # per_tensor_affine
        scale = (max_val - min_val) / (quant_max - quant_min)
        zero_point = int(quant_min - round(min_val / scale))

    return scale, int(zero_point)
```

---

## 二、FakeQuantize 源码分析 (`fake_quantize.py`)

### 2.1 核心类 `FakeQuantize` (:128)

```python
# fake_quantize.py:128
class FakeQuantize(FakeQuantizeBase):
    # :167 __init__
    def __init__(self, observer=MovingAverageMinMaxObserver, **kwargs):
        self.activation_post_process = observer(**kwargs)  # Observer 实例
        # :207 注册 scale/zero_point 为 buffer (persistent=True)
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))

    # :228 forward
    def forward(self, X):
        if self.observer_enabled[0] == 1:
            # T1: 更新统计量 (训练模式下持续更新 scale/zp)
            self.activation_post_process(X.detach())
            # 从 observer 计算结果中获取 scale/zero_point
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), ...
            self.scale.resize_(_scale.shape).copy_(_scale)
            self.zero_point.resize_(_zero_point.shape).copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            # T2: fake quantize (保持 float 精度, 模拟 int 误差)
            X = torch.fake_quantize_per_channel_affine(
                X, self.scale, self.zero_point,
                self.axis, self.quant_min, self.quant_max
            )
        return X
```

### 2.2 fake_quantize 的 C++ 实现

`torch.fake_quantize_per_tensor_affine` 在 C++ 端实现 (`aten/src/ATen/native/quantized/fake_quant_affine.cpp`):

```cpp
// 伪代码:
float fake_quantize(float x, float scale, int64_t zero_point, int64_t qmin, int64_t qmax):
    // 1. 映射到整数域
    int64_t xq = round(x / scale) + zero_point
    xq = clamp(xq, qmin, qmax)
    // 2. 映射回浮点域 (保持 float 精度)
    return (xq - zero_point) * scale
```

**backward 用 STE (Straight-Through Estimator)**: `∂output/∂input = 1` (梯度直接穿过, 不衰减)。

---

## 三、`convert()` 的模块替换机制

`convert()` 遍历模型的所有子模块，将 `nn.Conv2d` + `FakeQuantize`/`Observer` 替换为 `nn.quantized.Conv2d`:

```
Before convert:
  x → QuantStub → Conv2d → FakeQuantize(activation) → DeQuantStub → output
                        ↑
                   Conv2d weight (fp32)

After convert:
  x → Quantize → nn.quantized.Conv2d → DeQuantize → output
                        ↑
                   quantized weight (qint8, scale/zp baked in)
```

替换在 `quantize.py` 中通过 `swap_module()` 实现 — 检查模块类型，查表替换。

---

## 四、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `ObserverBase` | `observer.py` | 150 |
| `ObserverBase.forward` | `observer.py` | 172 |
| `MinMaxObserver` | `observer.py` | 440 |
| `MinMaxObserver.forward` | `observer.py` | 558 |
| `MovingAverageMinMaxObserver` | `observer.py` | 587 |
| `MovingAverageMinMaxObserver.forward` | `observer.py` | 668 |
| `_calculate_qparams` | `observer.py` | 349 |
| `FakeQuantize` | `fake_quantize.py` | 128 |
| `FakeQuantize.forward` | `fake_quantize.py` | 228 |
| `fuse_modules` | `fuse_modules.py` | 129 |
| fake_quantize C++ kernel | `aten/src/ATen/native/quantized/fake_quant_affine.cpp` | — |
| `prepare()` | `quantize.py` | — |
| `convert()` | `quantize.py` | — |
| `swap_module` | `quantize.py` | — |

---

## 五、可借鉴的工程技巧

1. **Observer 模式**: `forward(x)` 旁路记录统计量, 数据流原样传递 → 无损嵌入。

2. **FakeQuantize + STE**: 前向模拟截断误差, 反向直接用 identity gradient (STE) → 梯度可回传, 训练中学会补偿量化误差。

3. **EMA 统计**: 用指数移动平均代替 plain min/max → 对 outlier batch 不敏感, scale 更平滑。

4. **模块替换**: `convert()` 用 `swap_module` 做类型替换 → 用户代码无需修改模块定义。

---

## 六、实战常见坑点

*(见历史版本，此处省略以聚焦源码)*
