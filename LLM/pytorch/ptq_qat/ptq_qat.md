# PTQ & QAT 量化部署实践源码分析

> 源码路径: `torch/ao/quantization/` — PTQ/QAT eager mode 和 FX graph mode
> PTQ 入口: `quantize.py` — `prepare()` / `convert()` / `quantize()`
> QAT 入口: `quantize.py` — `prepare_qat()` / `convert()`
> FX mode: `quantize_fx.py` — `prepare_fx()` / `convert_fx()`
> 融合: `fuse_modules.py` — `fuse_modules()` / `fuse_modules_qat()`

## 0. 一句话总览

量化部署 = **校准** (observer 统计 min/max) → **量化** (将浮点权重/激活映射到 INT8) → **部署** (用 INT8 kernel 推理)。PTQ 无需训练，QAT 在训练中模拟量化误差。

---

## 一、PTQ (Post-Training Quantization) 三步走

### Step 1: 准备 — `prepare()`

```python
# quantize.py 核心逻辑
model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
# qconfig = QConfig(
#     activation=HistogramObserver.with_args(...),
#     weight=MinMaxObserver.with_args(dtype=torch.qint8, ...)
# )

model_prepared = torch.ao.quantization.prepare(model)
```

`prepare()` 做三件事:
1. **传播 qconfig**: `propagate_qconfig_()` → 每个子模块继承父模块的 qconfig
2. **插入 Observer**: 在需要量化的 op 前后插入 Observer 模块
3. **融合 Module**: 如果之前调用了 `fuse_modules()`，替换为量化友好的融合模块

### Step 2: 校准 — `observer` 收集统计量

```python
# 用代表性数据跑 inference
for data in calibration_data:
    model_prepared(data)

# Observer 在每次 forward 时更新 min/max 统计量
# Observer 内部: running_min = min(running_min, x.min())
#               running_max = max(running_max, x.max())
```

`observer.py` 中常见的 Observer:
- `MinMaxObserver` — 记录全局 min/max
- `MovingAverageMinMaxObserver` — EMA 更新 min/max
- `HistogramObserver` — 直方图分布估计（更精确）
- `PerChannelMinMaxObserver` — 逐通道统计

### Step 3: 转换 — `convert()`

```python
model_quantized = torch.ao.quantization.convert(model_prepared)
```

`convert()` 做:
1. **计算 scale + zero_point**: `scale = (max - min) / (qmax - qmin)`, `zero_point = qmin - round(min / scale)`
2. **量化权重**: 将 `nn.Conv2d` 的浮点 weight 转为 `torch.qint8`
3. **替换 Module**: `nn.Conv2d` → `nn.quantized.Conv2d`, `nn.Linear` → `nn.quantized.Linear`
4. **插入 Quantize/DeQuantize**: 在量化模块前后插入 `Quantize` / `DeQuantize` stub

---

## 二、QAT (Quantization-Aware Training)

与 PTQ 的区别在于 **Step 1** 用 `prepare_qat()` 而不是 `prepare()`:

```python
model_prepared = torch.ao.quantization.prepare_qat(model)
```

`prepare_qat()` 插入的是 `FakeQuantize` 而不是 `Observer`。

### FakeQuantize 原理 (`fake_quantize.py`):

```python
class FakeQuantize(nn.Module):
    def forward(self, x):
        # 1. Update running min/max (training mode)
        if self.training:
            self.activation_post_process(x.detach())

        # 2. Fake quantize: quantize + dequantize (simulates quantization error)
        x = torch.fake_quantize_per_tensor_affine(
            x, self.scale, self.zero_point,
            self.quant_min, self.quant_max
        )
        return x
```

**关键**: `fake_quantize` 在 forward 中模拟量化误差（float → int → float），但保持 float 精度以支持梯度回传。`scale` 和 `zero_point` 在训练中持续更新。

### QAT 流程:

```
prepare_qat() → 训练 (带 FakeQuantize) → convert() → int8 模型
```

QAT 比 PTQ 精度更高，因为模型在训练中**学会了补偿量化误差**。

---

## 三、Module Fusion — 量化前的关键步骤

`fuse_modules.py:129`:

```python
torch.ao.quantization.fuse_modules(
    model,
    modules_to_fuse=[
        ["conv1", "bn1", "relu1"],
        ["conv2", "bn2"],
        ["linear1", "relu1"],
    ],
)
```

融合支持的组合 (`fuser_method_mappings.py`):
- `[Conv2d, BatchNorm2d]` → `ConvBn2d`
- `[Conv2d, BatchNorm2d, ReLU]` → `ConvBnReLU2d`
- `[Conv2d, ReLU]` → `ConvReLU2d`
- `[Linear, ReLU]` → `LinearReLU`
- `[BatchNorm2d, ReLU]` → `BNReLU2d`

**为什么需要融合**: 量化时每个 op 边界都需要 quantize/dequantize 对。融合后 op 数量减少 → quant/dequant 对减少 → 精度损失减小。

---

## 四、FX Graph Mode 量化 (推荐方式)

FX mode 将量化表示为图级别的变换，而非 eager mode 的模块替换:

```python
# quantize_fx.py
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)
# calibrate...
model_quantized = convert_fx(model_prepared)
```

优势:
- 图级别优化 (DCE, constant folding) 可以在量化前后应用
- 支持更多 pattern 的自动识别和融合
- 统一的 IR 方便后端 codegen

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `prepare()` | `torch/ao/quantization/quantize.py` | — |
| `convert()` | `torch/ao/quantization/quantize.py` | — |
| `prepare_qat()` | `torch/ao/quantization/quantize.py` | — |
| `FakeQuantize` | `torch/ao/quantization/fake_quantize.py` | — |
| `Observer` 类 | `torch/ao/quantization/observer.py` | — |
| `fuse_modules` | `torch/ao/quantization/fuse_modules.py` | 129 |
| `fuse_modules_qat` | `torch/ao/quantization/fuse_modules.py` | 200 |
| `QConfig` 定义 | `torch/ao/quantization/qconfig.py` | — |
| 融合方法映射 | `torch/ao/quantization/fuser_method_mappings.py` | — |
| `prepare_fx` (FX mode) | `torch/ao/quantization/fx/prepare.py` | — |
| `convert_fx` (FX mode) | `torch/ao/quantization/fx/convert.py` | — |
| 后端配置 | `torch/ao/quantization/backend_config/` | — |
| 旧版兼容 | `torch/quantization/` (→ `torch.ao.quantization`) | — |

---

## 六、可借鉴的工程技巧

1. **Observer 模式**: `observer.forward(x)` 统计但不修改 x → 无损校准。类比：监控系统在不影响主流程的情况下采集指标。

2. **FakeQuantize 模拟误差**: `fake_quantize` 在浮点精度下模拟整数截断误差 → 保持可微性 → 梯度能正常回传。类比：浮点模拟定点运算。

3. **融合减少量化边界**: 每个 op 边界需要 quant/dequant 对 → 融合相邻 op 减少量化噪声累积。

4. **模块替换模式**: `convert()` 用 `quantized.Conv2d` 替换 `nn.Conv2d` → 用户无需改模型代码。类比：策略模式，运行时/部署时切换实现。

5. **配置传播**: `propagate_qconfig_()` 将 qconfig 从父模块传播到子模块 → 一处配置全局生效。类比：CSS 级联。

---

## 七、实战常见坑点

### 1. 量化后同一输入得到不同输出
**现象**: 每次 `model_int8(x)` 得到不同结果。
**原因**: Observer 在 `training=True` 时持续更新 min/max，即使模型标记为 `eval()`。
**解决**:
```python
model_prepared.eval()  # 先设置 eval
model_prepared.apply(torch.ao.quantization.disable_observer)  # 停掉 observer
# 或者 convert 之后再推理
model_int8 = quant.convert(model_prepared)
```

### 2. BatchNorm 量化后表现异常
**现象**: 量化模型比浮点模型差很多，排查发现 BN 层是元凶。
**原因**: BN 的 `running_mean`/`running_var` 被 observer 观测 + 量化 → 数值漂移。
**最佳实践**:
```python
# [1] 先 fuse conv+bn → BN 参数折叠进 conv weight
fused = quant.fuse_modules(model, ["conv", "bn", "relu"])
# [2] 再量化 — BN 层已消失，无 observer 误差
```

### 3. 首层/末层量化精度崩塌
**现象**: PTQ 后分类准确率暴跌 20%，但中间层误差不大。
**原因**: 输入层（RGB 图像 → quant）和输出层（logits → 类别）对量化噪声极其敏感。首层只有 3 个 channel → 量化 scale 由少量值决定 → 不稳定。
**解决**:
```python
model.qconfig = None  # 默认不量化
# 只量化中间层
model.layer1.qconfig = custom_qconfig
model.layer2.qconfig = custom_qconfig
# 首层和末层保持 fp32
model.conv1.qconfig = None
model.fc.qconfig = None
```

### 4. 动态量化用错了场景
**现象**: 在 CNN 上用 `torch.quantization.quantize_dynamic()` 没有加速。
**原因**: 动态量化只量化权重（int8），激活仍然是浮点 → 只对内存带宽有收益。CNN 是计算密集型 → 动态量化基本无加速。
**正确姿势**: 动态量化只适用于 LSTM/Transformer（带宽密集型）；CNN 用静态 PTQ（权重+激活都量化）。

### 5. QAT 训练时 scale 爆炸
**现象**: QAT 训练到后期 grad 出现 NaN。
**原因**: FakeQuantize 的 `scale` 在训练初期剧烈变化，"假量化"误差放大，梯度过大。
**解决**:
```python
# 先 PTQ 得到合理的 scale，再作为 QAT 初始值
model_prepared = quant.prepare(model)
# calibrate...
scale_init = model_prepared.activation_post_process.scale
# 将 scale 初始化给 QAT 模型
model_qat.activation_post_process.set_scale(scale_init)
```

### 6. 不同 batch size 下量化模型行为不一致
**现象**: batch=1 推理正常，batch=32 推理出错/崩溃。
**原因**: 量化 kernel（qconv2d, qlinear）有不同的内存布局和 padding 要求。fbgemm 后端要求特定对齐。
**解决**: 确保 batch size 是对齐的（如 fbgemm 要求 >=1 且 channel 对齐到 8）。

