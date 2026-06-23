import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# 基础线性量化
def linear_quantize(tensor, bits=8):
    """对张量进行线性量化"""
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1

    rmin, rmax = tensor.min(), tensor.max()
    print(f"raw 最小值、最大值：{rmin},{rmax}")
    scale = (rmax - rmin) / (qmax - qmin)
    print(f"缩放系数：{scale}")
    zero_point = qmin - torch.round(rmin / scale)
    print(f"零点：{zero_point}")
    zero_point = zero_point.clamp(qmin, qmax).to(torch.int32)

    # 量化
    q = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    q = q.to(torch.int8)

    # 反量化
    r = scale * (q.float() - zero_point.float())

    return q, scale, zero_point, r


# 测试
w = torch.randn(4, 4) * 3  # 模拟权重, std=3
q, s, z, r = linear_quantize(w, bits=8)
error = (w - r).abs().mean()
print(f"Weight: {w}")
print(f"Quantized: {q}")
print(f"Reconstructed: {r}")
print(f"Mean Abs Error: {error:.6f}, Scale: {s:.6f}, ZeroPoint: {z}")


# 量化感知训练(QAT)
class FakeQuantize(nn.Module):
    """模拟量化的模块, 用于 QAT"""

    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        qmin, qmax = -(2 ** (self.bits - 1)), (2 ** (self.bits - 1)) - 1
        # Fake量化: 前向使用量化值, 反向传播使用STE(Straight-Through Estimator)
        if self.training:
            # 1) 量化: 浮点 x 仿射映射到整数(round 不可导), 并截断到 [qmin, qmax]
            x_q = torch.clamp(torch.round(x / self.scale) + self.zero_point, qmin, qmax)
            # 2) 反量化: 整数还原回浮点, x_r 是“带量化误差”的浮点值
            x_r = self.scale * (x_q - self.zero_point)
            # 3) STE 技巧: 前向输出等于 x_r, 反向梯度却当作恒等(=1)直通
            #    - 前向: detach 不改变数值, x + (x_r - x) = x_r, 所以前向用的是量化值
            #    - 反向: (x_r - x).detach() 被切断梯度, 只剩 x 这一项, d(out)/d(x) = 1
            #      梯度原样穿过, 绕开不可导的 round, 避免梯度消失
            # 一句话: 前向假装量化, 反向假装没量化, 让量化网络能正常用梯度训练(QAT)
            return x + (x_r - x).detach()
        else:
            # 推理时真正量化
            x_q = torch.clamp(torch.round(x / self.scale) + self.zero_point, qmin, qmax)
            return self.scale * (x_q - self.zero_point)


class QATConv2d(nn.Module):
    """带 QAT 的卷积层"""

    def __init__(self, conv, weight_bits=8, act_bits=8):
        super().__init__()
        self.conv = conv
        self.weight_quant = FakeQuantize(weight_bits)
        self.act_quant = FakeQuantize(act_bits)

    def forward(self, x):
        w = self.weight_quant(self.conv.weight)
        x = self.act_quant(x)
        return nn.functional.conv2d(
            x,
            w,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


# INT8 推理模拟
def calibrate_activation_ranges(model, dataloader, num_batches=10):
    """校准激活值范围 (用于 PTQ 的“激活校准”环节)

    它在量化的哪个环节:
        训练好 FP32 模型
          -> 量化权重(静态, 直接看权重张量 min/max, 不需数据)
          -> 【激活校准: 跑校准数据收集每层激活的 min/max】  <- 本函数
          -> 用 min/max 算激活的 scale / zero_point
          -> 转成量化模型推理
    为什么激活需要喂数据: 权重是静态的, 但激活依赖输入数据, 模型自己不知道
    激活会落在什么范围, 必须用有代表性的校准数据统计出真实范围。
    """
    act_ranges = {}
    hooks = []

    def hook_fn(name):
        # forward hook: 每次前向时被调用, 拿到该层的 output
        def fn(module, input, output):
            if name not in act_ranges:
                act_ranges[name] = {"min": float("inf"), "max": float("-inf")}
            # 在多个 batch 上滚动更新, 取累计的全局 min/max
            act_ranges[name]["min"] = min(act_ranges[name]["min"], output.min().item())
            act_ranges[name]["max"] = max(act_ranges[name]["max"], output.max().item())

        return fn

    # 注册 hook: 给每个 ReLU/Conv2d/Linear 挂钩子, 用来捕获其输出激活
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 在校准数据集上运行: eval + no_grad, 纯推理, 不更新权重也不做反向,
    # 只为统计激活的数值分布; 仅跑 num_batches 个 batch 即可
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(data)

    # 移除 hooks, 避免影响后续正常前向
    for h in hooks:
        h.remove()

    # 注意: 这是最朴素的 min/max 校准, 对离群值(outlier)敏感 ——
    # 一个异常大的激活会撑大范围 -> scale 变大 -> 整体精度下降。
    # 工业界更鲁棒的做法: percentile(百分位)、KL 散度/entropy(TensorRT 默认)、MSE 最优等。
    return act_ranges


# 校准数据选择策略
# 校准数据是 PTQ 中最容易被低估的环节。校准数据的选择直接决定量化的成败
def calibration_data_strategy(dataloader, num_calibration_samples: int = 1024):
    """Production-grade calibration data selection strategy.

    CRITICAL RULES:
    1. Calibration data MUST match the distribution of inference-time data.
       Using ImageNet to calibrate a model deployed on surveillance footage
       → activation ranges mismatch → silent accuracy degradation on dark scenes.

    2. Sample diversity matters more than sample count.
       200 diverse samples > 10,000 homogeneous samples.
       Include: different lighting conditions, object scales, backgrounds.

    3. Always include some "worst-case" samples (edge cases):
       - Extreme lighting (very dark / very bright)
       - Rare object poses and occlusions
       - These samples define your activation range's upper bound
       → using only "easy" samples gives tight ranges → quantization error on
         real-world hard cases is amplified 5-10×.

    4. For NLP models: calibration data should cover all sequence lengths
       that appear in production (short queries AND long documents).
       Padding-only tokens create spurious zero-activations that skew ranges.
    """
    # 策略：从生产流量中做分层采样(stratified sampling)
    # 80% 从近期生产日志随机采样
    # 20% 人工挑选的边界样本（暗光、模糊、遮挡、稀有类别）
    # 下面用“样本亮度(mean)”作为可量化的分层维度，桶内均匀抽取以保证多样性。

    # 1) 先从 dataloader 收集候选样本，并记录每个样本的亮度统计量
    samples, brightness = [], []
    for data, _ in dataloader:
        for i in range(data.size(0)):
            samples.append(data[i])
            brightness.append(data[i].float().mean().item())
        # 多收集一些（4 倍），给分层采样留出挑选空间
        if len(samples) >= num_calibration_samples * 4:
            break

    if len(samples) == 0:
        return {"calibration_data": None, "stable": False, "reason": "empty dataloader"}

    samples = torch.stack(samples)
    brightness = torch.tensor(brightness)

    # 2) 分层采样：按亮度排序后分成若干桶，每个桶里随机取等量样本
    #    这样既覆盖暗场景也覆盖亮场景，避免“只用简单样本导致激活范围过窄”
    num_take = min(num_calibration_samples, samples.size(0))
    num_bins = 10
    order = torch.argsort(brightness)
    per_bin = max(1, num_take // num_bins)
    picked = []
    for b in torch.chunk(order, num_bins):
        sel = b[torch.randperm(len(b))[:per_bin]]
        picked.append(sel)
    picked = torch.cat(picked)[:num_take]
    calib = samples[picked]

    # 3) 关键检查：激活范围稳定性
    #    用不同随机种子反复在子集上估计 scale 上限(max abs)，看波动有多大。
    #    若变异系数(CV) > 10%，说明校准集太小或不具代表性。
    scales = []
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        half = max(1, calib.size(0) // 2)
        sub = calib[torch.randperm(calib.size(0), generator=g)[:half]]
        scales.append(sub.abs().max().item())
    scales = torch.tensor(scales)
    cv = (scales.std(unbiased=False) / scales.mean().clamp(min=1e-8)).item()
    stable = cv < 0.10

    return {
        "calibration_data": calib,
        "num_samples": int(calib.size(0)),
        "scale_cv": cv,
        "stable": stable,
    }


# REAL-WORLD BUG CASE:
# A team calibrated INT8 quantization on ImageNet validation set,
# achieved <0.3% accuracy drop. Deployed the model.
# Result: model accuracy on user-uploaded photos dropped 8%.
# Root cause: ImageNet photos are professionally shot (good lighting, centered objects).
# User photos are blurry, tilted, with random lighting.
# The activation ranges during inference were 3x wider than calibration predicted.
# Solution: re-calibrated using 1000 random user-uploaded photos from production logs.


# ---------------- 可运行 demo：激活校准 + 校准数据策略 + QAT/STE 验证 ----------------
def build_demo_model():
    """一个极小的 CNN，仅用于演示量化流程"""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def make_synthetic_loader(num_samples=512, batch_size=32):
    """构造亮度差异较大的合成图像，便于体现分层采样的意义（明/暗场景都有）"""
    xs, ys = [], []
    for _ in range(num_samples):
        base = torch.rand(1).item()  # 每张图不同的基础亮度
        img = (torch.randn(3, 16, 16) * 0.2 + base).clamp(0, 1)
        xs.append(img)
        ys.append(torch.randint(0, 10, (1,)).item())
    ds = TensorDataset(torch.stack(xs), torch.tensor(ys))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print("\n================ INT8 / QAT Demo ================")
    loader = make_synthetic_loader()
    model = build_demo_model()

    # 1) PTQ：在校准数据上统计各层激活值的 min/max 范围
    ranges = calibrate_activation_ranges(model, loader, num_batches=4)
    print(f"\n[激活范围校准] 采集到的层数: {len(ranges)}")
    for name, r in list(ranges.items())[:3]:
        print(f"  {name}: min={r['min']:.3f}, max={r['max']:.3f}")

    # 2) 校准数据选择策略 + 稳定性检查
    rep = calibration_data_strategy(loader, num_calibration_samples=128)
    print("\n[校准数据策略]")
    print(f"  选中样本数            = {rep['num_samples']}")
    print(f"  scale 变异系数(CV)    = {rep['scale_cv']:.4f}")
    print(f"  校准集是否稳定(<10%)  = {rep['stable']}")

    # 3) QAT 卷积层：验证 STE 让梯度能穿过“不可导的 round”
    print("\n[QAT / STE 验证]")
    conv = nn.Conv2d(3, 4, 3, padding=1)
    qconv = QATConv2d(conv, weight_bits=8, act_bits=8)
    qconv.train()
    # 给 fake-quant 设一个合理 scale，否则默认 scale=1 会把小权重全量化成 0
    with torch.no_grad():
        qconv.weight_quant.scale.fill_(conv.weight.abs().max().item() / 127 + 1e-8)
        qconv.act_quant.scale.fill_(0.05)

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    out = qconv(x)
    loss = out.pow(2).mean()
    loss.backward()
    print(f"  输出形状      = {tuple(out.shape)}")
    print(f"  输入梯度范数  = {float(x.grad.norm()):.4f}  (STE 生效则非 0)")
    print(f"  权重梯度范数  = {float(conv.weight.grad.norm()):.4f}")
