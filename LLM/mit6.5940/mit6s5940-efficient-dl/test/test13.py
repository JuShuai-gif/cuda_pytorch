import warnings

# pt2e 量化接口目前仍位于 torch.ao.quantization 命名空间下，会提示未来迁移到
# torchao。本环境未安装 torchao，故使用 torch 内置的 pt2e(图模式)接口 ——
# 它相对旧的 eager-mode 量化已经是 PyTorch 2.x 推荐的新范式。这里屏蔽该迁移提示。
warnings.filterwarnings("ignore", category=DeprecationWarning)

import copy
import torch
import torch.nn as nn
from torch.export import Dim
from torch.utils.data import TensorDataset, DataLoader

# pt2e(PyTorch 2 Export)量化新接口
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,  # PTQ：插入 Observer
    convert_pt2e,  # 转换为(参考)量化模型
    prepare_qat_pt2e,  # QAT：插入 FakeQuantize
)
from torch.ao.quantization import move_exported_model_to_eval
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


# ============================================================
# 说明：原脚本是一组引用了未定义对象(torchvision / *_loader / optimizer)的
# eager-mode 量化片段，已弃用且无法运行。这里改写为「自建小网络 + 合成数据」，
# 并改用 pt2e 图模式新接口跑通 PTQ / QAT，另含 LSQ 与 AMP 两段。
#
# 注意：
# 1) pt2e 基于 torch.export 导出计算图，再在图上做量化。
# 2) export 默认把 batch 维当作静态，需用 dynamic_shapes 声明为动态。
# 3) 当前 torch 版本下 conv+BN 的 QAT 折叠存在已知兼容问题(source_fn_stack)，
#    故网络不含 BatchNorm；pt2e 下 conv-relu 的融合由 quantizer 自动完成。
# ============================================================


class TinyNet(nn.Module):
    """无 BatchNorm 的小网络，便于 pt2e 图模式量化"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# export 用的示例输入；dynamic_shapes 把 batch 维(dim 0)声明为动态
EXAMPLE_INPUTS = (torch.randn(8, 3, 8, 8),)
DYNAMIC_SHAPES = ({0: Dim("batch", min=1, max=4096)},)


def export_graph(model):
    """用 torch.export 把 nn.Module 导出为可量化的训练图(GraphModule)"""
    return torch.export.export_for_training(
        model, EXAMPLE_INPUTS, dynamic_shapes=DYNAMIC_SHAPES
    ).module()


def make_loader(num_samples=128, batch_size=32):
    """合成图像数据(3x8x8)，替代原脚本里未定义的 calibration_loader/train_loader"""
    xs = torch.randn(num_samples, 3, 8, 8)
    ys = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size)


def count_quant_ops(gm):
    """统计图中 quantize/dequantize 节点数量，用于确认量化已生效"""
    return sum(1 for n in gm.graph.nodes if "quantize" in str(n.target))


# ---------------- 1) 训练后量化 PTQ (pt2e) ----------------
def run_ptq():
    print("\n[1] 训练后量化 PTQ (pt2e prepare_pt2e / convert_pt2e)")
    model = TinyNet().eval()
    graph = export_graph(model)

    # 选择 quantizer 并设置全局对称量化配置(XNNPACK 后端，CPU 友好)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    # 插入 Observer
    prepared = prepare_pt2e(graph, quantizer)

    # 用校准数据跑若干 batch，Observer 收集激活范围统计
    with torch.no_grad():
        for data, _ in make_loader():
            prepared(data)

    # 转换为量化模型(参考 q/dq 表示)
    quantized = convert_pt2e(prepared)
    print(f"  PTQ 完成，图中 quantize/dequantize 节点数: {count_quant_ops(quantized)}")
    return quantized


# ---------------- 2) 量化感知训练 QAT (pt2e) ----------------
def run_qat(steps=3):
    print("\n[2] 量化感知训练 QAT (pt2e prepare_qat_pt2e)")
    model = TinyNet().train()
    graph = export_graph(model)

    # is_qat=True：插入 FakeQuantize，前向模拟量化、反向用 STE
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_qat=True)
    )
    prepared = prepare_qat_pt2e(graph, quantizer)

    optimizer = torch.optim.SGD(prepared.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for i, (data, target) in enumerate(make_loader()):
        if i >= steps:
            break
        optimizer.zero_grad()
        loss = criterion(prepared(data), target)
        loss.backward()  # FakeQuantize 使用 STE
        optimizer.step()

    # 导出图不能直接用 .eval()，需用 pt2e 提供的切换函数
    move_exported_model_to_eval(prepared)
    quantized = convert_pt2e(prepared)
    print(f"  QAT 完成(最后一步 loss={loss.item():.4f})，已转为量化模型")
    return quantized


# ---------------- 3) LSQ：可学习步长量化 ----------------
class LSQQuantizer(torch.autograd.Function):
    """LSQ(Learned Step size Quantization)：把量化步长 scale 也作为可学习参数"""

    @staticmethod
    def forward(ctx, x, scale, nbits):
        # 量化：round 到整数并截断到 [-2^(n-1), 2^(n-1)-1]
        x_int = torch.clamp(
            torch.round(x / scale), -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
        )
        ctx.save_for_backward(x, scale)
        ctx.nbits = nbits
        return x_int * scale  # 反量化

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        nbits = ctx.nbits
        # 对 x 用 STE：超出量化范围的位置梯度置 0(被截断，无梯度)
        grad_x = grad_output.clone()
        grad_x[(x / scale).abs() > 2 ** (nbits - 1)] = 0
        # 对 scale 的梯度(简化版 LSQ 推导)
        x_div_s = x / scale
        grad_s = (x_div_s - x_div_s.round()).clamp(
            -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
        )
        grad_s = (grad_s * grad_output).sum()
        return grad_x, grad_s, None


def run_lsq():
    print("\n[3] LSQ 可学习步长量化")
    x = torch.randn(16, requires_grad=True)
    scale = torch.nn.Parameter(x.detach().abs().max() / (2**7 - 1))
    y = (LSQQuantizer.apply(x, scale, 8) ** 2).mean()
    y.backward()
    print(f"  x 梯度范数 = {float(x.grad.norm()):.4f}")
    print(f"  scale 梯度 = {float(scale.grad):.6f}  (LSQ 让步长可被学习)")


# ---------------- 4) AMP 自动混合精度训练 ----------------
def run_amp(steps=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[4] AMP 自动混合精度训练 (device={device})")
    use_cuda = device.type == "cuda"

    model = TinyNet().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    # 使用 torch.amp 新接口；CPU 上自动关闭 AMP，避免报错
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    for i, (data, target) in enumerate(make_loader()):
        if i >= steps:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_cuda):
            loss = criterion(model(data), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"  AMP 训练完成(最后一步 loss={loss.item():.4f})")


if __name__ == "__main__":
    print("================ 量化(pt2e) / AMP 全流程 Demo ================")
    run_ptq()
    run_qat()
    run_lsq()
    run_amp()
