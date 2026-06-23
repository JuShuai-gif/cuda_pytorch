import torch
import torch.quantization as quant
from typing import Dict


# QAT 模型上线前的“引擎一致性”校验
def production_qat_validation(
    model_qat, model_fp32, test_loader, target_inference_engine: str = "tensorrt"
):
    """校验 QAT 模型与生产推理引擎的数值一致性。

    关键：QAT 的 FakeQuantize 用的是 PyTorch 的取整(round)规则，
    而 TensorRT / ONNX Runtime / TFLite 可能使用不同的取整模式。

    取整模式对比：
    | 引擎          | 取整模式                       |
    |---------------|--------------------------------|
    | PyTorch       | round-half-to-even（四舍六入五成双，默认） |
    | TensorRT      | round-half-away-from-zero（逢五远离零） |
    | TFLite        | round-half-away-from-zero      |
    | ONNX Runtime  | round-half-to-even（默认）      |

    这种 1-ULP 的取整差异会带来系统性偏置(bias)，并在逐层累积，
    在深层 Transformer 中尤其明显。

    解决办法：上线前务必把 QAT 输出与“真实推理引擎”的输出做比对，
    而不仅仅是和 PyTorch eval 模式比对。
    """
    model_qat.eval()
    model_fp32.eval()

    total_cosine_sim = 0.0
    total_l2_error = 0.0
    n_batches = 0

    with torch.no_grad():
        for data, _ in test_loader:
            out_qat = model_qat(data)
            out_fp32 = model_fp32(data)

            # 余弦相似度：安全部署一般要求 > 0.999
            cos_sim = torch.nn.functional.cosine_similarity(
                out_qat.flatten(), out_fp32.flatten(), dim=0
            )
            # L2 相对误差：安全部署一般要求 < 0.01
            l2_err = torch.norm(out_qat - out_fp32) / torch.norm(out_fp32)

            total_cosine_sim += cos_sim.item()
            total_l2_error += l2_err.item()
            n_batches += 1

    avg_cos = total_cosine_sim / n_batches
    avg_l2 = total_l2_error / n_batches

    return {
        "avg_cosine_similarity": round(avg_cos, 6),
        "avg_l2_relative_error": round(avg_l2, 6),
        "safe_to_deploy": avg_cos > 0.999 and avg_l2 < 0.01,
        "engine": target_inference_engine,
    }


def amp_numerical_safety_check(model, data, target, criterion):
    """在正式训练前检查 AMP(自动混合精度)的数值稳定性。

    AMP 在以下场景可能悄悄产生 NaN/Inf 梯度：
    - 极深网络（梯度下溢，低于 FP16 最小值 6e-5）
    - loss 过大（溢出 FP16 最大值 65504）
    - 长序列注意力（>1024）导致 softmax 溢出
    """
    # 根据模型所在设备自适应：无 GPU 时关闭 AMP，避免在 CPU 上报错
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    with torch.amp.autocast(device.type, enabled=use_cuda):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()

    # 检查梯度中是否出现 NaN / Inf
    nan_grads = []
    inf_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)

    if nan_grads:
        print(f"WARNING: 出现 NaN 梯度: {nan_grads}")
        print("  → 尝试降低学习率，或增大 GradScaler 的 init_scale")
    if inf_grads:
        print(f"WARNING: 出现 Inf 梯度: {inf_grads}")
        print("  → loss 可能溢出 FP16 范围(max=65504)")

    return {"nan_params": nan_grads, "inf_params": inf_grads}


# ---------------- 可运行 demo ----------------
import copy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def build_demo_model():
    """一个极小的 CNN，仅用于演示量化一致性校验"""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )


def fake_quant_weights_(model, bits=8):
    """把模型权重做一次 8-bit 对称量化/反量化，模拟 QAT 后的权重扰动"""
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.parameters():
            scale = p.abs().max().clamp(min=1e-8) / qmax
            p.copy_(torch.round(p / scale).clamp(-qmax, qmax) * scale)


def make_synthetic_loader(num_samples=128, batch_size=32):
    xs = torch.randn(num_samples, 3, 8, 8)
    ys = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"================ QAT 校验 / AMP 安全检查 Demo (device={device}) ================"
    )

    # 1) 构造 fp32 模型与“伪量化”模型(同结构、权重做了 8-bit 量化)
    model_fp32 = build_demo_model().to(device).eval()
    model_qat = copy.deepcopy(model_fp32)
    fake_quant_weights_(model_qat, bits=8)
    model_qat = model_qat.to(device).eval()

    test_loader = [(x.to(device), y.to(device)) for x, y in make_synthetic_loader()]

    rep = production_qat_validation(model_qat, model_fp32, test_loader, "tensorrt")
    print("\n[QAT 一致性校验]")
    print(f"  平均余弦相似度   = {rep['avg_cosine_similarity']}")
    print(f"  平均 L2 相对误差 = {rep['avg_l2_relative_error']}")
    print(f"  是否可安全部署   = {rep['safe_to_deploy']}  (目标引擎: {rep['engine']})")

    # 2) AMP 数值稳定性检查(CPU 上会自动关闭 AMP)
    print("\n[AMP 数值稳定性检查]")
    model = build_demo_model().to(device)
    data = torch.randn(8, 3, 8, 8, device=device)
    target = torch.randint(0, 10, (8,), device=device)
    out = amp_numerical_safety_check(model, data, target, nn.CrossEntropyLoss())
    print(f"  NaN 参数: {out['nan_params'] or '无'}")
    print(f"  Inf 参数: {out['inf_params'] or '无'}")
