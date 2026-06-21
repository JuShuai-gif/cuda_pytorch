import torch
import torchvision.models as models

import time
from typing import Dict, Tuple


def production_model_audit(
    model: torch.nn.Module, input_shape: Tuple[int, ...], device: str = "cuda"
) -> Dict:
    """生产级模型审计，捕获常见陷阱。

    相比简单的参数计数，额外功能包括：
    - 检测融合与未融合的 BN 层（量化前需要融合）
    - 使用 CUDA 缓存分配器测量内存（而非仅 param * 4）
    - 使用 CUDA 事件进行精确 GPU 计时（而非 CPU 端时间戳）
    """
    # 道理：eval() 会禁用 Dropout 的随机丢弃和 BatchNorm 的 running stats
    # 更新，确保推理阶段行为确定且与训练时指数移动平均一致，避免每次推理结果不同
    model = model.to(device).eval()

    # 道理：reset_peak_memory_stats 清零峰值记录，empty_cache 释放 PyTorch
    # 缓存的显存块。若不清理，之前代码分配的碎片会影响本次测量结果，
    # 导致峰值读数偏高或偏低
    # 使用 CUDA 内存快照进行精确测量
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy = torch.randn(*input_shape, device=device)

    # 道理：CUDA kernel 首次启动时需 JIT 编译 PTX 代码，cuBLAS/cuDNN 会
    # 根据矩阵形状自动搜索最优算法（autotune），这些一次性开销可占总耗时的
    # 30% 以上。预热 3+ 次后这些开销消失，之后的计时才反映稳态推理延迟
    # 预热：至少 3 次迭代以触发 JIT 编译和自动调优
    for _ in range(3):
        _ = model(dummy)
    torch.cuda.synchronize()

    # 道理：CPU 端的 time.time() 只能记录 Python 发起 kernel launch 的时刻，
    # 无法感知 GPU 队列中的实际执行进度。CUDA Event 在 GPU 指令流中插入
    # 时间戳记录点，GPU 硬件自己记录执行到该点的时间，是真正的"壁钟精度"
    # 使用 CUDA 事件进行计时推理（壁钟精度）
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    with torch.no_grad():
        _ = model(dummy)

    ender.record()

    # 道理：synchronize() 阻塞 CPU 直到 GPU 上所有提交的操作执行完毕。
    # 若不调用，ender.record() 可能还未执行完成就读取时间，得到错误值
    torch.cuda.synchronize()
    latency = starter.elapsed_time(ender)

    # 道理：param.numel() * param.element_size() 只算参数自身的显存，
    # 遗漏了 cuDNN workspace、临时缓冲区、autograd 图等隐式分配。
    # max_memory_allocated() 统计了 PyTorch 缓存分配器分配过的所有显存峰值
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

    # 道理：BN（BatchNorm）在 INT8 量化时必须 fuse 到前面的 Conv 层
    # （即把 BN 的 scale/bias 折叠进 Conv 的 weight/bias）。独立的 BN 层
    # 需要额外的量化/反量化操作，导致精度严重下降且推理变慢
    # 检查量化前需要融合的 BN 层
    unfused_bn = []

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            unfused_bn.append(name)

    # 道理：混合精度训练/推理中不同层可能使用不同 dtype
    # （如 Embedding 用 fp32、Linear 用 fp16 或 bf16）。按 dtype 统计
    # 可以快速发现精度分配问题：例如某个层意外停留在 fp32 导致显存浪费
    # 按数据类型统计参数数量
    params_by_dtype = {}
    for name, p in model.named_parameters():
        if p.dtype not in params_by_dtype:
            params_by_dtype[p.dtype] = 0
        params_by_dtype[p.dtype] += p.numel()

    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "peak_memory_mb": round(peak_memory, 2),
        "latency_ms": round(latency, 2),
        "unfused_bn_layers": unfused_bn,
        "params_by_dtype": params_by_dtype,
    }


# 道理：执行此审计脚本可以快速获取模型的延迟、显存、BN 融合需求等
# 关键指标，方便在部署前判断模型是否满足生产环境的性能约束
# 使用示例
model = models.resnet50()
audit = production_model_audit(model, (1, 3, 224, 224), "cuda")
print(f"Latency: {audit['latency_ms']}ms | Memory: {audit['peak_memory_mb']}MB")
print(
    f"Total params: {audit['total_params']:,} | Trainable: {audit['trainable_params']:,}"
)
print(f"Params by dtype: { {str(k): v for k, v in audit['params_by_dtype'].items()} }")
# 道理：此处的 WARNING 是「待办提醒」而非报错，模型仍可正常推理，但提示
# 量化流水线还差一步 Conv-BN fusion。含义说明如下：
#   - 检出的 BN 数量（ResNet50 为 53）= 16 个 Bottleneck × 3 + 4 个 downsample
#     的 BN + stem 的 bn1，共 53 个 BatchNorm2d
#   - BN 推理公式 y = (x - μ)/sqrt(σ²+ε) * γ + β 是一组固定的线性缩放/偏移，
#     可在数学上等价地折叠进前一层 Conv 的 weight/bias（即 Conv-BN fusion）
#   - 若「不融合」就做 INT8 量化：每个独立 BN 都要插入额外的 quantize/dequantize，
#     既拖慢推理，又因多一次量化误差导致精度严重下降
#   - PyTorch 中可用 torch.quantization.fuse_modules 或 torch.ao.quantization
#     的 fuse 接口自动完成 Conv+BN(+ReLU) 的融合
if audit["unfused_bn_layers"]:
    print(
        f"WARNING: {len(audit['unfused_bn_layers'])} BN layers need fusion before INT8 quantization!"
    )
