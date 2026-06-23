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
    # 道理：autocast 让权重为 fp16 的 Conv/Linear 在低精度下计算，而对
    # 数值敏感的 BatchNorm 自动回退到 fp32，从而支持「权重混合精度」模型在
    # eager 模式下正常前向，避免 fp16 激活喂给 fp32 BN 时的 dtype 报错
    #
    # ── AMP 自动决定精度的内置算子清单（torch 逐算子判断，无需手动指定）──
    #   自动用 fp16（计算密集 + 对精度不敏感）：
    #       conv / linear / matmul / bmm / addmm 等
    #   自动强制 fp32（数值敏感 / 易溢出 / 做归约累加）：
    #       batchnorm / layernorm / softmax / exp / log / pow / sum /
    #       各类 loss（cross_entropy 等）
    # 关键：autocast 只在「计算时」临时转精度，权重存储仍是 fp32，
    #       所以单用 autocast 时 params_by_dtype 看到的依旧全是 fp32。
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )

    # 预热：至少 3 次迭代以触发 JIT 编译和自动调优
    for _ in range(3):
        with autocast_ctx:
            _ = model(dummy)

    torch.cuda.synchronize()

    # 道理：CPU 端的 time.time() 只能记录 Python 发起 kernel launch 的时刻，
    # 无法感知 GPU 队列中的实际执行进度。CUDA Event 在 GPU 指令流中插入
    # 时间戳记录点，GPU 硬件自己记录执行到该点的时间，是真正的"壁钟精度"
    # 使用 CUDA 事件进行计时推理（壁钟精度）
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    with torch.no_grad(), autocast_ctx:
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
        # 判断模型是否是 torch.nn.BatchNorm2d
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

# 道理：把 Conv/Linear 权重转成 fp16 以省显存/加速，但 BatchNorm 的均值方差
# 在 fp16 下易溢出或精度损失，工业界 AMP 标准做法是让 BN 保留 fp32。
# 这样模型权重就同时存在 fp16 与 fp32 两种 dtype，构成真正的「混合精度」
#
# ── .half() 是「物理转换」，不是临时视图/延迟计算 ──
#   1. 等价于 model.to(torch.float16)；递归遍历所有 Parameter 和 buffer，
#      为每个张量「新建一块 fp16 存储」并替换原 fp32 张量，原存储被释放，
#      显存实打实减半（本例 200MB → 106MB）。
#   2. 是 in-place 操作且返回 self；`model = model.half()` 中的赋值可省略。
#   3. 精度损失不可逆：数值被四舍五入到 fp16（10 位尾数、范围 ±65504），
#      过小下溢成 0、过大溢出成 inf，之后再 .float() 也救不回原始尾数。
#   4. 连 buffer 一起转：BatchNorm 的 running_mean/running_var 也会变 fp16，
#      所以下面对 BN 调 .float() 会把它的参数和 running stats 一起转回 fp32。
#
# 对比：.half() 永久改存储 dtype、真省显存、精度不可逆；
#       autocast 不改存储（权重仍 fp32）、只省计算时的中间量、权重无损。
model = model.half()
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.float()

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
#
# ── 量化能否照搬上面 AMP 那张表？方向对，但不能 1:1 ──
# 两者共享同一哲学：计算密集 + 对精度鲁棒的算子降精度，数值敏感/归约的留高精度。
# 重合点：Conv/Linear 同样是 INT8 量化的主力目标（对应 AMP 的 fp16 清单）。
# 但关键差异在于本质不同（fp16 是浮点、有指数位管动态范围；INT8 是整数、
# 靠 scale+zero_point 映射，怕的是数值分布/范围而非溢出累加）：
#   1. BatchNorm 处理方式相反：AMP 是「运行时保留 fp32」；量化是「直接折叠进
#      Conv 消灭掉」（即上面的 Conv-BN fusion），并不存在独立保 fp32 的 BN。
#   2. 量化有 AMP 没有的额外规则：通常把「首个 Conv」和「最后的分类 FC」留高
#      精度，因为它们对量化误差最敏感（输入分布/输出 logits 影响最大）。
#   3. Softmax/LayerNorm/GELU 等：两者都倾向留高精度，但量化往往需要专门的
#      量化算子实现，而非简单回退。
# 结论：可借鉴「哪类算子适合降精度」的直觉，但 BN、首尾层等具体规则要按量化
#       自己的方案走，不能直接套 AMP 的 fp16/fp32 清单。
if audit["unfused_bn_layers"]:
    print(
        f"WARNING: {len(audit['unfused_bn_layers'])} BN layers need fusion before INT8 quantization!"
    )
