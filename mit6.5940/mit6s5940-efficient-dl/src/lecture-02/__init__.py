"""
Lecture 02: Resource Accounting -- FLOPs, Arithmetic Intensity, and Mixed Precision
第 02 讲：资源核算 -- FLOPs、算术强度与混合精度

This package provides utilities for analysing the computational and memory
characteristics of deep learning models:
本包提供用于分析深度学习模型的计算和内存特性的工具：

  - flops_counter: parameter counting, training FLOPs, memory footprint
    flops_counter: 参数统计、训练 FLOPs、内存占用估算
  - arithmetic_intensity: compute arithmetic intensity for key operations,
    roofline model visualisation
    arithmetic_intensity: 计算关键操作的算术强度、Roofline 模型可视化
  - mixed_precision: demonstrate fp16/bf16/fp32 differences, AMP training loop
    mixed_precision: 演示 fp16/bf16/fp32 差异、AMP 训练循环
"""

from lecture_02.flops_counter import (
    count_parameters,
    estimate_memory_footprint,
    estimate_training_flops,
    format_memory_summary,
)
from lecture_02.arithmetic_intensity import (
    compute_attention_intensity,
    compute_elementwise_intensity,
    compute_matmul_intensity,
    plot_roofline,
)
from lecture_02.mixed_precision import (
    describe_float_formats,
    demonstrate_precision_differences,
    simple_amp_training_loop,
)

__all__ = [
    # flops_counter: FLOPs 计数器
    "count_parameters",
    "estimate_training_flops",
    "estimate_memory_footprint",
    "format_memory_summary",
    # arithmetic_intensity: 算术强度分析
    "compute_matmul_intensity",
    "compute_attention_intensity",
    "compute_elementwise_intensity",
    "plot_roofline",
    # mixed_precision: 混合精度训练
    "describe_float_formats",
    "demonstrate_precision_differences",
    "simple_amp_training_loop",
]
