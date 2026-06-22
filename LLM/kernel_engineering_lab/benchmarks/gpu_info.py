"""
GPU 硬件规格数据库 - 用于 Roofline 分析和性能预估。

借鉴: CUTLASS profiler, NVIDIA CUDA Programming Guide, TechPowerUp GPU DB

数据来源:
  - NVIDIA CUDA Programming Guide (Compute Capability)
  - NVIDIA 官方 whitepaper (架构规格)
  - techpowerup.com (消费级 GPU 规格)
  -实测数据 (bandwidth 验证)

用途:
  - Roofline 分析: 判断 kernel 是 memory-bound 还是 compute-bound
  - 性能预估: 根据算术强度预估 kernel 性能上限
  - 硬件对比: 跨 GPU 代际性能比较
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GpuSpec:
    """单个 GPU 的完整硬件规格。"""

    model: str
    architecture: str
    compute_capability: str
    sm_count: int
    max_threads_per_sm: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    max_registers_per_sm: int
    max_shared_memory_per_sm_bytes: int
    l1_cache_per_sm_bytes: int
    l2_cache_bytes: int
    memory_bus_width_bits: int
    memory_clock_mhz: float
    memory_bandwidth_gbps: float

    # 峰值理论吞吐量 (TFLOPS)
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_bf16_tflops: float
    peak_tensor_core_fp16_tflops: float
    peak_tensor_core_bf16_tflops: float
    peak_tensor_core_tf32_tflops: float

    # 派生属性: 操作强度脊点 (FLOP/Byte)
    @property
    def ridge_point_fp32(self) -> float:
        """FP32 Roofline 脊点: 峰值 TFLOPS / 峰值带宽 (GFLOP/GB)。"""
        return self.peak_fp32_tflops * 1000.0 / self.memory_bandwidth_gbps

    @property
    def ridge_point_fp16(self) -> float:
        """FP16 Roofline 脊点。"""
        return self.peak_fp16_tflops * 1000.0 / self.memory_bandwidth_gbps

    @property
    def ridge_point_tc_fp16(self) -> float:
        """Tensor Core FP16 Roofline 脊点。"""
        return self.peak_tensor_core_fp16_tflops * 1000.0 / self.memory_bandwidth_gbps


# ============================================================================
# GPU 硬件规格数据库
# ============================================================================

GPU_SPECS: Dict[str, GpuSpec] = {
    # ---- NVIDIA RTX 40 系列 (Ada Lovelace) ----
    "NVIDIA GeForce RTX 4070": GpuSpec(
        model="NVIDIA GeForce RTX 4070",
        architecture="Ada Lovelace",
        compute_capability="8.9",
        sm_count=46,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=37748736,  # 36MB
        memory_bus_width_bits=192,
        memory_clock_mhz=1313,
        memory_bandwidth_gbps=504.2,
        peak_fp32_tflops=29.15,
        peak_fp16_tflops=29.15,
        peak_bf16_tflops=29.15,
        peak_tensor_core_fp16_tflops=116.6,
        peak_tensor_core_bf16_tflops=116.6,
        peak_tensor_core_tf32_tflops=58.3,
    ),
    "NVIDIA GeForce RTX 4070 SUPER": GpuSpec(
        model="NVIDIA GeForce RTX 4070 SUPER",
        architecture="Ada Lovelace",
        compute_capability="8.9",
        sm_count=56,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=50331648,  # 48MB
        memory_bus_width_bits=192,
        memory_clock_mhz=1313,
        memory_bandwidth_gbps=504.2,
        peak_fp32_tflops=35.48,
        peak_fp16_tflops=35.48,
        peak_bf16_tflops=35.48,
        peak_tensor_core_fp16_tflops=141.9,
        peak_tensor_core_bf16_tflops=141.9,
        peak_tensor_core_tf32_tflops=71.0,
    ),
    "NVIDIA GeForce RTX 4070 Ti": GpuSpec(
        model="NVIDIA GeForce RTX 4070 Ti",
        architecture="Ada Lovelace",
        compute_capability="8.9",
        sm_count=60,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=50331648,
        memory_bus_width_bits=192,
        memory_clock_mhz=1313,
        memory_bandwidth_gbps=504.2,
        peak_fp32_tflops=40.09,
        peak_fp16_tflops=40.09,
        peak_bf16_tflops=40.09,
        peak_tensor_core_fp16_tflops=160.4,
        peak_tensor_core_bf16_tflops=160.4,
        peak_tensor_core_tf32_tflops=80.2,
    ),
    "NVIDIA GeForce RTX 4080": GpuSpec(
        model="NVIDIA GeForce RTX 4080",
        architecture="Ada Lovelace",
        compute_capability="8.9",
        sm_count=76,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=67108864,  # 64MB
        memory_bus_width_bits=256,
        memory_clock_mhz=1400,
        memory_bandwidth_gbps=716.8,
        peak_fp32_tflops=48.74,
        peak_fp16_tflops=48.74,
        peak_bf16_tflops=48.74,
        peak_tensor_core_fp16_tflops=195.0,
        peak_tensor_core_bf16_tflops=195.0,
        peak_tensor_core_tf32_tflops=97.5,
    ),
    "NVIDIA GeForce RTX 4090": GpuSpec(
        model="NVIDIA GeForce RTX 4090",
        architecture="Ada Lovelace",
        compute_capability="8.9",
        sm_count=128,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=75497472,  # 72MB
        memory_bus_width_bits=384,
        memory_clock_mhz=1313,
        memory_bandwidth_gbps=1008.0,
        peak_fp32_tflops=82.58,
        peak_fp16_tflops=82.58,
        peak_bf16_tflops=82.58,
        peak_tensor_core_fp16_tflops=330.3,
        peak_tensor_core_bf16_tflops=330.3,
        peak_tensor_core_tf32_tflops=165.2,
    ),
    # ---- NVIDIA RTX 30 系列 (Ampere) ----
    "NVIDIA GeForce RTX 3060": GpuSpec(
        model="NVIDIA GeForce RTX 3060",
        architecture="Ampere",
        compute_capability="8.6",
        sm_count=28,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=16,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=3145728,  # 3MB
        memory_bus_width_bits=192,
        memory_clock_mhz=1875,
        memory_bandwidth_gbps=360.0,
        peak_fp32_tflops=12.74,
        peak_fp16_tflops=12.74,
        peak_bf16_tflops=12.74,
        peak_tensor_core_fp16_tflops=51.0,
        peak_tensor_core_bf16_tflops=51.0,
        peak_tensor_core_tf32_tflops=25.5,
    ),
    "NVIDIA GeForce RTX 3080": GpuSpec(
        model="NVIDIA GeForce RTX 3080",
        architecture="Ampere",
        compute_capability="8.6",
        sm_count=68,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=16,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=5242880,
        memory_bus_width_bits=320,
        memory_clock_mhz=2375,
        memory_bandwidth_gbps=760.3,
        peak_fp32_tflops=29.77,
        peak_fp16_tflops=29.77,
        peak_bf16_tflops=29.77,
        peak_tensor_core_fp16_tflops=119.1,
        peak_tensor_core_bf16_tflops=119.1,
        peak_tensor_core_tf32_tflops=59.5,
    ),
    "NVIDIA GeForce RTX 3090": GpuSpec(
        model="NVIDIA GeForce RTX 3090",
        architecture="Ampere",
        compute_capability="8.6",
        sm_count=82,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=16,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=6291456,
        memory_bus_width_bits=384,
        memory_clock_mhz=2437,
        memory_bandwidth_gbps=936.2,
        peak_fp32_tflops=35.58,
        peak_fp16_tflops=35.58,
        peak_bf16_tflops=35.58,
        peak_tensor_core_fp16_tflops=142.3,
        peak_tensor_core_bf16_tflops=142.3,
        peak_tensor_core_tf32_tflops=71.2,
    ),
    # ---- 数据中心 GPU ----
    "NVIDIA A100": GpuSpec(
        model="NVIDIA A100",
        architecture="Ampere",
        compute_capability="8.0",
        sm_count=108,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=167936,
        l1_cache_per_sm_bytes=196608,
        l2_cache_bytes=41943040,  # 40MB
        memory_bus_width_bits=5120,
        memory_clock_mhz=1215,
        memory_bandwidth_gbps=2039.0,
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        peak_tensor_core_fp16_tflops=312.0,
        peak_tensor_core_bf16_tflops=312.0,
        peak_tensor_core_tf32_tflops=156.0,
    ),
    "NVIDIA H100": GpuSpec(
        model="NVIDIA H100",
        architecture="Hopper",
        compute_capability="9.0",
        sm_count=132,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=228352,
        l1_cache_per_sm_bytes=262144,
        l2_cache_bytes=52428800,  # 50MB
        memory_bus_width_bits=5120,
        memory_clock_mhz=1593,
        memory_bandwidth_gbps=3352.0,
        peak_fp32_tflops=60.0,
        peak_fp16_tflops=989.0,
        peak_bf16_tflops=989.0,
        peak_tensor_core_fp16_tflops=989.0,
        peak_tensor_core_bf16_tflops=989.0,
        peak_tensor_core_tf32_tflops=494.5,
    ),
    # ---- NVIDIA 专业/工作站 GPU ----
    "NVIDIA A10": GpuSpec(
        model="NVIDIA A10",
        architecture="Ampere",
        compute_capability="8.6",
        sm_count=72,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=16,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=6291456,
        memory_bus_width_bits=384,
        memory_clock_mhz=1563,
        memory_bandwidth_gbps=600.0,
        peak_fp32_tflops=31.2,
        peak_fp16_tflops=31.2,
        peak_bf16_tflops=31.2,
        peak_tensor_core_fp16_tflops=125.0,
        peak_tensor_core_bf16_tflops=125.0,
        peak_tensor_core_tf32_tflops=62.5,
    ),
    "NVIDIA T4": GpuSpec(
        model="NVIDIA T4",
        architecture="Turing",
        compute_capability="7.5",
        sm_count=40,
        max_threads_per_sm=1024,
        max_warps_per_sm=32,
        max_blocks_per_sm=16,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=65536,
        l1_cache_per_sm_bytes=65536,
        l2_cache_bytes=6291456,
        memory_bus_width_bits=256,
        memory_clock_mhz=1250,
        memory_bandwidth_gbps=320.0,
        peak_fp32_tflops=8.1,
        peak_fp16_tflops=65.0,
        peak_bf16_tflops=0.0,
        peak_tensor_core_fp16_tflops=65.0,
        peak_tensor_core_bf16_tflops=0.0,
        peak_tensor_core_tf32_tflops=0.0,
    ),
    "NVIDIA V100": GpuSpec(
        model="NVIDIA V100",
        architecture="Volta",
        compute_capability="7.0",
        sm_count=80,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=98304,
        l1_cache_per_sm_bytes=131072,
        l2_cache_bytes=6291456,
        memory_bus_width_bits=4096,
        memory_clock_mhz=877,
        memory_bandwidth_gbps=900.0,
        peak_fp32_tflops=15.7,
        peak_fp16_tflops=125.0,
        peak_bf16_tflops=0.0,
        peak_tensor_core_fp16_tflops=125.0,
        peak_tensor_core_bf16_tflops=0.0,
        peak_tensor_core_tf32_tflops=0.0,
    ),
}


def detect_gpu() -> Optional[GpuSpec]:
    """
    自动检测当前 GPU 并返回对应的硬件规格。

    Returns:
        GpuSpec 对象，若检测失败返回 None。
    """
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    device_name = torch.cuda.get_device_name(0)

    # 精确匹配
    if device_name in GPU_SPECS:
        return GPU_SPECS[device_name]

    # 模糊匹配：去掉 "NVIDIA " 前缀、大小写不敏感
    name_lower = device_name.lower().replace("nvidia ", "")
    for key, spec in GPU_SPECS.items():
        key_lower = key.lower().replace("nvidia ", "")
        if key_lower in name_lower or name_lower in key_lower:
            return spec

    # 尝试从 compute capability 推断
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    for key, spec in GPU_SPECS.items():
        if spec.compute_capability == cc:
            return spec

    return None


def get_gpu_spec(gpu_name: str) -> Optional[GpuSpec]:
    """
    根据 GPU 名称获取硬件规格。

    Args:
        gpu_name: GPU 名称，如 "NVIDIA GeForce RTX 4070" 或部分匹配 "RTX 4070"。

    Returns:
        GpuSpec 对象，若未找到返回 None。
    """
    if gpu_name in GPU_SPECS:
        return GPU_SPECS[gpu_name]

    name_lower = gpu_name.lower()
    for key, spec in GPU_SPECS.items():
        key_lower = key.lower()
        if key_lower in name_lower or name_lower in key_lower:
            return spec

    return None


def list_all_gpus() -> None:
    """打印所有已知 GPU 规格。"""
    headers = ["GPU Model", "Arch", "CC", "SMs", "FP32 TFLOPS", "FP16 TFLOPS", "BW GB/s"]
    rows = []
    for key, spec in GPU_SPECS.items():
        rows.append(
            [
                key,
                spec.architecture,
                spec.compute_capability,
                str(spec.sm_count),
                f"{spec.peak_fp32_tflops:.1f}",
                f"{spec.peak_tensor_core_fp16_tflops:.1f}",
                f"{spec.memory_bandwidth_gbps:.0f}",
            ]
        )

    col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    list_all_gpus()
    print()
    detected = detect_gpu()
    if detected:
        print(f"Detected GPU: {detected.model}")
        print(f"  Architecture:     {detected.architecture}")
        print(f"  Compute Cap.:     {detected.compute_capability}")
        print(f"  SM Count:         {detected.sm_count}")
        print(f"  Peak BW:          {detected.memory_bandwidth_gbps} GB/s")
        print(f"  Peak FP32:        {detected.peak_fp32_tflops} TFLOPS")
        print(f"  Peak TC FP16:     {detected.peak_tensor_core_fp16_tflops} TFLOPS")
        print(f"  Ridge FP32:       {detected.ridge_point_fp32:.1f} FLOP/Byte")
        print(f"  Ridge TC FP16:    {detected.ridge_point_tc_fp16:.1f} FLOP/Byte")
    else:
        print("No supported GPU detected.")
