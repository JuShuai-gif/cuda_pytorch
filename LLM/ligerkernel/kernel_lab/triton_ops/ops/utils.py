import functools
import importlib
import operator

from contextlib import contextmanager
from typing import Callable

import torch
import triton
import triton.language as tl

from packaging.version import Version
from liger_kernel.utils import infer_device


def is_hip() -> bool:
    return torch.version.hip is not None


@contextmanager
def device_context(device):
    # 上下文管理器：在 with 块内临时切换到指定设备（如 "cuda:0"）。
    # 根据设备类型解析对应后端（如 torch.cuda）；若该后端提供 .device() 上下文
    # 则使用之，否则（如 CPU 设备）直接 yield 原样通过，块结束后自动回到原设备。
    device = torch.device(device)
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield


def ensure_contiguous(fn):
    # 装饰器：调用 fn 前把 args/kwargs 中所有 torch.Tensor 通过 .contiguous() 强制
    # 转为内存连续（非张量保持原样），避免在 kernel 内部到处手动调用。
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    # 根据输入规模 n 计算 Triton kernel 的 BLOCK_SIZE 和 num_warps。
    # BLOCK_SIZE 取 n 的向上最近 2 的幂；若超过上限 65536 则报错，
    # 然后按 BLOCK_SIZE 大小逐级设定 num_warps（越大需要的 warp 越多，
    # HIP 平台上限降半）。
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def compare_version(package: str, operator: Callable, target: str):
    # 比较指定 Python 包的版本号：导入 package，用 operator（如 operator.ge）
    # 判断其版本是否满足与 target 的关系；包未安装时返回 False。
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def get_amp_custom_fwd_bwd() -> Callable:
    # 返回 AMP 混合精度前向/反向装饰器 custom_fwd/custom_bwd。
    # torch>=2.4 用 torch.amp.custom_fwd（按推断出的设备类型），
    # 否则优先 NPU 的 torch.npu.amp，最后回退到 torch.cuda.amp。
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    if hasattr(torch, "npu") and getattr(torch.npu, "amp", None) is not None:
        return torch.npu.amp.custom_fwd, torch.npu.amp.custom_bwd
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()


torch_to_triton_dtype = {
    # torch 到 triton 的数据类型映射表，供 kernel 按输入 dtype 选取对应的 tl 类型
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # 获取程序 ID（0 号维度）并转成 int64，防止大张量偏移量乘法时溢出
    program_id = tl.program_id(0).to(tl.int64)

    # 定位本程序负责的行起始地址
    X_ptr += program_id * X_stride

    # 加载标量梯度值（grad_output）
    grad_output = tl.load(grad_output_ptr)

    # 逐块对该行的 n_cols 个元素做原地乘法：X = X * grad_output
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def get_npu_core_count(default: int = 20) -> int:
    # 获取 NPU 向量核数量，用于 kernel 网格/并行度设定；
    # 若 Triton 运行时或 NPU 设备不可用，回退到默认值 default。
    try:
        utils = triton.runtime.driver.active.utils
        props = utils.get_device_properties(0)
        return int(props.get("num_vectorcore", default))
    except Exception:
        return default


def set_large_grf_mode(kernel_args: dict):
    # 为 XPU 设备开启大 GRF（General Register File，通用寄存器堆）模式，
    # 以提升寄存器容量。Triton 版本 >=3.6.0 使用字符串 "256"，更早版本用 "large"
    # （API 在 https://github.com/intel/intel-xpu-backend-for-triton/pull/5430 变更）。
    # 注意：随 pytorch-xpu 安装的 triton 包名是 pytorch-triton-xpu，源码安装的叫 triton。
    if compare_version("pytorch-triton-xpu", operator.ge, "3.6.0") or compare_version(
        "triton", operator.ge, "3.6.0"
    ):
        kernel_args["grf_mode"] = "256"
    else:
        kernel_args["grf_mode"] = "large"
