"""
构建配置：编译所有 CUDA C++ kernel 为 torch C++ 扩展。

编译命令:
    cd 01_cuda_basics && python setup.py build_ext --inplace

清理:
    rm -rf 01_cuda_basics/build/
"""

import glob
import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _get_cuda_arch_list() -> str:
    """
    获取目标 GPU 计算能力列表。
    优先级：TORCH_CUDA_ARCH_LIST > nvcc 自动检测 > 默认值。
    默认值 "8.0;8.6;8.9;9.0" 覆盖了大部分现代 GPU。
    """
    # 1. 环境变量优先
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if env_arch:
        return env_arch

    # 2. 尝试从 nvcc 获取支持的架构
    try:
        result = subprocess.run(["nvcc", "--help"], capture_output=True, text=True, timeout=10)
        # 检查输出中是否包含 gpu-code 相关信息，间接判断可用性
        if "gpu-code" in result.stdout or "gpu-code" in result.stderr:
            # nvcc 可用，使用 torch 自动检测已安装 GPU 的 compute capability
            import torch

            if torch.cuda.is_available():
                sms = set()
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    sms.add(f"{props.major}{props.minor}")
                if sms:
                    return ";".join(sorted(sms))
    except Exception:
        pass

    # 3. 默认值：覆盖 A100(8.0)、RTX3090(8.6)、RTX4090(8.9)、H100(9.0)
    return "8.0;8.6;8.9;9.0"


def _build_arch_flags(arch_list: str) -> list[str]:
    """
    将 "8.0;8.6" 格式的计算能力列表转换为 nvcc 的 gencode 编译标志。
    """
    flags = []
    for sm in arch_list.split(";"):
        sm = sm.strip()
        if sm:
            # 使用 --generate-code 生成针对特定 SM 架构的代码
            flags.append(f"--generate-code=arch=compute_{sm},code=sm_{sm}")
    return flags


# 计算能力配置
CUDA_ARCH_LIST = _get_cuda_arch_list()
ARCH_FLAGS = _build_arch_flags(CUDA_ARCH_LIST)

# 使用 glob 自动发现所有 .cu 和 .cpp 源文件
_csrc_dir = Path(__file__).parent / "csrc"
sources = glob.glob(str(_csrc_dir / "**" / "*.cu"), recursive=True) + glob.glob(
    str(_csrc_dir / "**" / "*.cpp"), recursive=True
)

# 收集所有子目录作为 include 路径
include_dirs = [str(_csrc_dir)]
for subdir in _csrc_dir.glob("**/"):
    include_dirs.append(str(subdir))

# 编译参数
extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
    ],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-std=c++17",
    ]
    + ARCH_FLAGS,
}

setup(
    name="cuda_kernels",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
)
