import platform
import sys

from importlib.metadata import version


def print_env_report():
    """
    Prints a report of the environment. Useful for debugging and reproducibility.

    Usage:
    ```
    python -m liger_kernel.env_report
    ```
    """
    print("Environment Report:")
    print("-------------------")
    # 打印操作系统信息（platform.platform() 返回详细系统描述）
    print(f"Operating System: {platform.platform()}")
    # 打印 Python 版本（sys.version.split()[0] 取版本号部分）
    print(f"Python version: {sys.version.split()[0]}")

    try:
        # 从已安装包的元数据中获取 liger-kernel 版本号
        print(f"Liger Kernel version: {version('liger-kernel')}")
    except ImportError:
        # 未安装 liger-kernel 时的提示
        print("Liger Kernel: Not installed")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        # 若 CUDA 可用则打印 CUDA 版本，否则提示不可用
        cuda_version = (
            torch.version.cuda if torch.cuda.is_available() else "Not available"
        )
        print(f"CUDA version: {cuda_version}")
        # 仅当 CUDA 可用且 HIP 版本信息存在时打印 HIP(ROCm) 版本，否则提示不可用
        hip_version = (
            torch.version.hip
            if torch.cuda.is_available() and torch.version.hip
            else "Not available"
        )
        print(f"HIP(ROCm) version: {hip_version}")

    except ImportError:
        # PyTorch 未安装时输出对应提示
        print("PyTorch: Not installed")
        print("CUDA version: Unable to query")
        print("HIP(ROCm) version: Unable to query")

    try:
        import triton

        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: Not installed")

    try:
        import transformers

        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")

    try:
        # 若 XPU (Intel GPU) 可用则打印版本，否则提示不可用
        xpu_version = (
            torch.version.xpu if torch.xpu.is_available() else "XPU Not Available"
        )
        print(f"XPU version: {xpu_version}")
    except ImportError:
        # 注意：此分支会捕获任何异常（如 torch 未安装导致的 ImportError）
        print("XPU version: Unable to query")


if __name__ == "__main__":
    # 仅当直接运行该文件时执行，作为模块导入时不触发
    print_env_report()
