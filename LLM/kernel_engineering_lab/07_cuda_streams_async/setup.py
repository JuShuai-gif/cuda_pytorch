"""
构建 CUDA stream 编程实战的 C++ 扩展模块。

Build:
    cd 07_cuda_streams_async && python setup.py build_ext --inplace

Clean:
    rm -rf 07_cuda_streams_async/build/
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _get_cuda_arch_flags() -> list[str]:
    """根据当前 GPU 计算能力返回 nvcc 架构编译选项。"""
    try:
        import torch

        if torch.cuda.is_available():
            major_minor = set()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                major_minor.add(f"{props.major}{props.minor}")

            flags = []
            for sm in sorted(major_minor):
                flags.extend(["-gencode", f"arch=compute_{sm},code=sm_{sm}"])
            return flags
    except Exception:
        pass

    fallback_sms = ["70", "75", "80", "86", "89", "90"]
    flags = []
    for sm in fallback_sms:
        flags.extend(["-gencode", f"arch=compute_{sm},code=sm_{sm}"])
    return flags


_csrc_dir = Path(__file__).parent / "csrc"

sources = [
    str(_csrc_dir / "stream_kernels.cu"),
    str(_csrc_dir / "bindings_stream.cpp"),
]

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
    ]
    + _get_cuda_arch_flags(),
}

import os

if "TORCH_CUDA_ARCH_LIST" in os.environ:
    arch_list = os.environ["TORCH_CUDA_ARCH_LIST"]
    cc_flags = []
    for sm in arch_list.split(";"):
        sm = sm.strip()
        if sm:
            cc_flags.extend(["-gencode", f"arch=compute_{sm},code=sm_{sm}"])
    nvcc_flags = extra_compile_args["nvcc"]
    nvcc_no_gencode = [f for f in nvcc_flags if not f.startswith("-gencode")]
    extra_compile_args["nvcc"] = nvcc_no_gencode + cc_flags


setup(
    name="cuda_stream_kernels",
    ext_modules=[
        CUDAExtension(
            name="cuda_stream_kernels",
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
)
