
from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_norm_module() -> JitSpec:
    nvcc_flags = [
        "-DENABLE_BF16",
        "-DENABLE_FP8",
    ]
    return gen_jit_spec(
        "norm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "norm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_norm_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )