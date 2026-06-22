
from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


def gen_fp4_kv_dequantization_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[8, 9, 10, 11, 12]
    )
    return gen_jit_spec(
        "fp4_kv_dequant",
        [jit_env.FLASHINFER_CSRC_DIR / "fp4_kv_dequantization.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16"],
    )