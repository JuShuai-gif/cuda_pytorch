from . import env as jit_env
from .core import JitSpec, gen_jit_spec, current_compilation_context


def gen_mla_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11]
    )
    return gen_jit_spec(
        "mla",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cutlass_mla.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mla_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )