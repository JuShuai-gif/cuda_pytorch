
from .core import JitSpec, gen_jit_spec, current_compilation_context
from . import env as jit_env


def gen_mxfp8_quantization_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "mxfp8_quantization_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp8Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=None
        )
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        ],
    )
