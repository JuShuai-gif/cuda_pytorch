
from .core import JitSpec, gen_jit_spec, current_compilation_context
from . import env as jit_env


def gen_comm_alltoall_module() -> JitSpec:
    return gen_jit_spec(
        "comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_alltoall.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_alltoall_prepare.cu",
        ],
    )


def gen_trtllm_mnnvl_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_mnnvl_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_mnnvl_allreduce.cu",
        ],
    )


def gen_trtllm_comm_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10]
    )
    return gen_jit_spec(
        "trtllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce_fusion.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_allreduce_fusion.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )


def gen_vllm_comm_module() -> JitSpec:
    return gen_jit_spec(
        "vllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "vllm_custom_all_reduce.cu",
        ],
    )


def gen_moe_alltoall_module() -> JitSpec:
    return gen_jit_spec(
        "mnnvl_moe_alltoall",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_alltoall.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "communicationKernels"
            / "moeAlltoAllKernels.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "cpp"
            / "common"
            / "envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "cpp"
            / "common"
            / "tllmException.cpp",
        ],
        extra_include_paths=[
            str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal"),
            str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include"),
        ],
    )




















