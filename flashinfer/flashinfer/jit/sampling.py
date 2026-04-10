from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_sampling_module() -> JitSpec:
    return gen_jit_spec(
        "sampling",
        [
            jit_env.FLASHINFER_CSRC_DIR / "sampling.cu",
            jit_env.FLASHINFER_CSRC_DIR / "renorm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_sampling_binding.cu",
        ],
    )
