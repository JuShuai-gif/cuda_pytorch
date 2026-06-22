from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_quantization_module() -> JitSpec:
    return gen_jit_spec(
        "quantization",
        [
            jit_env.FLASHINFER_CSRC_DIR / "quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_quantization_binding.cu",
        ],
    )