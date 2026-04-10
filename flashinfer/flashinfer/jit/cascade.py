from .core import JitSpec,gen_jit_spec
from . import env as jit_env

def gen_cascade_module() -> JitSpec:
    return gen_jit_spec(
        "cascade",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cascade.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_cascade_binding.cu"
        ],
    )
















