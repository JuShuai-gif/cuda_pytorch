from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_rope_module() -> JitSpec:
    return gen_jit_spec(
        "rope",
        [
            jit_env.FLASHINFER_CSRC_DIR / "rope.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_rope_binding.cu",
        ],
    )
