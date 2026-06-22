from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_topk_module() -> JitSpec:
    return gen_jit_spec(
        "topk",
        [
            jit_env.FLASHINFER_CSRC_DIR / "topk.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_topk_binding.cu",
        ],
    )
