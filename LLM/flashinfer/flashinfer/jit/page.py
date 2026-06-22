from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_page_module() -> JitSpec:
    return gen_jit_spec(
        "page",
        [
            jit_env.FLASHINFER_CSRC_DIR / "page.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_page_binding.cu",
        ],
    )