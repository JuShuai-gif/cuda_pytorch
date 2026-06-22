from . import env as jit_env
from .core import gen_jit_spec


def gen_spdlog_module():
    return gen_jit_spec(
        "spdlog",
        [
            jit_env.FLASHINFER_CSRC_DIR / "logging.cc",
        ],
        extra_include_paths=[
            jit_env.SPDLOG_INCLUDE_DIR,
            jit_env.FLASHINFER_INCLUDE_DIR,
        ],
    )
