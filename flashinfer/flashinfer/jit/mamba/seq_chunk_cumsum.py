from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec


def gen_seq_chunk_cumsum_module() -> JitSpec:
    """Generate JIT module for seq_chunk_cumsum kernel.

    No Jinja, no dtype parameterization — everything is int32.
    No architecture restrictions — plain CUDA (no tensor cores).
    """
    return gen_jit_spec(
        "mamba_seq_chunk_cumsum",
        [
            jit_env.FLASHINFER_CSRC_DIR / "seq_chunk_cumsum.cu",
            jit_env.FLASHINFER_CSRC_DIR / "seq_chunk_cumsum_jit_binding.cu",
        ],
    )
