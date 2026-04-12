import functools

from ..jit.cubin_loader import setup_cubin_loader
from ..jit import gen_cudnn_fmha_module


@functools.cache
def get_cudnn_fmha_gen_module():
    mod = gen_cudnn_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op
