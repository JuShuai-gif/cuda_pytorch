from types import SimpleNamespace


def gen_deepgemm_sm100_module() -> SimpleNamespace:
    from flashinfer.deep_gemm import load_all
    from flashinfer.gemm import (
        group_deepgemm_fp8_nt_groupwise,
        batch_deepgemm_fp8_nt_groupwise,
    )

    load_all()
    return SimpleNamespace(
        group_deepgemm_fp8_nt_groupwise=group_deepgemm_fp8_nt_groupwise,
        batch_deepgemm_fp8_nt_groupwise=batch_deepgemm_fp8_nt_groupwise,
    )
