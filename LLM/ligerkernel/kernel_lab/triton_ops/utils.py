try:
    import peft  # noqa: F401

    PEFT_AVAILABLE = True
except ImportError:
    # peft 未安装时置为 False
    PEFT_AVAILABLE = False

import functools

import torch


def is_peft_available():
    """返回 peft (Parameter-Efficient Fine-Tuning) 库是否可用。"""
    return PEFT_AVAILABLE


def infer_comm_backend():
    """
    Get communication backend name based on the environment.
    """
    # 按优先级探测可用的分布式通信后端
    if torch.distributed.is_nccl_available():
        # Works for Nvidia
        # TODO: nccl may not work for AMD decices that may require use of rccl.
        # NVIDIA GPU 首选后端
        return "nccl"
    elif is_npu_available():
        # Use Ascend NPU if available (torch.npu)
        # Ascend is not standard torch backend and requires extension.
        # Assume that it is installed if NPUs are being used in
        # multi device environment.
        # 昇腾 NPU 使用 extend 后端
        return "ascend"
    # XPU (Intel) if available
    elif torch.distributed.distributed_c10d.is_xccl_available():
        # Intel XPU 的 xccl 后端
        return "xccl"
    elif torch.distributed.is_mpi_available():
        # CPU backend, first option
        # CPU 首选 MPI
        return "mpi"
    elif torch.distributed.is_gloo_available():
        # CPU backend, backup option
        # CPU 后备 Gloo
        return "gloo"
    else:
        # 没有可用的分布式后端则报错
        raise RuntimeError("There is no distributed backend available.")


def infer_device():
    """
    Get current device name based on available devices
    """
    # 按优先级推断当前计算设备类型
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        # NVIDIA 与 AMD 在 torch 中都表现为 "cuda"
        return "cuda"
    # Use Ascend NPU if available (torch.npu)
    elif is_npu_available():
        return "npu"
    # XPU (Intel) if available
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def is_npu_available() -> bool:
    """Detect Ascend NPU availability."""
    try:
        # 借用 transformers 的 NPU 检测函数
        from transformers.utils import is_torch_npu_available

        return is_torch_npu_available()
    except Exception:
        # 任何异常（含 transformers 未安装）都视为 NPU 不可用
        return False


# NVIDIA: CUDA compute capability (major, minor) -> coarse arch family
# NVIDIA：CUDA 计算能力 (主版本, 次版本) -> 粗略的架构代际名
_NVIDIA_ARCH_BY_CC = {
    (7, 0): "volta_turing",  # Volta V100
    (7, 5): "volta_turing",  # Turing T4 / RTX 20xx
    (8, 0): "ampere_ada",  # Ampere A100
    (8, 6): "ampere_ada",  # Ampere RTX 30xx / A40
    (8, 9): "ampere_ada",  # Ada Lovelace RTX 40xx / L4 / L40
    (9, 0): "hopper",  # H100 / H200
    (10, 0): "blackwell",  # B100 / B200 / GB200 (sm_100)
    (10, 3): "blackwell_ultra",  # B300 / GB300 (sm_103)
    (12, 0): "blackwell_consumer",  # RTX 50xx (sm_120)
}

# AMD: gfx target (gcnArchName) -> coarse arch family
# AMD：gfx 目标名 -> 粗略的架构代际名
_AMD_ARCH_BY_GFX = {
    "gfx908": "cdna",  # MI100
    "gfx90a": "cdna2",  # MI200
    "gfx940": "cdna3",  # MI300
    "gfx941": "cdna3",
    "gfx942": "cdna3",  # MI300X/MI300A
    "gfx1100": "rdna3",  # RX 7900
    "gfx1101": "rdna3",
    "gfx1102": "rdna3",
}


def _infer_nvidia_arch(device_id: int) -> str:
    # 获取 CUDA 计算能力，查表；未收录时回退为 "sm_<major><minor>" 形式
    major, minor = torch.cuda.get_device_capability(device_id)
    return _NVIDIA_ARCH_BY_CC.get((major, minor), f"sm_{major}{minor}")


def _infer_amd_arch(device_id: int) -> str:
    # gcnArchName looks like "gfx942:sramecc+:xnack-"; keep the gfx target only.
    # gcnArchName 形如 "gfx942:sramecc+:xnack-"，只取冒号前的 gfx 目标名
    gfx = getattr(torch.cuda.get_device_properties(device_id), "gcnArchName", "").split(
        ":"
    )[0]
    # 查表；未收录时回退为原始 gfx 名，空串回退为 "cuda"
    return _AMD_ARCH_BY_GFX.get(gfx, gfx or "cuda")


def _infer_xpu_arch(device_id: int) -> str:
    # 通过设备名关键字判断 Intel GPU 架构
    name = torch.xpu.get_device_properties(device_id).name.lower()
    if any(tag in name for tag in ("max", "pvc", "ponte")):
        return "pvc"  # Ponte Vecchio / Data Center GPU Max
    if any(tag in name for tag in ("arc", "battlemage", "alchemist")):
        return "arc"
    return "xpu"


def _infer_npu_arch(device_id: int) -> str:
    # 通过设备名判断昇腾 NPU 芯片型号
    name = torch.npu.get_device_properties(device_id).name.lower()
    if "910" in name:
        return "ascend910"
    if "310" in name:
        return "ascend310"
    return "npu"


@functools.lru_cache(maxsize=None)
def infer_device_arch(device_id: int = 0) -> str:
    """
    Get a coarse architecture/generation name for the current device.

    Returns a family name when detectable, falling back to the device type
    from ``infer_device()`` (e.g. ``"cpu"``) otherwise:

      - NVIDIA: ``"volta_turing"``, ``"ampere_ada"``, ``"hopper"``, ``"blackwell"``,
                ``"blackwell_ultra"``, ``"blackwell_consumer"`` (else ``"sm_<major><minor>"``)
      - AMD:    ``"cdna"``, ``"cdna2"``, ``"cdna3"``, ``"rdna3"`` (else the raw gfx target)
      - Intel:  ``"pvc"``, ``"arc"`` (else ``"xpu"``)
      - Ascend: ``"ascend910"``, ``"ascend310"`` (else ``"npu"``)

    The result is cached; call ``infer_device_arch.cache_clear()`` to reset.
    """
    # 推断当前设备的粗略架构代际名（结果带 LRU 缓存，可调用 cache_clear() 重置）
    device = infer_device()
    try:
        if device == "cuda":
            # ROCm reports as "cuda" in torch; torch.version.hip distinguishes AMD.
            # ROCm 在 torch 中同样显示为 "cuda"，用 torch.version.hip 区分 AMD
            return (
                _infer_amd_arch(device_id)
                if torch.version.hip
                else _infer_nvidia_arch(device_id)
            )
        if device == "xpu":
            return _infer_xpu_arch(device_id)
        if device == "npu":
            return _infer_npu_arch(device_id)
    except Exception:
        # 探测失败时回退为设备类型名（如 "cuda"、"cpu"）
        return device
    return device


def transformers_version_dispatch(
    required_version: str,
    before_fn,
    after_fn,
    before_args: tuple = (),
    after_args: tuple = (),
    before_kwargs: dict = None,
    after_kwargs: dict = None,
):
    """
    Dispatches to different functions based on package version comparison.

    Args:
        required_version: Version to compare against (e.g. "4.48.0")
        before_fn: Function to call if package_version < required_version
        after_fn: Function to call if package_version >= required_version
        before_args: Positional arguments for before_fn
        after_args: Positional arguments for after_fn
        before_kwargs: Keyword arguments for before_fn
        after_kwargs: Keyword arguments for after_fn

    Returns:
        Result from either before_fn or after_fn

    Example:
        >>> rotary_emb = transformers_version_dispatch(
        ...     "4.48.0",
        ...     LlamaRotaryEmbedding,
        ...     LlamaRotaryEmbedding,
        ...     before_args=(head_dim,),
        ...     after_args=(LlamaConfig(head_dim=head_dim),),
        ...     before_kwargs={'device': device},
        ...     after_kwargs={'device': device}
        ... )
    """
    # 根据 transformers 版本号选择调用 before_fn 还是 after_fn
    from packaging import version
    from transformers import __version__ as transformers_version

    before_kwargs = before_kwargs or {}
    after_kwargs = after_kwargs or {}

    if version.parse(transformers_version) < version.parse(required_version):
        # 版本低于 required_version 时调用 before_fn
        return before_fn(*before_args, **before_kwargs)
    else:
        # 版本达到 required_version 时调用 after_fn
        return after_fn(*after_args, **after_kwargs)


def get_total_gpu_memory() -> int:
    """Returns total GPU memory in GBs."""
    # 返回设备总显存（单位 GB），按设备类型分发
    device = infer_device()
    if device == "cuda":
        # // (1024**3) 将字节数换算为 GiB
        return torch.cuda.get_device_properties(0).total_memory // (1024**3)
    elif device == "xpu":
        return torch.xpu.get_device_properties(0).total_memory // (1024**3)
    elif device == "npu":
        return torch.npu.get_device_properties(0).total_memory // (1024**3)
    else:
        raise RuntimeError(f"Unsupported device: {device}")
