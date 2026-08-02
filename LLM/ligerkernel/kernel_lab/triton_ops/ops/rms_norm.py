import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import set_large_grf_mode
from liger_kernel.ops.utils import torch_to_triton_dtype
from liger_kernel.utils import is_npu_available

# rsqrt（1/sqrt(x)，平方根倒数）的跨版本兼容导入。
# Triton >= 3.0 把数学函数从 tl.math 移到 tl.extra.libdevice（CUDA 专用），
# 而 NPU 后端没有 libdevice，因此仅对非 NPU 且版本达标的情况走 libdevice，
# 其余情况回退到 tl.math。
if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # 新版本标准路径，支持按设备自动分发
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # NGC 容器里 Triton 的路径不同，回退到 cuda.libdevice
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    # Triton < 3.0.0 或 NPU 设备走 tl.math
    from triton.language.math import rsqrt


# 部分 torch 构建不会主动加载 torch.distributed.tensor 子模块，
# 直接访问 torch.distributed.tensor.DTensor 可能抛
# AttributeError: module 'torch.distributed' has no attribute 'tensor'。
# 因此对 DTensor 做防御性导入：能导入时触发子模块加载并拿到一个可
# isinstance 判断的类；不可用时回退为 ()，使 isinstance 检查安全地变为
# no-op（普通非分布式张量永远不会是 DTensor）。
try:
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:
    _DTensor = ()

# RMSNorm 的三种 cast 模式常量，用 tl.constexpr 包装成 Triton 编译期常量
# （值在编译时确定，不占用运行时计算）。
# NONE：不做 dtype 转换；LLAMA：Llama 风格（x 与 x2 用不同精度处理）；
# GEMMA：Gemma 风格（fp32 输入 upcast 到 fp64 再做 RMSNorm）。
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """
    """
    定位本程序（线程块）负责处理的行，并准备列偏移与越界掩码。

    这个 kernel 的并行策略是：每个线程块处理矩阵的一行。
    - row_idx = tl.program_id(0)：当前线程块的编号（0 号维度），即要处理的
      行号。`.to(tl.int64)` 转成 64 位整数，防止大张量（行数多、偏移大）下
      row_idx * stride 这类偏移乘法发生 32 位整数溢出。
    - col_offsets = tl.arange(0, BLOCK_SIZE)：生成 [0,1,...,BLOCK_SIZE-1] 的
      列索引向量，用于向量化地同时访问一行的所有列。
    - mask = col_offsets < n_cols：越界掩码。真实列数 n_cols 可能不是 BLOCK_SIZE
      的整数倍（BLOCK_SIZE 是向上取整的 2 的幂），多出来的位置不存在数据，
      用 mask 屏蔽这些访问，避免读到越界内存。

    然后根据 row_idx 和每行的内存步长（stride），算出本行在 Y/X/RSTD 三个
    张量里的起始地址：
        base = Ptr + row_idx * row_stride
    stride 的作用是支持非连续内存布局（如 view 出来的子张量），不假设行与行
    紧挨着存放。
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 本行在三个张量中的起始地址
    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

    """
    向量化加载本行输入 X。tl.load 一次取整行 BLOCK_SIZE 个元素进寄存器：
    - 地址：x_base + col_offsets，即本行首地址 + 各列偏移；
    - mask=mask：越界（列号 >= n_cols）的位置不真正读内存；
    - other=0：被 mask 屏蔽的位置填充 0，保证后续计算（如 sum）不会受
      垃圾值影响。
    记录 X_row_dtype = X_row.dtype：X 的原始 dtype，后面 casting 模式要根据
    它决定什么时候转回原始精度。
    """
    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype

    """
    为什么要加载 W（权重）？

    RMSNorm 的完整公式是 y = (x / RMS) * W，而不是纯归一化 x / RMS。
    归一化后还要逐元素乘一个可学习的权重 W：每个维度一个标量，即 per-channel
    仿射，对应 PyTorch RMSNorm 的 elementwise_affine=True。所以必须先把 W 的
    本行加载进寄存器（W_ptr + col_offsets 定位每列权重地址，mask 屏蔽越界，
    越界填 0），后面与归一化结果 X_row * rstd 逐元素相乘，才得到最终输出。

    如果 elementwise_affine=False（纯归一化，不带权重），就不需要 W。该条件是
    constexpr，编译时已知，为 False 时这段代码直接不编译，零运行开销。
    """
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    """
    Llama 模式（casting_mode == LLAMA）：先把输入转成 fp32 再算。

    原因：sum(x^2)、sqrt 这类"累加型"计算对精度极其敏感。低精度（fp16/bf16）
    下每个元素的舍入误差会随列数 N 累积放大，导致 rstd 算不准。因此归一化
    （算 rstd 和做除法）这一段必须在 fp32 下完成。

    但 Llama 的实现只在 rstd 这里用 fp32 —— 后面乘权重、加 offset 这类逐元素
    仿射会转回原始 dtype 用低精度算（见下方对应注释），即"只有 rstd 用 fp32"。
    """
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    """
    Gemma 模式（casting_mode == GEMMA）：全程用 fp32 计算，与 Llama 相反。

    - 权重 W_row 和输入 X_row 都先转成 fp32；
    - 之后归一化、乘 W、加 offset 的仿射全部在 fp32 下完成，中间不降精度；
    - 只在最终输出 Y 时一次性转回原始 dtype（见下方输出处注释）。

    原因：Gemma 的实现对数值精度要求更严格，全程高精度计算以保证与参考实现
    的数值一致性，代价是比低精度直通稍慢。
    """
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    """
    NONE 模式（低精度直通）：全程不做任何精度转换，X_row/W_row 保持输入原始
    dtype（fp16/bf16）计算到底，fp16 进、fp16 出，不绕任何精度弯。

    eps/offset 是标量，原本是 Python float 或 fp32，与低精度张量直接运算会
    dtype 不匹配，故显式转成 X_row 的 dtype，以便参与 mean_square + eps 和
    offset + W_row。

    三种模式对比：
    - GEMMA：进来 fp16 → 先升到 fp32 算 → 最后降回 fp16 出去（全程高精度）
    - LLAMA：只有 rstd 那段升到 fp32 → 之后又降回 fp16 算仿射（部分直通）
    - NONE：完全不绕弯，fp16 进、fp16 出，一路 fp16（最快，但累加误差可能累积）
    """
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    """
    计算每行的 RMS（均方根），这是 RMSNorm 的核心：

    RMS = sqrt( sum(x_i^2) / N )

    分两步：
    1) mean_square = mean(x^2)：
       tl.sum 是 Triton 的归约函数（reduction，把多个值折叠成更少的值）。
       每个线程块处理矩阵的一行，这一行被表示成一维向量 X_row，BLOCK_SIZE 个
       元素"横向"排列，每个元素对应矩阵的一列：
           X_row = [ x_0,  x_1,  ...,  x_{N-1} ]
                     列0   列1          列N-1
       tl.sum(X_row * X_row, axis=0) 就是"按列方向规约"：先逐元素平方得到
       [x_0^2, x_1^2, ...]，再沿列（axis=0）把整行元素折叠成一个标量
       sum(x_i^2)。除以 n_cols 即得均值 mean(x^2)。
       GPU 上 BLOCK_SIZE 个元素分散在多个线程上，tl.sum 内部需做跨线程归约
       （线程间交换部分和、逐级累加，含隐式线程同步），最终汇聚成一个标量，
       因此它比逐元素运算"重"。

       数值例子：
           X_row        = [1, 2, 3, 4]
           X_row * X_row = [1, 4, 9, 16]
           sum(axis=0)  = 30
           / n_cols     = 7.5            # mean(x^2)

    2) rstd = rsqrt(mean_square + eps)：
       rsqrt(x) = 1/sqrt(x)，即开根号再取倒数，得到 1/RMS。
       + eps 是加一个极小常数（如 1e-5），防止 mean_square=0 时除零/开根号
       发散，保证数值稳定。
       之所以用 rsqrt 而非 1/sqrt(x)，是因为 GPU 有 rsqrt 硬件指令，一次完成
       开方+求倒数，速度更快且精度相当。

       得到 rstd 后，整行归一化就是 X_row * rstd（等价于除以 RMS）。
    """
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    """
    把算好的 rstd 缓存到 RSTD 张量（每行一个标量），供反向传播复用。

    为什么要缓存：反向计算 dx、dw 时需要 rstd（1/RMS）。与其在反向里重新算
    一次（*、sum、/、sqrt 共 4 次操作），不如前向顺手存下来。rstd 是每行一个
    标量，比整行 X_row 小得多，额外显存开销几乎可忽略，但反向省掉整块计算。
    这是典型的"以少量存储换计算"的换法。
    """
    tl.store(rstd_base, rstd)

    # 归一化：X_row * rstd 等价于 X_row / RMS
    X_row = X_row * rstd

    """
    Llama 模式：乘权重（做仿射）前转回原始 dtype（fp16/bf16）。

    Llama 只在 rstd 这一段用 fp32，而乘 W、加 offset 这类逐元素操作对精度要求
    低，用原始低精度算即可。这样既保住归一化的精度，仿射部分又省算力，且与
    Llama 参考实现的行为一致（其代码这一步就是低精度乘法）。这一步之后，X_row
    已回到原始 dtype，下面的仿射就在原始 dtype 下进行。
    """
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    """
    仿射变换：Y = X_norm * (offset + W)。

    RMSNorm 的前向分两步：归一化（X_row * rstd）和仿射（乘权重、加偏移）。
    归一化结果 X_row 已经是 x / RMS，这里再逐元素乘 (offset + W)：
    - W：可学习的逐元素权重（per-channel 仿射，elementwise_affine=True 时才有）；
    - offset：额外的可学习偏置/缩放项，先与 W 相加，再乘归一化结果。
    于是每个输出 y_i = (x_i / RMS) * (offset_i + w_i)，与 kernel 顶部公式一致。
    - elementwise_affine=False：纯归一化，不带 W，Y_row 直接就是 X_row。

    注意此处的精度由前面的 casting_mode 决定：
    - LLAMA：X_row 已在前面转回原始 dtype，这里在低精度下做仿射；
    - GEMMA：这里仍在 fp32 下计算，下面的 if 再统一转回原始 dtype；
    - NONE：一直保持原始 dtype。
    """
    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    """
    Gemma 模式：全程 fp32 计算完成后，在输出前一次性把结果转回原始 dtype
    （fp16/bf16）。这是 Gemma"全程高精度、最后统一降精度"策略的收尾一步，
    与前面把所有输入先升到 fp32 相呼应。
    """
    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    """
    把本行的最终结果 Y_row 写回 Y 张量：
    - 地址：y_base + col_offsets，即本行首地址 + 各列偏移；
    - mask=mask：越界位置（列号 >= n_cols）不写，保持目标内存原值；
    - Y 的输出布局与输入 X 逐行对应（同样按 Y_row_stride 存放）。
    到这里，这个线程块负责的整行 RMSNorm 前向计算就完成了。
    """
    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """
    # 前向：每个线程块处理 1 行 → row_idx = program_id
    # 反向：每个线程块处理 rows_per_program 行（连续的行块）→ 用 program_id 算出行的起止范围
    # 当前线程块编号(第几个块)，转成 int64 防止大张量偏移量乘法溢出
    row_block_id = tl.program_id(0).to(tl.int64)

    # 本块负责的起始行号。块0→行0，块1→rows_per_program，块2→2*rows_per_program...
    row_start = row_block_id * rows_per_program

    # 本块负责的结束行号（开区间，不包含）。min 是为了处理最后一块：
    # 总行数 n_rows 可能不是 rows_per_program 的整数倍，最后一块的行数可能不足，
    # 用 min 截断到 n_rows，避免越界
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)

    # 列索引向量 [0,1,...,BLOCK_SIZE-1]，同上，向量化访问一行所有列
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 越界掩码：真实列数可能不足 BLOCK_SIZE，屏蔽多余位置的读写
    mask = col_offsets < n_cols

    """
    权重准备（循环外只做一次，因为 W 对本块所有行共享）：
    - dW_row：dw 的块内累加器，用 fp32 并初始化为 0。dw 需要对本块负责的
      每一行累加贡献（W 被所有行共享，dw_i = Σ_行 dy_i·(x_i·rstd)），
      循环结束时把累加好的 partial 写回，由包装函数 _dW.sum(0) 跨块归约。
    - W_row：把权重 W 一次性加载进寄存器，并预先加上 offset（w' = w + offset），
      避免在每行循环里重复加载和相加。
    """
    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset

    """
    行循环：逐个处理本块负责的 [row_start, row_end) 行。
    每行依次：
    1) 按行号定位 dY/dX/X/RSTD 四个张量中该行的起始地址；
    2) 向量化加载 dY（上游梯度 dy）、X（前向输入 x）、rstd（前向缓存的 1/RMS）；
    3) X 转 fp32 再参与计算，保证累加精度。
    注意 rstd 是前向算好缓存下来的，这里直接复用，免去重算 sum/sqrt。
    """
    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride

        x_base = X_ptr + row_idx * X_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

        # 读取前向缓存下来的本行 rstd（1/RMS）
        rstd_row = tl.load(rstd_base)

        # 输入转 fp32 后计算，防止低精度累加误差
        X_row = X_row.to(tl.float32)

        """
        构造 m = dy·(w+offset)，公式 dx 里的第一项因子。
        三种 casting 分支只是为了匹配前向的精度行为：前向在哪个 dtype 下做的
        仿射，反向就在哪个 dtype 下还原，保证梯度与前向计算图一致：
        - LLAMA：前向仿射用原始低精度，所以 dy·W 在原始 dtype 算完再转 fp32；
        - GEMMA：前向全程 fp32，所以先把 dy 升到 fp32 再乘 W；
        - NONE：低精度直通，dy 原样参与。
        若 elementwise_affine=False（纯归一化），则 m = dy（没有 W）。
        """
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row

        """
        计算 dx，对应推导公式：
            dx = rstd·[ g - (rstd²/N)·(g·x)·x ]，其中 g = m = dy·(w+offset)

        第一项 dX_row = rstd·m：直接路径的梯度 dy·w'·rstd。
        第二项是修正项：x_i 通过改变整行 RMS 影响所有输出 y_j，因此要减去
        所有输出经 RMS 汇聚回来的共同贡献。其中 tl.sum(m * X_row, axis=0)
        就是内积 dot = Σ dy·w'·x（按列规约成标量），再乘 -(rstd²/N)·x。
        """
        dX_row = rstd_row * m

        dX_row += (rstd_row) * (
            -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
        )

        """
        累加 dw：每行贡献一份 dy·(x·rstd)，往块内累加器 dW_row 里 +=。
        dw_i = Σ_行 dy_i·(x_i·rstd)（W 被所有行共享，故逐行累加）。
        LLAMA 分支匹配前向：前向仿射用了低精度，这里把 x·rstd 先转回原始
        dtype 再乘 dy；其余模式 X_row 已在 fp32，直接 fp32 累加保精度。
        """
        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                # X_row 已经是 fp32（见上面 casting 分支）
                dW_row += dY_row * (X_row * rstd_row)

        # dx 每行算完立即写回，转回 X 原始 dtype 与前向输出一致
        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    """
    循环结束后写回 partial dw：本块把 [row_start, row_end) 行的贡献累加成了
    一个 partial，写到 _dW 的 row_block_id 行，最后由包装函数 _dW.sum(dim=0)
    做跨块归约，得到最终的 dw。
    """
    if elementwise_affine:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


"""
块版本（block）vs 行版本（row）kernel 的区别与选择：

_ rms_norm_forward_kernel（行版）：每个线程块处理 1 行，X 是一维向量 (BLOCK_SIZE,)，
  tl.sum(X*X, axis=0) 规约出 1 个标量 rstd，grid = (n_rows,)。

_block_rms_norm_forward_kernel（块版）：每个线程块处理 BLOCK_ROW（默认 16）行，
  一次加载 2D tile (BLOCK_ROW, BLOCK_SIZE)：
    - 行号：row_idx = program_id * BLOCK_ROW + arange(BLOCK_ROW)（16 个行号向量）
    - 掩码变成二维：row_mask[:, None] & col_mask[None, :]
    - tl.sum(X*X, axis=1) 沿列规约，一次给出 16 个行的 rstd
    - 权重 W 只加载一次，广播给 16 行共用

选择条件（包装函数里动态判断，三者同时满足才用块版）：
    if BLOCK_SIZE > 256 or n_rows < 4096*8 or row_mode:
        行版
    else:
        块版（BLOCK_ROW = 16）

块版的适用场景：hidden size 很小（<= 256）且 B×T 很大（>= 32768）。这类模型一行
才几百个元素，一行一个 block 会产生海量微型 block，调度开销占比高。块版把 16 行
并成一个 2D tile：block 数减少 16 倍、W 只加载一次、2D 连续读取吞吐更高。

注意：这是形状驱动的选择，不是某个特定模型专用。主流大模型 hidden size >= 512
（如 Llama/Qwen/Gemma），永远走行版本；块版只服务于小特征维 + 大行数的模型
（微型 Transformer、音频/语音编码器等）。
"""


@triton.jit
def _block_rms_norm_forward_kernel(
    # 输出/中间张量均为「指针 + 行步长」成对，用于支持非连续内存布局
    Y_ptr,  # 输出 Y 的指针
    Y_row_stride,  # Y 每行内存步长
    X_ptr,  # 输入 X 的指针
    X_row_stride,  # X 每行内存步长
    W_ptr,  # 权重 W 的指针（elementwise_affine=True 时使用）
    W_row_stride,  # W 每行内存步长
    RSTD_ptr,  # 每行 1/RMS（rstd）输出指针，前向缓存供反向复用
    RSTD_row_stride,  # RSTD 每行内存步长
    n_rows,  # 总行数（B×T）。块版一次处理 BLOCK_ROW 行，最后一块
    # 可能不足，用它构造行越界掩码 row_mask = row_idx < n_rows
    n_cols,  # 每行列数，构造列越界掩码 col_mask
    eps,  # 数值稳定项，加在 sum(x^2)/N 里防除零
    offset,  # 公式里的偏置/缩放项，与 W 相加后再乘归一化结果
    casting_mode: tl.constexpr,  # constexpr：选 NONE/LLAMA/GEMMA 三种精度模式，if 在编译期裁剪
    elementwise_affine: tl.constexpr,  # constexpr：是否使用逐元素权重 W
    BLOCK_SIZE: tl.constexpr,  # constexpr：列方向块大小，每行处理几个元素（由 calculate_settings(n_cols) 算得）
    BLOCK_ROW: tl.constexpr,  # constexpr：行方向块大小，一个线程块同时处理几行（默认 16）
    # BLOCK_SIZE × BLOCK_ROW 组成一个 2D tile (BLOCK_ROW, BLOCK_SIZE)
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """
    # 一次生成 BLOCK_ROW 个行号(一个 block 同时处理多行)
    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    # 列索引向量 [0,1,...,BLOCK_SIZE -1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 行越界掩码：最后一块可能不足 BLOCK_ROW 行，多余行号屏蔽
    row_mask = row_idx < n_rows
    # 列越界掩码：真实列数可能不足 BLOCK_SIZE
    col_mask = col_offsets < n_cols

    """
    1. 地址 = 行号 * 行步长 + 列偏移(广播展开)

    
    """
    # 行版是每个读一行，块版是一次读一个连续大块(16行)，吞吐更高
    # 把 16行 * 一行的列的连续内存一次性向量化读进寄存器，用广播把行号/列好/掩码都扩成二维，越界位置补0
    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    X_row_dtype = X_row.dtype

    """
    加载权重 W（可选）。注意与加载 X 的区别：
    - W 是 (BLOCK_SIZE,) 一维向量，因为权重对所有行共享（per-channel 仿射），
      只需按列加载一次，后面通过广播 [None, :] 应用于所有 BLOCK_ROW 行；
    - mask 只用一维 col_mask（不需要行掩码），other=0 越界填 0。
    elementwise_affine=False（纯归一化）时不加载，这段在编译期被裁剪。
    """
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)

    """
    Llama 模式：先把 X 转 fp32 再算。原因与行版相同：sum(x²)、sqrt 这类累加型
    计算对精度极敏感，低精度下误差随 N 累积放大，必须在 fp32 下算 rstd。
    后面乘权重前会转回原始 dtype（见下方注释），即"只有 rstd 用 fp32"。
    """
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    """
    Gemma 模式：全程用 fp32 计算（与 Llama 相反）。权重和输入都转 fp32，
    归一化与仿射都在 fp32 下完成，只在最终输出时一次性转回原始 dtype。
    Gemma 实现精度要求更严，牺牲一点速度换数值一致。
    """
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    """
    NONE 模式（低精度直通）：全程不做精度转换，X_row/W_row 保持输入原始 dtype
    （fp16/bf16）计算到底。eps/offset 是标量（Python float 或 fp32），与低精度
    张量运算会 dtype 不匹配，故显式转成 X_row 的 dtype 以便参与运算。
    """
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    """
    计算每行的 RMS。块版与行版唯一区别在规约轴：
    - X_row 是 (BLOCK_ROW, BLOCK_SIZE) 的二维 tile；
    - tl.sum(X_row * X_row, axis=1) 沿列方向（axis=1）规约，一次得到
      (BLOCK_ROW,) 的向量——每行一个 mean(x²)，再除以 n_cols 得均值；
    - rstd = rsqrt(mean_square + eps)，每行一个 1/RMS，同样 (BLOCK_ROW,)。
    对比行版用 axis=0 一次只出一个标量，块版用 axis=1 一次出 BLOCK_ROW 个。
    """
    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)

    """
    缓存 rstd 供反向复用：把 BLOCK_ROW 行的 rstd 一次性写回 RSTD 张量
    （rstd 是每行一个标量，比整行 X 小得多）。反向算 dx/dw 直接读取，
    省去重算 *、sum、/、sqrt 共 4 次操作。
    mask=row_mask：最后一块不足 BLOCK_ROW 行的越界行不写。
    """
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)

    # 归一化：每行乘自己的 rstd（rstd[:, None] 广播到二维 tile）
    X_row = X_row * rstd[:, None]

    """
    Llama 模式：乘权重（做仿射）前转回原始 dtype。Llama 只在 rstd 用 fp32，
    逐元素仿射对精度要求低，用原始低精度算，省算力且贴合参考实现行为。
    """
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    """
    仿射变换：Y = X_norm * (offset + W)。
    (offset + W_row) 是一维 (BLOCK_SIZE,)，用 [None, :] 广播成 (BLOCK_ROW,
    BLOCK_SIZE) 应用到所有行。elementwise_affine=False 时纯归一化直接输出。
    """
    if elementwise_affine:
        Y_row = X_row * (offset + W_row)[None, :]
    else:
        Y_row = X_row

    # Gemma 模式：全程 fp32 算完后，输出前一次性转回原始 dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    """
    写回结果 Y：地址用广播算 (BLOCK_ROW, BLOCK_SIZE) 的 2D tile 偏移，掩码用
    行 & 列的二维掩码，越界位置不写保持原值。至此本 block 负责的 BLOCK_ROW 行
    全部算完。
    """
    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _block_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_row = W_row + offset

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row[None, :]).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row

        dX_row = rstd_row[:, None] * m

        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols)
            * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None]
            * X_row
        )

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                # TODO(tcc): use tl.sum(..., dtype=tl.float32) once we upgrade to triton>=3.3.0
                dW_row += tl.sum(
                    (dY_row * (X_row * rstd_row[:, None]).to(X_dtype)).to(tl.float32), 0
                )
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    if elementwise_affine:
        tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)


# 字符串名 -> casting 模式值 的查找表。
# 把用户可见的字符串配置（"llama"/"gemma"/"none"）映射到内部 tl.constexpr 值，
# 这样 rms_norm_forward 同时接受字符串和 int 两种输入形式。
_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
    # 统一 casting_mode 的输入形式：允许传字符串（"llama"/"gemma"/"none"）
    # 或 int 值，校验合法性后统一转成 int。
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, (
            f"Invalid casting mode: {casting_mode}"
        )
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), (
            f"Invalid casting mode: {casting_mode}"
        )

    # 把输入展平成二维 [n_rows, n_cols]，方便按行处理（每行一个样本）。
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    # 根据列数自动计算 Triton kernel 的 BLOCK_SIZE 和 num_warps。
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD 缓存每一行的 rstd（均方根倒数）。
    # 使用 Llama/Gemma casting 模式时 RSTD 总是用 fp32 计算/存储（提高数值精度），
    # 否则与 X 保持相同 dtype。
    rstd_dtype = (
        torch.float32
        if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value)
        else X.dtype
    )
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # W 不为空表示做 elementwise_affine（逐元素缩放+偏移），否则不做。
    if W is not None:
        # 检查形状约束：列数必须等于 W 的长度。
        assert X.shape[1] == W.shape[0], (
            "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"
        )
        elementwise_affine = True
    else:
        elementwise_affine = False

    # XPU 特定优化：在 XPU 设备上启用 large GRF 模式（通过 kernel_args 传给 kernel）。
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)
    # 在正确设备上下文下启动 Triton kernel。
    with device_context(X.device):
        # 行数较少 / 列数较宽 / 显式指定时，用逐行 kernel；否则用分块（多行）kernel 减少启动开销。
        if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
            _rms_norm_forward_kernel[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_cols,
                eps,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
        else:
            # 分块 kernel：每个程序块一次处理 BLOCK_ROW 行。
            BLOCK_ROW = 16
            kernel_args["BLOCK_ROW"] = BLOCK_ROW
            _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
    # 返回结果：Y 恢复原形状；X（已展平）与 RSTD 留给反向传播复用，
    # 避免重复计算；BLOCK_SIZE/num_warps/casting_mode 也一并传回反向使用。
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode


def rms_norm_backward(
    dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode
):
    # 与 forward 相同：把梯度展平成二维 [n_rows, n_cols] 逐行处理。
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # 获取设备上的计算核心数量，用于把行数均分给各核心并行计算（每个核心处理 rows_per_program 行）。
    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    elif X.device.type == "npu":
        sm_count = get_npu_core_count()

    if W is not None:
        # 每个核心先在自身负责的行范围内累积部分梯度 _dW，最后跨核心求和得到 dW。
        # 用 fp32 累加以保证数值稳定性。
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
        elementwise_affine = True
    else:
        _dW = None
        elementwise_affine = False

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # 平均分配每行到每个核心，grid 大小 = 核心数。
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    # in_place 模式直接复用 dY 作为输出，节省内存；否则需要独立清零的 dX（因为 kernel 做累加）。
    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)

    # XPU 特定优化：在 XPU 设备上启用 large GRF 模式（通过 kernel_args 传给 kernel）。
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    with device_context(X.device):
        # 调度逻辑与 forward 一致：行少/列宽/显式指定时用逐行 kernel，否则用分块（多行）kernel。
        if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
            _rms_norm_backward_kernel[grid](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                torch_to_triton_dtype[X.dtype],
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                _dW,
                _dW.stride(0) if elementwise_affine else 0,
                n_rows,
                n_cols,
                offset,
                rows_per_program,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
        else:
            # 分块 kernel：每个程序块一次处理 BLOCK_ROW 行。
            BLOCK_ROW = 16
            kernel_args["BLOCK_ROW"] = BLOCK_ROW
            _block_rms_norm_backward_kernel[grid](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                torch_to_triton_dtype[X.dtype],
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                _dW,
                _dW.stride(0) if elementwise_affine else 0,
                n_rows,
                n_cols,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
    dX = dX.view(*shape)

    # 把各核心的局部梯度 _dW 沿第 0 维求和，得到完整的 dW，并转回 W 的 dtype。
    if elementwise_affine:
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        dW = None

    # 返回 dX、dW：
    #   dX: 输入 X 的梯度（∂L/∂X），形状与 X 相同，会继续向更前一层传播。
    #   dW: 权重 W 的梯度（∂L/∂W），用于更新 affine 参数；无 W 时为 None。
    # 二者均由输出梯度 dY（∂L/∂Y）经链式法则算出。
    return dX, dW


class LigerRMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset and casting mode.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 nefore starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    # 自定义 autograd.Function：forward 算 RMSNorm 前向，backward 算梯度。
    # @ensure_contiguous 保证输入输出是连续内存，便于直接启动 Triton kernel。

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None
    ):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if isinstance(X, _DTensor):
            # 输入若来自张量并行（tensor parallel）模块，先 gather 成本地完整张量，
            # 使每个 TP worker 都能对整个隐层维度做 RMSNorm。
            # TODO: support CP.
            X = X.full_tensor()

        # 调用 Triton 前向实现，得到输出 Y 以及反向所需的时间量：
        # X（展平后）、RSTD（每行 rstd）、以及 kernel 配置（BLOCK_SIZE/num_warps/casting_mode）。
        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(
            X, W, eps, offset, casting_mode, row_mode
        )
        # 把反向需要的参数暂存到 ctx，backward 时直接读取。
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        # 用 save_for_backward 保存前向中间量，供反向使用（框架会做内存优化）。
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        # 恢复 forward 保存的张量；无 W 时 W 记为 None。
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        if isinstance(dY, _DTensor):
            # 梯度同样可能是 TP 模块输出，先 gather 成本地张量再算反向。
            # TODO: support CP.
            dY = dY.full_tensor()

        # 调用 Triton 反向实现，用输出梯度 dY 结合前向保存的中间量算出 dX、dW。
        dX, dW = rms_norm_backward(
            dY,
            X,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.in_place,
            ctx.row_mode,
        )
        # 返回值必须与 forward 的入参一一对应（X, W, eps, offset, casting_mode, in_place, row_mode），
        # 只有 X、W 有梯度，其余参数返回 None。
        return dX, dW, None, None, None, None, None
