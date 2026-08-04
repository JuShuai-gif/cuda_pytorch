# 09 语言与 DSL 设计（深度版）

> 本文目标：深入 tl.* 原语的语义与编译期行为，重点理解 constexpr、specialization、块索引约束。

## 1. 核心原语语义

| 原语 | 语义 | 编译行为 |
| --- | --- | --- |
| `tl.program_id(axis)` | 当前 CTA 索引 | 硬件 blockIdx |
| `tl.arange(s, e)` | 1D 块索引（2的幂） | 布局系统分配线程 |
| `tl.load(ptr, mask, other)` | 块加载 | coalesce + 向量化 |
| `tl.store(ptr, val, mask)` | 块存储 | 同上 |
| `tl.dot(a, b, acc)` | 块乘加 | AccelerateMatmul → mma |
| `tl.constexpr` | 编译期常量 | specialization |

## 2. `tl.arange` 约束（深度）

- start/end 必须编译期常量。
- 大小必须 2 的幂。
- 原因：布局推断（`toLinearLayout`）需要静态块形状；2 的幂便于 warp 划分。

## 3. `tl.constexpr` 与 specialization（深度）

`JITFunction.run` 里 `_pack_args`（jit.py:704-746）：
- constexpr 参数被提取为 `constexprs`。
- 参与 `specialization` 与缓存键。

`KernelParam`（jit.py:303）控制 `do_not_specialize`/`do_not_specialize_on_alignment`。

## 4. `tl.dot`（深度）

`TT_DotOp`（`TritonOps.td:681`）：`d = a*b + c`。
输入要求（`TritonGPUAttrDefs.td:1434-1444`）：
- MMAv1/2：A/B 必须 `#ttg.dot_op` 编码（opIdx=0/1，parent=结果 mma 布局）。
- MMAv3+：主要驻留 shared（`NVMMASharedEncodingAttr`）。

`DotOperandEncodingAttr` 字段：`opIdx`/`parent`/`kWidth`。
- `kWidth` = `max(32/bitwidth, 1)`（fp16 → 2）。

## 5. `tl.load/store` 与 mask（深度）

```python
tl.load(ptr + offs, mask=offs < n, other=0.0)
```
- mask：边界处理。
- `other`：mask False 时填充。
- 编译器自动：coalescing、向量化（基于 AxisInfo contiguity）。
- 流水线内：转 `ttg.async_copy_global_to_local` + commit/wait（见 `16`）。

## 6. `tl.range` 的编译期 hint（深度）

`visit_For`（code_generator.py:1276-1289）读取：
- `num_stages`：→ `tt.num_stages` 属性（流水线）。
- `loop_unroll_factor`：→ `tt.loop_unroll_factor`。
- `warp_specialize`：→ `tt.warp_specialize`。
- `flatten`、`disable_licm`。

## 7. 完整例子：GEMM（深度）

```python
@triton.jit
def matmul(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs + k), tl.load(b_ptrs + k))
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

编译行为：
- `tl.arange(0, BM)` → 块索引，布局推断。
- `tl.load` → coalesce + 可能 cp.async。
- `tl.dot` → AccelerateMatmul → `#ttg.mma` 布局 + mma.sync。
- 外层 `for k` → 若 `num_stages>1` → 流水线。

## 8. 深入自测

1. `tl.arange` 两条约束及原因？
2. `tt.dot` 的输入布局要求（不同 mma 版本）？
3. constexpr 如何参与 specialization？
4. `tl.range` 的 4 个 hint？
5. GEMM 的编译行为链？

## 9. 下一步

进入 `10_运行时与JIT机制.md`（深度版）。
