# 09 语言与 DSL 设计（深度版）

> 本文目标：深入 DSL 的"Python 语法 → IR 节点 → TIR"完整翻译机制，理解 eager builder 的内部工作，以及 `T.Kernel`/`T.Parallel`/`T.gemm` 等的精确 IR 效果。

## 1. 翻译链全景

```mermaid
flowchart LR
    A["用户 Python 源码"] --> B["parser/ (T.xxx 解析)"]
    B --> C["ast/ (IR 节点定义)"]
    C --> D["eager/builder.py (逐句执行)"]
    D --> E["PrimFunc / IRModule"]
    E --> F["C++ Pass 流水线"]
```

## 2. eager builder 内部机制（深度）

`tilelang/language/eager/builder.py` 的核心是**维护当前 IR 构建上下文**，把 Python 语句翻译成 IR：

| Python 结构 | IR 效果 |
| --- | --- |
| `with T.Kernel(...)` | 建立 launch frame（block 循环） |
| `for i in T.Parallel(n)` | 生成并行循环（待划分线程） |
| `for i in T.serial(n)` | 生成串行 `For` |
| `for i in T.Pipelined(n, num_stages=k)` | 带 annotation 的串行 `For` |
| `if cond:` | 生成 IfThenElse（编译期/运行期判定） |
| `buf[i] = val` | `BufferStore` |
| `T.copy(a, b)` | tile 级 copy 调用 |
| `T.gemm(...)` | tile 级 gemm 调用 |

## 3. `T.Kernel` 精确机制

`tilelang/language/kernel.py:277-340`：
```python
def Kernel(grid_x, grid_y=None, threads=..., ...):
    return _ffi_api.KernelLaunch(grid_x, grid_y, ..., threads)
```
- 返回 `KernelLaunch`（launch frame）。
- `KernelLaunch.__enter__` 建立 block 循环，`as (bx, by)` 给出 block 索引。
- 对应 C++ 侧 `tl.KernelLaunch` FFI（`_ffi_api.KernelLaunch`，:340）。
- 之后由 `MaterializeKernelLaunch` pass 变成 `thread_extent` AttrStmt。

## 4. `T.Parallel` 与线程划分（深度）

`tilelang/language/loop.py` 的 `T.Parallel`：
```python
for i in T.Parallel(256):        # 256 是块内循环范围
    C[bx*256+i] = ...
```
- 语义：**这块循环应该并行到线程**。
- 具体线程划分由后续 pass（layout inference 的 parallel 循环布局 + LowerTileOp 的 `VisitStmt_(ForNode)` 展开）决定。
- 编译器根据 fragment layout 决定哪些迭代归哪个线程。

## 5. 缓冲区分配原语

| 原语 | scope | 对应 CUDA |
| --- | --- | --- |
| `T.alloc_global` | global | 全局内存 |
| `T.alloc_shared` | shared | `__shared__` |
| `T.alloc_fragment` | local.fragment | 寄存器（配合 mma） |
| `T.alloc_local` | local | 本地内存 |

位置：`tilelang/language/allocate.py`。

## 6. `T.gemm` 参数与语义（深度）

`tilelang/language/gemm_op.py:77-89` 的 shape 推导：
```python
M, N = C_shape[-2], C_shape[-1]
M_A = A_shape[-1] if transpose_A else A_shape[-2]
K = A_shape[-2] if transpose_A else A_shape[-1]
assert prim_expr_equal(M_A, M)  # M 一致性
assert prim_expr_equal(K, K_B)  # K 一致性
```
- `T.gemm` → `tirx.call_intrin(Op.get("tl.tileop.gemm"), ...)`（gemm_op.py:119）。
- `tl.tileop.gemm` C++ 注册（`src/op/gemm.cc:262`）。

## 7. 存储作用域变体（gemm 家族）

`tilelang/cuda/intrinsics/gemm/gemm_base.py:61-67`：
```python
def is_gemm_rs(self): return is_fragment(self.A) and is_shared(self.B)   # register-shared
def is_gemm_ss(self): return is_shared(self.A) and is_shared(self.B)      # shared-shared
def is_gemm_sr(self): return is_shared(self.A) and is_fragment(self.B)
def is_gemm_rr(self): return is_fragment(self.A) and is_fragment(self.B)
```
- `gemm_rs` = A 在寄存器、B 在 shared（**不是 reduce-sum！**）。
- 真正的归约是 `T.reduce_sum`（`language/reduce_op.py`）。

## 8. `T.copy` 与访存降级

`tilelang/language/copy_op.py`：
- `T.copy(src, dst)`：tile 级拷贝。
- 降级为：cp.async（受 `tl.enable_async_copy` 控制）、ldg/stg、TMA（Hopper）。
- 判断逻辑在 `src/cuda/op/copy_analysis.cc:713-799` `SelectCopyInstForLowering`。

## 9. 归约与 scan

- `T.reduce_max/min/sum`（`language/reduce_op.py`）：跨线程归约。
- 依赖 `LowerThreadAllreduce` pass 展开成 warp shuffle 等。
- `T.scan`（`language/scan_op.py`）：前缀和。

## 10. 符号运算（编译期常量）

`tilelang/language/symbolics.py`：
- `T.const("n")`：创建符号常量，编译时绑定。
- `T.ceildiv/min/max` 等：符号算术。
- 这些在 IR 里是 `PrimExpr`，参与 loop extent 和缓存键。

## 11. 动手实验（深度）

```bash
mkdir -p /home/hpc/ghr_code/cuda_pytorch/TileLang/experiments/09_dsl
# 写一个 vector add + 一个 T.Pipelined matmul
# 用 TL_ENABLE_DUMP_IR 观察 T.Kernel / T.Parallel / T.Pipelined 的初始 IR
```

## 12. 深入自测

1. eager builder 如何把 `with T.Kernel` 变成 IR？
2. `T.Parallel` 的线程划分在哪决定？
3. `T.gemm` 的 M/N/K 从哪推导？
4. `gemm_rs` 的 rs 是什么意思？
5. `T.copy` 能降级成哪三类指令？
6. `T.const` 参与哪三件事？

## 13. 下一步

进入 `10_运行时与JIT机制.md`（深度版）。
