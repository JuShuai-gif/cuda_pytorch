# 01 Triton 是什么（深度版）

> 本文目标：在概览基础上，深入 Triton 的设计动机、MLIR 技术选型、块级编程模型与编译管线，建立正确的思维模型。

## 1. 核心定位

Triton（OpenAI 发起，triton-lang 维护）是**块级（tile/block）GPU 编程语言**：

> 用户用 Python 写"块级计算"，编译器自动做布局推断、访存合并、mma/wgmma 降级与软件流水线，生成接近手写 CUDA 的性能。

它是 PyTorch 2.0 `torch.compile`（Inductor）的默认 GPU 后端。

## 2. 设计动机

| 方案 | 痛点 | Triton 的回应 |
| --- | --- | --- |
| 手写 CUDA | 线程/布局手工 | 块级抽象 + 自动并行 |
| 编译器（LLVM 直接） | 需自己管线程 | 自动布局 |
| cuBLAS | 无法融合 | 语言级自由 |

## 3. 技术选型：为什么用 MLIR（关键）

**这是理解 Triton 的分水岭**。Triton 基于 LLVM/MLIR：

```mermaid
flowchart LR
    DSL["Python DSL (tl.*)"] --> CG["code_generator (AST→ttir)"]
    CG --> TTIR["ttir (lib/Dialect/Triton)"]
    TTIR --> TTGIR["ttgir (lib/Dialect/TritonGPU)"]
    TTGIR --> LLIR["llir (lib/Conversion/TritonGPUToLLVM)"]
    LLIR --> PTX["PTX (lib/Target/LLVMIR)"]
    PTX --> CUBIN["cubin (ptxas)"]
```

**要点（已确认）**：
- 复用 MLIR 的 Dialect/Pass/TableGen 框架。
- 自研三个 dialect：ttir（无布局）、ttgir（布局+线程）、ttnvgpu（NVIDIA 专属）。
- 布局系统用 `LinearLayout`（GF(2) 矩阵表示），可做代数运算。

> 对比：TileLang 用 TVM TIRX。**Triton 学 MLIR，TileLang 学 TVM。**

## 4. 块级编程模型

```python
pid = tl.program_id(0)                     # 当前 CTA 索引
offs = pid * BLOCK + tl.arange(0, BLOCK)   # 块内索引（2的幂）
tl.store(y + offs, tl.load(x + offs) + 1, mask=offs < n)
```
- **program** = 一个 CTA（thread block）。
- `grid` 指定 program 数量。
- 编译器自动把 `tl.arange` 的块索引映射到线程（布局系统）。

## 5. 编译管线（概览，详见 07）

```
AST →(code_generator) ttir →(make_ttgir passes) ttgir →(make_llir) llir →(make_ptx) ptx →(make_cubin) cubin
```

- `make_ttgir`（`third_party/nvidia/backend/compiler.py:262`）是优化主战场：布局推断、加速 matmul、流水线。
- `make_cubin`（:513）调 ptxas。

## 6. 版本与环境

- 仓库版本：`3.8.0`（`python/triton/__init__.py:2`），commit `be81991971`。
- 后端：third_party 插件（nvidia/amd/proton）。
- 依赖：LLVM/MLIR（构建时拉取）+ CUDA Toolkit + ptxas。

## 7. 深入自测

1. Triton 为什么用 MLIR？三个自研 dialect 各管什么？
2. program 对应 GPU 的什么？
3. `tl.arange` 为什么要求 2 的幂？
4. 编译管线五个阶段？
5. 与 TileLang 的 IR 栈本质区别？

## 8. 下一步

进入 `02_仓库整体架构.md`（深度版）。
