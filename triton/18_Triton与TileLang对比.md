# 18 Triton 与 TileLang 对比（技术点 + 工程双维度深度版）

> 本文目标：从 Triton 视角与 TileLang 做双维度对比——技术点（IR/布局/pass/降级/缓存）与工程（构建/测试/调试/生态/贡献），并给出选型与双修建议。

## 第一部分：技术点对比

## 1. 技术栈全貌

| 维度 | Triton | TileLang | 本质差异 |
| --- | --- | --- | --- |
| 底层 IR | MLIR（ttir/ttgir/llir） | TIRX（TVM TIR 扩展） | 两套独立 IR 生态 |
| 依赖 | LLVM/MLIR + 自研 PTX | TVM 分叉 + CUTLASS | Triton 更自给自足 |
| Python↔C++ | nanobind | tvm_ffi | 绑定层不同 |
| 目标硬件 | 主 CUDA（插件架构） | CUDA/HIP/Metal/WebGPU/CPU | TileLang 覆盖更广 |
| 语言前端 | `ast.NodeVisitor`（整体） | eager builder（逐句） | 翻译策略不同 |

## 2. 布局系统对比（Triton 视角）

| | Triton | TileLang |
| --- | --- | --- |
| 表示 | `LinearLayout` GF(2) 矩阵（只存基向量） | `Layout{input_size, forward_index}` 显式函数 |
| 求值 | `apply()` xor 累加（LinearLayout.cpp:885） | `Forward()` 代入占位符 |
| 求逆 | `invertAndCompose` 伪逆 | `DetectIterMap` + `InverseAffineIterMap` |
| 组合 | `operator*`（直和） | 无通用代数运算 |
| 自动化 | 布局转换 = 矩阵代数，可大规模化简 | 求逆依赖启发式 |

**Triton 优势（已确认）**：LinearLayout 把所有布局操作统一成 GF(2) 矩阵运算。`RemoveLayoutConversions` 可以代数方式把 `convert_layout` 推到 load/splat 处吸收（`areLayoutsEquivalent` 直接比矩阵）。这让"布局自动优化"成为可能——编译器更"自动"。

**TileLang 优势**：显式函数表达力强（可含任意 floordiv/xor），fragment 线程映射更直接可见。

## 3. mma 降级对比（Triton 视角）

| | Triton | TileLang |
| --- | --- | --- |
| 降级位置 | C++ `DotOpToLLVM`（MMAv2/WGMMA/MMAv5） | C++ pass 调 `tl.gemm.lower`（Python 生成） |
| 指令模板 | 自研内联 PTX | CUTLASS `tl::mma_sync<>` |
| 版本选择 | `getMMAVersionSafe`（AccelerateMatmul.cpp:43） | `SelectInst`（cuda/op/gemm.cc:337） |
| 编译器算法语言 | C++ | Python（tileop/gemm/gemm_mma.py） |

**推论**：
- Triton 的 mma 降级全在 C++，性能好但改算法需重编。
- TileLang 把 gemm 算法放 Python（`gemm_mma.py`），改算法免重编——**编译器算法用 Python 开发**是重要工程差异。

## 4. 缓存机制对比（Triton 视角）

| | Triton | TileLang |
| --- | --- | --- |
| 进程内 | `kernel_cache` dict | `KernelCache`（单例） |
| 磁盘 | `~/.triton/cache/<base32>/` | `~/.tilelang/cache/<version>/.../kernels/<key>/` |
| 键输入 | triton_key+src+backend+options+env（5 元组） | IR脚本+args+target+configs+版本+lib指纹 |
| 原子写 | uuid + os.replace | staging + os.rename |
| 层级 | 单层 | 两层（KernelCache + CUDABinaryCache） |

**Triton 特点（已确认）**：`triton_key()` 覆盖全部编译器源码 + `libtriton` 二进制 hash——编译器任何改动都失效缓存，保证正确性优先。

## 5. pass 链对比（Triton 视角）

| | Triton | TileLang |
| --- | --- | --- |
| 定义处 | `make_ttgir`（nvidia/compiler.py:262） | `CUDAPassPipelineBody`（cuda/pipeline.py:145） |
| 可注入 | 否（固定） | 是 |
| 架构分支 | capability（<90/90/100/120） | target 相关 |
| 端到端 | TTGIR 直接含布局+mma 语义 | 分步 pass 链 |

## 6. 技术点总结表

| 技术点 | 谁更强 | 原因 |
| --- | --- | --- |
| 布局自动化 | Triton | GF(2) 矩阵代数 |
| 布局表达力 | TileLang | 任意表达式 |
| pass 可定制性 | TileLang | 可注入 |
| 编译器算法开发效率 | TileLang | Python 生成 IR |
| 端到端自动化 | Triton | TTGIR 一体化 |
| 硬件覆盖 | TileLang | 多后端 |
| 依赖精简 | Triton | 自研 PTX |
| 调试工具 | Triton | triton-opt/reduce |

## 第二部分：工程对比

## 7. 构建与安装

| | Triton | TileLang |
| --- | --- | --- |
| 构建后端 | setup.py + CMake | scikit-build-core |
| C++ 绑定 | nanobind | tvm_ffi |
| 大依赖 | LLVM/MLIR（构建拉取） | TVM 子模块 + CUTLASS |
| 首次编译 | 30-60 分钟（含 LLVM） | 10-30 分钟 |
| 增量 | `make`（ninja -C $BUILD_DIR） | `pip install -e .` |

**工程要点**：
- Triton 的 LLVM 是构建期拉取/查找，是首次编译慢的主因。
- Triton 的 `make` 工作流（dev-install → all → test-*）是成熟的 C++ 开发循环。

## 8. 测试体系（Triton 视角）

| | Triton | TileLang |
| --- | --- | --- |
| IR/pass 测试 | `test/`（lit + FileCheck，无需 GPU） | `testing/python/transform/`（pytest assert） |
| Python 测试 | `python/test/unit/`（pytest） | `testing/python/`（pytest） |
| C++ 测试 | `test-cpp`（gtest） | `testing/cpp/` |

**Triton 优势**：lit 测试 `.mlir` 文件无需 GPU、精确断言 pass 行为（FileCheck），是 MLIR 生态的标准做法，比纯 assert 更适合 pass 验证。

## 9. 调试工具链（Triton 明显优势）

| | Triton | TileLang |
| --- | --- | --- |
| IR dump | `TRITON_DUMP_IR=1` | `TL_ENABLE_DUMP_IR=1` |
| 单跑 pass | `triton-opt`（强大） | 无独立 CLI |
| 崩溃最小化 | `triton-reduce` | 无 |
| 崩溃复现 | `--run-reproducer` | lower_trace/pass_diff |
| 解释执行 | `TRITON_INTERPRET=1` | 无 |

**推论**：Triton 的调试链（dump→triton-opt→reduce→reproducer）得益于 MLIR 生态的成熟工具，这是学习"编译器调试"的宝贵资源。

## 10. 生态与社区

| | Triton | TileLang |
| --- | --- | --- |
| 社区 | 大 | 小 |
| PyTorch 集成 | **torch.compile 默认后端** | 独立 |
| 版本 | 3.x（较稳） | 0.1.x（快） |
| 文档 | 官方更全 | 本套文档 |

## 11. 选型决策（Triton 视角）

| 诉求 | 推荐 |
| --- | --- |
| 快速开发 + torch.compile + 生态 | Triton |
| 极致性能 + 深度定制 pass | TileLang |
| 多硬件 | TileLang |
| 调试工具 | Triton |
| 学习 MLIR 编译器 | Triton |
| 学习 TVM 体系 | TileLang |

## 12. 双修路线（Triton 视角）

1. **先 Triton 后 TileLang**：Triton 缓学，掌握块级编程后再看 TileLang 的 `T.Kernel/T.gemm`。
2. **关键迁移表**：
   - `tl.program_id` ↔ `T.Kernel(...) as (bx, by)`
   - `tl.arange` ↔ `T.Parallel`
   - `tl.load/store` ↔ `T.copy`
   - `tl.dot` ↔ `T.gemm`
   - `num_warps/num_stages` ↔ `threads/stages`
   - `TRITON_DUMP_IR` ↔ `TL_ENABLE_DUMP_IR`
   - `TRITON_ALWAYS_COMPILE` ↔ `compile(rebuild=True)`

## 13. 深入自测

1. LinearLayout 给 Triton 带来什么自动化能力？
2. Triton 的 mma 降级在哪、什么语言？
3. 缓存 key 的差异？
4. 调试工具链谁强？为什么？
5. Triton 在工程上最适合教什么（调试/lit/LLVM）？

## 14. 下一步

进入 `19_Triton与CUDA映射关系.md`（深度版）。
