# 18 与 Triton 对比（技术点 + 工程双维度深度版）

> 本文目标：从"机制对比"升级为"双维度对比"——技术点（IR/布局/pass/降级/缓存）与工程（构建/测试/调试/CI/生态/贡献）各成体系，并给出"选型决策"与"双修路线"。

## 第一部分：技术点对比

## 1. 技术栈全貌

| 维度 | TileLang | Triton | 本质差异 |
| --- | --- | --- | --- |
| 底层 IR | TIRX（TVM TIR 扩展） | MLIR（ttir/ttgir/llir） | 两套独立 IR 生态 |
| 依赖 | TVM 定制分叉 + CUTLASS | LLVM/MLIR + 自研 PTX | Triton 更"自给自足" |
| Python↔C++ | tvm_ffi | nanobind | 绑定层不同 |
| 目标硬件 | CUDA/HIP/Metal/WebGPU/CPU | 主 CUDA（插件架构） | TileLang 覆盖更广 |
| 语言前端 | eager builder（逐句） | `ast.NodeVisitor`（整体） | 翻译策略不同 |

## 2. 布局系统：数学本质对比（核心）

| | TileLang | Triton |
| --- | --- | --- |
| 表示 | `Layout{input_size, forward_index}` 显式函数 | `LinearLayout` GF(2) 矩阵（只存基向量） |
| 求值 | `Forward()` 代入占位符 | `apply()` 用 xor 累加基向量 |
| 求逆 | `DetectIterMap` + `InverseAffineIterMap` | `invertAndCompose`（伪逆，高斯消元） |
| 组合 | 无通用代数运算 | `operator*`（直和）、`invertAndCompose` |
| 表达力 | 强（可含 floordiv/xor） | 受限（GF(2) 线性） |
| 可计算性 | 弱 | 强（布局转换=矩阵运算） |

**推论**：
- Triton 的布局转换（`convert_layout`）本质是伪逆矩阵运算，可被 `RemoveLayoutConversions` 大规模代数化简 → 编译器更"自动"。
- TileLang 布局表达力强（fragment 线程映射可含任意表达式），但求逆依赖启发式 → 编译器更"可定制"但自动化程度较低。
- 都对应同一硬件事实（mma 每线程元素分布、swizzle bank 分配），只是编码方式不同。

## 3. mma 降级对比

| | TileLang | Triton |
| --- | --- | --- |
| 降级位置 | C++ pass 调 `tl.gemm.lower`（Python 生成 IR） | C++ `DotOpToLLVM` 分派 |
| 指令模板 | CUTLASS `tl::mma_sync<>` | 自研内联 PTX（MMAv2.cpp） |
| 版本选择 | `SelectInst`（cuda/op/gemm.cc:337） | `getMMAVersionSafe`（AccelerateMatmul.cpp:43） |
| FMA 兜底 | `GemmFMA` | `FMA.cpp` + `FMADotUtility.cpp` |
| 架构覆盖 | SM70/75/80/89/90/100 | Volta→Blackwell |

**推论**：TileLang 把"指令选择+IR 生成"下放到 Python（`tileop/gemm/gemm_mma.py`），改 gemm 算法只需改 Python；Triton 的 mma 降级在 C++，改动需重编。**这是"用 Python 做编译器算法"vs"用 C++ 做编译器算法"的代表性差异**。

## 4. 缓存机制对比

| | TileLang | Triton |
| --- | --- | --- |
| 进程内 | `KernelCache`（单例） | `kernel_cache` dict |
| 磁盘 | `~/.tilelang/cache/<version>/.../kernels/<key>/` | `~/.triton/cache/<base32(hash)>/` |
| 键输入 | IR脚本+args+target+configs+版本+lib指纹 | triton_key+src+backend+options+env |
| 原子写 | staging + os.rename | uuid临时目录 + os.replace |
| 二级缓存 | `CUDABinaryCache`（跳 nvcc） | 无（单层） |

**推论**：TileLang 有"整 kernel 缓存"+"设备二进制缓存"两层；Triton 一层但 key 含 libtriton 二进制 hash（编译器源码改动即失效）。两者都追求"改动源码/参数即失效、跨进程复用"。

## 5. pass 链对比

| | TileLang | Triton |
| --- | --- | --- |
| 定义处 | `CUDAPassPipelineBody`（cuda/pipeline.py:145） | `make_ttgir`（nvidia/compiler.py:262） |
| 可注入 | 是（注册自定义 pass） | 否（固定） |
| 配置 | `pass_configs`（30+ 项） | `CUDAOptions` + knobs |
| 架构分支 | target 相关 | capability 分支（<90/90/100/120） |

## 6. 技术点总结表

| 技术点 | 谁更强 | 原因 |
| --- | --- | --- |
| 布局自动化 | Triton | GF(2) 矩阵可代数化简 |
| 布局表达力 | TileLang | 任意表达式 |
| pass 可定制性 | TileLang | 可注入自定义 pass |
| 编译器算法开发效率 | TileLang | Python 生成 IR（免重编） |
| 端到端自动化 | Triton | TTGIR 直接含布局+mma |
| 硬件覆盖 | TileLang | 多后端 |
| 缓存架构 | TileLang | 两级缓存 |
| 依赖精简 | Triton | 自研 PTX，不强依赖 CUTLASS |

## 第二部分：工程对比

## 7. 构建与安装

| | TileLang | Triton |
| --- | --- | --- |
| 构建后端 | scikit-build-core（CMake+setuptools 混合） | setuptools + setup.py 调 CMake |
| C++ 绑定 | tvm_ffi | nanobind |
| 第三方大依赖 | TVM 分叉 + CUTLASS + CK | LLVM/MLIR（构建时拉取） |
| 首次编译时长 | 10-30 分钟 | 30-60 分钟（含 LLVM） |
| 安装方式 | `pip install -e .` | `pip install -e .` / `make dev-install` |
| wheel 分发 | 官方 wheel（含预编译 CUDA 二进制） | 官方 wheel（PyPI `triton`） |
| 编译隔离 | 无 build isolation | 同 |

**工程要点**：
- 两者都支持 editable install + 增量编译。
- TileLang 的 TVM 是**源码子模块**（`3rdparty/tvm`），构建耦合 TVM；Triton 的 LLVM 通过 CMake 拉取/查找。
- TileLang 改 C++ 重跑 `pip install -e .`；Triton 用 `make`（`ninja -C $BUILD_DIR`）。

## 8. 测试体系

| | TileLang | Triton |
| --- | --- | --- |
| IR/pass 测试 | `testing/python/transform/`（pytest assert） | `test/`（lit + FileCheck） |
| Python 测试 | `testing/python/`（pytest） | `python/test/unit/`（pytest） |
| C++ 测试 | `testing/cpp/`（gtest） | `test-cpp`（gtest） |
| GPU 依赖 | 大部分需要 | pytest 需要，lit 不需要 |
| 行为真相来源 | transform 测试的 assert | lit 的 CHECK 断言 |

**工程要点**：
- Triton 的 lit 测试（`.mlir` + FileCheck）更适合"pass 行为精确验证"，无需 GPU。
- TileLang 的 transform 测试用 pytest assert 验证 pass 输出（如 pipeline 的 `stage == [0,2]`）。
- 两者都以"可执行断言固定 pass 行为"为最佳实践。

## 9. 调试工具链

| | TileLang | Triton |
| --- | --- | --- |
| IR dump | `TL_ENABLE_DUMP_IR=1` | `TRITON_DUMP_IR=1` |
| 单跑 pass | 无独立 CLI（用 Python API） | `triton-opt`（功能强） |
| 崩溃最小化 | 无 | `triton-reduce` |
| 崩溃复现 | lower_trace/pass_diff | `--run-reproducer`（`{-# ... #-}` metadata） |
| 解释执行 | 无（eager 已近解释） | `TRITON_INTERPRET=1` |

**工程要点**：Triton 的调试工具链更成熟（triton-opt/triton-reduce/reproducer），因为它深度依赖 MLIR 生态；TileLang 的 `TL_ENABLE_DUMP_IR` + pass_diff 覆盖基本需求。

## 10. 环境变量体系

| | TileLang | Triton |
| --- | --- | --- |
| 管理文件 | `tilelang/env.py` + `PassConfigKey` | `python/triton/knobs.py` |
| 分组 | 编译/缓存/调试 | cache/compilation/runtime/nvidia/proton |
| 运行时 hook | 少 | 丰富（launch_enter_hook 等，供 profiler 用） |

## 11. 生态与社区（工程现实）

| | TileLang | Triton |
| --- | --- | --- |
| 发起方 | tile-ai（含中国团队贡献） | OpenAI（triton-lang 维护） |
| 社区规模 | 较小 | 大 |
| PyTorch 集成 | 独立 | **torch.compile 默认后端** |
| 算子覆盖 | 100+ 示例（含 fusedmoe/deepseek） | tutorials + 生态 |
| 版本迭代 | 0.1.x（快，API 可能变动） | 3.x（较稳） |
| 文档 | 本文档 + docs/ | 官方文档更全 |

## 12. 贡献工程对比

| | TileLang | Triton |
| --- | --- | --- |
| pass 加文件 | `src/transform/*.cc`，CMake glob 自动收集 | `lib/.../Transforms/`，**需改 CMakeLists** |
| 代码风格 | TVM 风格 | AGENTS.md（assert 无副作用等） |
| 测试要求 | transform assert | lit + pytest |
| 本地复现 CI | pytest | `make test-lit/test-unit` |

## 13. 选型决策表

| 你的诉求 | 推荐 |
| --- | --- |
| 极致性能 + 深度定制 pass | TileLang |
| 快速开发 + 生态 + torch.compile | Triton |
| 多硬件（AMD/Apple/Web） | TileLang |
| 调试工具成熟度 | Triton |
| 编译器算法用 Python 写 | TileLang |
| 布局自动化简 | Triton |

## 14. 双修路线（工程视角）

1. **先 Triton 后 TileLang**：Triton 学习曲线缓，掌握块级编程后，TileLang 的 `T.Parallel/T.gemm` 秒懂。
2. **先 TileLang 后 Triton**：TileLang 让你理解 TVM 体系，再学 Triton 的 MLIR 生态（`triton-opt` 等）。
3. **关键迁移表**：
   - `tl.arange` ↔ `T.Parallel`
   - `tl.load/store` ↔ `T.copy`
   - `tl.dot` ↔ `T.gemm`
   - `num_warps/num_stages` ↔ `threads/stages`
   - `TRITON_DUMP_IR` ↔ `TL_ENABLE_DUMP_IR`

## 15. 深入自测

1. 布局系统的数学本质差异？推论是什么？
2. mma 降级的位置差异（Python vs C++）？
3. 缓存机制的两层 vs 一层差异？
4. 构建系统差异（scikit-build-core vs setup.py+CMake）？
5. 测试体系差异（pytest assert vs lit FileCheck）？
6. 调试工具链谁更成熟？为什么？
7. 什么场景选 TileLang，什么场景选 Triton？

## 16. 下一步

进入 `19_CUDA与GPU概念映射.md`（深度版）。
