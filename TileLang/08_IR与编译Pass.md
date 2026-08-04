# 08 IR 与编译 Pass（深度版）

> 本文目标：从"pass 列表"升级为"pass 机制分析"——每个 pass 的内部算法、输入输出、以及"一个 kernel 如何被逐级变换"。这是阅读 TileLang 源码的核心地图。

## 1. IR 层级

```mermaid
flowchart LR
    H["高层 IR (TIRX, 含 tl.tileop.*)"] --> M["中低层 IR (LowerTileOp 展开后)"]
    M --> L["低层 IR (向量化/原子/同步后)"]
    L --> C["CUDA 源码"]
```

关键点：**tile 级算子（T.copy/T.gemm）在高层的 IR 里是函数调用节点，直到 LowerTileOp pass 才被展开成低层 load/copy/sync/mma**。

## 2. Pass 生命周期与执行环境

- 每个 pass = `tvm::transform::Pass`（`CreatePrimFuncPass` 包装，`layout_inference.cc:1266`）。
- 全部在 `PassContext(opt_level=3, config=pass_configs)` 中运行（`jit/kernel.py:274`）。
- pass 用 `PassContext::Current()` 读配置。

## 3. 核心 Pass 深度解析

### 3.1 MaterializeKernelLaunch
- 作用：把 `T.Kernel` 的 launch frame 变成 `thread_extent` AttrStmt 或串行 For。
- 位置：`src/transform/materialize_kernel_launch.cc`。
- 关联：`T.Kernel`（`kernel.py:277`）→ `_ffi_api.KernelLaunch`。

### 3.2 PipelinePlanning（深度）
位置：`src/transform/pipeline_planning.cc:1093-1348` `PipelinePlanner::VisitStmt_(ForNode)`。

**两条路径**：
1. **手动调度**（带 `tl_pipeline_order/stage` annotation）：转写为标准 annotation（:1097-1154）。
2. **自动规划**（带 `num_stages`）：核心流程（:1182-1340）：
   - `MakePipelineStageInfo`（:982）：对每条顶层语句分类（拷贝/消费）。
   - `IsPureCopyStmt`（:510）：判断 global→shared 拷贝。
   - `AnalyzeCopyLastUse`（:644）：计算每个拷贝的最后使用语句。
   - **stage 分配**（:1226-1254）：
     ```cpp
     pinfo.stage = num_stages;         // 非拷贝语句
     pinfo.stage = 0;                  // 拷贝（producer）
     ```
   - **旋转优化**（:1262-1287）：拷贝在 order 末尾时旋转到开头，消费方 stage-1。
   - 输出 annotation：`software_pipeline_stage/order/async_producers/async_groups`。

**典型结果（已确认，测试 :606-609）**：
```
num_stages=3, copy+gemm
=> stage: [0, 2]   (copy→0, gemm→2)
=> order: [0, 1]
```

### 3.3 InjectSoftwarePipeline（深度）
位置：`src/transform/inject_pipeline.cc:3370-3803` `PipelineInjector::VisitStmt_(ForNode)`。

**多缓冲**（`ComputeBufferVersions` :1474）：
- `num_versions = use_stage - def_stage + 1`。
- `RewriteAllocBuffer`（:1537）：shape 前插一维 num_versions → `A_shared[3,128,32]`。
- 读写改索引：`floormod(ko, 3)`（:1110）。

**三段式**（`EmitImpl` :2830）：
```cpp
prologue = EmitImpl(min, min + max_stage, true, true)      // 展开
body     = EmitImpl(min + max_stage, min + extent, false, false)  // serial
epilogue = EmitImpl(min + extent, min + extent + max_stage, true, true)  // 展开
```

**cp.async commit/wait**：
- commit：`async_commit_queue_scope` attr（:2808）→ `ptx_commit_group`（:356）。
- wait：`async_wait_queue_scope` + `async_wait_inflight_count`（:2720）→ `ptx_wait_group(n)`（:366）。
- 稳态 wait = `num_stages - 1`（测试 :407 确认 `loop_waits == [3]`）。

**IR 变化示例**（num_stages=3, copy+gemm）：
```
# 前
for ko in T.Pipelined(32, num_stages=3):
    T.copy(A[ko], A_shared)
    T.gemm(A_shared, B_shared, C_local)

# 后（简化）
# prologue (展开 2 次)
for ko in unrolled(0, 2):
    A_shared[floormod(ko,3)] = copy(A[ko])
# body (稳态)
for ko in serial(2, 30):
    A_shared[floormod(ko,3)] = copy(A[ko])
    commit_group()          # ptx_commit_group
    wait_group(2)           # ptx_wait_group(2)
    gemm(A_shared[floormod(ko-2,3)], ...)
# epilogue (排水, 展开 2 次)
for ko in unrolled(32, 34):
    wait_group(1)
    gemm(A_shared[floormod(ko-2,3)], ...)
```

### 3.4 LayoutInference（深度）
位置：`src/transform/layout_inference.cc`。完整算法见 `20_核心数据结构.md` 深度版。核心流程（`BufferUseDefCollector::Run` :316-464）：
1. 漂浮 fragment → FullyReplicated。
2. kStrict 推断。
3. BFS 队列 kCommon 推断。
4. kFree 模式：Union-Find 连通分量 + 最小寄存器搜索（:1012-1167）。
5. 别名传播 + 注解回写。

### 3.5 LowerTileOp（深度）
位置：`src/transform/lower_tile_op.cc:1117-1198` `VisitStmt_(EvaluateNode)`：
```cpp
auto tile_op = ParseOperator(...);      // 识别 tileop
auto lowered = tile_op->Lower(lower_args, analyzer_);  // 生成替换
return IRMutatorWithAnalyzer::VisitStmt(lowered);       // 递归改写
```
- `LowerArgs` 携带 target/thread_bounds/layout_map 等（:1178-1193）。
- `GemmNode::Lower` → `tl.gemm.lower`（Python）。

### 3.6 其他 pass 速览

| Pass | 作用 | 位置 |
| --- | --- | --- |
| VectorizeLoop | 循环向量化 | `vectorize_loop.cc` |
| LegalizeVectorizedLoop | 向量化合法性 | `legalize_vectorized_loop.cc` |
| LegalizeSafeMemoryAccess | 越界安全 | `legalize_safe_memory_access.cc` |
| StorageRewrite | 存储优化/重写 | `storage_rewrite.cc` |
| FlattenBuffer | buffer 扁平化 | `flatten_buffer.cc` |
| LowerThreadAllreduce | 跨线程归约 | `lower_thread_allreduce.cc` |
| SplitHostDevice | host/device 分离 | `split_host_device.cc` |
| MergeSharedMemoryAllocations | 共享内存合并 | `merge_shared_memory_allocations.cc` |
| ThreadSync | 插入同步 | `thread_storage_sync.cc` |
| MakePackedAPI | 打包 ABI | `make_packed_api.cc` |
| LowerDeviceKernelLaunch | launch 代码 | `lower_device_kernel_launch.cc` |

## 4. Pass 流水线（CUDA）

`tilelang/cuda/pipeline.py:145-258` 的 `CUDAPassPipelineBody`。完整序列：
```
MaterializeKernelLaunch
→ PipelinePlanning
→ InjectSoftwarePipeline
→ Simplify
→ LayoutInference
→ LayoutVisual
→ LowerTileOp
→ (更多，以源码为准)
→ VectorizeLoop → LegalizeVectorizedLoop
→ StorageRewrite
→ LowerThreadAllreduce
→ SplitHostDevice
→ MergeSharedMemoryAllocations
→ ThreadSync
→ MakePackedAPI
→ LowerDeviceKernelLaunch
```

## 5. 如何调试 pass（升级版）

| 工具 | 用法 | 定位 |
| --- | --- | --- |
| Dump IR | `TL_ENABLE_DUMP_IR=1` + `TL_DUMP_IR_DIR` | pass 间 IR |
| lower_trace | `TL_LOWER_TRACE=1` | 逐 pass 追踪 |
| pass_diff | `TILELANG_PASS_DIFF=terminal` | pass 前后 diff |
| pass_visualizer | `tilelang/tools/pass_visualizer` | IR 树浏览 |
| 计时 | `TL_PASS_PROFILE=1` | pass 耗时 |

## 6. pass_configs 速查（深度版）

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `tl.enable_fast_math` | False | nvcc `--use_fast_math` |
| `tl.enable_async_copy` | True | cp.async 降级 |
| `tl.disable_vectorize_256` | False | 256-bit 向量化 |
| `tl.disable_wgmma` | False | 禁用 wgmma |
| `tl.disable_warp_specialized` | False | 禁用 warp 特化 |
| `tl.disable_data_race_check` | True(关) | 数据竞争检查 |
| `tl.disable_safe_memory_legalize` | False | 越界安全 |
| `tl.enable_lower_ldgstg` | False | ldg/stg 降级 |
| `tirx.disable_cse_tir` | False | CSE |
| `tirx.noalias` | True | 非别名假设 |
| `tl.ptxas_register_usage_level` | None | ptxas 寄存器级别 |
| `tl.device_compile_flags` | None | 额外 nvcc 参数 |

## 7. 深入自测

1. `T.Pipelined` 在 IR 层产生什么？两个 pass 各做什么？
2. InjectSoftwarePipeline 的"多缓冲"如何在 IR 上体现？
3. LayoutInference 的四步流程？
4. LowerTileOp 如何处理 `T.gemm`？
5. 稳态 wait_group 为什么是 num_stages-1？
6. 举 5 个 pass 及其作用。

## 8. 下一步

进入 `09_语言与DSL设计.md`（深度版）。
