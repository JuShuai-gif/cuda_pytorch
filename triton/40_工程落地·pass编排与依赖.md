# 40 工程落地·pass 编排与依赖（深度版）

> 本类问题的共性：编译器是 pass 流水线，pass 之间有**顺序依赖**。难点在于"**顺序必须正确、架构分支必须清晰、可重入性必须保证**"。

## 机制 1：make_ttgir 的 pass 顺序

### 约束
- pass 之间有严格先后（coalesce 在 accelerate_matmul 前等）。
- 顺序错了 → 错误产物。

### 实现（`nvidia/backend/compiler.py:262-340`）
```python
passes.ttir.add_convert_to_ttgpuir(pm, ...)   # ① 布局推断
passes.ttgpuir.add_coalesce(pm)               # ② 访存合并
passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)    # ③ TF32
passes.ttgpuir.add_remove_layout_conversions(pm)  # ④ 布局消除
passes.ttgpuir.add_accelerate_matmul(pm)      # ⑤ mma 布局
passes.ttgpuir.add_optimize_dot_operands(pm)  # ⑥ dot 操作数
...
```

> **不变量 PO1**：pass 顺序满足每个 pass 的"输入前提"（如 accelerate_matmul 前必须先有布局）。

### 工程判断
- **顺序 = 依赖显式化**：写死在 `make_ttgir` 里。
- 例证：`coalesce`（产 Blocked 布局）必须先于 `accelerate_matmul`（基于已有布局选 mma）——顺序违反则 mma 选择无依据。

## 机制 2：架构分支（capability 分叉）

### 约束
- 不同 GPU 架构需要不同 pass 集（Hopper vs Blackwell）。

### 实现（`make_ttgir` 内 if）
```python
if capability // 10 in [8, 9]:
    add_hopper_warpspec; add_assign_latencies; add_schedule_loops; add_pipeline
elif capability // 10 >= 10:
    add_warp_specialize; add_pipeline; add_hoist_tmem_alloc; add_optimize_partition_warps
else:
    add_triton_licm
```

> **不变量 PO2**：同一 kernel 在不同架构上走正确的 pass 分支。

### 工程判断
- **架构分支显式化**：`capability//10` 作为分叉键，每个分支是完整且自洽的 pass 集。
- **新架构 = 新分支**：维护成本随架构数线性增长——**这是硬件加速编译器的必然工程形态**。
- 每个分支末尾 `add_pipeline` 共性是流水线——**共性 pass 共享，差异 pass 分支**。

## 机制 3：架构分支的可测试性（lit）

### 约束
- 分支逻辑必须可验证，不能靠跑真机。

### 实现（`test/TritonGPU/*.mlir` + FileCheck）
```mlir
// RUN: triton-opt %s --ttgpu-accelerate-matmul | FileCheck %s
// CHECK: tt.dot ... #ttg.dot_op<...>
```

> **不变量 PO3**：每个 pass 的关键行为都有 lit 测试固定。

### 工程判断
- **lit 免 GPU 验证 pass 行为**：这是 MLIR 生态的标准做法，让 pass 开发"纯编译器测试"。
- 比 TileLang 的 pytest assert 更适合 pass 精确验证（见 `18` 对比）。

## 机制 4：pass 可重入性（同一 IRModule 多次跑）

### 约束
- 同一个 pass 可能跑多次（不同 num_stages 等）。

### 实现
- MLIR pass 是 `OperationPass`，`runOnOperation` 修改 ModuleOp。
- 幂等性：pass 重复跑应稳定（或 pass 显式声明非幂等）。

> **不变量 PO4**：pass 在相同输入 + 相同配置下产出相同结果（确定性）。

### 工程判断
- **确定性 = 可重放**：autotune 需要"同一 kernel 多配置"的重放能力。
- MLIR 的 pass 管理保证重复 run 是合法的（不破坏 IR 结构）。

## 机制 5：阶段链（add_stages）的编排

### 约束
- `add_stages` 注册 ttir/ttgir/llir/ptx/cubin 五阶段。
- 每阶段产物喂下一阶段。

### 实现（`nvidia/backend/compiler.py:598-609`）
```python
def add_stages(self, stages, options, language):
    stages["ttir"] = ... make_ttir ...
    stages["ttgir"] = ... make_ttgir ...
    stages["llir"] = ... make_llir ...
    stages["ptx"] = ... make_ptx ...
    stages["cubin"] = ... make_cubin ...
```

> **不变量 PO5**：阶段链顺序固定，且每阶段可独立产出（缓存/调试）。

### 工程判断
- **阶段即缓存粒度**：每阶段产物独立缓存（`.ttir/.ttgir/...`），可部分复用——**阶段化 = 缓存粒度化**。
- `IRSource` 可从任意阶段开始（`first_stage` 逻辑，compiler.py:292）——**支持从 IR 文件调试**。

## 本类工程判断总结

1. **顺序 = 依赖显式化**（写死在 make_ttgir）。
2. **架构分支显式化**（capability 分叉，共性共享、差异分支）。
3. **lit 固定 pass 行为**（免 GPU 可测试）。
4. **确定性 = 可重放**（autotune 需要）。
5. **阶段化 = 缓存粒度化**（每阶段独立产物，支持从 IR 调试）。

## 深入自测

1. 为什么 coalesce 必须在 accelerate_matmul 前？
2. SM89 与 SM100 的 pass 分支差异？共性是什么？
3. lit 测试如何固定 pass 行为？
4. 阶段链如何支持缓存复用与 IR 调试？
5. 对比 TileLang 的 CUDAPassPipelineBody。

## 下一步

回顾 `32_工程落地设计·总纲` 与 `31_架构设计与工程权衡.md`，形成工程维度全景。
