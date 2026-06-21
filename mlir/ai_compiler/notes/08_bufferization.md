# 08 · Bufferization：Tensor → MemRef

> 对应代码：`src/Conversion/Pipelines.cpp`（`edge-lower-to-loops` / `edge-lower-to-llvm`
> 流水线中的 one-shot-bufferize 段）
> 验证：`ninja -C build check-edge`（lower-pipeline 测试已通过，Edge 一路降到 LLVM 方言）

---

## 1. 中文原理讲解

Bufferization 是把"值语义的张量"(`tensor`, 不可变, SSA) 转成"引用语义的缓冲区"(`memref`, 可变,
有地址) 的过程。它是从"数学表达"跨入"真实内存执行"的分水岭。

本项目用 MLIR 的 **One-Shot Bufferize**（`one-shot-bufferize{bufferize-function-boundaries=true}`）：
- 一次性、全模块地把所有 `tensor` 算子转成 `memref` 算子；
- 通过别名/写冲突分析尽量**就地复用**缓冲区（in-place），减少 alloc/copy；
- `bufferize-function-boundaries=true` 让函数签名也转成 memref（配合 `buffer-results-to-out-params`
  把返回的张量改为输出参数, 这是 C ABI / 嵌入式运行时的常见约定）。

验证结果：`edge.relu`/`edge.matmul` 经 Edge→Linalg→one-shot-bufferize 后变为 `memref.alloc` +
`scf.for` + `memref.load/store`，再继续降到 LLVM 方言（69 行 `llvm.*`, 零错误）。

## 2. 工业背景

任何要在真实硬件上跑的张量程序最终都要落到"buffer + 读写"。Bufferization 决定了内存分配/复用的
质量，直接影响峰值内存与带宽。它和 Module 09 的内存规划是一对：bufferization 决定"有哪些 buffer",
memory planner 决定"这些 buffer 如何共享地址"。

## 3. TensorRT 对应模块

TensorRT 的 `IExecutionContext` + workspace/`ITensor` 绑定就是 buffer 层；builder 决定每个张量是否
需要独立显存、能否原地复用——对应 one-shot-bufferize 的 in-place 分析。

## 4. TVM 对应模块

≈ TVM 的 `StorageRewrite` / `LowerTE` 把 TE 的 tensor 降到带 `Allocate` 的 TIR buffer；in-place 复用
≈ TVM 的 storage planning。

## 5. TPU-MLIR 对应模块

TPU-MLIR 同样用 MLIR bufferization 思路把 `tpu` 张量降到带地址的内存对象, 再做 LMEM/GMEM 分配。

## 6. Ascend CANN 对应模块

≈ GE 的内存分配阶段 (MemoryAssigner): 给每个算子输出分配 device 内存, 并做复用; UB 融合也涉及
片上 buffer 的就地复用。

## 7. 性能收益

- in-place bufferization 消除冗余拷贝与分配, 直接降低带宽与峰值内存。
- 函数边界 bufferization + out-params 让运行时可预分配所有 buffer, 推理期零动态分配 (实时性关键)。

## 8. Trade-off

- 过度就地复用会引入写后读 (WAR) 依赖, 限制并行/流水; one-shot 分析需在"省内存"与"可并行"间权衡。
- 函数边界 bufferization 改变 ABI, 与上层调用约定耦合, 需要 `buffer-results-to-out-params` 配套。

## 9. 常见 Bug（本模块真实注意点）

1. **`tensor.empty` 无法 bufferize**：需要 one-shot-bufferize 正确处理 (或先 `empty-tensor-to-alloc-tensor`);
   本项目的配方直接可用。
2. **返回张量到 LLVM 报错**：必须 `buffer-results-to-out-params` 把返回 memref 改为出参, 否则
   `convert-func-to-llvm` 处理返回的 memref 结构体时会出问题。
3. **strided memref 到 LLVM**：函数参数是 `memref<...strided>`，需 `expand-strided-metadata` +
   `finalize-memref-to-llvm` 才能正确降级。
4. **顺序敏感**：bufferize 必须在 `convert-linalg-to-loops` 之前 (loops 跑在 memref 上)。

## 10. 调试方法

- `--one-shot-bufferize="...test-analysis-only"`：只看别名/就地分析结果, 不改 IR。
- `--mlir-print-ir-after=one-shot-bufferize`：看 bufferize 后的 memref IR。
- 出现意外 copy 时, 用 `--debug-only=one-shot-analysis` 看为何无法就地。

## 11. Profiling 方法

- bufferize 后统计 `memref.alloc` 数量与总字节数 ≈ 估算峰值内存 (精确版见 Module 09)。
- `--mlir-timing` 看 bufferize/loops/llvm 各段编译耗时。

## 12. 在机器人 / VLA 中的应用

机器人推理要求**确定性、无运行期分配**。函数边界 bufferization + out-params 让我们把 VLA 策略网络
编译成"调用方预分配所有 buffer、推理期零 malloc"的形态, 满足硬实时控制环 (10–50 Hz) 的延迟可预测性。

> 下一步（Module 09）：在 bufferize 产生的 buffer 集合上做生命周期分析 + graph-coloring 复用, 量化并压低峰值内存。
