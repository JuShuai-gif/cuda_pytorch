# 08｜TensorRT Plugin：把自定义算子接入 Engine

## 本模块解决的问题

TensorRT 通过 layer fusion + tactic selection 优化模型，但它只能优化**它认识的算子**。当模型里有 TensorRT 不支持的算子（自定义归一化、新的激活函数、量化 GEMM、fused attention 变体）时，整条链路会断掉。Plugin 就是"自己写 CUDA kernel 并接入 TensorRT engine"的机制。

本章回答：

```text
TensorRT 不支持某个 operator 时怎么解决？
Plugin 类需要实现哪些方法，各有什么作用？
为什么 engine 反序列化还需要 Creator？
IPluginV2 和 IPluginV3 的区别是什么？
```

配套代码：`src/inference/tensorrt/plugin/`（`rmsnorm_plugin.cu` + `rmsnorm_demo.cu`）。

---

## 1. 工业场景：算子缺失

以 RMSNorm 为例：它是 LLaMA/VLA 系模型的核心归一化算子，但 TensorRT 原生并不总是提供 fused RMSNorm。此时有三条路：

```text
1. 用现有算子组合模拟（LayerNorm 近似 / 手写多算子）→ 丢失 fusion，性能差
2. 等 TensorRT 版本更新支持 → 不可控
3. 自己写 Plugin → 完全可控，这是工业标准做法
```

Plugin 的本质：**把你的 CUDA kernel 包装成 TensorRT 认识的"层"，参与它的图优化和 engine 序列化**。

---

## 2. Plugin 的完整生命周期

一个 RMSNorm plugin 从写入到运行要经过：

```text
CUDA kernel（rmsnorm_kernel）
   ↓ 包装
RMSNormPlugin（IPluginV2DynamicExt 实现，enqueue 里调 kernel）
   ↓ 注册
RMSNormPluginCreator（IPluginCreator，负责创建/反序列化）
   ↓ REGISTER_TENSORRT_PLUGIN 静态注册到 registry
   ↓
构建期：registry 取 creator → createPlugin → addPluginV2 插入 network → 构建 engine → 序列化
   ↓
推理期：反序列化 engine（此时需要 creator 恢复 plugin）→ enqueue
```

关键点：**engine 序列化时 plugin 的状态（本例子里的 eps）被 `serialize` 写进文件，反序列化时 `deserializePlugin` 重建 plugin 实例**。这就是为什么"只有 Plugin 类"不够，还必须有 Creator。

---

## 3. 必须实现的方法（IPluginV2DynamicExt）

| 方法 | 作用 |
|---|---|
| `getOutputDimensions` | 声明输出 shape（dynamic 维度用表达式） |
| `supportsFormatCombination` | 声明支持的数据类型/格式组合 |
| `enqueue` | **核心**：调用你的 CUDA kernel，拿到 stream |
| `getWorkspaceSize` | 声明额外 workspace 大小 |
| `getSerializationSize` / `serialize` | 把 plugin 状态写进 engine 文件 |
| `clone` | 复制（TensorRT 构建时会 clone） |
| `initialize` / `terminate` | 资源初始化/释放 |
| `destroy` | 释放自己 |
| `getPluginType` / `getPluginVersion` | 类型/版本标识（registry 用它查找） |

本例子里的 `enqueue`：

```cuda
int32_t enqueue(... cudaStream_t stream) noexcept override {
    auto dims = inputDesc[0].dims;      // 运行时拿到实际 shape
    int rows = dims.d[0] * dims.d[1];
    int cols = dims.d[2];
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, y, rows, cols, mEps);
    return 0;
}
```

注意：**kernel 必须 launch 在 TensorRT 传入的 stream 上**，这样才能和 engine 里其他层正确串行/重叠。

---

## 4. 实测：完整链路跑通

本机 Thor/sm_110，RMSNorm plugin（fp32，dynamic batch/seq）：

```text
build → serialize → deserialize → infer → verify
correct_max_abs_diff = 0.0   （fp32 + 简单运算，精确）
engine_size = 3652 bytes     （只有 plugin 层 + eps，无权重）
```

diff 为 0 说明 plugin 的 enqueue、shape 传递、反序列化全都正确——这一步验证的是**链路正确性**，比性能更重要（性能在 Stage 4 的 CUDA kernel 优化里单独做）。

---

## 5. V2 vs V3：为什么编译时看到 deprecated 警告

本机 TensorRT 10.13 编译 plugin 时，`IPluginV2DynamicExt` 会报 deprecated 警告。原因是：

- **IPluginV2 / IPluginV2DynamicExt**：旧 API，从 TensorRT 6 一路沿用，**功能完全可用**，绝大多数工业代码、博客、教程都基于它。
- **IPluginV3**：TensorRT 10.x 引入的新 API，把 plugin 拆成 `IPluginV3OneCore`（核心）、`IPluginV3OneBuild`（构建期）、`IPluginV3OneRuntime`（推理期）三个接口，目的是把"构建期状态"和"推理期状态"分离，支持更精细的 shape/format 协商。

**本模块用 V2**，因为它是理解 Plugin 机制最清晰的入口，且兼容性最好。掌握了 V2 的方法职责后，迁移 V3 只是把同一个职责拆分到三个接口。生产新代码如果必须跟 10.x 的新特性，再迁 V3。

---

## 6. Plugin 在工业推理中的位置

```text
cuBLAS/CUTLASS 覆盖的算子 → 不用 plugin，交给 TensorRT 内置
TensorRT 不支持的算子   → 自己写 Plugin（本模块）
需要极致融合的自定义算子 → Plugin + 手写 CUDA kernel（Stage 2/4 的能力）
```

vLLM、SGLang、TensorRT-LLM 里大量用 plugin 实现自定义 attention、量化 GEMM、fused 归一化。所以"会写 Plugin" = "能把 Stage 2/4 的 CUDA kernel 能力接到 TensorRT 的图优化里"，是岗位 A 的关键技能。

---

## 7. 本模块闭环小结

```text
问题：TensorRT 不支持某算子怎么办
      ↓
机制：CUDA kernel → Plugin（enqueue）→ Creator（序列化）→ registry → addPluginV2
      ↓
验证：build → serialize → deserialize → infer → verify（diff=0）
      ↓
结论：Plugin 是把自定义 kernel 接入 engine 的标准机制
```

---

## 8. Stage 7 收尾总结

Stage 7 完成了 TensorRT 的完整能力链：

```text
07_tensorrt.md      PyTorch → ONNX → Engine（FP32/FP16，dynamic shape，快 4x）
08_tensorrt_plugin.md  自定义算子接入（RMSNorm plugin 全链路）
```

下一模块进入 **Stage 8 量化**：

```text
FP32/TF32/FP16/BF16/FP8/INT8/INT4 精度体系
量化公式 x_q = round(x/scale)、scale/zero-point/clipping/calibration
per-tensor / per-channel / per-token / per-group / weight-only
PTQ/QAT、SmoothQuant、AWQ、GPTQ
实测精度 vs 延迟 vs 吞吐 vs 显存 vs 模型大小
```

要继续就说「继续」。
