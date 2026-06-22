# 如何学习大模型推理平台技术栈？

> 来源：知乎文章，约 8796 字
> 主题：以 SGLang Piecewise CUDA Graph 为例，系统讲解大模型推理平台的调试与优化

## 0. 先理解问题的背景

### 大模型推理的一个核心矛盾

大模型推理分两个阶段：
- **Prefill（预填充）**：一口气处理整个输入 prompt。计算量大，但 input 长度不固定
- **Decode（解码）**：逐 token 生成输出。每个 token 计算量小，但 GPU kernel launch 有开销

**CUDA Graph** 是解决 decode 阶段 kernel launch overhead 的利器——把整个 decode 的计算图提前录制好，每次只需重放，免去逐 kernel 调度的 CPU 开销。

### 为什么 Prefill 一直做不了 CUDA Graph？

因为 prefill 的 input shape 不固定——每次请求的 prompt 长度不一样，attention 的计算量就不同。传统 CUDA Graph 要求所有 tensor shape 固定，所以 prefill 图做不了。

### Piecewise CUDA Graph 的思路

**既然整张图做不了，那就切成小块——对能做图的部分做图。**

---

## 1. Piecewise CUDA Graph 原理

### 1.1 核心思想

通过 `torch.compile`（dynamo 前端）把 model forward 拆成若干 submodules：

```
submodule_0: token_embedding  → 做 CUDA Graph（shape 固定）
submodule_1: layer_0_attn     → eager 执行（shape 不固定，不做图）
submodule_2: layer_0_mlp      → 做 CUDA Graph（shape 固定）
submodule_3: layer_1_attn     → eager 执行
submodule_4: layer_1_mlp      → 做 CUDA Graph
...
```

**Attention 不走图**（因为 input shape 变），**Embedding 和 MLP 走图**（因为 shape 固定）。

### 1.2 怎么开启

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --enable-piecewise-cuda-graph
```

### 1.3 重要说明

SGLang 目前只用 torch dynamo 做 **graph rewrite**（把 forward 拆成 submodules），并没有使用 Inductor 做 kernel 融合。所以这里 dynamo 的角色是"切图器"，不是"优化器"。

---

## 2. Compile 阶段常见问题

torch.compile（dynamo）在工作时会 trace 整个 forward 路径，以下操作会导致 trace 失败：

| 需避免的操作 | 为什么 | 怎么办 |
|-------------|--------|--------|
| **print / logging** | dynamo 无法 trace Python 的 print | 用 `direct_register_custom_op` 包装，compile 时走 fake 函数 |
| **文件读写** | 副作用操作无法被 trace | 绕开 forward 路径中的 IO |
| **with Context Manager** | dynamo 可能无法识别自定义 context manager | 检查是否必要，用装饰器替代 |
| **数据依赖控制流** | 根据 input shape 选择不同 kernel 路径会触发 recompile / graph break | 先归束到一条固定的 kernel 执行路径 |

### 2.1 自定义 CUDA/C++ 算子怎么接入

用 `direct_register_custom_op` 包装即可。参考：
- https://github.com/sgl-project/sglang/pull/12518
- https://github.com/sgl-project/sglang/pull/13272

---

## 3. Recompile 问题排查

### 3.1 怎么观察 recompile

```bash
TORCH_LOGS=recompiles python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --enable-piecewise-cuda-graph
```

### 3.2 常见 recompile 原因和修复

| 日志提示 | 原因 | 修复 |
|---------|------|------|
| `GLOBAL_STATE changed: grad_mode` | forward 路径上 grad mode 变了 | 在 forward 上加 `@torch.no_grad()` |
| 动态 shape 变化 | 不同请求的 shape 不一致触发了新的 trace | 先归束到同一个 kernel 执行路径 |

**recompile 意味着 dynamo 认为"代码变了"重新 trace，非常耗时，要避免。**

---

## 4. 精度对拍技巧（调试 Piecewise 的精华）

### 4.1 技巧 #1：解耦 Torch Compile 和 CUDA Graph

修改 `backend.py` 中的 `submod_names_to_compile`：

```python
# 设为 []：只用 torch.compile 做图重写，不用 CUDA Graph
submod_names_to_compile = []

# 切片 [10:]：排除前 10 个子模块，逐步缩小问题范围
submod_names_to_compile = submod_names_to_compile[10:]
```

这样你可以二分定位：是 torch.compile 的问题，还是 CUDA Graph capture 的问题。

### 4.2 技巧 #2：安全打印的位置

找对地方 print 是调试的关键：

- **`unified_attention_with_output`**：Attention 不走 CUDA Graph，可以**自由 print**
- **被 `direct_register_custom_op` 包装的函数**：compile 时用 fake 函数不报错，真正执行时走真实函数
- **临时用 `direct_register_custom_op` 包装可疑函数做排查**

### 4.3 技巧 #3：两边对拍

最暴力的定位方式：

1. 一台机器开 piecewise，一台关 piecewise
2. 跑同一个输入，比较每一层的输出
3. 找到**第一个差异的层** → 问题就在那层的 submodule
4. 启用 `--enable-deterministic-inference` 保持 bitwise 一致
5. TP > 1 时设置 `NCCL_ALGO=allreduce:tree` 保证 all_reduce 确定性

---

## 5. Full Graph 是否必要？

### 5.1 不同场景结论不同

| 场景 | GPU 状态 | Full Graph 收益 |
|------|---------|----------------|
| **GPU 已打满**（GLM4.5-Air + TP4 + Input 1024） | Attention kernel 本身就在排队 | **收益小**，GPU 不差你那点 kernel launch |
| **CPU overhead 大**（QWen3-30B-A3B + TP1 + Input 32） | GPU 有明显 bubble（空闲间隙） | **收益大**，CUDA Graph 可以填掉这些空隙 |

### 5.2 怎么判断

不要凭感觉，看 profiling 数据：观察 P50 / P90 / P99 / P999 延迟分布，找到真正的瓶颈在哪。

---

## 6. 显存分析

### 6.1 CUDA Graph 的显存从哪来

**不是图对象本身占显存**（图对象很小），而是 CUDA Graph 需要静态分配的**激活值**和**临时缓冲区（scratch buffer）**。

### 6.2 Prefill CUDA Graph 真的增加显存压力吗？

```
不开启 Prefill CUDA Graph：
  Decode CUDA Graph Buffer + Prefill 临时 Buffer
  = K × (decode_max_bs + chunk_prefill_size)

开启 Prefill CUDA Graph：
  max(Decode Buffer, Prefill Buffer) + 临时 Buffer（可选）
  = K × max(chunk_prefill_size, decode_max_bs)
```

**结论：Prefill CUDA Graph 不会真正增加显存压力**，反而因为用 max 合并了 buffer，带来更可控的显存布局。

---

## 7. RL 场景的特殊性

- RL（强化学习）场景下，Prefill CUDA Graph 应该**默认开启**
- 原因：RL 对 TTFT（首 token 延迟）要求不高，但频繁有 **Retract/Abort + Re-prefill**，CUDA Graph 的无痛加速在重复 prefill 场景下收益明显

---

## 8. 调试方法论总结

调试 Piecewise CUDA Graph 的五个步骤：

```
1. torch.compile 兼容性排查  → 去掉 print、IO、动态控制流
2. Recompile 归因与修复       → TORCH_LOGS=recompiles 看是谁触发的
3. 二分定位                   → 解耦 Torch Compile / CUDA Graph
4. 精度对拍                   → 两台机器对比，找第一个出错的层
5. Profiling 判断             → 看是否需要 Full Graph
```

### 8.1 核心心智模型

Piecewise CUDA Graph 本质上是**用 torch.compile 做编译器级别的模块拆分，再对 shape 固定的部分做 CUDA Graph**。调试时的关键问题是：

- 是 dynamo trace 出错了？（compile 阶段）
- 还是 CUDA Graph capture/replay 出错了？（执行阶段）
- 还是精度本身就漂了？（数值问题）

把这三个问题分开来对待，定位效率会高很多。
