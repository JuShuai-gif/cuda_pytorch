# PyTorch 混合精度与量化源码分析

> 基于 PyTorch 源码 `~/code/pytorch-main` 的实现剖析，对应 `test/test1.py` 里的现象。
> 三个机制：`torch.autocast`（计算混合精度）、`model.half()`（存储混合精度）、Conv-BN 融合（量化前置步骤）。

---

## 一、`torch.autocast` 如何自动决定精度

**核心认知：Python 端只是个开关，真正的"逐算子选精度"发生在 C++ dispatcher 层。**

### 1. Python 端只设置线程局部状态

`torch/amp/autocast_mode.py:308` 的 `__enter__` 不做任何计算，只翻三个 TLS 标志：

```python
torch.set_autocast_enabled(self.device, self._enabled)   # 317
torch.set_autocast_dtype(self.device, self.fast_dtype)   # 318  cuda 默认 fp16
torch.autocast_increment_nesting()                       # 319
```

`__exit__`（`:343`）退栈，嵌套归零时 `clear_autocast_cache()`。

### 2. "开关"本质是打开一个 dispatch key

`aten/src/ATen/autocast_mode.cpp:16`：

```cpp
void set_autocast_enabled(...) {
  // 启用 = 不排除 Autocast key
  ... tls_set_dispatch_key_excluded(dispatch_key, !enabled);
}
```

启用 autocast = 让 `Autocast` 这个 dispatch key 参与分发，于是每个算子调用会**先经过 autocast 包装层**。

### 3. 算子分成 5 类策略（"自动决定精度"那张表的源头）

`aten/src/ATen/autocast_mode.h:416` 的 `enum class CastPolicy`：
`lower_precision_fp / fp32 / fp32_set_opt_dtype / fp32_append_dtype / promote`。

具体哪个算子归哪类，写死在 `autocast_mode.h` 的宏清单里：

| 策略 | 宏清单（文件位置） | 典型算子 |
| --- | --- | --- |
| 降精度 fp16 | `AT_FORALL_LOWER_PRECISION_FP` (`autocast_mode.h:819`) | conv1d/2d/3d、matmul、mm、bmm、addmm、linear、einsum、attention |
| 强制 fp32 | `AT_FORALL_FP32` (`:854`) | exp、log、pow、softplus、layer_norm、group_norm、各种 loss、cdist |
| fp32 + 可选输出 dtype | `AT_FORALL_FP32_SET_OPT_DTYPE` (`:915`) | softmax、log_softmax、sum、prod、cumsum |
| 取最宽 dtype | `AT_FORALL_PROMOTE` (`:945`) | addcdiv、atan2、cross、dot、index_put |

注册发生在 `autocast_mode.cpp:181` `TORCH_LIBRARY_IMPL(aten, Autocast, m)`，用宏把每个算子绑到对应 policy 的包装函数。

> 注意：BatchNorm 在 CUDA autocast 里没有显式列入 fp32 清单，但 MPS 后端把 `batch_norm` 明确归为 fp32（`autocast_mode.cpp:280`）；CUDA 上 BN 走 cuDNN 时内部对均值/方差按 fp32 累加。

### 4. 每个 policy 怎么转——模板特化

`autocast_mode.h` 里 `WrapFunction_` 对每种 policy 各有一份实现：

- `lower_precision_fp`（`:470`）：把所有输入 `cached_cast` 成 fp16 再跑

  ```cpp
  return (*F)(cached_cast(get_lower_precision_fp_from_device_type(device_type), args, ...));
  ```

- `fp32`（`:494`）：`cached_cast(at::kFloat, args...)` 全转回 fp32
- `fp32_set_opt_dtype`（`:515`）：softmax 类，仅在用户没指定输出 dtype 时设为 fp32
- `fp32_append_dtype`（`:543`）：norm 类，追加 fp32 dtype 并重分发到带 dtype 的重载
- `promote`（`:566`）：`promote_type(...)` 选出最宽 dtype，全部对齐到它

**防递归关键**：每个 `call` 开头都有 `ExcludeDispatchKeyGuard no_autocast(...)`（如 `:478`）——转完精度后把 Autocast key 排除，再重新分发到真正的 kernel，避免无限递归。

### 5. 为什么权重还是 fp32（对应 test1.py 现象）

实际转换在 `cached_cast`（`autocast_mode.cpp:122`）：它只对**传入算子的参数**临时 `arg.to(to_type)`（`:147/152`），并把 fp32 叶子权重的 fp16 结果**缓存**复用（Apex 套路，`:129`）。**它从不修改 `Parameter` 本身的存储**。

> 所以单用 autocast，`params_by_dtype` 永远全是 fp32。

默认 dtype 写死在 `autocast_mode.cpp:59` 的数组：`CPU → kBFloat16`、`CUDA → kHalf`。

---

## 二、`model.half()` 的物理转换

### 1. 入口：一行 `_apply`

`torch/nn/modules/module.py:1202`：

```python
def half(self):
    return self._apply(lambda t: t.half() if t.is_floating_point() else t)
```

`is_floating_point()` 守卫 —— **只转浮点**，像 BN 的 `num_batches_tracked`（int64）不会被动。
`float()`（`:1180`）、`double()`（`:1191`）、`bfloat16()`（`:1213`）同理。

### 2. `_apply` 真正做物理替换

`module.py:930`：

1. 递归子模块（`:931-933`）
2. 对每个 parameter（`:957`），在 `no_grad` 下 **`param_applied = fn(param)`（`:964`）** —— 这一步 `t.half()` 会**新建一块 fp16 张量**（底层走 aten `_to_copy`，分配新存储并按 fp16 舍入拷贝）
3. 把新张量换进模块。默认路径 `param.data = param_applied`（`:995`），即**原地替换底层数据**，旧 fp32 存储无引用后释放 → 显存减半。（新版可选 `swap_tensors` `:986` 或重建 `Parameter` `:1003`）
4. 梯度一起转（`:1006-1030`）
5. **buffer 也转**：`:1032` `self._buffers[key] = fn(buf)` → BN 的 `running_mean / running_var` 跟着变 fp16
6. `:1036 return self` → in-place、返回自身

**结论四点**：物理新建存储、in-place 返回 self、连 buffer 一起转、`is_floating_point` 跳过整型。

### 3. 与 autocast 的本质区别

| 操作 | 改变存储 dtype? | 物理省显存 | 精度损失 |
| --- | --- | --- | --- |
| `.half()` | 是（永久） | 是（减半） | 是（不可逆，fp16 仅 10 位尾数、范围 ±65504） |
| `torch.autocast` | 否（权重仍 fp32） | 否（只省计算时中间量） | 计算时临时，权重无损 |

---

## 三、Conv-BN 融合（量化前置步骤，对应 test1.py 的 WARNING）

### 1. 入口：要求 eval、深拷贝、折权重

`torch/nn/utils/fusion.py:20` `fuse_conv_bn_eval`：

```python
if conv.training or bn.training: raise ...   # 38 必须 eval（running stats 已固定）
fused_conv = copy.deepcopy(conv)             # 40
fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(...)  # 44
```

### 2. 数学核心：把 BN 折进 Conv 的 weight/bias

`fusion.py:58` `fuse_conv_bn_weights`，本质是合并两条公式：

- Conv：`x = W·input + b`
- BN 推理：`y = (x - rm) / sqrt(rv + eps) · gamma + beta`

代码（`:91-103`）：

```python
bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)                    # 1/sqrt(rv+eps)
fused_conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)  # 98 逐输出通道缩放 W
fused_conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b  # 101 折叠 bias
```

即 `W' = W·gamma/sqrt(rv+eps)`、`b' = (b−rm)·gamma/sqrt(rv+eps)+beta`。
融合后**单个 Conv 的输出等价于原 Conv→BN**，独立 BN 层消失。

还处理了缺省项：无 conv bias → 零、无 gamma → 1、无 beta → 0（`:85-90`），并保持原 dtype（`:83-84, 98-99`）。
`fuse_linear_bn_weights`（`:162`）是 Linear 版同款逻辑。

### 3. 对量化的意义

test1.py 的 `WARNING: 53 BN layers need fusion` 指的就是这 53 个 BN 还没被折掉。
一旦折掉，INT8 量化时就不必给每个 BN 插额外的 quantize/dequantize，精度和速度都更好。

---

## 四、串起来：三种"混合精度"的区别

- **autocast（计算混合精度）**：dispatcher 拦截 → 按 policy 表 `cached_cast` 输入 → 权重存储不变（dtype 统计仍 fp32）。
- **`.half()`（存储混合精度）**：`_apply` 物理新建 fp16 存储替换参数/buffer（dtype 统计才会变）。
- **量化的 BN**：不是"运行时保 fp32"，而是用上面的数学**直接折进 Conv 消灭**——这是 autocast 的 fp16/fp32 表不能 1:1 套到量化的关键差异。

### 量化能否照搬 AMP 那张表？

方向对，但不能 1:1：

1. **重合点**：Conv/Linear 同样是 INT8 量化的主力目标（对应 AMP 的 fp16 清单）。
2. **BN 处理相反**：AMP 运行时保 fp32；量化直接折叠进 Conv 消灭。
3. **量化额外规则**：通常把首个 Conv 和最后的分类 FC 留高精度（对量化误差最敏感）。
4. **根因不同**：fp16 怕溢出/累加误差（有指数位管动态范围）；INT8 怕数值分布/范围（靠 scale+zero_point 映射）。

---

## 关键源码位置速查

| 主题 | 文件 | 行 |
| --- | --- | --- |
| autocast Python 上下文 | `torch/amp/autocast_mode.py` | `__enter__` 308 / `__exit__` 343 |
| autocast 开关=dispatch key | `aten/src/ATen/autocast_mode.cpp` | 16 |
| autocast 默认 dtype 数组 | `aten/src/ATen/autocast_mode.cpp` | 59 |
| autocast 算子注册 | `aten/src/ATen/autocast_mode.cpp` | 181 |
| `cached_cast`（实际转换+缓存） | `aten/src/ATen/autocast_mode.cpp` | 122 |
| CastPolicy 枚举 | `aten/src/ATen/autocast_mode.h` | 416 |
| WrapFunction_ 模板特化 | `aten/src/ATen/autocast_mode.h` | 461-582 |
| 算子清单宏 | `aten/src/ATen/autocast_mode.h` | 819 / 854 / 915 / 945 |
| `Module.half` | `torch/nn/modules/module.py` | 1202 |
| `Module._apply` | `torch/nn/modules/module.py` | 930 |
| `fuse_conv_bn_eval` | `torch/nn/utils/fusion.py` | 20 |
| `fuse_conv_bn_weights` | `torch/nn/utils/fusion.py` | 58 |
| `fuse_linear_bn_weights` | `torch/nn/utils/fusion.py` | 162 |

---

## 五、可借鉴的内存 / 计算 / 工程优化技巧

从上述源码里提炼出的、可移植到自己工程的优化技巧。

### 内存优化

1. **权重 cast 缓存（Apex 套路）** — `autocast_mode.cpp:122` `cached_cast`
   一次 forward 里同一 fp32 权重被多个算子复用，只转一次 fp16 并按 `TensorImpl*` 缓存。
   > 可借鉴：对「确定性 + 重复使用 + 转换昂贵」的中间结果，按稳定身份 memoize。

2. **缓存键用裸指针但持 weakref 防地址复用** — `autocast_mode.cpp:28-44`
   key 是裸 `TensorImpl*`，value 持 `weak_intrusive_ptr` 钉住该 impl 不释放；否则原张量回收后新张量复用同一地址 → 缓存假命中（罕见、间歇、极难查）。
   > 可借鉴：**拿地址当缓存键时务必持弱引用防地址回收复用**。教科书级踩坑预防。

3. **backward 也存低精度** — `autocast_mode.h:613-628` 注释
   在 autograd 记录前打补丁，让 saved-for-backward 张量存 fp16 而非 fp32，显著降显存。
   > 可借鉴：中间量「存在哪层、用什么精度存」比单纯算得快更影响显存峰值。

4. **`to_empty` 只搬结构不拷数据** — `module.py:1224`
   用 `torch.empty_like(t, device=...)` 移动模型而不拷贝数据，用于 meta device → 真实设备的大模型初始化。
   > 可借鉴：知道马上要覆写的内存，先 `empty` 分配、跳过无意义拷贝。

5. **融合保持原 dtype，不偷偷升精度** — `fusion.py:83-84,98-103`
   折叠算完 `.to(dtype=conv_weight_dtype)` 转回原精度，避免融合把 fp16 升成 fp32 撑大显存。

### 计算优化

6. **Conv-BN 折叠 = 推理期常量折叠** — `fusion.py:58`
   BN 推理是固定线性变换，折进前一层 Conv 的 weight/bias，BN 算子在推理图里彻底消失，零运行时开销。
   > 可借鉴：把推理期恒定的变换折进相邻线性算子。

7. **dispatch key 实现「关闭即零成本」** — `autocast_mode.cpp:178`
   关闭时把 key 从 TLS 排除 + `makeFallthrough()` 直接穿透，热路径无散落 `if 开关`。
   > 可借鉴：用分层 dispatch / fallthrough 做特性开关，而非到处插 `if enabled`。

8. **`rsqrt` 单算子代替 `1/sqrt`** — `fusion.py:91`
   `torch.rsqrt(rv+eps)` 一个融合的倒数平方根，省一次除法。

9. **缓存只对真正稳定的权重生效** — `autocast_mode.cpp:129-133`
   命中条件卡死 `requires_grad && is_leaf && !is_view && !inference_mode`，避免缓存会变的东西。
   > 可借鉴：缓存命中条件写到「只对不变量生效」，宁严勿松。

### 工程设计

10. **声明式算子分类 + 模板生成包装** — `autocast_mode.h:819/461`
    宏清单做单一事实来源，`WrapFunction_` 模板特化按 policy 自动生成代码，CPU/CUDA/XPU/MTIA 多后端复用同一份清单（`:818` 注释明说）。
    > 可借鉴：规则数据化（清单）+ 代码生成（模板），改一处全后端生效。

11. **in-place 改 `.data` 而非重建对象** — `module.py:995`
    `.half()` 默认走 `param.data = param_applied`，保持 Parameter 对象身份不变——因为 optimizer 持有参数引用，重建对象会让优化器状态失联（`compute_should_use_set_data` `:937` 整段为此）。
    > 可借鉴：批量改张量存储时保留对象 identity，别让外部引用悬空。

12. **批量改参数包在 `no_grad` 里** — `module.py:963`
    对叶子参数转换不该建 autograd 图。
    > 可借鉴：任何 bulk 参数变更前先 `torch.no_grad()`。

13. **RAII 防递归** — `autocast_mode.h:478` `ExcludeDispatchKeyGuard`
    转完精度后排除自身 key 再重分发，RAII 自动恢复，避免无限递归。

14. **缺省项用恒等默认值统一代码路径** — `fusion.py:85-90`
    无 conv bias→0、无 gamma→1、无 beta→0，一套公式覆盖所有配置，无需特判。
    > 可借鉴：用「恒等元」填补可选参数，消除特判分支。

### 最值得直接搬的三个

- **#2 弱引用防地址复用**（裸指针缓存的安全模式）
- **#11 改 `.data` 保对象身份**（批量改参数不破坏外部引用）
- **#10 清单 + 模板的声明式扩展**（多后端单一事实来源）

---

## 六、Inductor kernel 融合策略（为什么 transformer 能加速那么多）

`torch.compile()` 默认后端 Inductor 的核心价值：**把成百上千个孤立小算子融合进寥寥几个 Triton kernel**。节省的不只是 kernel launch 开销（GPU 每次 launch 都有微秒级延迟），更关键的是**减少 global memory 的读写次数**——pointwise 链（add+relu+mul+add）融合后，中间结果直接在寄存器/共享内存传递，不用写回显存再读出来。

### 1. 融合流程概览

```
Scheduler.fuse_nodes()           [scheduler.py:5102]     ← 入口，最多迭代 10 轮
  └─ fuse_nodes_once()           [scheduler.py:6216]
       ├─ get_possible_fusions() [scheduler.py:6639]     ← 找候选：共享 buffer 的节点对
       │    └─ 每个对 → can_fuse(node1, node2)
       │         ├─ _can_fuse() [scheduler.py:7610]      ← 通用检查
       │         │    ├─ 流/设备/拓扑 ✓
       │         │    ├─ 模板 prologue/epilogue 规则
       │         │    ├─ _score_fusion_memory()           ← 内存收益估算
       │         │    └─ 判断方向：
       │         │         ├─ node2 依赖 node1 → 垂直融合 → can_fuse_vertical()
       │         │         └─ 无依赖 → 水平融合 → can_fuse_horizontal()
       │         └─ SIMDScheduling.can_fuse() [simd.py:2029]  ← 后端具体判断
       │              ├─ reduction+reduction: 相同 shape 或 mix/nested
       │              ├─ pointwise+pointwise: 相同 shape + 兼容 tiling
       │              └─ pointwise+reduction: 广播 shape + 兼容 tiling
       └─ Backend.fuse() [scheduler.py:10071]             ← 创建 FusedSchedulerNode

代码生成：
SIMDScheduling._codegen_nodes()  [simd.py:2987]
  ├─ generate_node_schedule()    [simd.py:2214]           ← 在 reduction 循环内外排序节点
  ├─ codegen_node_schedule()     [simd.py:3103]
  │    ├─ get_tiling_and_scores  [simd.py:4437]           ← 按 stride 选 tiling
  │    ├─ TritonKernel(tiling)                            ← 所有融合节点共享一个 kernel 对象
  │    └─ codegen_node_schedule_with_kernel() [simd.py:3219]
  │         ├─ 第一遍：收集索引、decide_inplace_update
  │         └─ 第二遍：node.codegen(index_vars) → 调用 self._body
  └─ kernel.codegen_kernel()                              ← 生成最终 Triton 源码
```

### 2. 什么能融合——`GroupKey` 是核心判据

每个 `SchedulerNode` 创建时算出一个 `group` 元组（`scheduler.py:2248`）：

```python
device = self.node.get_device_or_error()
group_fn = self.scheduler.get_backend(device).group_fn
self.group = (device, group_fn(self._sizes))  # 结果是 (device, (numel, rnumel))
```

`group_fn` 在 `SIMDScheduling` 里（`simd.py:2026`）就是把索引维度和 reduction 维度各自乘起来：

```python
def group_fn(self, sizes):
    return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)
```

**融合的门槛**：`can_fuse()` 先看 `group` 是否匹配，再看 **tiling 兼容性**——两个节点能不能塞进同一套循环嵌套里。

具体到三种组合（`simd.py:2029` 往后）：

| 组合 | 条件 | 例子 |
| --- | --- | --- |
| pointwise + pointwise | `numel1 == numel2 && rnumel1 == rnumel2` + 相同 tiling | `add → relu → mul` |
| pointwise + reduction | `numel1 == numel2 * rnumel2`（pointwise 广播到 reduction） | `add → sum` |
| reduction + reduction | `numel/rnumel` 一致 或 `MixOrderReduction` / `NestedReduction` | `max → sum`（layer_norm 场景） |

**`ExternKernel`（回退的 aten 算子）不与任何东西融合**，除了 `UserDefinedTritonKernel` 可以吞进 epilogue pointwise（`scheduler.py:7734`）。

### 3. 怎么找融合候选

`get_possible_fusions()`（`scheduler.py:6639`）的策略：

1. **按 buffer 名称分组**（`:6670`）：任何共享同一 buffer（生产者-消费者关系或共享输入）的节点对都去问 `can_fuse()`。
2. **激进融合**（`:6679`，对应 `aggressive_fusion` 配置）：同一 `group` 的节点对也去问 `can_fuse()`——哪怕没有直接 buffer 连接。

候选找到后按 `score_fusion_key`（`:8684`）排序，后者委托给 `V.choices.score_fusion()`，主要考虑 `score_fusion_memory()` 估算的**访存节省量**以及融合后对其他候选的阻塞影响。

### 4. 垂直 vs 水平融合

`_can_fuse()`（`:7893-7930`）按依赖关系路由：

```python
if node1.get_operation_names() & node2.ancestors:
    # node2 依赖 node1 → 垂直融合
    return self.can_fuse_vertical(node1, node2, ...)
else:
    # 无依赖 → 水平融合
    return self.can_fuse_horizontal(node1, node2, ...)
```

- **垂直融合**（生产-消费）：`add → relu`，融合后 relu 直接消费 add 的输出，中间值不进显存。`can_fuse_vertical()`（`:7932`）验证消费者的所有读取是否都能由生产者的写入在索引上等价满足，用 `fusable_read_and_write()` 做索引匹配。
- **水平融合**（相同消费者）：`(add, mul) → cat`，两个 pointwise 共享同一个后续 op，合并它们可以共享循环结构。

在 Triton 后端（`simd.py:2211-2212`），`can_fuse_horizontal` 和 `can_fuse_vertical` 是同一个函数——垂直/水平的区别只在调度器层面的路由不同。

### 5. 融合后的两遍代码生成

`codegen_node_schedule_with_kernel()`（`simd.py:3219`）为什么分两遍：

- **第一遍**：遍历每个节点，调用 `node.decide_inplace_update()` 决定哪些 buffer 可以原地改写，以及 `kernel.split_and_set_ranges(node.get_ranges())` 把节点的迭代空间映射到 kernel 的平铺循环上。
- **第二遍**：`node.codegen(index_vars)` → 调用 `self._body(*index_vars)`，每个节点把自己的计算（`tl.load` → CSE 表达式 → `tl.store`）追加到**同一个 `TritonKernel` 对象**的 buffer（`self.body`）。

关键机制：所有融合节点**共享一个 `TritonKernel` 实例**（`triton.py:3131`），包括：
- 同一组 `range_trees`（`x、y、z` 循环变量）
- 同一个 `cse`（公共子表达式缓存）——节点间自动共享重复计算
- 同一个 `prologue/body/suffix` 缓冲区——最终拼成单个 Triton 函数

这就是为什么 `add + relu + add` 三个独立算子经过编译后只剩一个 `tl.load → tl.add → tl.add → tl.store`，中间结果不需要写回显存。

### 6. Tiling 怎么选——按 stride 决定内存合并

`get_tiling_and_scores()`（`simd.py:4437`）选择 tiling，核心原则是**最大化内存合并**：

- 对于 **pointwise kernel**：按 stride-1 维度优先平铺（访存连续的维度尽量做大 tile）。
- 对于 **reduction kernel**：reduction 维度的 tiling 由 `tile_reductions` 配置控制。
- 对于 **原生 matmul（`"dot"` reduction type）**：强制 `{y: M, x: N, r0_: K}` 平铺——这就是 triton 矩阵乘法的经典 tiling。

融合多个节点时，`select_tiling()` 必须为所有节点选出一致的 tiling，如果两个 node 的 stride 模式不同导致无法达成一致，融合就被否决。

### 7. 可借鉴的设计要点

与之前各节类似的视角提炼：

**15. GroupKey 作为融合的「第一道门」** — `scheduler.py:2248`
用 `(device, (numel, rnumel))` 这个简单元组做预筛选，避免低质量的融合候选。粗筛快、细筛准。
> 可借鉴：融合/分组时先算一个"形状签名"快速过滤，再跑精确合法性检查。

**16. 两遍代码生成：先分析、再发射** — `simd.py:3219`
第一遍收集所有节点的索引和原地更新决策（只读不写），第二遍才写 code buffer。避免了生成一半发现需要修改决策的尴尬。
> 可借鉴：需要多源输出的代码生成，分 analyze → emit 两阶段，不要在 emit 里做决策。

**17. 平铺与融合的耦合打平** — `simd.py:1010` `split_and_set_ranges`
融合节点可能有不同形状（如 pointwise 广播到 reduction 时），`split_and_set_ranges` 把扁平范围映射到 kernel 的 tiling 循环，让不同形状的节点共享同一套循环嵌套。
> 可借鉴：不同粒度 → 统一成同一套坐标系的映射逻辑，降低融合的形状兼容门槛。

**18. `FusedSchedulerNode` 作为组合模式** — `scheduler.py:2664`
不修改各 node 的原有数据结构，用一个 `FusedSchedulerNode.snodes` 列表把它们包起来，融合只是**组合/拆装/重新排序** `snodes` 列表，不影响单个 node 的 codegen。融合失败了恢复也不费力。
> 可借鉴：用组合而不是侵入式修改来打包可融合单元，拆装灵活。

**19. 内存收益评分引导融合** — `scheduler.py:8364` `score_fusion_memory`
不为所有可融合对都融合，而是算共享 buffer 的简化程度（精确索引匹配→省掉整个 buffer 的 global memory；部分索引重叠→按重叠比率折算）。分数低于阈值就跳过。
> 可借鉴：优化决策要量化（节省的字节 / 预计加速比），不要只靠启发式瞎猜。

**20. "激进融合"开关** — `scheduler.py:6679`
默认只融合有直接 buffer 连接的节点对，打开 `aggressive_fusion` 后允许同 `group` 任何节点对尝试融合，覆盖了"通过其余 buffer 间接连接"的场景。
> 可借鉴：给优化器配置一个"激进"档位，保守 vs 激进由用户按场景选，不要硬编码一个策略。

### 与前面三节的交叉关系

| 机制 | 对 Inductor 融合的影响 |
| --- | --- |
| `autocast` | 不改权重 dtype，但复算路径上的算子精度选择会影响 IR 节点类型，不影响融合逻辑本身 |
| `.half()` | 权重变成 fp16 → Inductor 生成 fp16 的 Triton kernel（可以用 Tensor Core），显存减半的同时访存带宽也减半 → 融合后收益更大 |
| Conv-BN fusion | 图里 BN 节点被提前消掉 → Inductor 不用处理 BN 的 IR，融合逻辑更干净，也避免了 BN 的 ExternKernel 导致的图断裂 |
| **compile + 三者** | 完全可以同时使用：先 `.half()` + BN 保 fp32，再 fuse BN，最后 `torch.compile()` + forward 时 `with autocast()` |
