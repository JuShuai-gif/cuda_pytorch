# Custom Triton Kernel & PyTorch Inductor 代码生成源码分析

> Inductor Triton codegen: `torch/_inductor/codegen/triton.py:3131` — `TritonKernel` 类
> `codegen_body`: `triton.py:5903` — 核心代码生成
> `codegen_kernel`: `triton.py:6454` — 最终 Triton 代码输出
> Scheduler fusion: `torch/_inductor/scheduler.py:5102` — 算子融合入口

## 0. 一句话总览

Inductor 的 Triton 代码生成 = **从融合后的 Inductor IR 节点 → 单个 Triton kernel 源码**。所有融合节点共享一个 `TritonKernel` 实例（包括同一组 `range_trees`、CSE 缓存、`prologue/body/suffix` 缓冲区），最终拼接成一个 `@triton.jit` 装饰的 GPU kernel。

---

## 一、`TritonKernel` 核心架构 (`triton.py:3131`)

```python
# triton.py:3131
class TritonKernel(SIMDKernel[TritonCSEVariable]):
    """
    代表一个即将生成的 Triton kernel。

    关键成员:
      range_trees: list[IterationTree]  — 循环变量树 (x/y/z/r 维度)
      cse: CSE[TritonCSEVariable]       — 公共子表达式缓存
      body: IndentedBuffer              — kernel body 缓冲区
      indexing_code: IndentedBuffer     — 索引计算代码
      loads:   IndentedBuffer           — tl.load 语句
      compute: IndentedBuffer           — 数学运算
      stores:  IndentedBuffer           — tl.store 语句
      suffix:  IndentedBuffer           — 最终 reduction/linalg 后缀
    """
```

### 1.1 融合节点共享 kernel 实例

当 Scheduler 决定将多个节点融合到一个 kernel 时，所有节点**共用同一个 `TritonKernel` 实例**:

```python
# 伪代码 (scheduler.py → simd.py → triton.py):
kernel = TritonKernel(tiling, ...)
for node in fused_nodes:
    node.codegen(kernel.index_vars)  # 每个节点追加自己的 body 代码
    #   → Pointwise.codegen: 调用 self._body(*index_vars)
    #     → inner_fn 被重放到 kernel.body buffer 中
```

**这就解释了** 为什么 `a * b + c` 三个 op 最终生成一个 Triton kernel —— 它们都追加到同一个 `kernel.body` buffer 中，中间结果在寄存器中传递。

---

## 二、`codegen_body` 源码分析 (`triton.py:5903`)

```python
# triton.py:5903
def codegen_body(self):
    """
    将 index_code / loads / compute / stores / suffix
    拼接成 self.body。

    Pointwise kernel:  调用一次 → 生成平坦 body
    Reduction kernel:  在 reduction loop 内调用多次
    """
    if not (self.indexing_code or self.loads or ...):
        return  # 空 kernel

    loop_trees = [tree for tree in self.range_trees if tree.is_loop]  # :5923

    # Pointwise 路径 — 无循环 (所有维度都被 flatten)
    if not loop_trees:
        self.body.splice(self.indexing_code)   # 索引: x0 = xindex
        self.body.splice(self.loads)           # load: tmp0 = tl.load(in_ptr0 + x0)
        self.body.splice(self.compute)         # compute: tmp1 = tmp0 + 1
        self.body.splice(self.stores)          # store: tl.store(out_ptr0 + x0, tmp1)
        return

    # Reduction 路径 — 生成 for loop
    # 代码结构:
    #   xmask = ...                             ← index
    #   for roffset in range(0, rnumel, RBLOCK): ← loop
    #       tmp0 = tl.load(in_ptr + ..., rmask)  ← load
    #       tmp1 = floor(tmp0)                   ← compute
    #       _tmp_acc = _tmp_acc + tmp1           ← accumulate
    #   tmp_final = tl.sum(_tmp_acc, 0)           ← suffix
    #   tl.store(out_ptr + ..., tmp_final)       ← store
    ...
```

### 2.1 Indexing Code — 索引生成

```python
# 通过 RangeTree 生成索引代码:
#   对每个 range_tree:
#     x0 = xoffset + 0    (或 xoffset + tl.arange(0, XBLOCK))
#     y0 = yoffset + 0    (y 维度被 flatten)
```

### 2.2 CSE 在 codegen 中的作用

`kernel.cse` 是一个 `CSE[TritonCSEVariable]` 实例，在生成代码时去重:

```python
# CSE.generate(buffer, expression, dtype) 
#   如果 expression 已存在于 cache → 返回已有变量名
#   否则 → 创建新变量, 写入 buffer, 返回新变量名
#   :5790 (CSE.generate)
```

这保证了同一 kernel 内的重复子表达式（如两次 `load(in_ptr + offset)`）只生成一次。

---

## 三、从 IR 节点到 Triton 代码的完整路径

```
Inductor IR 节点 (Pointwise / Reduction / ComputedBuffer)
   │
   │ 1. Scheduler.fuse_nodes() 决定融合哪些节点
   ▼
   │ 2. SIMDScheduling._codegen_nodes() → 创建 TritonKernel
   ▼
   │ 3. 每个 FusedSchedulerNode → node.codegen(index_vars)
   │    └─ Pointwise.codegen → inner_fn(index_vars)
   │       └─ 每个 ops.load / ops.floor / ops.add 等被重放
   ▼
   │ 4. 所有 op 追加到 kernel.compute / kernel.loads / kernel.stores
   ▼
   │ 5. kernel.codegen_body() 拼接 buffer
   ▼
   │ 6. kernel.codegen_kernel() 输出最终 Triton 源码
   ▼
   @triton.jit
   def triton_(in_ptr0, out_ptr0, ...):
       xoffset = tl.program_id(0) * XBLOCK
       xindex = xoffset + tl.arange(0, XBLOCK)[:]
       x0 = xindex
       tmp0 = tl.load(in_ptr0 + x0)
       tmp1 = tl.math.floor(tmp0)
       tmp2 = tl.math.ceil(tmp0)
       tmp3 = tmp1 + tmp2
       tl.store(out_ptr0 + x0, tmp3)
```

---

## 四、PyTorch 端集成模式: `autograd.Function` 包裹 Triton kernel

`torch/autograd/function.py` 提供了 `Function.apply()` 接口:

```python
class MyTritonOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        output = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        my_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output * y  # 自定义梯度
        grad_y = grad_output * x
        return grad_x, grad_y

output = MyTritonOp.apply(x, y)  # 完全 autograd 兼容
```

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `TritonKernel` 类 | `triton.py` | 3131 |
| `codegen_body` (拼接代码) | `triton.py` | 5903 |
| `codegen_kernel` (最终输出) | `triton.py` | 6454 |
| `CSE.generate` (去重) | `triton.py` | ~5790 |
| `Scheduler.fuse_nodes` (融合入口) | `scheduler.py` | 5102 |
| `SIMDScheduling._codegen_nodes` | `simd.py` | 2987 |
| `Pointwise.codegen` (追加 inner_fn) | `triton.py` | — |
| `torch.autograd.Function` | `autograd/function.py` | — |
| `Function.apply` | `autograd/function.py` | — |

---

## 六、可借鉴的工程技巧

1. **多节点共享 kernel**: 融合节点都写同一个 `body` buffer → 中间结果自动在寄存器中传递。

2. **CSE 自动去重**: 同一 kernel 内重复的 load/compute 只生成一次。

3. **分段代码生成**: `indexing_code / loads / compute / stores / suffix` 五段式 → 每段独立生成，最后 `splice` 拼接。

4. **RangeTree 管理循环**: 多维循环通过 `range_trees` 树管理 → tiling/parsing/调度统一抽象。

5. **STE (Straight-Through Estimator)**: FakeQuantize backward 用 identity gradient → 梯度可穿过多层量化操作。
