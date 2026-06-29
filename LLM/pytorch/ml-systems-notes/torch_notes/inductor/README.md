### TorchInductor

## 阶段 1 - 理解 IR

在这个阶段，TorchDynamo 产生的 FX 图被转换为 **Inductor IR**。

```python
@torch.compile
def f(x):
    b = torch.floor(x) + torch.ceil(x)
    c = b.sum(dim=-1)
    d = c + 1
    return d
```

变为：

```
x
│
├── floor
├── ceil
│
└── add
     │
     sum
      │
     add(1)
      │
    output
```

GraphLowering 遍历每个 FX 节点并将其转换为 Inductor IR 节点。

---

## 1. Placeholder（占位符）

placeholder `x` 变为：

```python
TensorBox(
    StorageBox(
        InputBuffer(
            name="arg0_1",
            layout=FixedLayout(
                device="cuda:0",
                dtype=torch.float32,
                size=[32,512,1024],
                stride=[524288,1024,1]
            )
        )
    )
)
```

```python
InputBuffer(...)
```

它存储关于输入张量的元数据，但不发生实际计算：
* device
* dtype
* shape
* strides
* layout

## 2. Pointwise 运算

FX 节点：

```python
floor(x)
```

变为：

```python
TensorBox(
 StorageBox(
  Pointwise(
    'cuda:0',
    torch.float32,

    def inner_fn(index):

        i0,i1,i2 = index

        tmp0 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp1 = ops.floor(tmp0)

        return tmp1

    ranges=[32,512,1024]
)))
```

Inductor 存储了一个**计算配方**（recipe）来描述如何计算一个输出元素，但从不实际执行它。

`ceil(x)` 产生几乎相同的 IR：

```python
TensorBox(
 StorageBox(
  Pointwise(

    def inner_fn(index):

        i0,i1,i2 = index

        tmp0 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp1 = ops.ceil(tmp0)

        return tmp1
)))
```

## 3. Pointwise 融合

下一个 FX 节点：

```python
add(floor(x), ceil(x))
```

变为：

```python
TensorBox(
 StorageBox(
  Pointwise(

    def inner_fn(index):

        i0,i1,i2 = index

        tmp0 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp1 = ops.floor(tmp0)

        tmp2 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp3 = ops.ceil(tmp2)

        tmp4 = tmp1 + tmp3

        return tmp4
)))
```

之前独立的 `Pointwise` 节点消失了，但它们的 `inner_fn` 函数被复制到了这个新节点中。

不再是这样：

```
floor → 临时张量 → ceil → 临时张量 → add
```

Inductor 现在存储的是：

```
load → floor → load → ceil → add
```

作为一个单一的配方——这就是**融合**！

## 4. Reduction（约简）

现在我们遇到：

```python
sum(dim=-1)
```

这不能用 pointwise IR 表示，因为每个输出元素依赖于多个输入元素。

所以 Inductor 创建：

```python
TensorBox(
 StorageBox(
  ComputedBuffer(

    name="buf0",

    data=Reduction(

      ranges=[32,512],
      reduction_ranges=[1024],

      def inner_fn(index,rindex):

          i0,i1=index

          r0=rindex

          tmp0 = ops.load(
              arg0_1,
              r0 + 1024*i1 + 524288*i0
          )

          tmp1 = ops.floor(tmp0)

          tmp2 = ops.load(
              arg0_1,
              r0 + 1024*i1 + 524288*i0
          )

          tmp3 = ops.ceil(tmp2)

          tmp4 = tmp1 + tmp3

          return tmp4
)))
```

概念上类似：

```python
for i0 in range(32):
    for i1 in range(512):

        total = 0

        for r0 in range(1024):

            total += floor(x[i0,i1,r0]) + ceil(x[i0,i1,r0])

        output[i0,i1] = total
```

注意 floor、ceil 和 add 计算再次出现了，但这**不是重复计算**。

因为这些张量从未实际存在过——只有它们的配方存在。Reduction 只是把这些配方复制到了自己的计算中。

---

## 5. ComputedBuffer

Reduction 结果被包装为：

```python
ComputedBuffer(
    name="buf0"
)
```

这意味着这个 reduction 的结果现在是一个逻辑张量，后续运算可以读取它。

它**不**一定意味着内存已被分配——它是在寄存器、共享内存还是全局内存中，由调度器后续决定。

## 6. 剩余的 Pointwise 运算

下一个运算：

```python
d = c + 1
```

变为：

```python
TensorBox(
 StorageBox(
  Pointwise(

    def inner_fn(index):

        i0,i1 = index

        tmp0 = ops.load(
            buf0,
            i1 + 512*i0
        )

        tmp1 = ops.constant(
            1,
            torch.float32
        )

        tmp2 = tmp0 + tmp1

        return tmp2
)))
```

与之前的 pointwise 节点不同，这个节点从 `buf0` 加载数据，而不是从输入张量加载。

## 7. 输出

最后，output 节点将前面的 pointwise 计算包装为另一个：

```python
ComputedBuffer(
    name="buf1"
)
```

这是编译函数返回的张量。

我们只看了少数几个运算，但 PyTorch 有数千个其他运算，它们各自有对应的 IR。

## 阶段 2 - Lowering 如何工作

现在了解了 Inductor IR 的样子，让我们看看 FX 图实际上是如何被降低为 IR 的。

### 2.1 Lowering 注册表

每个 ATen 运算都有一个通过 `@register_lowering` 注册的 lowering 函数。以 `ceil` 为例：

```python
@register_lowering(aten.ceil)
def ceil(x):
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("ceil")
    return make_pointwise(fn)(x)
```

这里的 `x` 是一个 Inductor IR 节点（可能是 `InputBuffer`、`ComputedBuffer`，甚至是尚未实现的 `Pointwise`）。

`make_pointwise` 是一个构建 pointwise IR 节点的辅助函数。以下是简化版：

```python
def make_pointwise(fn, ...):
    def inner(*inputs: List[TensorBox], alpha=None):
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()

        def inner_fn(index):
            return fn(*[load(index) for load in loaders])

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )
    return inner
```

注意三层嵌套：
1. `make_pointwise(fn)` - 配置数学函数
2. `inner(*inputs)` - 接收 Inductor IR 节点，构建 `inner_fn` 但**不调用它**
3. `inner_fn(index)` - 每个元素的配方，在 codegen 时被调用

`inner` 只将 `inner_fn` 包装成 `Pointwise` IR 节点。对 `inner_fn` 的实际调用发生在 codegen 阶段。

以 `ceil(x)` 为例追踪其过程：

```
step 1: ceil(x) 在 lowering 期间被调用
        x 是一个 InputBuffer IR 节点 (shape [32,512,1024])

step 2: ceil 调用 make_pointwise(ops_wrapper("ceil"))(x)
        └── make_pointwise(fn) 返回 inner (闭包)
        └── inner(x) 现在被调用

step 3: 在 inner(x) 内部：
        loaders = [x.make_loader()]        # InputBuffer.make_loader()
        ranges = [32, 512, 1024]

        def inner_fn(index):               # 定义但未调用
            return ops.ceil(loaders[0](index))

        return Pointwise.create(           # 返回 IR 节点，而非值
            inner_fn=inner_fn,             # inner_fn 作为配方存储
            ranges=[32,512,1024]
        )

step 4: 结果：一个 Pointwise IR 节点（配方）。没有发生计算。
```

`inner` 构建 IR 图。`inner_fn` 被 codegen 检查以生成实际的内核代码。二者都不实际对数据执行 `ceil`——那发生在编译后的 Triton 内核在 GPU 上执行的时候。

### 2.3 Loader 如何工作

每种 IR 类型都有自己的 `make_loader()`。这是理解融合的关键：

- **`InputBuffer.make_loader()`** - 返回一个调用 `ops.load(buf, offset)` 的函数，直接从输入张量读取数据。

- **`Pointwise.make_loader()`** - 返回它自己的 `inner_fn`！所以当你对 pointwise 节点调用 `load(index)` 时，你得到的是它的计算配方。

融合就是这样发生的。当 `add(floor, ceil)` 被 lowering 时：

1. 调用 `make_pointwise(add_fn)(floor_ir, ceil_ir)`
2. `floor_ir.make_loader()` 返回 floor 的 `inner_fn`（load → floor）
3. `ceil_ir.make_loader()` 返回 ceil 的 `inner_fn`（load → ceil）
4. 新的 `inner_fn` 同时调用两个 loader，然后相加：

```python
def inner_fn(index):
    floor_val = floor_loader(index)  # floor 的 inner_fn
    ceil_val = ceil_loader(index)    # ceil 的 inner_fn
    return floor_val + ceil_val
```

## 阶段 3 - 从 inner_fn 到 Triton 代码

我们已经看到 `inner_fn` 只是一个定义每个元素计算的 Python 函数。现在看看 Inductor 如何将该函数转换为实际的 Triton 内核代码。

### 3.1 美化打印 inner_fn

Inductor 如何打印我们在阶段 1 中看到的 `inner_fn` 代码？它使用 `KernelFormatterHandler` 以 fake ops handler 运行 `inner_fn`，将每个 op 捕获为字符串：

```python
class KernelFormatterHandler:
    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None):
        with V.set_ops_handler(formatter):
            result = ir_fn(*args)
            return formatter.getvalue(result)
```

当 `inner_fn` 在这个 handler 下运行时，每个 `ops.load`、`ops.floor` 等操作生成类似 `"tl.load(in_ptr0 + (x0), None)"` 的字符串，而不是实际计算。每个 op 将其结果赋给一个临时变量：

```python
def inner(*args, **kwargs):
    line = getattr(self.parent_handler, name)(*args, **kwargs)
    varname = f"tmp{next(self.var_counter)}"
    self.output.writeline(f"{varname} = {line}")
    return varname  # 下个 op 使用此 varname 作为输入
```

所以 `inner_fn` 通过执行来追踪——索引值不重要，只有操作的**结构**才重要。

### 3.2 实际 Codegen 路径

对于真正的 Triton codegen，Inductor 做了更深入的工作：

```
inner_fn
    
LoopBodyBlock 将 inner_fn 转换为 FX 图
    
在末尾追加 ops.store / ops.store_reduction
    
TritonKernel 遍历 FX 图并生成 Triton 代码
```

### 3.3 Pointwise 内核

看最简单的情况：

```python
@torch.compile
def fa(x):
    a = torch.floor(x) + torch.ceil(x)
    return a
```

这产生一个平坦的 pointwise Triton 内核：

```python
@pointwise(size_hints=[16777216], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.math.floor(tmp0)
    tmp2 = tl.math.ceil(tmp0)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
```

注意两点：

1. **维度合并** - 3D 形状 [32,512,1024] 变为 1D [16777216]。Inductor 合并连续维度以简化循环逻辑。

2. **XBLOCK 自动调优** - `@pointwise` 提供 block 大小（通常 1024 或 512）并基准测试它们。内核是 `xnumel` 个元素的单个平坦循环。

### 3.4 Reduction 内核

Reduction 根据 reduction 维度大小产生不同的 Triton 代码。

**情况 1：Reduction 维度中等大小（rnumel=1024）**

整个 reduction 维度适合一个 RBLOCK。Inductor 使用 `persistent_reduction`：

```python
@persistent_reduction(size_hints=[16384, 1024], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1          # 每个 program 一行
    RBLOCK: tl.constexpr = 1024       # 整个 reduction 维度
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    rindex = tl.arange(0, RBLOCK)[:]
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), rmask, other=0)
    tmp1 = tl.math.floor(tmp0)
    tmp2 = tl.math.ceil(tmp0)
    tmp3 = tmp1 + tmp2
    tmp7 = tl.sum(tmp3, 0)             # 沿 RBLOCK 约简
    tl.store(out_ptr0 + (x0), tmp7, None)
```

XBLOCK=1，因为每个线程块处理一行。加载整行，执行计算，然后 `tl.sum` 约简。

**情况 2：Reduction 维度非常小（rnumel=16）**

由于 reduction 很廉价，XBLOCK 可以更大：

```python
@persistent_reduction(size_hints=[16384, 16], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # 2D 网格
    rindex = tl.arange(0, RBLOCK)[None, :]
    ...
    tmp7 = tl.sum(tmp6, 1)[:, None]    # 沿 RBLOCK 每 XBLOCK 行约简
    tl.store(out_ptr0 + (x0), tmp7, None)
```

这里 XBLOCK 从 [1,8,32,128] 自动调优，因为 reduction 足够廉价，多行可以共享一个线程块。

**情况 3：Reduction 维度很大（rnumel=32768）**

整行不适合一个 RBLOCK。Inductor 使用带 for 循环的 `reduction`：

```python
@reduction(size_hints=[16384, 32768], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        tmp0 = tl.load(in_ptr0 + (r1 + 32768*x0), rmask, other=0)
        tmp1 = tl.math.floor(tmp0)
        tmp2 = tl.math.ceil(tmp0)
        tmp3 = tmp1 + tmp2
        tmp6 = _tmp5 + tmp3              # 累积
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]    # 最终约简
    tl.store(out_ptr0 + (x0), tmp5, None)
```

每次迭代加载 [xblock, rblock] 个元素，累积，循环结束后用 `tl.sum` 最终约简。RBLOCK 自动调优。

### 3.5 CSE - 公共子表达式消除

在 IR 中，`inner_fn` 加载同一个输入两次（一次给 floor，一次给 ceil）：

```python
tmp0 = ops.load(arg0_1, offset)   # for floor
tmp2 = ops.load(arg0_1, offset)   # for ceil
```

但在生成的 Triton 代码中，只加载一次：

```python
tmp0 = tl.load(in_ptr0 + (x0), None)
tmp1 = tl.math.floor(tmp0)
tmp2 = tl.math.ceil(tmp0)          # 复用 tmp0，没有第二次加载
```

这是因为 Inductor 的 `CSE` 类去重了相同表达式：

```python
class CSE:
    def generate(self, buffer, expr):
        var = self.cache.get(expr)
        if not var:                  # 首次：创建新变量
            var = self.newvar()
            self.cache[expr] = var
            buffer.writeline(f"{var} = {expr}")
        return var                   # 缓存命中：复用之前的变量
```

两个 load 有相同的地址表达式，所以第二个返回同一个 `tmp0` 变量。

### 3.6 FX 图中间表示

在生成 Triton 代码之前，Inductor 将 `inner_fn` 转换为一个小型 FX 图。`fa`（pointwise）的图如下：

```
[get_index, load, floor, get_index, load, ceil, add, get_index, store, output]
```

`fb`（reduction）则有额外节点：

```
[get_index, load, floor, get_index, load, ceil, add, reduction, get_index, store_reduction, output]
```

`store` / `store_reduction` 节点由 `ComputedBuffer.get_store_function()` 追加：

```python
class Pointwise(Loops):
    def store_output(self, output_name, indexer, vars):
        loader = self.make_loader()
        return ops.store(output_name, indexer(vars), loader(vars))

class Reduction(Loops):
    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        value = ops.reduction(self.dtype, self.src_dtype,
                              self.reduction_type, self.inner_fn(vars, reduction_vars))
        return ops.store_reduction(output_name, indexer(vars), value)
```

FX 图随后被 `TritonKernel.codegen_body()` 消费，它将代码拆分为：

- **indexing_code** - 索引计算（r1 = rindex, x0 = xindex）
- **loads** - tl.load 语句
- **compute** - 数学运算（floor、ceil、add）
- **stores** - tl.store 语句
- **suffix** - 最终约简（tl.sum）

对于有大维度的 reduction 内核，`codegen_body` 将 loads/compute/stores 包裹在 `for roffset` 循环中。对于 persistent reduction，所有内容都是平坦的。
