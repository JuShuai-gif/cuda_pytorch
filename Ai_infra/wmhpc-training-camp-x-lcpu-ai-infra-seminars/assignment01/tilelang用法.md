# TileLang 用法总结

## 基础工作流

```python
import tilelang
import tilelang.language as T

# 1. 定义 prim_func（算子描述）
# 2. tilelang.compile(prim_func, out_idx=[输出索引])
# 3. 调用编译后的 kernel(tensor_args)

# 可选：按形状缓存编译结果（推荐）
_cache = {}
def get_kernel(M, N):
    key = (M, N)
    if key not in _cache:
        _cache[key] = tilelang.compile(make_xxx(M, N), out_idx=[1])
    return _cache[key]
```

---

## 1. 算子定义：`@T.prim_func`

```python
@T.prim_func
def kernel(
    X: T.Buffer((M, N), "float32"),   # 输入 buffer
    Y: T.Buffer((M, N), "float32"),   # 输出 buffer
):
    ...
```

- `T.Buffer((shape), dtype)` 声明输入/输出 buffer
- dtype: `"float32"`, `"float16"`, `"float32"`（accum_dtype）

---

## 2. 并行模型

### 二维 CTA Grid

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

- `grid_x`：列方向（N）的 block 数，通常 `T.ceildiv(N, BLOCK_N)`
- `grid_y`：行方向（M）的 block 数，通常 `T.ceildiv(M, BLOCK_M)`
- `threads`：每个 block 的线程数
- `bx`：当前 block 的 x 坐标（列方向）
- `by`：当前 block 的 y 坐标（行方向）

### 一维 CTA Grid

```python
with T.Kernel(1, M, threads=128) as (bx, by):
    # bx 忽略，by 指定处理第几行
```

### 全局坐标计算

```python
gi = by * block_M + i      # 全局行号
gj = bx * block_N + j      # 全局列号
```

### Block 内并行：`T.Parallel`

```python
for i, j in T.Parallel(block_M, block_N):
    # 每个线程处理 tile 内的一个元素
```

- `T.Parallel(dim1, dim2)` 将 tile 内的迭代展开为线程级并行

---

## 3. 内存层级

### Shared Memory：`T.alloc_shared`

```python
X_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
```

- 在 shared memory 中分配 tile 大小的缓冲区

### Fragment（寄存器）：`T.alloc_fragment`

```python
C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
x_local = T.alloc_fragment((BLOCK_N,), "float32")     # 一维 fragment
```

- Fragment 存放在寄存器中，用于累加/规约
- 通常是 2D（矩阵 tile）或 1D（向量）

### 清零 Fragment

```python
T.clear(C_local)
```

---

## 4. 数据搬运：`T.copy`

### 全局 → Shared

```python
T.copy(A[by * BLOCK_M, bx * BLOCK_N], X_shared)
```

- 第一个参数是全局内存的切片（起点坐标）
- 第二个参数是 shared memory buffer
- 自动处理越界（边界保护）

### Shared → 全局

```python
T.copy(X_shared, Y[by * BLOCK_M, bx * BLOCK_N])
```

### Fragment → 全局

```python
T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

---

## 5. 自述运算

### 逐元素操作

```python
X_shared[i, j] = X_shared[i, j] * 2.0     # shared memory 上的逐元素
Y[gi, gj] = X[gi, gj] * 2.0 + 1.0         # 全局内存直接写
```

### 条件语句：`T.if_then_else`

```python
x_local[j] = T.if_then_else(
    j < N,                      # 条件
    X[by, j],                   # 真值：有效数据
    -T.infinity("float32"),     # 假值：越界填充
)
```

- 用于边界保护或 mask 操作
- `T.infinity("float32")` 返回指定类型的正无穷

### Exp：`T.exp`

```python
x_local[j] = T.exp(x_local[j] - x_max)
```

---

## 6. 规约运算

### Max Reduction

```python
x_max = T.reduce_max(x_local, axis=0)
```

### Sum Reduction

```python
x_sum = T.reduce_sum(x_local, axis=0)
```

- Fragment 沿指定轴规约为标量
- 配合 broadcasting：`x_local - x_max` 自动广播

### 数值稳定的 softmax 模式

```python
x_local = T.alloc_fragment((BLOCK_N,), "float32")

# 加载 + 补 -inf
for j in T.Parallel(BLOCK_N):
    x_local[j] = T.if_then_else(j < N, X[by, j], -T.infinity("float32"))

# 减 max → exp → sum → normalize
x_max = T.reduce_max(x_local, axis=0)
for j in T.Parallel(BLOCK_N):
    x_local[j] = T.exp(x_local[j] - x_max)
x_sum = T.reduce_sum(x_local, axis=0)
for j in T.Parallel(BLOCK_N):
    Y[by, j] = T.if_then_else(j < N, x_local[j] / x_sum, T.float32(0.0))
```

---

## 7. Tiled Matmul 模式

### 完整结构

```python
def make_matmul(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                threads=128, num_stages=3,
                dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M),
                      threads=threads) as (bx, by):
            # Step 1: 分配 shared memory tile
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)

            # Step 2: 分配 fragment 累加器
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(C_local)

            # Step 3: 沿 K 维流水循环
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                # Step 4: 搬运 tile 到 shared
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                # Step 5: tile 级矩阵乘累加
                T.gemm(A_shared, B_shared, C_local)

            # Step 6: 写回
            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return main
```

### T.Pipelined：软件流水线

```python
for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=N):
    T.copy(...)  # 搬运当前 tile
    T.gemm(...)  # 计算上一个 tile（重叠）
```

- `num_stages`：流水线深度（典型值 2-3）
- 实现 copy 和 compute 重叠，隐藏内存延迟

### T.gemm：Tile 级矩阵乘累加

```python
T.gemm(A_shared, B_shared, C_local)
```

- A: `[BLOCK_M, BLOCK_K]`，B: `[BLOCK_K, BLOCK_N]`
- 结果累加到 `C_local` `[BLOCK_M, BLOCK_N]`

### Global 起点坐标

```python
A[by * BLOCK_M, k * BLOCK_K]    # A 的当前 tile 起点
B[k * BLOCK_K, bx * BLOCK_N]    # B 的当前 tile 起点
C[by * BLOCK_M, bx * BLOCK_N]   # C 的当前 tile 起点
```

---

## 8. 编译

```python
kernel = tilelang.compile(make_scale_add(M, N), out_idx=[1])
```

- `out_idx=[i]` 指定哪些 buffer 参数是输出
- 例：`[1]` 表示第二个 `T.Buffer` 是输出，`[2]` 表示第三个

### 编译缓存模式

```python
_cache = {}

def softmax(x):
    M, N = x.shape
    key = (M, N)
    if key not in _cache:
        _cache[key] = tilelang.compile(make_softmax(M, N), out_idx=[1])
    kernel = _cache[key]
    y = torch.empty_like(x)
    kernel(x, y)
    return y
```

- TileLang 按形状编译，相同的 `(M, N)` 复用编译结果

---

## 9. 工具

| API                    | 作用                     |
| ---------------------- | ------------------------ |
| `T.ceildiv(a, b)`        | 向上取整 `ceil(a/b)`         |
| `T.infinity("float32")`  | 正无穷                   |
| `T.float32(0.0)`         | float32 字面量           |

---

## 10. Triton vs TileLang 对比

| 操作           | Triton                                   | TileLang                                        |
| -------------- | ---------------------------------------- | ----------------------------------------------- |
| Kernel 声明      | `@triton.jit` def `f(...)`                 | `@T.prim_func` def `f(...)`                       |
| Buffer 参数       | 裸指针 `x_ptr`                               | `T.Buffer((M, N), dtype)`                         |
| Block 索引        | `tl.program_id(0)`                         | `with T.Kernel(...) as (bx, by):`                 |
| Block 内并行       | 隐式（每个 thread 处理一个 `offset`）                | `for i, j in T.Parallel(M, N):`                  |
| 地址计算         | 手动 `pid * BLOCK + tl.arange(...)`            | `by * BLOCK + i`（全局）/ `i`（shared）                      |
| Shared memory  | 不支持                                   | `T.alloc_shared((M,N), dtype)`                   |
| Fragment（寄存器）  | 隐式（`acc = tl.zeros(...)`）                | `T.alloc_fragment((M,N), dtype)`                 |
| 数据搬运         | `tl.load` / `tl.store`                     | `T.copy(src, dst)`                                |
| 边界保护         | `mask=mask, other=val`（手动）               | `T.copy` 自动处理 / `T.if_then_else`                |
| Reduction     | `tl.max(x, axis=0)`, `tl.sum(x, axis=0)`   | `T.reduce_max(x, axis=0)`, `T.reduce_sum(x, axis=0)` |
| Matmul       | `tl.dot(a, b)`                            | `T.gemm(A_shared, B_shared, C_local)`            |
| 软件流水线       | 手动 `for k0 in range(0,K,BLOCK_K)`          | `for k in T.Pipelined(steps, num_stages=N):`    |

---

## 11. 完整示例速查

### 二维 scale-add

```python
def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def kernel(X: T.Buffer((M, N), dtype), Y: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                gi, gj = by * block_M + i, bx * block_N + j
                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0
    return kernel
```

### 二维 copy + compute（shared memory）

```python
def make_scale2d(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def kernel(X: T.Buffer((M, N), dtype), Y: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(X[by * block_M, bx * block_N], X_shared)
            for i, j in T.Parallel(block_M, block_N):
                X_shared[i, j] = X_shared[i, j] * 2.0
            T.copy(X_shared, Y[by * block_M, bx * block_N])
    return kernel
```

### 一维 softmax

```python
def make_softmax(M, N):
    BLOCK_N = 1
    while BLOCK_N < N:
        BLOCK_N *= 2

    @T.prim_func
    def kernel(X: T.Buffer((M, N), "float32"), Y: T.Buffer((M, N), "float32")):
        with T.Kernel(1, M, threads=128) as (bx, by):
            x_local = T.alloc_fragment((BLOCK_N,), "float32")
            for j in T.Parallel(BLOCK_N):
                x_local[j] = T.if_then_else(
                    j < N, X[by, j], -T.infinity("float32"))
            x_max = T.reduce_max(x_local, axis=0)
            for j in T.Parallel(BLOCK_N):
                x_local[j] = T.exp(x_local[j] - x_max)
            x_sum = T.reduce_sum(x_local, axis=0)
            for j in T.Parallel(BLOCK_N):
                Y[by, j] = T.if_then_else(
                    j < N, x_local[j] / x_sum, T.float32(0.0))
    return kernel
```
