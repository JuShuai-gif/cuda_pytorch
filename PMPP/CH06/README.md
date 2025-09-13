
---

## Exercise 1

你写的 kernel：

```cpp
__global__ void TiledMatrixMulKernelColMajorOrder(float *M, float *N, float *P, int m, int n, int o)
```

✅ 代码逻辑正确：

* `M` 按行取，`N` 做 **corner turning** 以保证全局内存访问 coalesced。
* 两个 `__shared__` tile，分别缓存 `M` 和转置访问后的 `N`。
* 边界处理：`if (row < m && col < o)` 确保越界填 `0`。
* 内层 `for(k=0;k<TILE_WIDTH;k++)` 是 tile 乘加。

这个就是 Fig. 6.4 的实现。

---

## Exercise 2

> “For what BLOCK\_SIZE values will we avoid uncoalesced accesses?”

* CUDA warp = 32 个线程。
* 如果 `BLOCK_SIZE < 32`，warp 会跨多行 → 访问不连续。
* 所以 **BLOCK\_SIZE 必须是 32 的倍数**：`32, 64, 128, …`。
* 受 shared memory 限制，实际不会超过 `64`。

✅ 你的答案正确。

---

## Exercise 3

访问模式逐行看：

* **a. a (line 05)** → 全局内存，相邻线程读相邻位置 → **coalesced**
* **b. a\_s (line 05)** → shared memory → **不适用**
* **c. b (line 07)** → 相邻线程读相邻位置 → **coalesced**
* **d. c (line 07)** → 每个线程 stride=4 → **uncoalesced**
* **e. bc\_s (line 07)** → shared memory → **不适用**
* **f. a\_s (line 10)** → shared memory → **不适用**
* **g. d (line 10)** → 相邻线程读相邻位置 → **coalesced**
* **h. bc\_s (line 11)** → shared memory → **不适用**
* **i. e (line 11)** → stride=8 → **uncoalesced**

✅ 你写的判断和理由完全正确。

---

## Exercise 4

浮点操作/内存访问比 (OP/B)：

假设矩阵维度：`M (m×n)`，`N (n×o)`，`P (m×o)`，元素 4B float。

### (a) Naïve kernel

* 每个 `P[row,col]`：

  * Load `n` from M, load `n` from N → `2n` loads
  * Compute `n` mul + `n` add → `2n` ops
* 总内存字节：`2n * 4`
* 比值：`2n / (8n) = 0.25 OP/B`

✅ 你的答案正确。

---

### (b) Shared-memory tiling (`32×32`)

* 每个线程只负责 `n/32` 元素加载（因为一个 warp 协同加载 32 个元素 tile）。
* 所以每个结果需要：`2n/32 = n/16` loads
* ops 还是 `2n`
* 比值：`2n / (4*(n/16)) = 8 OP/B`

✅ 正确。

---

### (c) Tiling + Coarsening (factor=4)

* `M` tile 被重用 4 次 → `n/128` loads
* `N` 还是 `n/32` loads
* 总共：`n/32 + n/128 = 5n/128` loads
* 内存字节：`4 * 5n/128 = 20n/128 = 5n/32`
* 比值：`2n / (5n/32) = 64/5 ≈ 12.8 OP/B`

✅ 完全正确。

---

### 总结

你写的四道题答案都对，尤其 Exercise 4 的算式很漂亮。
我建议你在写报告或者笔记时，可以用表格展示：

| Kernel type | Global loads per element | Ops per element | Bytes   | OP/B |
| ----------- | ------------------------ | --------------- | ------- | ---- |
| Naïve       | `2n`                     | `2n`            | `8n`    | 0.25 |
| Tiled       | `2n/32`                  | `2n`            | `8n/32` | 8    |
| Tiled+Coars | `n/32 + n/128`           | `2n`            | `5n/32` | 12.8 |

---


