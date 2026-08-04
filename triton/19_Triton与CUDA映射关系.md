# 19 Triton 与 CUDA 映射关系（深度版）

> 本文目标：建立精确的 CUDA 映射，重点深入布局系统（BlockedEncodingAttr）如何决定线程/寄存器分配。

## 1. 映射表

| CUDA | Triton |
| --- | --- |
| grid | program 网格 |
| blockIdx | `tl.program_id(axis)` |
| blockDim | `num_warps`（×32） |
| threadIdx | 布局系统自动 |
| global memory | `tl.load/store` |
| shared memory | `local_alloc`（自动） |
| `__syncthreads()` | membar/布局转换自动 |
| warp | 布局系统 |
| mma.sync | `tl.dot` 自动 |
| cp.async | Pipeliner 自动 |
| TMA/tcgen05 | Hopper/Blackwell 自动 |

## 2. BlockedEncodingAttr 深度（核心）

`TritonGPUAttrDefs.td:818`：
```td
$sizePerThread    // 每线程寄存器元素数
$threadsPerWarp   // warp 内 lane 分布
$warpsPerCTA      // CTA 内 warp 分布
$order            // 最快维在前
```

### 转 LinearLayout（`LinearLayoutConversions.cpp:866-875`）
```cpp
ctaLayout =
    identityStandardND(register, sizePerThread, order) *
    identityStandardND(lane, threadsPerWarp, order) *
    identityStandardND(warp, warpsPerCTA, order);
return combineCtaCgaWithShape(ctaLayout, CGALayout, shape);
```

### 具体例子（128x128, 4 warps, fp16）
设 `numWarps=4, order=[1,0], sizePerThread=[1,4]`：
```
#ttg.blocked<{sizePerThread=[1,4], threadsPerWarp=[1,32], warpsPerCTA=[4,1], order=[1,0]}>
```
基向量：
- register: (0,1),(0,2)
- lane: (0,1)..(0,16)
- warp: (1,0),(2,0)

## 3. MmaEncodingAttr（深度）

`NvidiaMmaEncodingAttr`（:1272）：`versionMajor/versionMinor/warpsPerCTA/instrShape`。
- `nvidiaMmaTile`（`LinearLayoutConversions.cpp:917`）：
```cpp
ctaLayout = ctaLayout *
    identity1D(kWidth, register, inner) *       # kWidth 个寄存器
    identity1D(4, lane, inner) *                # 4 个 lane
    identity1D(8, lane, outer) *                # 8 个 lane
    identity1D(m/8, register, outer) *          # 行重复
    identity1D(n/(kWidth*4), register, inner);  # 列重复
```

## 4. 布局转换（深度）

- 代价本质：跨 warp/block 转换走 shared + barrier；warp 内走 shuffle。
- `minimalCvtLayout` = `dstLayout.invertAndCompose(srcLayout)`。
- `areLayoutsEquivalent`（Dialect.cpp:4388）：转 LL 后比相等。

## 5. 共享内存布局（深度）

`SwizzledSharedEncodingAttr` 编码 swizzle 进基向量（`LinearLayoutConversions.cpp:50-96`）：
```cpp
for (int row = 1; row < numRows; row *= 2) {
  int vec = shared.getVec(); int perPhase = shared.getPerPhase(); int maxPhase = shared.getMaxPhase();
  bases2D.push_back({row, (vec * ((row / perPhase) % maxPhase)) % numCols});
}
```

## 6. 同步语义

- program 间无同步。
- program 内：membar/布局转换自动插 barrier。

## 7. 深入自测

1. BlockedEncodingAttr 的 4 字段？
2. 128x128/4warps/fp16 的具体布局？
3. nvidiaMmaTile 的 5 个 identity1D？
4. 布局转换何时走 shared？
5. swizzle 如何编码进基向量？

## 8. 下一步

进入 `20_核心数据结构.md`（深度版）。
