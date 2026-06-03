# ARM SIMD 演进全景

## NEON → SVE → SVE2：从固定宽度到向量长度无关

ARM SIMD 指令集的发展可以分为三个主要阶段：

| 阶段 | 架构版本 | SIMD 扩展 | 寄存器宽度 | 出现年份 |
|------|----------|-----------|-----------|----------|
| ARMv7 | Cortex-A8/A9/A15 | NEON (Advanced SIMD) | 128-bit | 2009 |
| ARMv8-A | Cortex-A53/A57/A72 | NEON (AArch64 增强) | 128-bit, 32 个寄存器 | 2014 |
| ARMv8.2-A | Cortex-A75/A76 | NEON 增强 + SVE (可选) | NEON 128-bit, SVE 128-2048-bit | 2019 |
| ARMv9-A | Cortex-X1/X2, Neoverse V1 | SVE2 | 128-2048-bit | 2021 |

### NEON：固定宽度 128-bit SIMD

NEON 是 ARM 最广泛部署的 SIMD 指令集：

- **寄存器宽度**：固定 128-bit（Q0-Q31，AArch64 模式下 32 个）
- **数据处理能力**（单条指令）：
  - 4 × float32
  - 8 × float16
  - 16 × int8
  - 8 × int16
  - 4 × int32
  - 2 × float64

```
Q0 (128-bit): [ lane3 | lane2 | lane1 | lane0 ]   -- float32x4_t
Q0 (128-bit): [ l15 | l14 | ... | l1 | l0 ]        -- int8x16_t
```

**核心设计哲学**：显式、固定、硬件已知的向量长度。编译器在生成代码时知道确切的寄存器布局，可以进行精确的指令调度和寄存器分配。代价是：你的代码只能跑在一种向量宽度上。如果一个操作需要更大吞吐，你必须手动展开循环或依赖编译器自动向量化。

### SVE：可伸缩向量扩展

SVE 引入了"向量长度无关"（Vector-Length Agnostic, VLA）的概念：

- **寄存器宽度**：128-bit 到 2048-bit，具体取决于硬件实现
- **核心创新**：
  - **谓词寄存器** (p0-p15)：每个位控制对应向量 lane 的启用/禁用，彻底消除了标量尾部循环
  - **VLA 编程模型**：同一份二进制代码可以在不同向量宽度的硬件上运行，无需重新编译

```
Z0 (N-bit, N ∈ {128, 256, 512, 1024, 2048}):

[ lane_vl-1 | lane_vl-2 | ... | lane1 | lane0 ]
                 |
    --- p0 predicate register (per-lane enable mask) ---
```

**为什么 VLA 重要**：

1. **可移植性**：一次编译，在各种硬件上自动获得最优向量利用率。AWS Graviton3 的 SVE 宽 256-bit，Fujitsu A64FX 宽 512-bit，同一份二进制在两者上都能被充分利用
2. **消除尾部循环**：传统 SIMD 最常见的 bug 来源就是标量尾部处理。SVE 的 `svwhilelt` + 谓词加载/存储 自动处理
3. **未来兼容**：当新的 SVE 实现带宽更宽时，已有代码无需改动即可受益

### SVE2：SVE 的功能补全

ARMv9 引入的 SVE2 在 SVE 基础上添加了：

- **复数整数乘法**（对信号处理至关重要）
- **位操作谓词**（predicate 之间的 AND/OR/XOR/NOT）
- **增强的 scatter/gather**（非连续内存访问）
- **BE / C 指令** 用于 AES 和 SM4 加解密加速
- **DSP 类操作**：SVE2 将 NEON 的所有 DSP 能力搬到了 VLA 编程模型

本质上，**SVE2 = SVE + NEON 的全部功能**，是 ARM 生态的长期 SIMD 方向。

---

## 工业应用场景

### 1. 移动端 ML 推理

```
模型格式：ONNX / TFLite / NCNN
执行后端：XNNPACK / ARM Compute Library / QNNPACK
SIMD 角色：卷积的 im2col+GEMM，全连接层的 GEMV，激活函数，量化/反量化
```

典型代码流程（以 NCNN 为例）：
```
Input → 量化 (int8) → im2col → GEMM (vdotq_s32) → 反量化 → Output
                                                        ↑
                                              NEON int8 dot product
                                              每周期 4× int8 乘加
```

移动端 ML 推理是 NEON 的最大战场。NCNN, MNN, TFLite XNNPACK 都重度使用 NEON intrinsics 来加速：
- **卷积层**：im2col 将卷积转为矩阵乘法，使用 NEON GEMM 微核心（4×4 或 8×8 分块）
- **全连接层**：GEMV，权重已预打包为适合 NEON 加载的布局
- **激活函数**：ReLU、Sigmoid、Tanh 全部使用 NEON 查找表或多项式近似
- **池化**：max/average pooling 使用 NEON 向量比较和归约

性能数据（Cortex-A76, 单核）：
- fp32 GEMM: ~40 GFLOPS（NEON 手动优化）
- int8 GEMM: ~120 GOPS（vdotq_s32）

### 2. 云端 ARM 服务器 ML 推理

AWS Graviton3, Ampere Altra, Google Axion 正在大规模取代 x86 在 ML 推理场景中的位置：

```
平台            CPU          SIMD        向量宽度    内存带宽
AWS m7g         Graviton3    SVE+NEON    256-bit     307 GB/s
Ampere Altra    Altra Max    NEON        128-bit     200 GB/s
Google Axion    Axion        SVE+NEON    256-bit     356 GB/s
```

- **模型服务**：LLaMA、BERT 等 Transformer 模型的高吞吐服务
- **微批推理**：batch_size=1 的在线推理，受限于内存带宽而非计算
- **推荐系统**：嵌入表查找 + 特征交叉的 SIMD 优化

### 3. 嵌入式视觉

- 目标检测（YOLO nano/tiny 变体）在 ARM Cortex-M55/M85 上的部署
- 图像预处理：RGB→YUV 转换、直方图均衡化、双边滤波，全部使用 NEON
- 特征提取：ORB/SIFT 关键点检测的 SIMD 加速

### 4. 移动游戏引擎数学库

Unity/Unreal 等引擎在 ARM 上的数学库（如 `Unity.Mathematics`, `DirectXMath`）重度依赖 NEON：
- 4×4 矩阵乘法 (`float4x4`)、四元数运算
- 碰撞检测（AABB × 三角形）
- 蒙皮矩阵变换

### 5. 视频编解码加速

libaom, x264, libvpx 等编解码器对 ARM 平台的 NEON 优化：
- 运动估计（SAD/SSE 计算）
- DCT/IDCT 变换
- 去块滤波
- 色彩空间转换（YUV ↔ RGB）

---

## SIMD 宽度与编译器标志

### NEON 编程

```c
// 编译器标志
// GCC/Clang:
//   -march=armv8-a+simd    - NEON 在 ARMv8-A
//   -march=armv8.2-a+simd  - 包含 int8 dot product (vdotq)
//   -march=armv9-a         - SVE2 + NEON
//
// 头文件
#include <arm_neon.h>
```

### SVE 编程

```c
// 编译器标志
//   -march=armv8-a+sve       - 基础 SVE
//   -march=armv8.2-a+sve     - SVE + NEON int8 dot
//   -march=armv9-a           - SVE2 + NEON
//
// 头文件
#include <arm_sve.h>
```

---

## 如何检测 CPU 特性

### Linux 运行时检测

```c
// 方法 1：/proc/cpuinfo
// $ cat /proc/cpuinfo | grep Features
// Features : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp
//               ^^^^^ NEON (Advanced SIMD)
//                           ^^^^  ^^^^ ^^^^ 加密扩展
//                                               ^^^^^ 原子操作(ARMv8.1)
//                                                      ^^^^ fp16 NEON
//                                                               ^^^^^^^ int8 dot product(ARMv8.2)

// 方法 2：getauxval (推荐用于程序内检测)
#include <sys/auxv.h>
#include <asm/hwcap.h>

void detect_features() {
    unsigned long hwcaps = getauxval(AT_HWCAP);

    if (hwcaps & HWCAP_ASIMD)     printf("NEON: yes\n");
    if (hwcaps & HWCAP_ASIMDHP)   printf("NEON fp16: yes\n");    // ARMv8.2
    if (hwcaps & HWCAP_ASIMDDP)   printf("NEON int8 dot: yes\n"); // ARMv8.2
    if (hwcaps & HWCAP_FPHP)      printf("Scalar fp16: yes\n");
    if (hwcaps & HWCAP_CRC32)     printf("CRC32: yes\n");         // ARMv8.1

    unsigned long hwcaps2 = getauxval(AT_HWCAP2);
    if (hwcaps2 & HWCAP2_SVE)     printf("SVE: yes\n");           // ARMv8.2+
    if (hwcaps2 & HWCAP2_SVE2)    printf("SVE2: yes\n");          // ARMv9
}
```

### 方法 3：lscpu

```bash
# 查看 CPU 架构和 flags
$ lscpu
Architecture:            aarch64
CPU op-mode(s):          64-bit
Model name:              Cortex-A76
Flags:                   fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
                         fphp asimdhp cpuid asimdrdm dcpop
#                        ^^^^^ = NEON            ^^^^^^^ = fp16 NEON
```

---

## NEON vs SVE：编程模型对比

| 特性 | NEON | SVE | 胜者 |
|------|------|-----|------|
| 便携性 | 固定 128-bit，需手动处理多种宽度 | VLA，一次编译处处最优 | SVE |
| 尾部循环 | 需要标量尾部或手动 padding | 谓词自动处理，零尾部处理 | SVE |
| 部署成熟度 | 所有 ARMv7/v8/v9 芯片 | 仅高端芯片（Neoverse V1, M4） | NEON |
| 调试工具 | 完善（ARM DS, gdb） | 较新，支持有限 | NEON |
| 社区生态 | 大量开源示例 | 案例较少 | NEON |
| 指令复杂度 | 中等 | 较高（需要理解谓词） | NEON |
| 内存访问模式 | Load/Store 固定宽度 | 谓词控制的 gather/scatter | SVE |
| 未来保证 | 遗留指令集 | ARM 长期方向 | SVE |

### 实际建议

**2024-2026 年的正确策略**：

1. **主力使用 NEON + 编译器自动向量化**：覆盖 99% 的 ARM 设备
2. **SVE 用于云端专用代码路径**：在服务器端利用 SVE 提升吞吐
3. **使用运行时指令调度（function multi-versioning）**：

```c
// 运行时根据 CPU 特性选择最优实现
typedef void (*gemm_func)(int M, int N, int K, const float* A, const float* B, float* C);

gemm_func select_gemm() {
    unsigned long hwcaps2 = getauxval(AT_HWCAP2);
    if (hwcaps2 & HWCAP2_SVE) return gemm_sve_256bit;  // Graviton3
    if (hwcaps & HWCAP_ASIMDDP) return gemm_neon_dot;  // Cortex-A76
    return gemm_neon_baseline;                          // Cortex-A53
}
```

---

## NEON 寄存器文件结构

```
AArch64 NEON 寄存器视图:

128-bit Q registers (32个):
Q0  [127:0] ─── V0 (低128位)，同时也是 V0.B16 / V0.H8 / V0.S4 / V0.D2
Q1  [127:0] ─── V1
...
Q31 [127:0] ─── V31

64-bit D registers (也可作为半宽使用):
D0  [63:0]  ─── Q0 的低64位
D1  [63:0]  ─── Q0 的高64位
D2  [63:0]  ─── Q1 的低64位
...

NEON 和浮点寄存器是同一个物理寄存器文件：
S0 [31:0]   ─── D0 的低32位 (标量浮点)
S1 [31:0]   ─── D0 的高32位
```

**关键限制**：
- NEON 不能直接做整数除法（没有 vdivq_s32，需要标量模拟或使用浮点倒数）
- `vdivq_f32` 在 Cortex-A76 上延迟 ~9 周期，吞吐量 ~1/5 周期（比乘法慢 3-4 倍）
- 跨 lane 操作（如 shuffle、extract）在端口受限的核心上可能成为瓶颈

---

## 性能模型速览

ARM 微架构关键性能参数（用于估算速度）：

| 核心 | 发布 | NEON 吞吐 | NEON fma 延迟 | vdiv f32 延迟 | L1 大小 | L2 大小 |
|------|------|-----------|---------------|---------------|---------|---------|
| Cortex-A53 | 2014 | 1 NEON op/cycle | 8 cycles | 17 cycles | 8-64KB | 128KB-2MB |
| Cortex-A72 | 2016 | 2 NEON op/cycle | 5 cycles | 10 cycles | 48KB I+32KB D | 512KB-4MB |
| Cortex-A76 | 2018 | 2 NEON op/cycle | 4 cycles | 9 cycles | 64KB I+64KB D | 256-512KB |
| Cortex-X1 | 2021 | 4 NEON op/cycle | 3 cycles | 7 cycles | 64KB I+64KB D | 1MB |
| Neoverse V1 | 2021 | 4 NEON + SVE(256b) | 3 cycles | 6 cycles | 64KB I+64KB D | 1MB |
| Apple M1 (Firestorm) | 2020 | 4 NEON op/cycle | 3 cycles | ~5 cycles | 192KB I+128KB D | 12-16MB |

**经验法则**：
- 浮点 FMA 吞吐量 ≈ 2× NEON 发射宽度（因为每条 FMA 是 2 FLOP）
- int8 dot product 吞吐量 ≈ 4× NEON 发射宽度（每条 vdotq_s32 含 16 个乘加 = 32 次操作）
- **NEON 代码性能通常受限于内存带宽，而非计算**，除非数据完全在 L1 缓存中

---

## 练习建议

如果你是 SIMD 新手，建议按以下顺序学习：

1. **NEON 基础操作**（`02_neon_basics.md`）：掌握 load/store, add/mul/fma, compare/select
2. **内存布局**（`03_memory_layout.md`）：学会 AoS→SoA 转换，理解为什么布局决定性能
3. **NEON 模式**（`04_neon_intrinsics_patterns.md`）：map/reduce/filter/convolution 等常见模式
4. **SVE 入门**（`05_sve_basics.md`）：理解谓词、VLA 编程模型
5. **工业案例**（`06_industrial_cases.md`）：真实生产级代码的架构设计

每个主题都有可编译运行的代码片段。建议在 ARM 开发板（如树莓派 4/5）或 QEMU 用户模式下实际运行和测量。
