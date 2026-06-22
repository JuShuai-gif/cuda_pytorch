#include "io_persistence.h"

#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

// ============================================================================
// 模拟 rknn_input_set: 每次调用都分配/释放 IO 缓冲区
//
// 真实场景:
//   1. dma_buf_alloc(fd, size)     → 内核态 DMA-BUF 分配
//   2. mmap(fd, ...)               → 用户态映射
//   3. rknn_set_input(ctx, buf)    → 设置输入 tensor
//   4. rknn_run(ctx)               → 推理
//   5. munmap + close(fd)          → 释放资源
//
// 模拟方式:
//   - 分配大块内存 (模拟 DMA 分配开销)
//   - memset 写入数据 (模拟 DMA 同步)
//   - 立即释放
// ============================================================================
double simulate_rknn_input_set(size_t bufferSize, int iterations) {
    std::cout << "\n  [模拟 rknn_input_set - 每次重新分配 DMA-BUF]\n";
    std::cout << "  缓冲区大小: " << bufferSize / 1024 << " KB, 迭代 " << iterations << " 次\n";

    auto total_start = std::chrono::high_resolution_clock::now();
    long long total_alloc_ns = 0;
    long long total_free_ns = 0;

    for (int i = 0; i < iterations; ++i) {
        // 模拟 dma_buf_alloc + mmap
        auto t1 = std::chrono::high_resolution_clock::now();
        uint8_t *buf = new (std::nothrow) uint8_t[bufferSize];
        auto t2 = std::chrono::high_resolution_clock::now();
        total_alloc_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        if (buf) {
            // 模拟 DMA 同步: 写入数据到缓冲区
            memset(buf, 0xAB, bufferSize);

            // 模拟 munmap + dma_buf_free
            auto t3 = std::chrono::high_resolution_clock::now();
            delete[] buf;
            auto t4 = std::chrono::high_resolution_clock::now();
            total_free_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    double avg_ms = total_ms / iterations;
    double avg_alloc_ms = total_alloc_ns / (iterations * 1e6);
    double avg_free_ms = total_free_ns / (iterations * 1e6);

    std::cout << "  总耗时: " << total_ms << " ms\n";
    std::cout << "  平均每次分配 (dma_buf_alloc+mmap): " << avg_alloc_ms << " ms\n";
    std::cout << "  平均每次释放 (munmap+close): " << avg_free_ms << " ms\n";
    std::cout << "  每次完整 IO 操作 (分配+memset+释放): " << avg_ms << " ms\n";
    std::cout << "  ⚠ 注意: 在真实系统上，dma_buf_alloc 涉及内核态调用，实际开销可达 ~27ms\n";

    return avg_ms;
}

// ============================================================================
// 模拟 rknn_set_io_mem: 一次分配，持续复用
//
// 真实场景:
//   1. rknn_create_memory(ctx, size)     → 预先分配 IO 内存 (仅一次)
//   2. rknn_set_io_mem(ctx, mem, ...)    → 注册 IO 内存 (仅一次)
//   在后续每次推理中:
//   3. 仅做 rknn_run()                   → 推理 (无需重新分配)
//   4. 同步 fence                        → 等待 DMA 完成
//
// 模拟方式: 预先分配一次内存，后续仅做 memset (模拟 DMA 同步)
// ============================================================================
double simulate_io_persistent(size_t bufferSize, int iterations) {
    std::cout << "\n  [模拟 rknn_set_io_mem - IO 持久化]\n";
    std::cout << "  缓冲区大小: " << bufferSize / 1024 << " KB, 迭代 " << iterations << " 次\n";

    // 一次性预分配 (模拟 rknn_create_memory)
    auto alloc_start = std::chrono::high_resolution_clock::now();
    uint8_t *persistent_buf = new (std::nothrow) uint8_t[bufferSize];
    auto alloc_end = std::chrono::high_resolution_clock::now();
    double one_time_alloc_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();

    if (!persistent_buf) {
        std::cerr << "  内存分配失败！\n";
        return -1;
    }

    // 模拟 rknn_set_io_mem: 注册 (仅一次)
    std::cout << "  一次性预分配耗时: " << one_time_alloc_ms << " ms (仅执行一次)\n";

    // 迭代推理
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        // 模拟 DMA 同步: 仅写入数据, 不分配/释放
        memset(persistent_buf, static_cast<uint8_t>(i & 0xFF), bufferSize);

        // 模拟 fence 同步 (DMA 完成等待)
        // 在真实系统中这是硬件级别的同步，耗时极短
        __asm__ volatile("" : : "r"(persistent_buf) : "memory");
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    double avg_ms = total_ms / iterations;

    delete[] persistent_buf;

    std::cout << "  总耗时 (" << iterations << " 次推理): " << total_ms << " ms\n";
    std::cout << "  平均每次 IO 操作 (仅 sync): " << avg_ms << " ms\n";
    std::cout << "  ✓ 相比每次重新分配，IO 持久化消除了 DMA-BUF 分配和映射的开销\n";

    return avg_ms;
}

// ============================================================================
// 预计算 FP16 查找表 (LUT)
//
// FP16 (IEEE 754 half-precision) 格式: 1 符号位 + 5 指数位 + 10 尾数位
// 这里预计算 256 个值的 FP16 表示，用于 NEON 查表法
//
// 真实 NEON 指令:
//   vld1q_u8  → 一次加载 16 个 uint8 索引
//   vtbl      → 并行查表 (8 路)
//   vst1q_f16 → 一次存储 8 个 FP16 值
// ============================================================================
void precompute_fp16_lut(uint16_t *lut) {
    // 将 0-255 映射到 FP16 的 half-float 表示
    // 简化: 假设输入 0-255 对应 FP16 值 0.0f - 1.0f (归一化)
    union {
        float f;
        uint32_t u;
    };

    for (int i = 0; i < 256; ++i) {
        f = static_cast<float>(i) / 255.0f;

        // 提取 FP32 的符号位、指数位、尾数位
        uint32_t sign = (u >> 31) & 0x1;
        uint32_t exp = (u >> 23) & 0xFF;
        uint32_t mant = u & 0x7FFFFF;

        // 转换为 FP16
        uint16_t fp16;
        if (exp == 0) {
            // 零或次正规数
            fp16 = static_cast<uint16_t>((sign << 15) | 0);
        } else if (exp == 0xFF) {
            // 无穷大或 NaN
            fp16 = static_cast<uint16_t>((sign << 15) | 0x7C00 | ((mant != 0) ? 0x0200 : 0));
        } else {
            int newExp = static_cast<int>(exp) - 127 + 15;
            if (newExp >= 31) {
                // 溢出到无穷大
                fp16 = static_cast<uint16_t>((sign << 15) | 0x7C00);
            } else if (newExp <= 0) {
                // 下溢: 次正规数或零
                if (newExp >= -10) {
                    uint32_t shiftedMant = (mant | 0x800000) >> (1 - newExp);
                    fp16 = static_cast<uint16_t>((sign << 15) | (shiftedMant >> 13));
                } else {
                    fp16 = static_cast<uint16_t>(sign << 15);
                }
            } else {
                fp16 = static_cast<uint16_t>(
                    (sign << 15) | (newExp << 10) | (mant >> 13));
            }
        }
        lut[i] = fp16;
    }
}

// ============================================================================
// 逐元素 C 转换 FP16 → FP32 (模拟原始慢速路径)
// 每个元素单独做位操作
// ============================================================================
static float fp16_to_fp32_scalar(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp_half = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t fp32;
    if (exp_half == 0) {
        if (mant == 0) {
            fp32 = (sign << 31);
        } else {
            // 次正规数
            uint32_t mant_norm = mant;
            int e = -14;
            while (!(mant_norm & 0x400)) {
                mant_norm <<= 1;
                e -= 1;
            }
            mant_norm &= 0x3FF;
            uint32_t exp32 = static_cast<uint32_t>(e + 127) << 23;
            fp32 = (sign << 31) | exp32 | (mant_norm << 13);
        }
    } else if (exp_half == 31) {
        fp32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        uint32_t exp32 = static_cast<uint32_t>(exp_half - 15 + 127) << 23;
        fp32 = (sign << 31) | exp32 | (mant << 13);
    }

    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = fp32;
    return conv.f;
}

double convert_fp16_naive(const uint8_t *input, float *output, size_t count, int iterations) {
    std::cout << "\n  [逐元素 C 转换 FP16→FP32 - " << count << " 个元素]\n";

    // 将 input[0..255] 视作 FP16 查找键 (0-255 对应亮度值 0.0-1.0)
    // 这里制造一个预计算的 FP16 LUT，用 input 值作为索引
    uint16_t fp16_lut[256];
    precompute_fp16_lut(fp16_lut);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t fp16_val = fp16_lut[input[i]];
            output[i] = fp16_to_fp32_scalar(fp16_val);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    double avg_ms = total_ms / iterations;

    double elements_per_sec = (count * iterations) / (total_ms / 1000.0);
    double gb_per_sec = (count * sizeof(float) * iterations) / (total_ms / 1000.0) / 1e9;

    std::cout << "  总耗时 (" << iterations << " 轮): " << total_ms << " ms\n";
    std::cout << "  平均每轮: " << avg_ms << " ms\n";
    std::cout << "  吞吐量: " << (elements_per_sec / 1e6) << " M 元素/秒\n";
    std::cout << "  带宽: " << gb_per_sec << " GB/s\n";

    return avg_ms;
}

// ============================================================================
// NEON LUT 查表法转换 - 模拟 8 元素并行查表
//
// 真实 NEON 实现:
//   uint8x8_t idx = vld1_u8(input + i);
//   uint16x8_t result = vtbl1q_u8(lut, idx);  // 并行查表
//   vst1q_f16(output + i, result);
//
// 模拟: 8 元素一组处理 (编译器可能自动向量化)
// ============================================================================
double convert_fp16_lut(const uint8_t *input, float *output, size_t count, int iterations) {
    std::cout << "\n  [NEON LUT 查表法 FP16→FP32 - " << count << " 个元素]\n";

    uint16_t fp16_lut[256];
    precompute_fp16_lut(fp16_lut);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // 8 元素一组处理 (模拟 NEON vld1q_u8 → vtbl → vst1q_f16)
        // 真实 NEON 一次处理 16 个元素 (128-bit NEON) 或 8 个 (64-bit aarch32)
        size_t i = 0;

        // 主循环: 每次处理 16 个 (模拟 NEON 128-bit 寄存器)
        for (; i + 16 <= count; i += 16) {
            // 模拟 vld1q_u8: 加载 16 个 uint8 索引
            uint8_t indices[16];
            for (int j = 0; j < 16; ++j) indices[j] = input[i + j];

            // 模拟 vtbl 查表: 用索引从 LUT 中并行获取 FP16 值
            // 实际的 vtbl 指令是硬件级别的并行操作
            uint16_t fp16_vals[16];
            for (int j = 0; j < 16; ++j) {
                fp16_vals[j] = fp16_lut[indices[j]];
            }

            // 模拟 vst1q_f16: 存储 8 个 FP16 → 转换为 FP32
            // 注意: 真实 NEON 有专用指令完成 FP16→FP32 转换
            for (int j = 0; j < 16; ++j) {
                output[i + j] = fp16_to_fp32_scalar(fp16_vals[j]);
            }
        }

        // 尾数处理 (不足 16 个元素)
        for (; i < count; ++i) {
            output[i] = fp16_to_fp32_scalar(fp16_lut[input[i]]);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    double avg_ms = total_ms / iterations;

    double elements_per_sec = (count * iterations) / (total_ms / 1000.0);
    double gb_per_sec = (count * sizeof(float) * iterations) / (total_ms / 1000.0) / 1e9;

    std::cout << "  总耗时 (" << iterations << " 轮): " << total_ms << " ms\n";
    std::cout << "  平均每轮: " << avg_ms << " ms\n";
    std::cout << "  吞吐量: " << (elements_per_sec / 1e6) << " M 元素/秒\n";
    std::cout << "  带宽: " << gb_per_sec << " GB/s\n";

    return avg_ms;
}
