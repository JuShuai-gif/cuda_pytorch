#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

// ============================================================================
// 模拟 RK3588 NPU IO 持久化机制
//
// 核心概念：
// - rknn_input_set(): 每次推理都分配 DMA-BUF + 建立映射 → 27ms
// - rknn_set_io_mem(): 预注册 IO 缓冲区，运行时仅 sync+run → 0.1ms
// ============================================================================

// 模拟 DMA-BUF 分配的 IOBuffer
struct IOBuffer {
    void *data;  // 映射后的 CPU 可访问指针
    int fd;      // DMA-BUF 文件描述符
    size_t size; // 缓冲区大小

    IOBuffer() : data(nullptr), fd(-1), size(0) {
    }
};

// ============================================================================
// 模拟 rknn_input_set：每次调用都分配/释放 IO 内存
// 在真实系统中涉及 dma_buf_alloc + mmap + dma_buf_free 等内核调用
// ============================================================================
double simulate_rknn_input_set(size_t bufferSize, int iterations);

// ============================================================================
// 模拟 rknn_set_io_mem：预分配 IO 缓冲区，运行时仅触发同步
// 在真实系统中使用 rknn_set_io_mem() 注册持久化 buffer
// ============================================================================
double simulate_io_persistent(size_t bufferSize, int iterations);

// ============================================================================
// NEON LUT FP16 转换模拟
// 真实 NEON 指令: vld1q_u8 + vst1q_f16 + tbl 查表
// 这里用 SSE/标量模拟 8 元素并行查表
// ============================================================================
void precompute_fp16_lut(uint16_t *lut); // 预计算 256 个 FP16 值的 LUT
double convert_fp16_naive(const uint8_t *input, float *output, size_t count, int iterations);
double convert_fp16_lut(const uint8_t *input, float *output, size_t count, int iterations);
