#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// FP16 (uint16_t) → FP32 转换
void scalar_fp16_to_f32(const uint16_t *src, float *dst, size_t count);

void neon_fp16_to_f32(const uint16_t *src, float *dst, size_t count);

// BGR uint8 → FP16 RGB 转换（通道重排 + 类型转换）
void scalar_bgr_to_fp16_rgb(const uint8_t *bgr, uint16_t *fp16_rgb,
                            size_t pixel_count);

void neon_bgr_to_fp16_rgb(const uint8_t *bgr, uint16_t *fp16_rgb,
                          size_t pixel_count);

// 运行所有 NEON 转换基准测试
void demo_neon_conversion();
