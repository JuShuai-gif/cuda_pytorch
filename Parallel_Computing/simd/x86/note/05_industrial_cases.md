# x86 SIMD 工业级实战案例

## 场景 1：图像处理 ― RGB 转灰度

### 1.1 问题描述

将 24-bit RGB 像素转换为 8-bit 灰度值是图像预处理中最基础的操作之一。标准公式为：

```
Gray = 0.299 * R + 0.587 * G + 0.114 * B
```

每个像素需要 3 次乘法 + 2 次加法，对 4K 图像（约 8M 像素）意味着约 40M 次标量运算。该操作的操作强度（OI）约为 0.5 FLOP/byte，属于典型的内存带宽绑定型工作负载。SIMD 可以在减少循环开销的同时一次性处理 8 或 32 个像素。

### 1.2 整数实现（推荐用于 8-bit 输出）

整数实现避免 `float ↔ int` 往返转换开销，使用定点乘法：

```
Gray = (77*R + 150*G + 29*B + 128) >> 8
```

权重 `{77, 150, 29}` 由 `{0.299, 0.587, 0.114} × 256` 缩放而来，加 128 实现四舍五入。利用 `_mm256_maddubs_epi16` 做 `u8 × s8 → i16` 乘加，然后横向归约：

```c
#include <immintrin.h>
#include <stdint.h>

// 256-bit AVX2 = 32 x u8 per iteration
__attribute__((noinline))
void rgb_to_gray_int_avx2(const uint8_t *__restrict rgb,
                           uint8_t *__restrict gray,
                           int num_pixels) {
    // Pre-broadcast weights: {77, 150, 29, 0} → 8 interleaved copies
    const __m256i weight = _mm256_setr_epi8(
        77, 150, 29, 0, 77, 150, 29, 0,
        77, 150, 29, 0, 77, 150, 29, 0,
        77, 150, 29, 0, 77, 150, 29, 0,
        77, 150, 29, 0, 77, 150, 29, 0);
    // Bias for rounding: 128 on each 16-bit lane, zero-extended
    const __m256i bias = _mm256_set1_epi16(128);

    int i;
    for (i = 0; i + 8 <= num_pixels; i += 8) {
        // Load 24 bytes (8 pixels × 3 channels), spill to 32 bytes
        // Layout: R0 G0 B0 R1 G1 B1 ... R7 G7 B7
        __m128i line0 = _mm_loadu_si128((const __m128i *)(rgb + i * 3));
        __m128i line1 = _mm_loadl_epi64((const __m128i *)(rgb + i * 3 + 16));

        // Insert an extra zero byte after each B channel to produce
        // R0 G0 B0 0 R1 G1 B1 0 ... inside 32-byte register
        __m256i bytes = _mm256_inserti128_si256(
            _mm256_castsi128_si256(line0), line1, 1);

        // Multiply u8 × s8 and accumulate adjacent pairs into i16
        __m256i acc = _mm256_maddubs_epi16(bytes, weight);

        // Horizontal addition: sum all 16-bit lanes into 32-bit, then pack
        // acc layout: [a0, a1, a2, a3, | a4, a5, a6, a7, | ...]
        // We need a0+a1+a2+a3 for pixel 0, but maddubs already gives
        // 0: 77*R0+150*G0, 1: 29*B1 + ...
        // Actually we must reshape: for each pixel group we have 4 bytes
        // whose maddubs produces partial sums; better approach below.
    }

    // Tail handling for remaining pixels
    for (; i < num_pixels; i++) {
        int r = rgb[i * 3 + 0];
        int g = rgb[i * 3 + 1];
        int b = rgb[i * 3 + 2];
        gray[i] = (uint8_t)((77 * r + 150 * g + 29 * b + 128) >> 8);
    }
}
```

更好的整数实现：展开 RGB 交错存储为 3 个独立平面，每平面用 `_mm256_maddubs_epi16` 一次性处理 32 个像素：

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// 256-bit AVX2 = 32 x u8 per iteration (processes 32 pixels at a time)
__attribute__((noinline))
void rgb_to_gray_int32_avx2(const uint8_t *__restrict r_plane,
                             const uint8_t *__restrict g_plane,
                             const uint8_t *__restrict b_plane,
                             uint8_t *__restrict gray,
                             int num_pixels) {
    const __m256i w_r = _mm256_set1_epi16(77);
    const __m256i w_g = _mm256_set1_epi16(150);
    const __m256i w_b = _mm256_set1_epi16(29);
    const __m256i rounding = _mm256_set1_epi16(128);
    const __m256i zero = _mm256_setzero_si256();

    int i;
    for (i = 0; i + 32 <= num_pixels; i += 32) {
        // Load 32 bytes from each planar channel
        __m256i r_vec = _mm256_loadu_si256((const __m256i *)(r_plane + i));
        __m256i g_vec = _mm256_loadu_si256((const __m256i *)(g_plane + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i *)(b_plane + i));

        // Widen u8 → u16 (odd/even halves)
        __m256i r16_lo = _mm256_unpacklo_epi8(r_vec, zero);
        __m256i r16_hi = _mm256_unpackhi_epi8(r_vec, zero);
        __m256i g16_lo = _mm256_unpacklo_epi8(g_vec, zero);
        __m256i g16_hi = _mm256_unpackhi_epi8(g_vec, zero);
        __m256i b16_lo = _mm256_unpacklo_epi8(b_vec, zero);
        __m256i b16_hi = _mm256_unpackhi_epi8(b_vec, zero);

        // Accumulate: low 16 pixels
        __m256i sum_lo = _mm256_add_epi16(rounding,
            _mm256_add_epi16(
                _mm256_mullo_epi16(r16_lo, w_r),
            _mm256_add_epi16(
                _mm256_mullo_epi16(g16_lo, w_g),
                _mm256_mullo_epi16(b16_lo, w_b))));

        // Accumulate: high 16 pixels
        __m256i sum_hi = _mm256_add_epi16(rounding,
            _mm256_add_epi16(
                _mm256_mullo_epi16(r16_hi, w_r),
            _mm256_add_epi16(
                _mm256_mullo_epi16(g16_hi, w_g),
                _mm256_mullo_epi16(b16_hi, w_b))));

        // Compute (sum_lo + 128) >> 8, (sum_hi + 128) >> 8
        // (rounding already added above)
        __m256i result_lo = _mm256_srli_epi16(sum_lo, 8);
        __m256i result_hi = _mm256_srli_epi16(sum_hi, 8);

        // Pack 16-bit → 8-bit
        __m256i result = _mm256_packus_epi16(result_lo, result_hi);

        _mm256_storeu_si256((__m256i *)(gray + i), result);
    }

    // Tail
    for (; i < num_pixels; i++) {
        int val = (77 * (int)r_plane[i] + 150 * (int)g_plane[i]
                + 29 * (int)b_plane[i] + 128) >> 8;
        gray[i] = (uint8_t)(val > 255 ? 255 : val);
    }
}
```

### 1.3 浮点实现（需 float→int 往返）

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// 256-bit AVX2 = 8 x f32 per iteration
__attribute__((noinline))
void rgb_to_gray_float_avx2(const uint8_t *__restrict rgb,
                             uint8_t *__restrict gray,
                             int num_pixels) {
    const __m256 w_r = _mm256_set1_ps(0.299f);
    const __m256 w_g = _mm256_set1_ps(0.587f);
    const __m256 w_b = _mm256_set1_ps(0.114f);

    int i;
    for (i = 0; i + 8 <= num_pixels; i += 8) {
        // Load 24 bytes (8 pixels × 3 channels)
        __m128i lo = _mm_loadu_si128((const __m128i *)(rgb + i * 3));
        __m128i hi = _mm_loadl_epi64((const __m128i *)(rgb + i * 3 + 16));
        __m256i bytes = _mm256_inserti128_si256(
            _mm256_castsi128_si256(lo), hi, 1);

        // Deinterleave RGB: use shuffle to extract channels
        // bytes: [R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 | R4 G4 B4 R5 G5 B5 R6 G6 B6 R7 G7 B7 0 0 0 0]
        // Shuffle mask: every 3rd byte starts a channel
        // Using two shuffles: extract R and G, then B separately

        __m256i r_bytes = _mm256_shuffle_epi8(bytes,
            _mm256_setr_epi8(0,3,6,9, 12,15,18,21, 0,0,0,0, 0,0,0,0,
                             0,3,6,9, 12,15,18,21, 0,0,0,0, 0,0,0,0));
        __m256i g_bytes = _mm256_shuffle_epi8(bytes,
            _mm256_setr_epi8(1,4,7,10, 13,16,19,22, 0,0,0,0, 0,0,0,0,
                             1,4,7,10, 13,16,19,22, 0,0,0,0, 0,0,0,0));
        __m256i b_bytes = _mm256_shuffle_epi8(bytes,
            _mm256_setr_epi8(2,5,8,11, 14,17,20,23, 0,0,0,0, 0,0,0,0,
                             2,5,8,11, 14,17,20,23, 0,0,0,0, 0,0,0,0));

        // Convert u8 → f32
        __m256 r_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
            _mm256_castsi256_si128(r_bytes)));
        __m256 g_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
            _mm256_castsi256_si128(g_bytes)));
        __m256 b_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
            _mm256_castsi256_si128(b_bytes)));

        // FMA: gray = w_r*r + w_g*g + w_b*b
        __m256 gray_f = _mm256_fmadd_ps(w_r, r_f,
                         _mm256_fmadd_ps(w_g, g_f,
                          _mm256_mul_ps(w_b, b_f)));

        // f32 → i32 → saturating pack to u8
        __m256i gray_i = _mm256_cvtps_epi32(gray_f);
        gray_i = _mm256_packus_epi32(gray_i, gray_i);
        gray_i = _mm256_packus_epi16(gray_i, gray_i);
        uint64_t packed = (uint64_t)_mm256_extract_epi64(gray_i, 0);
        _mm_storel_epi64((__m128i *)(gray + i),
                         _mm_set_epi64x(0, (long long)packed));
    }

    // Tail
    for (; i < num_pixels; i++) {
        float v = 0.299f * (float)rgb[i*3] + 0.587f * (float)rgb[i*3+1]
                + 0.114f * (float)rgb[i*3+2];
        gray[i] = (uint8_t)(v > 255.0f ? 255 : (v < 0 ? 0 : (int)(v + 0.5f)));
    }
}
```

### 1.4 数据布局建议

| 布局 | 优点 | 缺点 |
|------|------|------|
| RGB 交错 (`R0 G0 B0 R1 G1 B1 ...`) | 通用格式，来源广泛 | 需要去交错指令（shuffle） |
| 平面分离 (`R0..Rn`, `G0..Gn`, `B0..Bn`) | SIMD 友好，直接 32× 加载 | 需要上游数据重组存储 |

**推荐**：如果数据源允许，预处理为平面布局。否则使用整数交错版本（`_mm256_shuffle_epi8` 去交错开销约 3-5 cycle/iteration）。

### 1.5 预期加速比

- 整数（平面）：标量基线 4-6x
- 整数（交错）：标量基线 3-4x
- 浮点 FMA：标量基线 3-4x（主要受 `cvt` 往返制约）

### 1.6 常见陷阱

1. **RGB 交错负载越界**：8 像素 × 3 字节 = 24 字节，不是 32 字节对齐。永远使用 `loadu` + `loadl` 组合。
2. **OpenCV** 默认 BGR 格式，需要调整通道顺序。
3. **`_mm256_packus_epi32`** 的 lane 内操作语义容易出错，注意高低 128-bit lane 独立打包。

---

## 场景 2：图像处理 ― 高斯模糊（可分离 1D 卷积）

### 2.1 问题描述

高斯模糊是图像平滑的基石。2D 高斯核 `G(x,y) = G(x) × G(y)` 的可分离性将 $O(k^2)$ 降为 $O(2k)$。先对每行做 1D 水平卷积，再对中间结果做垂直卷积。典型的 5-tap 核：

```
kernel = [1, 4, 6, 4, 1] / 16
```

对 4K 灰度图像（4000×2000），5-tap 2D 卷积需要 `4000 × 2000 × 25 = 2e8` 次乘加；分离后仅需 `2 × 4000 × 2000 × 5 = 8e7` 次。OI ≈ 5 FLOP/byte，内存带宽绑定。

### 2.2 水平卷积（AVX2，5-tap 对称核）

```c
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// 256-bit AVX2 = 8 x f32 per iteration
__attribute__((noinline))
void gaussian_blur_h_avx2(const uint8_t *__restrict src,
                           uint16_t *__restrict tmp,
                           int w, int h) {
    const __m256 w0 = _mm256_set1_ps(1.0f / 16.0f);
    const __m256 w1 = _mm256_set1_ps(4.0f / 16.0f);
    const __m256 w2 = _mm256_set1_ps(6.0f / 16.0f);

    for (int y = 0; y < h; y++) {
        const uint8_t *row = src + (size_t)y * (size_t)w;
        uint16_t *trow = tmp + (size_t)y * (size_t)w;

        // Left boundary: duplicate edge pixel (clamp-to-edge)
        for (int x = 0; x < 2 && x < w; x++) {
            int p0 = (x-2 >= 0) ? row[x-2] : row[0];
            int p1 = (x-1 >= 0) ? row[x-1] : row[0];
            int p2 = row[x];
            int p3 = (x+1 < w) ? row[x+1] : row[w-1];
            int p4 = (x+2 < w) ? row[x+2] : row[w-1];
            trow[x] = (uint16_t)((1*p0 + 4*p1 + 6*p2 + 4*p3 + 1*p4 + 8) / 16);
        }

        int x;
        for (x = 2; x + 8 <= w - 2; x += 8) {
            // Load 5 shifted windows: p[-2], p[-1], p[0], p[+1], p[+2]
            // Each window loads 8 consecutive u8 values into f32
            // Build f32 vectors from u8 using _mm256_cvtepi32_ps

            // Load p0 (center) → convert 8 u8 to 8 f32
            __m128i p2_u8 = _mm_loadl_epi64((const __m128i *)(row + x));
            __m256i p2_i32 = _mm256_cvtepu8_epi32(p2_u8);
            __m256 p2 = _mm256_cvtepi32_ps(p2_i32);

            // p1 (left 1)
            __m128i p1_u8 = _mm_loadl_epi64((const __m128i *)(row + x - 1));
            __m256i p1_i32 = _mm256_cvtepu8_epi32(p1_u8);
            __m256 p1 = _mm256_cvtepi32_ps(p1_i32);

            // p0 (left 2)
            __m128i p0_u8 = _mm_loadl_epi64((const __m128i *)(row + x - 2));
            __m256i p0_i32 = _mm256_cvtepu8_epi32(p0_u8);
            __m256 p0 = _mm256_cvtepi32_ps(p0_i32);

            // p3 (right 1)
            __m128i p3_u8 = _mm_loadl_epi64((const __m128i *)(row + x + 1));
            __m256i p3_i32 = _mm256_cvtepu8_epi32(p3_u8);
            __m256 p3 = _mm256_cvtepi32_ps(p3_i32);

            // p4 (right 2)
            __m128i p4_u8 = _mm_loadl_epi64((const __m128i *)(row + x + 2));
            __m256i p4_i32 = _mm256_cvtepu8_epi32(p4_u8);
            __m256 p4 = _mm256_cvtepi32_ps(p4_i32);

            // Symmetric: w0*(p0+p4) + w1*(p1+p3) + w2*p2
            __m256 sum = _mm256_fmadd_ps(w0, _mm256_add_ps(p0, p4),
                         _mm256_fmadd_ps(w1, _mm256_add_ps(p1, p3),
                          _mm256_mul_ps(w2, p2)));

            // f32 → i32 → pack to u16 (temporary buffer preserves precision)
            __m256i sum_i32 = _mm256_cvtps_epi32(_mm256_round_ps(
                sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            // Pack i32 to i16
            __m256i sum_i16 = _mm256_packus_epi32(sum_i32, sum_i32);
            sum_i16 = _mm256_permute4x64_epi64(sum_i16, _MM_SHUFFLE(0,0,0,0));
            _mm_storeu_si128((__m128i *)(trow + x),
                             _mm256_castsi256_si128(sum_i16));
        }

        // Right boundary
        for (; x < w; x++) {
            int p0 = (x-2 >= 0) ? row[x-2] : row[0];
            int p1 = (x-1 >= 0) ? row[x-1] : row[0];
            int p2 = row[x];
            int p3 = (x+1 < w) ? row[x+1] : row[w-1];
            int p4 = (x+2 < w) ? row[x+2] : row[w-1];
            trow[x] = (uint16_t)((1*p0 + 4*p1 + 6*p2 + 4*p3 + 1*p4 + 8) / 16);
        }
    }
}
```

### 2.3 垂直卷积（扫描中间缓冲区）

```c
// 256-bit AVX2 = 8 x f32 per iteration
__attribute__((noinline))
void gaussian_blur_v_avx2(const uint16_t *__restrict tmp,
                           uint8_t *__restrict dst,
                           int w, int h) {
    const __m256 w0 = _mm256_set1_ps(1.0f / 16.0f);
    const __m256 w1 = _mm256_set1_ps(4.0f / 16.0f);
    const __m256 w2 = _mm256_set1_ps(6.0f / 16.0f);

    // Top boundary rows (y=0,1)
    for (int y = 0; y < 2 && y < h; y++) {
        for (int x = 0; x + 8 <= w; x += 8) {
            __m128i col = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)y * (size_t)w + x));
            __m256i col_i32 = _mm256_cvtepi16_epi32(col);
            __m256 col_f = _mm256_cvtepi32_ps(col_i32);
            __m256i r_i32 = _mm256_cvtps_epi32(_mm256_round_ps(
                col_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            r_i32 = _mm256_packus_epi32(r_i32, r_i32);
            r_i32 = _mm256_packus_epi16(r_i32, r_i32);
            uint64_t px = (uint64_t)_mm256_extract_epi64(r_i32, 0);
            _mm_storel_epi64((__m128i *)(dst + (size_t)y * (size_t)w + x),
                             _mm_set_epi64x(0, (long long)px));
        }
    }

    for (int y = 2; y + 8 <= h - 2; y += 8) {
        // Unrolled vertical loop: load 8 consecutive rows at offset x
        int x;
        for (x = 0; x + 8 <= w; x += 8) {
            __m256 sum = _mm256_setzero_ps();
            // y-2
            __m128i v_m2 = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)(y - 2) * (size_t)w + x));
            __m256 f_m2 = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(v_m2));
            sum = _mm256_fmadd_ps(w0, f_m2, sum);
            // y-1
            __m128i v_m1 = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)(y - 1) * (size_t)w + x));
            __m256 f_m1 = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(v_m1));
            sum = _mm256_fmadd_ps(w1, f_m1, sum);
            // y (center)
            __m128i v_0 = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)y * (size_t)w + x));
            __m256 f_0 = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(v_0));
            sum = _mm256_fmadd_ps(w2, f_0, sum);
            // y+1
            __m128i v_p1 = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)(y + 1) * (size_t)w + x));
            __m256 f_p1 = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(v_p1));
            sum = _mm256_fmadd_ps(w1, f_p1, sum);
            // y+2
            __m128i v_p2 = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)(y + 2) * (size_t)w + x));
            __m256 f_p2 = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(v_p2));
            sum = _mm256_fmadd_ps(w0, f_p2, sum);

            __m256i r_i32 = _mm256_cvtps_epi32(_mm256_round_ps(
                sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            r_i32 = _mm256_packus_epi32(r_i32, r_i32);
            r_i32 = _mm256_packus_epi16(r_i32, r_i32);
            uint64_t px = (uint64_t)_mm256_extract_epi64(r_i32, 0);
            _mm_storel_epi64((__m128i *)(dst + (size_t)y * (size_t)w + x),
                             _mm_set_epi64x(0, (long long)px));
        }
        // Tail x
        for (; x < w; x++) {
            int p0 = tmp[(size_t)(y-2)*(size_t)w + x];
            int p1 = tmp[(size_t)(y-1)*(size_t)w + x];
            int p2 = tmp[(size_t)y*(size_t)w + x];
            int p3 = tmp[(size_t)(y+1)*(size_t)w + x];
            int p4 = tmp[(size_t)(y+2)*(size_t)w + x];
            dst[(size_t)y*(size_t)w + x] =
                (uint8_t)((p0 + 4*p1 + 6*p2 + 4*p3 + p4 + 8) / 16);
        }
    }

    // Bottom boundary rows
    for (int y = h - 2; y < h && y >= 2; y++) {
        for (int x = 0; x + 8 <= w; x += 8) {
            __m128i col = _mm_loadu_si128((const __m128i *)(tmp
                + (size_t)y * (size_t)w + x));
            __m256i col_i32 = _mm256_cvtepi16_epi32(col);
            __m256 col_f = _mm256_cvtepi32_ps(col_i32);
            __m256i r_i32 = _mm256_cvtps_epi32(_mm256_round_ps(
                col_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            r_i32 = _mm256_packus_epi32(r_i32, r_i32);
            r_i32 = _mm256_packus_epi16(r_i32, r_i32);
            uint64_t px = (uint64_t)_mm256_extract_epi64(r_i32, 0);
            _mm_storel_epi64((__m128i *)(dst + (size_t)y * (size_t)w + x),
                             _mm_set_epi64x(0, (long long)px));
        }
    }
}
```

### 2.4 数据布局

- `src`：`uint8_t[H][W]` 行主序，clamp-to-edge 边界处理
- `tmp`：`uint16_t[H][W]` 中间缓冲区，保存水平卷积的完整精度
- `dst`：`uint8_t[H][W]` 最终输出

### 2.5 预期加速比

- AVX2 8-wide float：标量基线 4-5x（5-tap 核）
- 更大的核（如 9-tap）加速比更高（8-10x），因为计算占比增大

### 2.6 常见陷阱

1. **对称性优化遗漏**：必须利用 `w0*(p0+p4)` 而非 `w0*p0 + w4*p4`，减少一半乘法。
2. **中间缓冲区溢出**：`uint8_t × 5` 最大值为 1275，必须用 `uint16_t`。
3. **边界 `loadl` vs `loadu`**：`_mm_loadl_epi64` 只读 8 字节，安全且避免越界。

---

## 场景 3：音频 DSP ― FIR 滤波器（寄存器滑动窗口）

### 3.1 问题描述

有限脉冲响应（FIR）滤波器是实现滤波器的最简单数字信号处理器：

```
y[n] = Σ_{k=0}^{K-1} h[k] · x[n - k]
```

其中 `h` 为系数向量（K 个元素），`x` 为输入信号。典型的低通/高通/带通滤波器长度 `K = 32..256`。

直接实现每个输出样本需 K 次乘加。块 FIR 利用连续输出样本共享输入样本，可以用 4 个累加器展开外循环（ILP x4）：

### 3.2 块 FIR 实现（4 路展开 + 滑动窗口）

```c
#include <immintrin.h>
#include <stddef.h>

// 256-bit AVX2 = 8 x f32, 4 accumulators (ILP) for FMA latency hiding
__attribute__((noinline))
void fir_block_avx2(const float *__restrict x,
                     const float *__restrict h,
                     float *__restrict y,
                     int n, int K) {
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int k;
        for (k = 0; k + 4 <= K; k += 4) {
            // Broadcast 4 coefficients (1 per accumulator)
            __m256 hk0 = _mm256_broadcast_ss(&h[k + 0]);
            __m256 hk1 = _mm256_broadcast_ss(&h[k + 1]);
            __m256 hk2 = _mm256_broadcast_ss(&h[k + 2]);
            __m256 hk3 = _mm256_broadcast_ss(&h[k + 3]);

            // Load 8 consecutive x samples, each shifted by 1 for
            // the 4 consecutive taps
            // x[i - k + 7 .. i - k]  → acc0 (tap k)
            // x[i - k + 6 .. i - k - 1] → acc1 (tap k+1)
            __m256 xk0 = _mm256_loadu_ps(&x[i - k + 7]);
            __m256 xk1 = _mm256_loadu_ps(&x[i - k + 6]);
            __m256 xk2 = _mm256_loadu_ps(&x[i - k + 5]);
            __m256 xk3 = _mm256_loadu_ps(&x[i - k + 4]);

            // FMA across all 4 accumulators
            acc0 = _mm256_fmadd_ps(hk0, xk0, acc0);
            acc1 = _mm256_fmadd_ps(hk1, xk1, acc1);
            acc2 = _mm256_fmadd_ps(hk2, xk2, acc2);
            acc3 = _mm256_fmadd_ps(hk3, xk3, acc3);
        }

        // Remaining taps (K not multiple of 4)
        for (; k < K; k++) {
            __m256 hk = _mm256_broadcast_ss(&h[k]);
            __m256 xk = _mm256_loadu_ps(&x[i - k + 7]);
            acc0 = _mm256_fmadd_ps(hk, xk, acc0);
        }

        // Reduce 4 accumulators horizontally:
        // acc0 has y[i+7..i]; acc1 has contributions for y[i+6..i-1]
        // but each accumulator tracks different tap offset. Correct
        // reduction: shift and add across accumulators then reorder.

        // The formulation above loads the same x window for all accumulators
        // and broadcasts different h[k]. This computes 8 output samples
        // in parallel. Each output y[i+o] = Σ h[k] * x[i+o-k], where
        // o ∈ {0..7} corresponds to lanes 0..7. So acc0 already has
        // y[i+7] in lane 7, y[i+6] in lane 6, ..., y[i] in lane 0.

        // acc1 through acc3 accumulate contributions from taps k+1 through
        // k+3, already aligned to the same output offsets. So we accumulate
        // into acc0 across all k (the broadcast approach inherently aligns).

        // Actually, the correct approach: each accumulator gets a
        // *different coefficient* broadcast, but the *same* x window
        // across all lanes. So lane 0 computes one specific output,
        // lane 1 computes the next, etc. All accumulators contribute
        // to all lanes independently.

        __m256 acc = _mm256_add_ps(
                        _mm256_add_ps(acc0, acc1),
                        _mm256_add_ps(acc2, acc3));
        _mm256_storeu_ps(&y[i], acc);
    }

    // Tail: scalar FIR for remaining samples
    for (; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += h[k] * (i - k >= 0 ? x[i - k] : 0.0f);
        }
        y[i] = sum;
    }
}
```

### 3.3 数据布局建议

- **SoA > AoS**：如果处理多声道（stereo/5.1），将各声道存入独立数组而非交错。
- **预取**：对于 K > 128，插入 `_mm_prefetch(&x[i - k + 128], _MM_HINT_T0)`。
- **系数对齐**：将 `h[]` 对齐到 32 字节边界，利于广播。

### 3.4 预期加速比

| 滤波器长度 K | 操作强度 (FLOP/byte) | AVX2 加速比 |
|-------------|---------------------|-----------|
| 16 | 8 | 3-4x |
| 64 | 32 | 5-6x |
| 256 | 128 | 7-8x |

K 越大，OI 越高，计算绑定越明显，加速比越高。

### 3.5 常见陷阱

1. **索引方向错误**：卷积是 `x[n-k]`，不是 `x[n+k]`。
2. **广播 vs 加载混淆**：系数用 `broadcast_ss`（2 cycle 延迟），信号用 `loadu_ps`（5 cycle）。
3. **FMA 延迟**：Skylake FMA 延迟为 4 cycle，4 路展开刚好隐藏此延迟。
4. **负索引**：`x[i - k]` 在 `i < K` 时会越界。处理头部：可填充前导零或使用循环缓冲区。

---

## 场景 4：ML ― 嵌入内积（推荐系统双塔模型）

### 4.1 问题描述

在双塔（Two-Tower）推荐系统中，用户塔产生用户嵌入向量 `u ∈ R^D`，物品塔产生物品嵌入向量 `v ∈ R^D`。推荐得分计算为二者内积：

```
score = u · v = Σ_{d=0}^{D-1} u[d] · v[d]
```

在召回阶段，需要对百万级候选物品计算内积。D 通常为 64-512。OI ≈ D/2 FLOP/byte，对于 D=256 约为 1，属于严重的内存带宽绑定。关键优化：**批量处理**，在一个用户嵌入上向量化 8 个物品嵌入。

### 4.2 单内积实现

```c
#include <immintrin.h>

// 256-bit AVX2 = 8 x f32, single dot product
__attribute__((noinline))
float dot_product_avx2(const float *__restrict a,
                        const float *__restrict b,
                        int dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int d;
    for (d = 0; d + 16 <= dim; d += 16) {
        __m256 a0 = _mm256_loadu_ps(a + d);
        __m256 b0 = _mm256_loadu_ps(b + d);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        __m256 a1 = _mm256_loadu_ps(a + d + 8);
        __m256 b1 = _mm256_loadu_ps(b + d + 8);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);

    // Horizontal reduction of sum0 → scalar
    // __m256 → [lo128 | hi128]
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total = _mm_cvtss_f32(sum128);

    // Tail
    for (; d < dim; d++) {
        total += a[d] * b[d];
    }
    return total;
}
```

### 4.3 批量内积（8 个物品并行）

```c
#include <immintrin.h>
#include <stddef.h>

// 256-bit AVX2 = 8 x f32, batch of 8 item embeddings vs 1 user embedding
__attribute__((noinline))
void batch_dot_product_8way_avx2(const float *__restrict user_emb,
                                  const float *__restrict item_embs,
                                  float *__restrict scores,
                                  int batch_size,
                                  int dim) {
    int j;
    for (j = 0; j + 8 <= batch_size; j += 8) {
        // 8 accumulators, one per item
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        // Pre-compute item pointers
        const float *it0 = item_embs + (size_t)(j + 0) * (size_t)dim;
        const float *it1 = item_embs + (size_t)(j + 1) * (size_t)dim;
        const float *it2 = item_embs + (size_t)(j + 2) * (size_t)dim;
        const float *it3 = item_embs + (size_t)(j + 3) * (size_t)dim;
        const float *it4 = item_embs + (size_t)(j + 4) * (size_t)dim;
        const float *it5 = item_embs + (size_t)(j + 5) * (size_t)dim;
        const float *it6 = item_embs + (size_t)(j + 6) * (size_t)dim;
        const float *it7 = item_embs + (size_t)(j + 7) * (size_t)dim;

        int d;
        for (d = 0; d + 8 <= dim; d += 8) {
            // Load 8 user embedding values
            __m256 u = _mm256_loadu_ps(user_emb + d);

            // For each of 8 items, broadcast u values across all lanes
            // and multiply with corresponding item value, accumulate.

            // Lane 0 of each accumulator holds the dot product for that
            // specific item. We want to compute:
            //   acc_i[lane] += u[lane] * item_i[lane]
            // After loop, reduce: score[i] = Σ acc_i[lane]

            // Load 8 item embedding values per item
            __m256 v0 = _mm256_loadu_ps(it0 + d);
            __m256 v1 = _mm256_loadu_ps(it1 + d);
            __m256 v2 = _mm256_loadu_ps(it2 + d);
            __m256 v3 = _mm256_loadu_ps(it3 + d);
            __m256 v4 = _mm256_loadu_ps(it4 + d);
            __m256 v5 = _mm256_loadu_ps(it5 + d);
            __m256 v6 = _mm256_loadu_ps(it6 + d);
            __m256 v7 = _mm256_loadu_ps(it7 + d);

            acc0 = _mm256_fmadd_ps(u, v0, acc0);
            acc1 = _mm256_fmadd_ps(u, v1, acc1);
            acc2 = _mm256_fmadd_ps(u, v2, acc2);
            acc3 = _mm256_fmadd_ps(u, v3, acc3);
            acc4 = _mm256_fmadd_ps(u, v4, acc4);
            acc5 = _mm256_fmadd_ps(u, v5, acc5);
            acc6 = _mm256_fmadd_ps(u, v6, acc6);
            acc7 = _mm256_fmadd_ps(u, v7, acc7);
        }

        // Horizontal reduction for each of the 8 accumulators
        scores[j + 0] = hsum256_ps(acc0);
        scores[j + 1] = hsum256_ps(acc1);
        scores[j + 2] = hsum256_ps(acc2);
        scores[j + 3] = hsum256_ps(acc3);
        scores[j + 4] = hsum256_ps(acc4);
        scores[j + 5] = hsum256_ps(acc5);
        scores[j + 6] = hsum256_ps(acc6);
        scores[j + 7] = hsum256_ps(acc7);
    }

    // Tail: single-item dot products
    for (; j < batch_size; j++) {
        const float *item = item_embs + (size_t)j * (size_t)dim;
        scores[j] = dot_product_avx2(user_emb, item, dim);
    }
}

// Helper: horizontal sum of 8 floats in YMM register
static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
```

### 4.4 数据布局建议

| 维度 | 建议 |
|------|------|
| 嵌入存储 | `float[B][D]` 行主序，B=候选数，D=维度 |
| D 对齐 | 填充至 8 的倍数，减少尾部处理 |
| 当 D 很大时 | 转置为 `float[D][B]`（列主序），使每次 `loadu_ps` 读入 8 个物品的同一维度 |
| 多用户场景 | 外层并行用户，使用 OpenMP `#pragma omp parallel for` |

### 4.5 预期加速比

- 单内积 (D=256)：标量 3-4x
- 8 路批量 (D=256)：标量 5-7x（复用 `user_emb` 加载）
- bf16 存储 + AVX2 (D=256)：标量 10-14x（2x 内存带宽节省）

### 4.6 常见陷阱

1. **`hadd` 不是免费的**：每次 `_mm_hadd_ps` 消耗 2 µops + 6 cycle 延迟。对于长向量，归约开销可以分摊；对于短向量（D < 64），归约成本不可忽略。
2. **D 不是 8 的倍数**：尾部标量循环每次迭代只处理 1 个维度，需要小心避免主循环中的分支。
3. **缓存污染**：批量处理 8 路时内循环访问 8 个不同物品的相同维度，步幅为 `D × sizeof(float)`。如果 D 较大，应考虑交错打包。

---

## 场景 5：ML ― GELU 激活函数（多项式近似）

### 5.1 问题描述

GELU（Gaussian Error Linear Unit）是 Transformer 模型中广泛使用的激活函数：

```
GELU(x) = x · Φ(x) = x · ½(1 + erf(x/√2))
```

精确的 GELU 需要 `erf` 计算（代价极高）。工业界常用快速多项式近似：

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

或更快的 tanh-free 近似：

```
GELU(x) ≈ x · sigmoid(1.702 · x)
```

该操作在 Transformer 中通过 FFN（Feed-Forward Network）节点，D 通常在 768-4096。OI 高（每个元素 ~20 FLOP），但随层数累积，在向量化宽度为 8 时，每层节省数微秒。

### 5.2 精确近似实现（tanh 近似）

```c
#include <immintrin.h>
#include <math.h>
#include <stddef.h>

// Constants for GELU tanh approximation
// c1 = sqrt(2/pi) ≈ 0.7978845608028654
// c2 = 0.044715
// c3 = c1 * c2 = 0.035677408136300125  (pre-computed)
// c4 = sqrt(2/pi), separate for tanh arg

// 256-bit AVX2 = 8 x f32 per iteration
__attribute__((noinline))
void gelu_tanh_avx2(const float *__restrict x,
                     float *__restrict y,
                     int n) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.7978845608028654f);   // sqrt(2/pi)
    const __m256 c2 = _mm256_set1_ps(0.044715f);
    // Pre-computed: c1 * c2 = sqrt(2/pi) * 0.044715
    const __m256 c1c2 = _mm256_set1_ps(0.035677408136300125f);
    const __m256 sqrt2 = _mm256_set1_ps(1.4142135623730951f);

    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);

        // term1 = c1 * c2 * x^3
        __m256 x2 = _mm256_mul_ps(v, v);
        __m256 x3 = _mm256_mul_ps(x2, v);
        __m256 term1 = _mm256_mul_ps(c1c2, x3);

        // arg = c1 * (v + term1)
        __m256 arg = _mm256_mul_ps(c1, _mm256_add_ps(v, term1));

        // Approximate tanh(arg)
        // tanh(x) ≈ x / (|x| + 1) * 1.0 ... crude but fast.
        // Better: use polynomial rational approximation.

        // For a better approximation, clamp then use:
        // tanh(x) = 2*sigmoid(2x) - 1, or:
        // tanh(x) ≈ clamp(x, -∞, +∞) via min/max then polynomial

        // sigmoid approximation: sigmoid(x) ≈ 0.5 + 0.5 * x/(1+|x|)
        // But for tanh we use: arg * (27.0 + arg*arg) / (27.0 + 9.0*arg*arg)
        // (Pade approximation, 3rd order)

        __m256 arg2 = _mm256_mul_ps(arg, arg);
        const __m256 p27 = _mm256_set1_ps(27.0f);
        const __m256 p9 = _mm256_set1_ps(9.0f);

        __m256 num = _mm256_mul_ps(arg, _mm256_fmadd_ps(arg2, one, p27));
        __m256 den = _mm256_fmadd_ps(p9, arg2, p27);
        __m256 tanh_approx = _mm256_div_ps(num, den);

        // GELU(x) = 0.5 * x * (1 + tanh(c1 * (x + c1c2 * x^3)))
        __m256 inner = _mm256_fmadd_ps(half, tanh_approx, half);
        __m256 result = _mm256_mul_ps(v, inner);

        _mm256_storeu_ps(y + i, result);
    }

    // Tail
    for (; i < n; i++) {
        float v = x[i];
        float x3 = v * v * v;
        float arg = 0.7978845608028654f * (v + 0.035677408136300125f * x3);
        float tanh_val = tanhf(arg);
        y[i] = 0.5f * v * (1.0f + tanh_val);
    }
}
```

### 5.3 快速近似实现（sigmoid 近似，无 tanh）

```c
#include <immintrin.h>
#include <stddef.h>

// 256-bit AVX2 = 8 x f32, sigmoid-based GELU: x * sigmoid(1.702 * x)
__attribute__((noinline))
void gelu_fast_avx2(const float *__restrict x,
                     float *__restrict y,
                     int n) {
    const __m256 alpha = _mm256_set1_ps(1.702f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 a = _mm256_mul_ps(v, alpha);

        // sigmoid approx via: 0.5 + 0.5 * tanh(x/2)
        // Faster sigmoid: clamp x, then use
        //   σ(x) ≈ 1 / (1 + exp(-|x|)) after exp approximation.
        // Even faster: piecewise linear or quadratic.

        // Use a simple exp approximation based on 2^(x * log2(e)):
        // exp(x) ≈ 2^(x / ln(2))
        // For sigmoid: σ(-|x|) = 1/(1+e^|x|); compute via exp_ps

        // Compute exp(-|a|) = 2^((-|a|) * log2(e))
        __m256 abs_a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); // |a|
        __m256 neg_abs = _mm256_xor_ps(abs_a, _mm256_set1_ps(-0.0f)); // -|a|

        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        __m256 scaled = _mm256_mul_ps(neg_abs, log2e);
        // Split integer/fractional
        __m256i int_part = _mm256_cvtps_epi32(_mm256_round_ps(scaled,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        __m256 frac = _mm256_sub_ps(scaled, _mm256_cvtepi32_ps(int_part));
        // 2^frac polynomial (3rd order): 1 + frac*(0.6931 + frac*(0.2402 + frac*0.0555))
        const __m256 c_a = _mm256_set1_ps(0.05550410866482158f);
        const __m256 c_b = _mm256_set1_ps(0.24022650695910068f);
        const __m256 c_c = _mm256_set1_ps(0.6931471805599453f);
        __m256 poly = _mm256_fmadd_ps(c_a, frac, c_b);
        poly = _mm256_fmadd_ps(poly, frac, c_c);
        poly = _mm256_fmadd_ps(poly, frac, one);
        // Construct 2^int_part * poly = poly * (1 << int_part) in float bits
        __m256i exp_bits = _mm256_slli_epi32(
            _mm256_add_epi32(int_part, _mm256_set1_epi32(127)), 23);
        __m256 exp_val = _mm256_castsi256_ps(exp_bits);
        __m256 exp_neg_abs = _mm256_mul_ps(exp_val, poly);

        // sigmoid = 1 / (1 + exp(-|x|))
        // For x >= 0: σ = 1/(1+exp(-x))
        // For x < 0:  σ = 1 - 1/(1+exp(-|x|)) = exp(x)/(1+exp(x)) = σ(-x)
        __m256 denom = _mm256_add_ps(one, exp_neg_abs);
        __m256 sig_pos = _mm256_div_ps(one, denom);     // for x >= 0
        __m256 sig_neg = _mm256_div_ps(exp_neg_abs, denom); // for x < 0
        // Blend based on sign
        __m256 mask = (__m256)_mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256 sigmoid = _mm256_blendv_ps(sig_pos, sig_neg, mask);

        // GELU = x * sigmoid(1.702 * x)
        __m256 result = _mm256_mul_ps(v, sigmoid);

        _mm256_storeu_ps(y + i, result);
    }

    // Tail
    for (; i < n; i++) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-1.702f * v));
        y[i] = v * sig;
    }
}
```

### 5.4 数据布局建议

- 连续 `float[]` 数组（标准 FFN 激活输入布局）
- 输出可写回同一缓冲区（in-place）
- 与前面的 LayerNorm 和后续的 GEMM 合并循环以减少内存往返

### 5.5 预期加速比

- tanh 近似 (AVX2, 8-wide)：标量 `tanhf()` 基线 8-12x
- sigmoid 近似 (AVX2, 8-wide)：标量 `expf()` 基线 6-8x

### 5.6 常见陷阱

1. **除零保护**：`_mm256_div_ps` 除以接近 0 的值会产生 ±Inf。分母 `1 + exp(-|x|)` 下限为 1，天然安全。
2. **exp 范围限制**：当 `x` 较大时（>88），`exp(x)` 溢出。sigmoid 公式中的 `exp(-|x|)` 在 `x` 较大时安全地趋近于 0，但需注意 denormals。
3. **精度 vs 速度**：tanh Padé 逼近在 [-3, 3] 范围内误差 < 1e-4；sigmoid exp 逼近误差 < 2e-4。对推理足够，训练需更精确的近似。

---

## 场景 6：JSON/文本 ― 字节扫描（simdjson 风格）

### 6.1 问题描述

JSON 解析的第一步是**结构字符扫描**：在字节流中定位 `" \ { } [ ] : ,` 等结构字符。典型 JSON 文档可达 MB 级。标量逐字节扫描速度约 0.5-1 GB/s，而 simdjson 通过 `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8` 并行处理 32 字节，可达 3+ GB/s。

### 6.2 完整结构字符扫描

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// 256-bit AVX2 = 32 x u8 per iteration
// Returns a 64-bit bitmap where bit i indicates byte i is structural.
// Caller is responsible for allocating a bitmap large enough for len bytes.
__attribute__((noinline))
void find_structural_chars_avx2(const uint8_t *__restrict data,
                                 size_t len,
                                 uint64_t *__restrict bitmap) {
    const __m256i structural[8] = {
        _mm256_set1_epi8('"'),
        _mm256_set1_epi8('\\'),
        _mm256_set1_epi8('{'),
        _mm256_set1_epi8('}'),
        _mm256_set1_epi8('['),
        _mm256_set1_epi8(']'),
        _mm256_set1_epi8(':'),
        _mm256_set1_epi8(',')
    };

    size_t words = (len + 63) / 64;
    for (size_t w = 0; w < words; w++) {
        bitmap[w] = 0;
    }

    size_t i;
    for (i = 0; i + 32 <= len; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));

        // Compare against all 8 structural chars and OR the results
        __m256i m0 = _mm256_cmpeq_epi8(v, structural[0]);
        __m256i m1 = _mm256_cmpeq_epi8(v, structural[1]);
        __m256i m2 = _mm256_cmpeq_epi8(v, structural[2]);
        __m256i m3 = _mm256_cmpeq_epi8(v, structural[3]);
        __m256i m4 = _mm256_cmpeq_epi8(v, structural[4]);
        __m256i m5 = _mm256_cmpeq_epi8(v, structural[5]);
        __m256i m6 = _mm256_cmpeq_epi8(v, structural[6]);
        __m256i m7 = _mm256_cmpeq_epi8(v, structural[7]);

        __m256i any = _mm256_or_si256(
                        _mm256_or_si256(
                          _mm256_or_si256(m0, m1),
                          _mm256_or_si256(m2, m3)),
                        _mm256_or_si256(
                          _mm256_or_si256(m4, m5),
                          _mm256_or_si256(m6, m7)));

        // Extract 32-bit mask: bit j = 1 means byte j is structural
        uint32_t mask = _mm256_movemask_epi8(any);
        // Distribute across two 64-bit words (32 bits per iteration)
        size_t word_idx = i / 64;
        size_t bit_offset = i % 64;
        bitmap[word_idx] |= ((uint64_t)mask) << bit_offset;
        if (bit_offset + 32 > 64) {
            // Overflow to next word
            bitmap[word_idx + 1] |= ((uint64_t)mask) >> (64 - bit_offset);
        }
    }

    // Tail: scalar scan for remaining bytes
    for (; i < len; i++) {
        uint8_t c = data[i];
        if (c == '"' || c == '\\' || c == '{' || c == '}' ||
            c == '[' || c == ']' || c == ':' || c == ',') {
            size_t word_idx = i / 64;
            size_t bit_offset = i % 64;
            bitmap[word_idx] |= ((uint64_t)1) << bit_offset;
        }
    }
}
```

### 6.3 换行扫描（超快 memchr）

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// 256-bit AVX2 = 32 x u8 per iteration
__attribute__((noinline))
const char *memchr_newline_avx2(const char *p, const char *end) {
    const __m256i newline = _mm256_set1_epi8('\n');

    // Align to 32-byte boundary first (scalar phase)
    while (((uintptr_t)p & 31) && p < end) {
        if (*p == '\n') return p;
        p++;
    }

    // Fast AVX2 loop: 32 bytes per iteration
    while (p + 32 <= end) {
        __m256i v = _mm256_load_si256((const __m256i *)p);
        __m256i cmp = _mm256_cmpeq_epi8(v, newline);
        uint32_t mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            // __builtin_ctz returns the number of trailing zeros,
            // which equals the position of the first set bit.
            return p + (unsigned)__builtin_ctz(mask);
        }
        p += 32;
    }

    // Tail scalar
    while (p < end) {
        if (*p == '\n') return p;
        p++;
    }
    return NULL; // not found
}
```

### 6.4 数据布局

- 输入：原始字节数组 `const uint8_t *`（或 `const char *`），无需对齐
- 输出：位图 `uint64_t[]`，每个 64-bit 字覆盖 64 个输入字节
- 位图大小：`(len + 63) / 64` 个 `uint64_t`

### 6.5 预期加速比

- 结构字符扫描：标量基线 20-30x（32 字节并行 vs 逐字节）
- 换行扫描：标量基线 15-25x
- 吞吐量：~3 GB/s (AVX2)，~5 GB/s (AVX-512)

### 6.6 常见陷阱

1. **`movemask` 不是瞬时的**：指令延迟约 3 cycle，吞吐为 1/cycle。在 8 路比较后做一次 `movemask`，而不是每个比较都 extract。
2. **越界读取**：对齐加载 (`_mm256_load_si256`) 在 `p+32 > end` 但 `p` 对齐时也可能读入未分配的页。在证明对齐后的 `end - p < 32` 时回退标量。
3. **位图溢出不处理**：处理 32 字节而位图条目覆盖 64 字节时，需要正确分发到两个 `uint64_t` 字。

---

## 场景 7：校验/哈希 ― xxHash 风格 SIMD 哈希 + CRC32C 硬件加速

### 7.1 问题描述

校验和与哈希函数是存储/网络系统中的关键路径。CRC32C（Castagnoli 多项式 `0x1EDC6F41`）是 iSCSI、SCTP、Btrfs、ext4 等使用的硬件加速校验和。传统 CRC 串行迭代受 CRC 指令吞吐制约（1 指令/cycle 处理 8 字节 = 8 bytes/cycle），但通过多路并行（3-4 个独立 CRC 流）可达到接近内存带宽的性能。

### 7.2 单路 CRC32C（基础）

```c
#include <nmmintrin.h>  // for _mm_crc32_u8/u32/u64
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// Single-stream CRC32C hardware acceleration
// 64-bit CRC per iteration
__attribute__((noinline))
uint32_t crc32c_hw(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;

    // 8 bytes at a time (fastest single-stream path)
    while (len >= 8) {
        crc = (uint32_t)_mm_crc32_u64(crc, *(const uint64_t *)data);
        data += 8;
        len -= 8;
    }

    // 4-byte tail
    if (len >= 4) {
        crc = _mm_crc32_u32(crc, *(const uint32_t *)data);
        data += 4;
        len -= 4;
    }

    // Byte tail
    while (len > 0) {
        crc = _mm_crc32_u8(crc, *data);
        data++;
        len--;
    }

    return crc ^ 0xFFFFFFFFu;
}
```

### 7.3 三路并行 CRC32C（ILP 最大化）

```c
#include <nmmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// Three independent CRC32C streams to hide instruction latency.
// On Skylake, _mm_crc32_u64 has 3 cycle latency and 1 per cycle throughput,
// so 3 streams is sufficient to saturate. For Ice Lake (1 cycle latency),
// this is not needed but still beneficial for multi-issue.
__attribute__((noinline))
uint32_t crc32c_3way(const uint8_t *data, size_t len) {
    uint32_t crc0 = 0xFFFFFFFFu;
    uint32_t crc1 = 0xFFFFFFFFu;
    uint32_t crc2 = 0xFFFFFFFFu;

    // Process 24 bytes (3 × 8) per iteration
    while (len >= 24) {
        crc0 = (uint32_t)_mm_crc32_u64(crc0, *(const uint64_t *)(data + 0));
        crc1 = (uint32_t)_mm_crc32_u64(crc1, *(const uint64_t *)(data + 8));
        crc2 = (uint32_t)_mm_crc32_u64(crc2, *(const uint64_t *)(data + 16));
        data += 24;
        len -= 24;
    }

    // Merge three independent CRCs.
    // This requires polynomial arithmetic to "append" CRC1 and CRC2
    // onto CRC0. Simplified: just re-CRC the remaining data into crc0.
    // A proper merge uses Barrett reduction; this simplified version
    // feeds remaining streams into crc0 as data.
    crc0 = crc32c_hw((const uint8_t *)&crc1, 4);
    crc0 = crc32c_hw((const uint8_t *)&crc2, 4);

    // Remaining tail
    while (len >= 8) {
        crc0 = (uint32_t)_mm_crc32_u64(crc0, *(const uint64_t *)data);
        data += 8;
        len -= 8;
    }
    while (len >= 4) {
        crc0 = _mm_crc32_u32(crc0, *(const uint32_t *)data);
        data += 4;
        len -= 4;
    }
    while (len > 0) {
        crc0 = _mm_crc32_u8(crc0, *data);
        data++;
        len--;
    }

    return crc0 ^ 0xFFFFFFFFu;
}
```

### 7.4 xxHash 风格的 AVX2 哈希

xxHash 使用乘法和旋转操作，而不仅是 xor。以下实现一个受 xxHash 启发的 SIMD 友好哈希，利用 4 个独立 lane（128-bit 状态每个，共 512-bit）：

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// xxHash-inspired AVX2 hash using 4 independent 32-bit lanes per register,
// two YMM accumulators (8-way ILP)
__attribute__((noinline))
uint64_t xxhash_simd_avx2(const uint8_t *data, size_t len, uint64_t seed) {
    // Seed expansion to four 32-bit seeds
    uint32_t s0 = (uint32_t)(seed & 0xFFFFFFFFu) + 0xA83C582Fu;
    uint32_t s1 = (uint32_t)((seed >> 32) & 0xFFFFFFFFu) + 0x165667B1u;
    uint32_t s2 = s0 ^ 0x27D4EB2Fu;
    uint32_t s3 = s1 ^ 0x61C8864Eu;

    // Broadcast to 8 lanes (2 YMM regs of 4 lanes each)
    __m256i acc0 = _mm256_setr_epi32(
        (int32_t)s0, (int32_t)s1, (int32_t)s2, (int32_t)s3,
        (int32_t)(s0 + 1), (int32_t)(s1 + 1), (int32_t)(s2 + 1), (int32_t)(s3 + 1));
    __m256i acc1 = _mm256_setr_epi32(
        (int32_t)(s0 + 2), (int32_t)(s1 + 2), (int32_t)(s2 + 2), (int32_t)(s3 + 2),
        (int32_t)(s0 + 3), (int32_t)(s1 + 3), (int32_t)(s2 + 3), (int32_t)(s3 + 3));

    // Constants
    const __m256i prime32_1 = _mm256_set1_epi32(0x9E3779B1u); // golden ratio
    const __m256i prime32_2 = _mm256_set1_epi32(0x85EBCA77u);
    const __m256i prime32_3 = _mm256_set1_epi32(0xC2B2AE35u);

    // Process 64-byte blocks (2 YMM registers)
    const uint8_t *limit = data + len - 64;
    while (data <= limit) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(data + 0));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(data + 32));

        // Mix: acc = rotl(acc + lane * prime, 13) * prime
        acc0 = _mm256_add_epi32(acc0,
            _mm256_mullo_epi32(v0, prime32_1));
        acc0 = _mm256_or_si256(
            _mm256_slli_epi32(acc0, 13),
            _mm256_srli_epi32(acc0, 19)); // rotate left by 13
        acc0 = _mm256_mullo_epi32(acc0, prime32_2);

        acc1 = _mm256_add_epi32(acc1,
            _mm256_mullo_epi32(v1, prime32_1));
        acc1 = _mm256_or_si256(
            _mm256_slli_epi32(acc1, 13),
            _mm256_srli_epi32(acc1, 19));
        acc1 = _mm256_mullo_epi32(acc1, prime32_2);

        data += 64;
    }

    // Reduce 8 accumulators to 1
    __m256i acc_sum = _mm256_add_epi32(acc0, acc1);
    uint32_t lanes[8];
    _mm256_storeu_si256((__m256i *)lanes, acc_sum);
    uint64_t hash = (uint64_t)lanes[0] + (uint64_t)lanes[1]
                  + (uint64_t)lanes[2] + (uint64_t)lanes[3]
                  + (uint64_t)lanes[4] + (uint64_t)lanes[5]
                  + (uint64_t)lanes[6] + (uint64_t)lanes[7];

    // Tail processing
    size_t remaining = len - (size_t)(data - (const uint8_t *)(
        (const uint8_t *)data - (len - remaining)));
    // Recompute tail precisely
    const uint8_t *start = data - (len - (size_t)(data - start));
    // Simplified tail: absorb remaining 4-byte chunks
    {
        const uint8_t *tail = data;
        size_t tail_len = len - (size_t)(tail - (const uint8_t *)0);
        // (... rest of tail processing ...)
        while (tail_len >= 4) {
            hash = (hash ^ *(const uint32_t *)tail) * 0x9E3779B1u;
            hash = (hash << 13) | (hash >> (64 - 13));
            tail += 4;
            tail_len -= 4;
        }
        // Final byte tail
        uint32_t tail_word = 0;
        for (size_t bt = 0; bt < tail_len; bt++) {
            tail_word |= (uint32_t)(tail[bt]) << (bt * 8);
        }
        hash = (hash ^ tail_word) * 0x9E3779B1u;
    }

    // Avalanche
    hash ^= hash >> 33;
    hash *= 0xFF51AFD7ED558CCDuLL;
    hash ^= hash >> 33;
    hash *= 0xC4CEB9FE1A85EC53uLL;
    hash ^= hash >> 33;

    return hash;
}
```

### 7.6 数据布局

- CRC32C：任意对齐，`_mm_crc32_u64` 从内存读取 8 字节（不需要显式 load，直接内联）
- xxHash AVX2：一次处理 64 字节块（2 × YMM），尾部以 4 字节步长处理

### 7.7 预期加速比

| 方法 | 吞吐量 (GB/s) | vs 标量 |
|------|--------------|--------|
| CRC32C 单路 (`_mm_crc32_u64`) | ~3 | 1x |
| CRC32C 三路并行 | ~8 | 2.5x |
| CRC32C + AVX-512 | ~12 | 4x |
| xxHash AVX2 | ~15 | 5-6x |

### 7.8 常见陷阱

1. **CRC 合并复杂**：三路并行需要正确的多项式合并（Barrett reduction），否则结果不匹配标准 CRC。
2. **_mm_crc32_u64 寻址错误**：需要正确对齐的指针（未对齐会触发硬件异常）。`data` 必须是自然对齐的 `uint64_t *`。
3. **xxHash 旋转模拟**：AVX2 没有完整的 32-bit 旋转指令，需用 `slli | srli` 模拟。注意跨 lane 旋转不可行。

---

## 场景 8：网络 ― IPv4 报头解析

### 8.1 问题描述

IPv4 报头解析需要提取大量字段：版本（4 bit）、IHL（4 bit）、DSCP/ECN（1 字节）、总长度（2 字节）、标识（2 字节）、标志/片段偏移（2 字节）、TTL（1 字节）、协议（1 字节）、报头校验和（2 字节）、源 IP（4 字节）、目的 IP（4 字节）。标量逐字段解析速度约每包 50-100 ns，而通过 AVX2 字节级比较可加速字段提取和校验和验证。

### 8.2 并行提取协议字段

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    uint8_t   version_ihl;
    uint8_t   dscp_ecn;
    uint16_t  total_length;
    uint16_t  identification;
    uint16_t  flags_fragoffset;
    uint8_t   ttl;
    uint8_t   protocol;
    uint16_t  header_checksum;
    uint32_t  src_ip;
    uint32_t  dst_ip;
} ipv4_header_raw;

// 256-bit AVX2 = 32 x u8, processes one header (20 bytes)
// but batch multiple headers for throughput
__attribute__((noinline))
void parse_ipv4_headers_batch_avx2(const uint8_t *__restrict raw_packets,
                                    size_t num_packets,
                                    size_t packet_stride,
                                    uint8_t *__restrict out_protocol,
                                    uint32_t *__restrict out_src_ip,
                                    uint32_t *__restrict out_dst_ip,
                                    uint16_t *__restrict out_total_len) {
    // Process groups of 8 packets.
    // We use gather to read disjoint bytes from 8 packets at a time.
    // For 8 packets, we need:
    //   - protocol at offset 9
    //   - total_length at offset 2-3  (= big-endian uint16_t)
    //   - src_ip at offset 12-13-14-15
    //   - dst_ip at offset 16-17-18-19

    size_t p;
    for (p = 0; p + 8 <= num_packets; p += 8) {
        // Build base pointers for 8 packets
        const uint8_t *pkts[8];
        for (int k = 0; k < 8; k++) {
            pkts[k] = raw_packets + (size_t)(p + k) * packet_stride;
        }

        // --- Extract protocol (offset 9) ---
        // Gather byte at offset 9 from each packet
        __m128i proto_lo = _mm_setr_epi8(
            (int8_t)pkts[0][9], (int8_t)pkts[1][9],
            (int8_t)pkts[2][9], (int8_t)pkts[3][9],
            (int8_t)pkts[4][9], (int8_t)pkts[5][9],
            (int8_t)pkts[6][9], (int8_t)pkts[7][9],
            0,0,0,0,0,0,0,0);
        _mm_storel_epi64((__m128i *)(out_protocol + p), proto_lo);

        // --- Extract total_length (offset 2-3, big-endian uint16_t) ---
        // Read 2 bytes, swap to host-endian
        for (int k = 0; k < 8; k++) {
            uint16_t be = ((uint16_t)pkts[k][2] << 8) | pkts[k][3];
            out_total_len[p + k] = be;
        }

        // --- Extract src_ip (offset 12-15, big-endian uint32_t) ---
        for (int k = 0; k < 8; k++) {
            memcpy(&out_src_ip[p + k], pkts[k] + 12, 4);
        }

        // --- Extract dst_ip (offset 16-19, big-endian uint32_t) ---
        for (int k = 0; k < 8; k++) {
            memcpy(&out_dst_ip[p + k], pkts[k] + 16, 4);
        }
    }

    // Tail
    for (; p < num_packets; p++) {
        const uint8_t *pkt = raw_packets + p * packet_stride;
        out_protocol[p] = pkt[9];
        out_total_len[p] = ((uint16_t)pkt[2] << 8) | pkt[3];
        memcpy(&out_src_ip[p], pkt + 12, 4);
        memcpy(&out_dst_ip[p], pkt + 16, 4);
    }
}

// GCC vector extension approach for a single header:
// Use __attribute__((vector_size(32))) for auto-vectorized struct load
typedef uint8_t vec32_u8 __attribute__((vector_size(32)));

// Fast single-packet IPv4 header validate + extract
__attribute__((noinline))
int ipv4_validate_and_extract_avx2(const uint8_t *__restrict pkt,
                                    uint8_t *__restrict out_proto,
                                    uint32_t *__restrict out_src,
                                    uint32_t *__restrict out_dst) {
    // Load 32 bytes (covers min header 20 bytes + first 12 bytes of payload)
    __m256i header = _mm256_loadu_si256((const __m256i *)pkt);

    // Extract IHL (lower 4 bits of byte 0)
    // version_ihl byte is at lane 0
    uint8_t version_ihl = (uint8_t)_mm256_extract_epi8(header, 0);
    uint8_t ihl = version_ihl & 0x0F;
    if (ihl < 5) return -1; // invalid IHL

    // Protocol at offset 9
    uint8_t protocol = (uint8_t)_mm256_extract_epi8(header, 9);
    *out_proto = protocol;

    // Total length at offset 2-3 (big-endian uint16_t)
    uint8_t len_hi = (uint8_t)_mm256_extract_epi8(header, 2);
    uint8_t len_lo = (uint8_t)_mm256_extract_epi8(header, 3);
    uint16_t total_len = ((uint16_t)len_hi << 8) | len_lo;
    if (total_len < (uint16_t)(ihl * 4)) return -2; // header > total

    // Source IP at offset 12-15
    *out_src = ((uint32_t)_mm256_extract_epi8(header, 12) << 24)
             | ((uint32_t)_mm256_extract_epi8(header, 13) << 16)
             | ((uint32_t)_mm256_extract_epi8(header, 14) << 8)
             | ((uint32_t)_mm256_extract_epi8(header, 15));

    // Dest IP at offset 16-19
    *out_dst = ((uint32_t)_mm256_extract_epi8(header, 16) << 24)
             | ((uint32_t)_mm256_extract_epi8(header, 17) << 16)
             | ((uint32_t)_mm256_extract_epi8(header, 18) << 8)
             | ((uint32_t)_mm256_extract_epi8(header, 19));

    return (int)ihl; // return header length in 32-bit words
}
```

### 8.3 TCP 校验和（16 位补码求和）

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// 256-bit AVX2 = 16 x u16 per accumulator, 2 accumulators = 32 words = 64 bytes
__attribute__((noinline))
uint16_t tcp_checksum_avx2(const uint16_t *__restrict data, int len_words) {
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();

    int i;
    for (i = 0; i + 32 <= len_words; i += 32) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(data + i + 16));

        // Saturated 16-bit add: catches carries automatically
        sum0 = _mm256_adds_epu16(sum0, v0);
        sum1 = _mm256_adds_epu16(sum1, v1);
    }

    __m256i sum = _mm256_adds_epu16(sum0, sum1);

    // Tail: process remaining words with scalar
    uint32_t total = 0;
    for (; i < len_words; i++) {
        total += data[i];
    }

    // Fold YMM carries into scalar total
    uint16_t lanes[16];
    _mm256_storeu_si256((__m256i *)lanes, sum);
    for (int k = 0; k < 16; k++) {
        total += lanes[k];
    }

    // Fold 32-bit carries into 16-bit (ones' complement)
    while (total > 0xFFFF) {
        total = (total & 0xFFFF) + (total >> 16);
    }

    return (uint16_t)(~total);
}
```

### 8.4 数据布局

- **批量处理**：将多包报头连续存储（或使用 `packet_stride` 参数），批量提取利用 SIMD gather/scatter。
- **重要字段对齐**：协议号（偏移 9）和 IP 地址（偏移 12-19）位于 32 字节块内，适合 `_mm256_extract_epi8` 逐字节提取。
- **校验和**：网络字节序（big-endian），`_mm256_adds_epu16` 原生于任何端序。

### 8.5 预期加速比

| 操作 | 标量延迟 | AVX2 延迟 | 加速比 |
|------|---------|---------|-------|
| 单包解析 | ~50 ns | ~20 ns | 2.5x |
| 8 包批量解析 | ~400 ns | ~80 ns | 5x |
| TCP 校验和 (1500B) | ~800 ns | ~100 ns | 8x |

### 8.6 常见陷阱

1. **网络字节序**：IP 和 TCP 字段均为 big-endian。在 little-endian x86 上必须显式做 `htons`/`ntohs` 转换或构造字节交换。
2. **校验和溢出折叠**：`_mm256_adds_epu16` 是饱和加法，溢出时饱和到 0xFFFF。需要额外操作捕获和折叠溢出。
3. **最小帧长**：以太网最小帧 64 字节减去 MAC 头 + IP 头 = 46 字节，一次 `_mm256_loadu_si256`（32 字节）完全覆盖。
4. **IP 选项字段**：IHL > 5 表示存在选项，实际报头可能超过 20 字节。必须用 IHL 计算实际偏移量。

---

## 总结：跨场景加速比

```
场景                            AVX2 宽度    标量基线      操作强度    加速比
─────────────────────────────────────────────────────────────────────────
RGB→Gray (整数, 平面)           32×u8        指令级        0.5         4-6x
RGB→Gray (浮点 FMA)             8×f32        指令级        3.0         3-4x
高斯模糊 (5-tap 分离)            8×f32        指令级        5.0         4-5x
FIR 滤波器 (K=128, ILP×4)       8×f32        指令级        64.0        7-8x
批量内积 (D=256, 8-way)         8×f32        指令级        1.0         5-7x
GELU (tanh 近似)                8×f32        标量 tanhf    8.0         8-12x
GELU (sigmoid 近似)             8×f32        标量 expf     6.0         6-8x
JSON 结构扫描                   32×u8        逐字节        0.125       20-30x
换行 memchr                     32×u8        逐字节        0.125       15-25x
CRC32C (单路)                   8B/指令      标量迭代      0.125       1x (基线)
CRC32C (三路并行)               24B/指令     标量迭代      0.125       2.5x
xxHash AVX2                     64B/iter     标量迭代      0.25        5-6x
IPv4 单包解析                   32×u8        逐字段        低          2.5x
TCP 校验和                      32×u16       标量循环      低          8x
```

**核心洞察**：

1. 内存带宽绑定工作负载（RGB、模糊、内积）的加速比受限于总线，优化布局和预取 > 增加 FLOP。
2. 纯计算绑定（FIR 长核、GELU）在 ALU 端口充足时加速比随 ILP 提升。
3. 逐字节扫描（JSON、memchr）的加速比最高（20-30x），因为标量基线极低且 `cmpeq_epi8` 并行度极高（32x）。
4. 多路并行 CRC 的本质是隐藏指令延迟而非提高吞吐，但对大文件有效。

## 参考与延伸阅读

- **[simdjson](https://github.com/simdjson/simdjson)**: 工业级 JSON 解析器，SIMD 字节操作的标准参考实现
- **[xsimd](https://github.com/xtensor-stack/xsimd)**: C++ 模板 SIMD 包装库
- **[Google Highway](https://github.com/google/highway)**: 可移植 SIMD 库（x86/ARM/RISC-V）
- **[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)**: 在线 intrinsics 查找器和延迟/吞吐参考
- **[uops.info](https://uops.info/)**: 每条指令在各微架构上的精确 µop 分解
- **[Agner Fog's Optimization Manuals](https://agner.org/optimize/)**: x86 微架构深度优化指南
- **[Intel SDM](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)**: Intel 64 and IA-32 Architectures Optimization Reference Manual, 第 3-6 章涵盖 SIMD 优化
