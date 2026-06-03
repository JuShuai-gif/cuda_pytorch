# x86 SIMD 工业级实战案例

## 1. 图像处理

### 1.1 RGB 转灰度

```c
// Gray = 0.299*R + 0.587*G + 0.114*B
// 利用 FMA 指令：Gray = (0.299*R + 0.587*G) + 0.114*B
void rgb_to_gray_avx2(const uint8_t* rgb, uint8_t* gray, int num_pixels) {
    const __m256 w_r = _mm256_set1_ps(0.299f);
    const __m256 w_g = _mm256_set1_ps(0.587f);
    const __m256 w_b = _mm256_set1_ps(0.114f);

    for (int i = 0; i + 8 <= num_pixels; i += 8) {
        // Load 8 pixels × 3 bytes each = 24 bytes → unpack to floats
        // rgb layout: R0 G0 B0 R1 G1 B1 R2 G2 B2 ...
        __m128i rgb_low = _mm_loadu_si128((__m128i*)(rgb + i * 3));
        __m128i rgb_hi  = _mm_loadl_epi64((__m128i*)(rgb + i * 3 + 16));

        // Unpack bytes to 16-bit, then to 32-bit
        __m256i r = /* extract R channel */ ...;
        __m256i g = /* extract G channel */ ...;
        __m256i b = /* extract B channel */ ...;

        // Convert to float
        __m256 rf = _mm256_cvtepi32_ps(r);
        __m256 gf = _mm256_cvtepi32_ps(g);
        __m256 bf = _mm256_cvtepi32_ps(b);

        // FMA: gray = w_r*r + (w_g*g + w_b*b)
        __m256 gray_f = _mm256_fmadd_ps(w_r, rf,
                         _mm256_fmadd_ps(w_g, gf, _mm256_mul_ps(w_b, bf)));

        // Convert back to uint8_t and pack
        __m256i gray_i = _mm256_cvtps_epi32(gray_f);
        // ... pack and store ...
    }
}

// Optimized short version using integer arithmetic (no float conversion)
void rgb_to_gray_int_avx2(const uint8_t* rgb, uint8_t* gray, int n) {
    // Pre-scaled weights: 77, 150, 29 (scaled by 256)
    // Gray = (77*R + 150*G + 29*B) >> 8
    for (int i = 0; i + 32 <= n; i += 32) {
        // More complex but avoids float conversion overhead
        // Use _mm256_maddubs_epi16 for u8*s8 multiply-accumulate
    }
}
```

### 1.2 高斯模糊（可分离 1D 卷积）

2D 高斯模糊可分离为两个 1D 卷积，将 O(k²) 降为 O(2k)：

```c
// 5-tap Gaussian blur: weights = [1, 4, 6, 4, 1] / 16
void gaussian_blur_h_avx2(const uint8_t* src, uint8_t* dst, int w, int h) {
    __m256 w0 = _mm256_set1_ps(1.0f / 16.0f);
    __m256 w1 = _mm256_set1_ps(4.0f / 16.0f);
    __m256 w2 = _mm256_set1_ps(6.0f / 16.0f);

    for (int y = 0; y < h; y++) {
        const uint8_t* row = src + y * w;
        for (int x = 0; x + 8 <= w; x += 8) {
            // Load 5 shifted windows (8 pixels each)
            __m256 p0 = loadu8_to_f32(row + x);
            __m256 p1 = loadu8_to_f32(row + x + 1);
            __m256 p2 = loadu8_to_f32(row + x + 2);
            __m256 p3 = loadu8_to_f32(row + x + 3);
            __m256 p4 = loadu8_to_f32(row + x + 4);

            // Symmetric: result = w0*(p0+p4) + w1*(p1+p3) + w2*p2
            __m256 sum = _mm256_fmadd_ps(w0, _mm256_add_ps(p0, p4),
                        _mm256_fmadd_ps(w1, _mm256_add_ps(p1, p3),
                         _mm256_mul_ps(w2, p2)));
            // ... store sum as uint8 ...
        }
    }
}
```

### 1.3 双线性图像缩放（Bilinear Resize）

```c
// Bilinear interpolation: v = (1-dx)*(1-dy)*p00 + dx*(1-dy)*p10 + (1-dx)*dy*p01 + dx*dy*p11
void bilinear_resize_avx2(const uint8_t* src, int sw, int sh,
                           uint8_t* dst, int dw, int dh) {
    float x_ratio = (float)(sw - 1) / dw;
    float y_ratio = (float)(sh - 1) / dh;

    for (int y = 0; y < dh; y++) {
        float sy = y * y_ratio;
        int sy_int = (int)sy;
        float dy = sy - sy_int;

        for (int x = 0; x + 8 <= dw; x += 8) {
            float sx_arr[8];
            int sx_int_arr[8];
            float dx_arr[8];
            for (int k = 0; k < 8; k++) {
                float sx = (x + k) * x_ratio;
                sx_int_arr[k] = (int)sx;
                dx_arr[k] = sx - sx_int_arr[k];
            }

            // Gather 4 corners of 8 pixels
            __m256 p00 = gather_u8_8(src, sy_int, sx_int_arr, sw);
            __m256 p10 = gather_u8_8(src, sy_int, sx_int_arr_add1, sw);
            __m256 p01 = gather_u8_8(src, sy_int + 1, sx_int_arr, sw);
            __m256 p11 = gather_u8_8(src, sy_int + 1, sx_int_arr_add1, sw);

            // Interpolate: use FMA for all 4 products
            __m256 vdx = _mm256_loadu_ps(dx_arr);
            __m256 vdy = _mm256_set1_ps(dy);

            __m256 p0 = _mm256_fmadd_ps(vdx, _mm256_sub_ps(p10, p00), p00);
            __m256 p1 = _mm256_fmadd_ps(vdx, _mm256_sub_ps(p11, p01), p01);
            __m256 result = _mm256_fmadd_ps(vdy, _mm256_sub_ps(p1, p0), p0);

            // ... clamp to [0,255] and store as uint8 ...
        }
    }
}
```

## 2. 音频 DSP

### 2.1 FIR 滤波器（块 FIR + AVX2 寄存器轮转）

FIR 滤波器：`y[n] = Σ_{k=0}^{K-1} h[k] * x[n-k]`

直接实现是计算密集型的。使用块 FIR（分块处理）和寄存器轮转优化：

```c
// Block FIR filter: process 8 output samples at a time
// Input: x (signal), h (filter coefficients), K (filter length)
void fir_block_avx2(const float* x, const float* h, float* y, int n, int K) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 acc = _mm256_setzero_ps();

        for (int k = 0; k < K; k++) {
            // Broadcast filter coefficient h[k] to all lanes
            __m256 hk = _mm256_broadcast_ss(&h[k]);

            // Load x[i-k .. i-k+7] (8 consecutive samples)
            // Note: x indices go backwards (convolution)
            __m256 xk = _mm256_loadu_ps(x + i - k);

            // FMA: acc += h[k] * x[n-k]
            acc = _mm256_fmadd_ps(hk, xk, acc);
        }
        _mm256_storeu_ps(y + i, acc);
    }
}

// Optimized: use 4 accumulators for FMA latency hiding
void fir_block_avx2_opt(const float* x, const float* h, float* y, int n, int K) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int k = 0; k + 4 <= K; k += 4) {
            __m256 hk0 = _mm256_broadcast_ss(&h[k]);
            __m256 hk1 = _mm256_broadcast_ss(&h[k+1]);
            __m256 hk2 = _mm256_broadcast_ss(&h[k+2]);
            __m256 hk3 = _mm256_broadcast_ss(&h[k+3]);

            __m256 xk0 = _mm256_loadu_ps(x + i - k);
            __m256 xk1 = _mm256_loadu_ps(x + i - k - 1);
            __m256 xk2 = _mm256_loadu_ps(x + i - k - 2);
            __m256 xk3 = _mm256_loadu_ps(x + i - k - 3);

            acc0 = _mm256_fmadd_ps(hk0, xk0, acc0);
            acc1 = _mm256_fmadd_ps(hk1, xk1, acc1);
            acc2 = _mm256_fmadd_ps(hk2, xk2, acc2);
            acc3 = _mm256_fmadd_ps(hk3, xk3, acc3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        _mm256_storeu_ps(y + i, acc);
    }
}
```

**Roofline 分析**：
- 计算：K 次 FMA/输出样本
- 内存：读取 x（2×）和 h（1×），写入 y（1×）
- OI ≈ 2K FLOP / (4 + 1/8) bytes ≈ K/2 FLOP/byte
- 当 K > 64 时（OI > 32），在 DRAM 绑定场景下转为计算绑定

### 2.2 IIR Biquad 滤波器（难以向量化）

IIR 滤波器有反馈回路：`y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`

反馈依赖使单通道向量化几乎不可能。但**多通道**可以向量化：

```c
// Process 8 audio channels simultaneously with same coefficients
void biquad_8ch_avx2(float* x[8], float* y[8], int n,
                      float b0, float b1, float b2, float a1, float a2) {
    __m256 vb0 = _mm256_set1_ps(b0);
    __m256 vb1 = _mm256_set1_ps(b1);
    __m256 vb2 = _mm256_set1_ps(b2);
    __m256 va1 = _mm256_set1_ps(-a1);  // negative because formula: -a1*y[n-1]
    __m256 va2 = _mm256_set1_ps(-a2);

    // Per-channel state (interleaved in SIMD lanes)
    __m256 x1 = _mm256_setzero_ps();  // x[n-1] for 8 channels
    __m256 x2 = _mm256_setzero_ps();  // x[n-2] for 8 channels
    __m256 y1 = _mm256_setzero_ps();  // y[n-1] for 8 channels
    __m256 y2 = _mm256_setzero_ps();  // y[n-2] for 8 channels

    for (int i = 0; i < n; i++) {
        // Gather x[i] from 8 channels into one YMM register
        __m256 x0;
        for (int ch = 0; ch < 8; ch++)
            ((float*)&x0)[ch] = x[ch][i];

        // Biquad formula
        __m256 y0 = _mm256_mul_ps(vb0, x0);
        y0 = _mm256_fmadd_ps(vb1, x1, y0);
        y0 = _mm256_fmadd_ps(vb2, x2, y0);
        y0 = _mm256_fmadd_ps(va1, y1, y0);
        y0 = _mm256_fmadd_ps(va2, y2, y0);

        // Shift states
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;

        // Scatter result back to 8 channels
        for (int ch = 0; ch < 8; ch++)
            y[ch][i] = ((float*)&y0)[ch];
    }
}
```

### 2.3 FFT Radix-4 蝶形运算

```c
// Radix-4 FFT butterfly: 4 complex inputs → 4 complex outputs
// twiddle: 3 complex twiddle factors (W^1, W^2, W^3)
// Each complex number = (real, imag) interleaved in memory

// Helper: complex multiply (a_re, a_im) * (b_re, b_im)
// result_re = a_re * b_re - a_im * b_im
// result_im = a_re * b_im + a_im * b_re
static inline void cmul_avx2(__m256 ar, __m256 ai, __m256 br, __m256 bi,
                              __m256* out_r, __m256* out_i) {
    // FMA: out_re = a_re * b_re - a_im * b_im
    *out_r = _mm256_fnmadd_ps(ai, bi, _mm256_mul_ps(ar, br));
    // out_im = a_re * b_im + a_im * b_re
    *out_i = _mm256_fmadd_ps(ai, br, _mm256_mul_ps(ar, bi));
}

// Radix-4 butterfly processes 8 complex values at a time (2 butterflies in parallel)
void fft_radix4_butterfly_avx2(
    __m256 r0, __m256 i0, __m256 r1, __m256 i1,
    __m256 r2, __m256 i2, __m256 r3, __m256 i3,
    __m256 wr1, __m256 wi1, __m256 wr2, __m256 wi2, __m256 wr3, __m256 wi3,
    __m256* out0r, __m256* out0i, __m256* out1r, __m256* out1i,
    __m256* out2r, __m256* out2i, __m256* out3r, __m256* out3i)
{
    // Stage 1: cross additions
    __m256 t0r = _mm256_add_ps(r0, r2);
    __m256 t0i = _mm256_add_ps(i0, i2);
    __m256 t1r = _mm256_sub_ps(r0, r2);
    __m256 t1i = _mm256_sub_ps(i0, i2);
    __m256 t2r = _mm256_add_ps(r1, r3);
    __m256 t2i = _mm256_add_ps(i1, i3);
    __m256 t3r = _mm256_sub_ps(r1, r3);
    __m256 t3i = _mm256_sub_ps(i1, i3);

    // Stage 2: twiddle multiplication
    __m256 m1r, m1i;
    cmul_avx2(t2r, t2i, wr2, wi2, &m1r, &m1i);
    __m256 m2r, m2i;
    cmul_avx2(t3r, t3i, wr1, wi1, &m2r, &m2i);
    __m256 m3r, m3i;
    cmul_avx2(t3r, t3i, wr3, wi3, &m3r, &m3i);

    // Stage 3: final cross additions
    *out0r = _mm256_add_ps(t0r, m1r);
    *out0i = _mm256_add_ps(t0i, m1i);
    *out1r = _mm256_add_ps(t1r, m2r);
    *out1i = _mm256_add_ps(t1i, m2i);
    *out2r = _mm256_sub_ps(t0r, m1r);
    *out2i = _mm256_sub_ps(t0i, m1i);
    *out3r = _mm256_sub_ps(t1r, m3r);
    *out3i = _mm256_sub_ps(t1i, m3i);
}
```

## 3. ML 嵌入内积（推荐系统双塔模型）

在推荐系统中，双塔模型的核心运算是：`score = user_emb · item_emb`（两个稠密向量内积）。对海量候选物品做内积是计算瓶颈。

### 3.1 AVX2 批量内积

```c
// Compute dot(user_emb, item_emb[j]) for j in [0, batch_size)
// user_emb: dim-dimensional vector
// item_embs: batch_size × dim matrix (row-major)
void batch_dot_product_avx2(const float* user_emb,
                             const float* item_embs,
                             float* scores,
                             int batch_size, int dim) {
    for (int j = 0; j < batch_size; j++) {
        const float* item = item_embs + j * dim;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        int i;
        for (i = 0; i + 16 <= dim; i += 16) {
            __m256 u0 = _mm256_loadu_ps(user_emb + i);
            __m256 v0 = _mm256_loadu_ps(item + i);
            sum0 = _mm256_fmadd_ps(u0, v0, sum0);

            __m256 u1 = _mm256_loadu_ps(user_emb + i + 8);
            __m256 v1 = _mm256_loadu_ps(item + i + 8);
            sum1 = _mm256_fmadd_ps(u1, v1, sum1);
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        float score = reduce_sum_avx2(sum0);
        for (; i < dim; i++) score += user_emb[i] * item[i];
        scores[j] = score;
    }
}

// Optimized: process 8 items at a time using same user_emb row
void batch_dot_product_8way_avx2(const float* user_emb,
                                  const float* item_embs,
                                  float* scores,
                                  int batch_size, int dim) {
    for (int j = 0; j + 8 <= batch_size; j += 8) {
        __m256 acc[8] = {};  // 8 accumulators, one per item
        for (int i = 0; i < dim; i++) {
            __m256 u = _mm256_broadcast_ss(user_emb + i);
            for (int k = 0; k < 8; k++) {
                acc[k] = _mm256_fmadd_ps(u, 
                    _mm256_set1_ps(item_embs[(j + k) * dim + i]), acc[k]);
            }
        }
        for (int k = 0; k < 8; k++)
            scores[j + k] = reduce_sum_avx2(acc[k]);
    }
}
```

### 3.2 AVX-512 单周期处理 16 个维度

```c
void batch_dot_product_avx512(const float* user_emb,
                               const float* item_embs,
                               float* scores,
                               int batch_size, int dim) {
    for (int j = 0; j < batch_size; j++) {
        const float* item = item_embs + j * dim;
        __m512 sum = _mm512_setzero_ps();
        int i;
        for (i = 0; i + 16 <= dim; i += 16) {
            __m512 u = _mm512_loadu_ps(user_emb + i);
            __m512 v = _mm512_loadu_ps(item + i);
            sum = _mm512_fmadd_ps(u, v, sum);
        }
        float s = _mm512_reduce_add_ps(sum);
        for (; i < dim; i++) s += user_emb[i] * item[i];
        scores[j] = s;
    }
}
```

### 3.3 bf16 嵌入（2x 内存带宽）

```c
// bf16 embedding allows 2x more embeddings in the same memory bandwidth
// Each embedding dimension is stored as uint16_t (bf16)
void batch_dot_bf16_avx2(const uint16_t* user_emb,
                          const uint16_t* item_embs,
                          float* scores,
                          int batch_size, int dim) {
    for (int j = 0; j < batch_size; j++) {
        const uint16_t* item = item_embs + j * dim;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        int i;
        for (i = 0; i + 16 <= dim; i += 16) {
            // Load 8 bf16 values, convert to fp32
            __m128i u16 = _mm_loadu_si128((__m128i*)(user_emb + i));
            __m128i v16 = _mm_loadu_si128((__m128i*)(item + i));

            // Shift left 16 to restore to fp32 representation
            __m256i u32 = _mm256_cvtepu16_epi32(u16);
            __m256i v32 = _mm256_cvtepu16_epi32(v16);
            u32 = _mm256_slli_epi32(u32, 16);
            v32 = _mm256_slli_epi32(v32, 16);

            __m256 u = _mm256_castsi256_ps(u32);
            __m256 v = _mm256_castsi256_ps(v32);
            sum0 = _mm256_fmadd_ps(u, v, sum0);

            // Process second half of 8
            // ... similar ...
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        scores[j] = reduce_sum_avx2(sum0);
    }
}
```

**内存带宽优势**：bf16 存储大小是 fp32 的 1/2，对内存带宽绑定的大内积推荐场景，这意味着 ~2x 的吞吐提升。对于典型的 dim=256 内积 + 大规模候选，从 fp32 切换到 bf16 可以实现 ~1.8x 加速（接近理论 2x，受计算开销影响略有折损）。

## 4. GEMM/GEMV 微内核

### 4.1 GEMV with Tiling

GEMV 通常受内存带宽限制，通过分块提高 L1/L2 重用：

```c
// Tiled GEMV: break N into tiles that fit in L2 cache
// A: M×N row-major, x: N-vector, y: M-vector
void gemv_tiled_avx2(const float* A, const float* x, float* y,
                      int M, int N, int tile_size) {
    for (int t = 0; t < N; t += tile_size) {
        int tn = (t + tile_size <= N) ? tile_size : N - t;
        for (int i = 0; i + 4 <= M; i += 4) {
            __m256 y0 = _mm256_setzero_ps();
            // ... up to y3 ...
            for (int j = 0; j + 8 <= tn; j += 8) {
                // Process 8 columns at a time for 4 rows
            }
            y[i+0] += reduce_sum_avx2(y0);
            // ...
        }
    }
}
```

**典型分块参数**（Skylake L2=1MB）：
- `tile_size` = 2048（2048 × 4 bytes ≈ 8KB，远小于 1MB L2）
- L2 重用率：每个 tile 内的 x 值被 M 行重用，重用倍数 = M/tile_size

### 4.2 高效 GEMM Packing

对矩阵 B 做 packing 以利用 TLB 和连续访问：

```c
// Pack B into column-major format, transposing for SIMD-friendly access
void pack_b_avx2(const float* B, float* B_packed, int K, int N, int ldb) {
    int NR = 8;  // micro-kernel column block
    for (int n = 0; n < N; n += NR) {
        for (int k = 0; k < K; k++) {
            for (int nr = 0; nr < NR; nr++) {
                if (n + nr < N)
                    B_packed[(n/NR) * K * NR + k * NR + nr] = B[k * ldb + n + nr];
            }
        }
    }
}

// Pack A into panel format for broadcasting
void pack_a_avx2(const float* A, float* A_packed, int M, int K, int lda) {
    int MR = 8;  // micro-kernel row block
    for (int m = 0; m < M; m += MR) {
        for (int k = 0; k < K; k++) {
            for (int mr = 0; mr < MR; mr++) {
                if (m + mr < M)
                    A_packed[(m/MR) * K * MR + k * MR + mr] = A[(m + mr) * lda + k];
            }
        }
    }
}
```

## 5. LayerNorm / Softmax 完整实现

LayerNorm 和 Softmax 在 NLP Transformer 模型中占比可达推理时间的 15-30%。快速实现使用 2-3 遍算法。

### 5.1 AVX-512 LayerNorm

```c
void layernorm_avx512(const float* x, float* y, int n, float eps,
                       const float* gamma, const float* beta) {
    // Pass 1: compute mean and M2 (Welford's algorithm vectorized)
    int count = 0;
    __m512 mean = _mm512_setzero_ps();
    __m512 m2 = _mm512_setzero_ps();

    for (int i = 0; i + 16 <= n; i += 16, count += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 delta = _mm512_sub_ps(v, mean);
        mean = _mm512_add_ps(mean, _mm512_mul_ps(delta, 
            _mm512_set1_ps(1.0f / (float)(i + 16))));
        __m512 delta2 = _mm512_sub_ps(v, mean);
        m2 = _mm512_add_ps(m2, _mm512_mul_ps(delta, delta2));
    }

    float mean_f = _mm512_reduce_add_ps(mean) / n;
    float var_f = _mm512_reduce_add_ps(m2) / n;
    float inv_std = 1.0f / sqrtf(var_f + eps);

    // Pass 2: normalize and apply scale/shift
    __m512 vmean = _mm512_set1_ps(mean_f);
    __m512 vinv = _mm512_set1_ps(inv_std);

    for (int i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        v = _mm512_mul_ps(_mm512_sub_ps(v, vmean), vinv);
        if (gamma) v = _mm512_mul_ps(v, _mm512_loadu_ps(gamma + i));
        if (beta)  v = _mm512_add_ps(v, _mm512_loadu_ps(beta + i));
        _mm512_storeu_ps(y + i, v);
    }
}
```

### 5.2 AVX-512 Softmax

```c
// Fast exp approximation for softmax using polynomial
static inline __m512 exp_approx_avx512(__m512 x) {
    // exp(x) ≈ 2^(x * log2(e))
    // Use range reduction + polynomial for accurate approx
    __m512 c1 = _mm512_set1_ps(1.4426950408889634f);   // 1/ln(2)
    __m512 c2 = _mm512_set1_ps(0.6931471805599453f);   // ln(2)
    __m512 c3 = _mm512_set1_ps(1.0f / 256.0f);
    __m512 c4 = _mm512_set1_ps(1.0f);

    // Scale to base-2 range
    __m512 t = _mm512_fmadd_ps(x, c1, _mm512_set1_ps(126.0f));
    __m512i t_int = _mm512_cvtps_epi32(t);  // integer part
    __m512 f = _mm512_cvtepi32_ps(t_int);

    // Polynomial approximation for 2^fractional
    __m512 frac = _mm512_sub_ps(t, f);
    // 2^x ≈ 1 + x*(0.6931 + x*(0.2402 + x*(0.0555)))
    __m512 poly = _mm512_set1_ps(0.05550410866482158f);
    poly = _mm512_fmadd_ps(poly, frac, _mm512_set1_ps(0.24022650695910068f));
    poly = _mm512_fmadd_ps(poly, frac, _mm512_set1_ps(0.6931471805599453f));
    poly = _mm512_fmadd_ps(poly, frac, c4);

    // Combine with integer exponent via float exponent manipulation
    __m512i exponent = _mm512_slli_epi32(t_int, 23);
    return _mm512_castsi512_ps(exponent) * poly;
}

void softmax_avx512(const float* x, float* y, int n) {
    // Pass 1: find max
    __m512 max_vec = _mm512_set1_ps(-INFINITY);
    int i;
    for (i = 0; i + 16 <= n; i += 16)
        max_vec = _mm512_max_ps(max_vec, _mm512_loadu_ps(x + i));
    float max_val = _mm512_reduce_max_ps(max_vec);
    for (; i < n; i++) max_val = fmaxf(max_val, x[i]);

    // Pass 2: exp and sum
    __m512 sum_vec = _mm512_setzero_ps();
    __m512 max_bc = _mm512_set1_ps(max_val);
    for (i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        v = _mm512_sub_ps(v, max_bc);
        v = exp_approx_avx512(v);
        _mm512_storeu_ps(y + i, v);
        sum_vec = _mm512_add_ps(sum_vec, v);
    }
    float sum_val = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; i++) { y[i] = expf(x[i] - max_val); sum_val += y[i]; }

    // Pass 3: normalize
    __m512 inv = _mm512_set1_ps(1.0f / sum_val);
    for (i = 0; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), inv));
    for (; i < n; i++) y[i] /= sum_val;
}
```

## 6. JSON/字节扫描（simdjson 风格）

### 6.1 找引号和反斜杠

```c
// Find all structural characters in a JSON string
// Key characters: " \ { } [ ] : ,
uint64_t find_structural_chars_avx2(const uint8_t* data, int len) {
    __m256i quote = _mm256_set1_epi8('"');
    __m256i bslash = _mm256_set1_epi8('\\');
    __m256i lbrace = _mm256_set1_epi8('{');
    __m256i rbrace = _mm256_set1_epi8('}');
    __m256i lbracket = _mm256_set1_epi8('[');
    __m256i rbracket = _mm256_set1_epi8(']');
    __m256i colon = _mm256_set1_epi8(':');
    __m256i comma = _mm256_set1_epi8(',');

    uint64_t bitmap = 0;
    // Process 32 bytes at a time
    for (int i = 0; i + 32 <= len; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)(data + i));

        // Compare against each structural character
        __m256i m_quote = _mm256_cmpeq_epi8(v, quote);
        __m256i m_bslash = _mm256_cmpeq_epi8(v, bslash);
        // ... etc

        // Bitwise OR all masks together
        __m256i structural = m_quote;
        structural = _mm256_or_si256(structural, m_bslash);
        structural = _mm256_or_si256(structural, lbrace);
        structural = _mm256_or_si256(structural, rbrace);
        structural = _mm256_or_si256(structural, lbracket);
        structural = _mm256_or_si256(structural, rbracket);
        structural = _mm256_or_si256(structural, colon);
        structural = _mm256_or_si256(structural, comma);

        // _mm256_movemask_epi8: extract 32 bits (one per byte)
        // bit j = 1 means byte j was a structural character
        uint32_t mask = _mm256_movemask_epi8(structural);
        bitmap |= ((uint64_t)mask) << i;
    }
    return bitmap;
}
```

**simdjson 性能**：在 AVX2 上可达 ~3 GB/s 的结构化字符扫描。在 AVX-512 上使用 `_mm512_cmpeq_epi8_mask` 直接获取 64 位掩码可进一步加速。

### 6.2 字节分类（用 PSHUFB 做查表）

```c
// Use PSHUFB as a 16-entry lookup table per 128-bit lane
// Classify each byte: 0=space, 1=alpha, 2=digit, 3=punct, 4=other
void classify_bytes_avx2(const uint8_t* input, uint8_t* output, int n) {
    // Build lookup table: table[byte_value] = classification
    // 256 byte values → only 16 can be mapped in one PSHUFB
    // Use MSB bits as nibbles (high nibble = row, low nibble = column)
    // Process each nibble separately: hi_nibble selects sub-table, lo_nibble indexes it
    // Complex but feasible with 2 PSHUFB operations per 16 bytes

    __m256i hi_nibble_mask = _mm256_set1_epi8(0xF0);
    __m256i lo_nibble_mask = _mm256_set1_epi8(0x0F);

    // Build 16 tables for each possible high nibble (0x00-0xF0)
    // table[hi][lo] = classification for ((hi<<4) | lo)
    __m256i tables[16];  // one YMM register per high nibble table
    // ... initialize tables ...

    for (int i = 0; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)(input + i));
        // In reality this is complex due to 128-bit lane constraint
        // simdjson uses a series of clever PSHUFB tricks
    }
}
```

### 6.3 memchr 与新行扫描

```c
// Find next newline: extremely fast with SIMD
// Expected throughput: ~32 GB/s on modern Intel (memory bandwidth limited)
const char* memchr_newline_avx2(const char* p, const char* end) {
    __m256i newline = _mm256_set1_epi8('\n');

    // Align to 32-byte boundary first
    while (((uintptr_t)p & 31) && p < end) {
        if (*p == '\n') return p;
        p++;
    }

    while (p + 32 <= end) {
        __m256i v = _mm256_load_si256((__m256i*)p);
        __m256i cmp = _mm256_cmpeq_epi8(v, newline);
        uint32_t mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            return p + __builtin_ctz(mask);
        }
        p += 32;
    }

    // Tail
    while (p < end) {
        if (*p == '\n') return p;
        p++;
    }
    return NULL;
}
```

## 7. 压缩与校验

### 7.1 硬件 CRC32C

```c
// CRC32C (Castagnoli) - hardware accelerated
// Used in storage, networking, iSCSI, SCTP
uint32_t crc32c_avx2(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;

    // Process 8 bytes at a time with 64-bit CRC instruction
    while (len >= 8) {
        crc = (uint32_t)_mm_crc32_u64(crc, *(uint64_t*)data);
        data += 8;
        len -= 8;
    }

    // Tail
    while (len >= 4) {
        crc = _mm_crc32_u32(crc, *(uint32_t*)data);
        data += 4;
        len -= 4;
    }
    while (len > 0) {
        crc = _mm_crc32_u8(crc, *data);
        data++;
        len--;
    }
    return crc ^ 0xFFFFFFFFu;
}

// CRC32C using multiple independent streams for ILP
// Can reach 50+ GB/s on modern Intel
uint32_t crc32c_3way(const uint8_t* data, size_t len) {
    uint32_t crc0 = 0xFFFFFFFFu, crc1 = 0xFFFFFFFFu, crc2 = 0xFFFFFFFFu;

    while (len >= 24) {
        crc0 = (uint32_t)_mm_crc32_u64(crc0, *(uint64_t*)(data));
        crc1 = (uint32_t)_mm_crc32_u64(crc1, *(uint64_t*)(data + 8));
        crc2 = (uint32_t)_mm_crc32_u64(crc2, *(uint64_t*)(data + 16));
        data += 24;
        len -= 24;
    }

    // Combine three streams (requires polynomial arithmetic)
    // ... complex combination code ...
    return crc0;  // simplified
}
```

### 7.2 Base64 编解码

```c
// AVX2 Base64 decode: use PSHUFB as lookup table for 6-bit → 8-bit expansion
// Process 32 bytes of encoded data → 24 bytes of decoded data

void base64_decode_avx2(const char* input, uint8_t* output, int encoded_len) {
    // Lookup table for Base64 char to 6-bit value
    // Valid chars: A-Z, a-z, 0-9, +, /
    static const uint8_t lut[32] __attribute__((aligned(32))) = {
        /* 0x00-0x1F */ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        /* 0x20-0x3F */ 0,0,0,0,0,0,0,0,0,0,0,62,0,0,0,63,
        /* 0x40-0x5F */ 52,53,54,55,56,57,58,59,60,61,0,0,0,0,0,0,
        /* 0x60-0x7F */ 0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
    };

    __m256i shuffle_mask = _mm256_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1
    );

    for (int i = 0; i + 32 <= encoded_len; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)(input + i));

        // Step 1: look up each byte in LUT (requires nibble splitting for full 256-entry LUT)
        // This is a simplified version using PSHUFB
        __m256i hi_nibble = _mm256_and_si256(_mm256_srli_epi32(v, 4), _mm256_set1_epi8(0x0F));
        __m256i lo_nibble = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));

        // ... complex nibble-based lookup ...

        // Step 2: pack 4x6-bit values into 3 bytes
        // ... use PSHUFB + OR/shift to combine ...

        // Store 24 decoded bytes
        _mm256_storeu_si256((__m256i*)(output + (i/4)*3), decoded);
    }
}
```

## 8. 网络数据包处理

### 8.1 TCP 校验和（16 位补码和）

```c
// TCP checksum: 16-bit ones' complement sum over TCP pseudo-header + segment
// AVX2 implementation processes 16 x 16-bit = 32 bytes at a time
uint16_t tcp_checksum_avx2(const uint16_t* data, int len_words) {
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();

    // Process 32 words (64 bytes) at a time
    for (int i = 0; i + 32 <= len_words; i += 32) {
        __m256i v0 = _mm256_loadu_si256((__m256i*)(data + i));
        __m256i v1 = _mm256_loadu_si256((__m256i*)(data + i + 16));
        sum0 = _mm256_adds_epu16(sum0, v0);  // saturated add to detect carry
        sum1 = _mm256_adds_epu16(sum1, v1);
    }

    // Combine and fold carries
    __m256i sum = _mm256_adds_epu16(sum0, sum1);
    // ... extract to scalars, fold 32-bit carries into 16-bit ...
    // ... then ones' complement ...
    return ~folded_sum;
}
```

### 8.2 Bloom Filter with SIMD Hash

```c
// SIMD-accelerated Bloom filter lookup
// Compute k hash functions in parallel for a single key
void bloom_lookup_avx2(const uint8_t* bloom, int m,
                        const uint8_t* key, int key_len,
                        uint8_t* result) {
    // Use _mm256_crc32_u64 as a fast hash function
    // Process key in 8-byte chunks with CRC32C
    uint64_t h = 0;
    for (int i = 0; i + 8 <= key_len; i += 8) {
        h = _mm_crc32_u64(h, *(uint64_t*)(key + i));
    }

    // Generate k hash values using double-hashing: h_i = h1 + i*h2
    // Then test k Bloom filter positions simultaneously using gather
    // ...
}

// Batch bloom filter lookup: test 8 keys at once
void bloom_batch_avx2(const uint8_t* bloom, int m,
                       const uint8_t** keys, int key_len, int batch_size,
                       uint8_t* results) {
    for (int j = 0; j + 8 <= batch_size; j += 8) {
        // Compute 8 hashes (one per key) in parallel
        __m256i hashes = _mm256_setzero_si256();
        for (int i = 0; i + 8 <= key_len; i += 8) {
            // Load 8 bytes from 8 different keys → broadcast and CRC
            // ... gather 8 key bytes, CRC each lane individually ...
        }

        // Modulo m to get bit positions
        // Test bits in bloom filter via gather
        // ...
    }
}
```

## 9. Roofline 分析总结

将本章所有案例的 roofline 特性汇总：

```
案例                         操作强度(FLOP/B)  限制因素     预期加速(vs scalar)
─────────────────────────────────────────────────────────────────────────────
RGB→Gray (int)               0.5              内存带宽     3-4x
RGB→Gray (float FMA)         3.0              内存带宽     4-5x
高斯模糊 (5-tap)             5.0              内存带宽     4-5x
FIR 滤波器 (K=128)           64.0             计算         7-8x
批量内积 (dim=256)           1.0              内存带宽     5-7x (bf16: 10-14x)
int8 点积 (AVX2)             2.0              内存带宽     6-8x
int8 点积 (VNNI)             4.0              内存带宽     12-16x
GEMV (M=1)                   2.0              内存带宽     4-6x
GEMM (N=1024)                340              计算         20-30x
LayerNorm                    2.0              内存带宽     5-7x
Softmax (exp approx)         8.0              内存+计算    6-8x
JSON 结构扫描                0.125            内存带宽     20-30x
CRC32C                       0.125            内存+计算    10-50x
TCP 校验和                   0.5              内存带宽     8-12x
```

**关键洞察**：

1. 大多数实际工作负载受**内存带宽限制**，而非计算限制。这意味着优化内存访问模式（SoA、对齐、预取）比增加 FLOP 更有效。

2. **bf16** 是内存带宽绑定场景的免费午餐：2x 带宽 = 近 2x 加速，几乎无需算法改变。

3. **AVX-512 的主要优势不是 2x 宽度**，而是掩码、压缩/扩展、更多的寄存器和更强大的 permute。在内存带宽限定场景中，512 位宽度本身不值得与 AVX2 比较（因为两个都在等内存）。

4. 对于**低操作强度的内存绑定工作负载**（如 JSON 解析、memchr、checksum），AVX2 已经足够接近 AVX-512 的最优性能。在这些场景下，投资 AVX2 优化带来最大的 ROI。

5. 对于**高操作强度的计算绑定工作负载**（如 GEMM、FFT），AVX-512 的 2x FMA 吞吐和 2x 寄存器数量带来近 2x 加速。

## 10. 参考与延伸阅读

- **[simdjson](https://github.com/simdjson/simdjson)**: 工业生产级 JSON 解析器，阅读源码理解 SIMD 字节操作的最佳方式
- **[xsimd](https://github.com/xtensor-stack/xsimd)**: C++ 模板 SIMD 包装，学习如何用现代 C++ 组织 SIMD 代码
- **[Google Highway](https://github.com/google/highway)**: 可移植 SIMD 库，支持 x86/ARM/RISC-V
- **[uops.info](https://uops.info/)**: 每条指令在每个微架构上的精确 µop 分解
- **[Agner Fog's Optimization Manuals](https://agner.org/optimize/)**: x86 微架构深度指南
- **[Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)**: Intel 官方优化手册，章节 3-6 专门讨论 SIMD 优化
