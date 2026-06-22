#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>

// ============================================================================
// memory bandwidth benchmark kernels
// 演示 vectorized 访问（float/float2/float4）、coalesced vs strided 访问
//     对内存带宽的致命影响
// ============================================================================

// ---------------------------------------------------------------------------
// 1. float 标量访问：常规 coalesced 访问，每个线程读取一个 float
// ---------------------------------------------------------------------------

__global__ void copy_float_kernel(const float* src, float* dst, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ---------------------------------------------------------------------------
// 2. float2 向量化访问：每个线程一次读取两个 float（8 bytes）
//    GPU 的 L2 cache line 是 32 bytes，float2 能更好地利用 cache line
// ---------------------------------------------------------------------------

__global__ void copy_float2_kernel(const float* src, float* dst, int64_t n) {
    // n 是以 float 为单位的元素数量，必须是偶数
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t idx2 = idx * 2;
    if (idx2 + 1 < n) {
        float2 val = reinterpret_cast<const float2*>(src)[idx];
        reinterpret_cast<float2*>(dst)[idx] = val;
    } else if (idx2 < n) {
        // 处理最后一个奇数元素
        dst[idx2] = src[idx2];
    }
}

// ---------------------------------------------------------------------------
// 3. float4 向量化访问：每个线程一次读取四个 float（16 bytes）
//    这是单线程最大向量宽度，能最大化内存带宽利用率
//    条件：n 必须是 4 的倍数才能完全利用
// ---------------------------------------------------------------------------

__global__ void copy_float4_kernel(const float* src, float* dst, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t idx4 = idx * 4;
    if (idx4 + 3 < n) {
        float4 val = reinterpret_cast<const float4*>(src)[idx];
        reinterpret_cast<float4*>(dst)[idx] = val;
    } else {
        // 处理尾部不足 4 个的元素
        for (int i = 0; i < 4 && (idx4 + i) < n; ++i) {
            dst[idx4 + i] = src[idx4 + i];
        }
    }
}

// ---------------------------------------------------------------------------
// 4. 元素级乘加（float4 向量化版本）
//    模拟 ML 中常见的 elementwise 操作，演示向量化对实际计算带宽的影响
// ---------------------------------------------------------------------------

__global__ void elem_mul_float4_kernel(const float* a, const float* b, float* c, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t idx4 = idx * 4;
    if (idx4 + 3 < n) {
        float4 va = reinterpret_cast<const float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        float4 vc;
        vc.x = va.x * vb.x;
        vc.y = va.y * vb.y;
        vc.z = va.z * vb.z;
        vc.w = va.w * vb.w;
        reinterpret_cast<float4*>(c)[idx] = vc;
    } else {
        for (int i = 0; i < 4 && (idx4 + i) < n; ++i) {
            c[idx4 + i] = a[idx4 + i] * b[idx4 + i];
        }
    }
}

// ---------------------------------------------------------------------------
// 5. strided 访问 kernel（模拟非 contiguous 内存访问）
//    每个线程读取 stride 间隔的元素，演示未合并访问对带宽的致命影响
//    例如 stride=1 是 coalesced，stride=32 则一个 warp 的访问散布在
//    32 个不同的 cache line 上，造成严重的带宽浪费
// ---------------------------------------------------------------------------

__global__ void copy_strided_kernel(const float* src, float* dst, int64_t n, int64_t stride) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t src_idx = idx * stride;
    if (src_idx < n * stride && idx < n) {
        dst[idx] = src[src_idx];
    }
}

// ---------------------------------------------------------------------------
// 6. 非合并写入 kernel：将 stride 应用于写入而非读取
//    演示写操作的合并性同样影响带宽
// ---------------------------------------------------------------------------

__global__ void write_strided_kernel(const float* src, float* dst, int64_t n, int64_t stride) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t dst_idx = idx * stride;
    if (dst_idx < n * stride && idx < n) {
        dst[dst_idx] = src[idx];
    }
}

// ============================================================================
// Host 端 launch 函数
// ============================================================================

// 通用计时辅助函数
static float time_copy_kernel(
    const torch::Tensor& src, torch::Tensor& dst,
    void (*kernel)(const float*, float*, int64_t),
    int64_t n,
    int warmup, int measure) {

    const int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;

    // 预热
    for (int i = 0; i < warmup; ++i) {
        kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < measure; ++i) {
        kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms / measure;  // 平均每次 kernel 时间
}

torch::Tensor bench_copy_float(const torch::Tensor& src) {
    TORCH_CHECK(src.device().is_cuda() && src.is_contiguous(), "输入必须是 contiguous CUDA 张量");
    int64_t n = src.numel();
    auto dst = torch::empty_like(src);
    float avg_ms = time_copy_kernel(src, dst, copy_float_kernel, n, 5, 30);
    auto result = torch::empty({1}, torch::dtype(torch::kFloat32));
    result[0] = avg_ms * 1000.0f;  // 转换为微秒
    return result;
}

torch::Tensor bench_copy_float2(const torch::Tensor& src) {
    TORCH_CHECK(src.device().is_cuda() && src.is_contiguous(), "输入必须是 contiguous CUDA 张量");
    int64_t n = src.numel();
    if (n < 2) { n = 2; }  // 确保至少有 2 个元素
    auto dst = torch::empty_like(src);
    const int threads = 256;
    int blocks = (static_cast<int>(n / 2) + threads - 1) / threads;

    for (int i = 0; i < 5; ++i) {
        copy_float2_kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 30; ++i) {
        copy_float2_kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "float2 kernel error: %s\n", cudaGetErrorString(err));
    }

    auto result = torch::empty({1}, torch::dtype(torch::kFloat32));
    result[0] = (ms / 30.0f) * 1000.0f;  // 微秒
    return result;
}

torch::Tensor bench_copy_float4(const torch::Tensor& src) {
    TORCH_CHECK(src.device().is_cuda() && src.is_contiguous(), "输入必须是 contiguous CUDA 张量");
    int64_t n = src.numel();
    if (n < 4) { n = 4; }
    auto dst = torch::empty_like(src);
    const int threads = 256;
    int blocks = (static_cast<int>(n / 4) + threads - 1) / threads;

    for (int i = 0; i < 5; ++i) {
        copy_float4_kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 30; ++i) {
        copy_float4_kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "float4 kernel error: %s\n", cudaGetErrorString(err));
    }

    auto result = torch::empty({1}, torch::dtype(torch::kFloat32));
    result[0] = (ms / 30.0f) * 1000.0f;  // 微秒
    return result;
}

torch::Tensor bench_strided_copy(const torch::Tensor& src, int64_t stride) {
    TORCH_CHECK(src.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(stride >= 1, "stride 必须 >= 1");
    int64_t n = src.numel() / stride;
    if (n < 1) { n = 1; }
    auto dst = torch::empty({n}, src.options());
    const int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;

    for (int i = 0; i < 5; ++i) {
        copy_strided_kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n, stride);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 30; ++i) {
        copy_strided_kernel<<<blocks, threads>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n, stride);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "strided kernel error: %s\n", cudaGetErrorString(err));
    }

    auto result = torch::empty({1}, torch::dtype(torch::kFloat32));
    result[0] = (ms / 30.0f) * 1000.0f;  // 微秒
    return result;
}

torch::Tensor bench_elem_mul_float4(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(a.sizes() == b.sizes(), "形状必须相同");
    int64_t n = a.numel();
    if (n < 4) { n = 4; }
    auto c = torch::empty_like(a);
    const int threads = 256;
    int blocks = (static_cast<int>(n / 4) + threads - 1) / threads;

    for (int i = 0; i < 5; ++i) {
        elem_mul_float4_kernel<<<blocks, threads>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 30; ++i) {
        elem_mul_float4_kernel<<<blocks, threads>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "float4 mul kernel error: %s\n", cudaGetErrorString(err));
    }

    auto result = torch::empty({2}, torch::dtype(torch::kFloat32));
    result[0] = (ms / 30.0f) * 1000.0f;  // 微秒
    result[1] = c;
    return result;
}
