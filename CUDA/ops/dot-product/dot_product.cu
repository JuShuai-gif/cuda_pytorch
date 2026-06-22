#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <c10/cuda/CUDAStream.h>
#include <type_traits>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// Warp Reduce Sum
// 这段代码实现的逻辑被称为 Butterfly Reduction。它不需要访问慢速的共享内存（Shared Memory），直接在线程的 寄存器 之间交换数据
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    // 强制编译器展开循环。因为 kWarpSize 是常数（32），
    // 循环次数确定（5次：16->8->4->2->1），展开能消除分支跳转开销。
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        // __shfl_xor_sync 是核心指令：
        // 1. 0xffffffff: 掩码，表示参与该操作的所有 32 个线程都是活跃的。
        // 2. val: 当前线程拥有的局部累加值。
        // 3. mask: XOR 操作数，用于决定当前线程与哪一个“邻居”交换数据。
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Dot Product
// grid(N/256), block(256)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    // keep the data in register is enough for warp operaion.
    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    // warp leaders store the data to shared memory.
    // 每个 warp 的第 0 位置记录当前warp的所有和
    if (lane == 0) {
        reduce_smem[warp] = prod;
    }

    __syncthreads(); // make sure the data is in shared memory.

    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;


    if (warp == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }
    if (tid == 0) {
        atomicAdd(y, prod);
    }
}

// Dot Product + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);

    float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // perform warp sync reduce.
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);

    // warp leaders store the data to shared memory.
    if (lane == 0) {
        reduce_smem[warp] = prod;
    }
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (tid == 0) {
        atomicAdd(y, prod);
    }
}

// FP16
// Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
        // val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


/*
在高性能计算（HPC）和深度学习算子开发中，FP16（半精度浮点数） 的最大痛点在于其极窄的数值表示范围：
它能表示的最大值仅为 65,504。在一个点积运算中，如果向量长度过长（比如 $10^5$ 级），中间累加值很容易超过这个界限，
导致结果变成 inf（无穷大）。此外，FP16 的精度位（Mantissa）有限，长序列加法会产生严重的截断误差。
🛠️ 混合精度规约的工程实现解决办法很简单：输入用 FP16（省带宽），累加用 FP32（保精度）。
*/
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f16_f32_kernel(half *a, half *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    // keep the data in register is enough for warp operaion.
    half prod_f16 = (idx < N) ? __hmul(a[idx], b[idx]) : __float2half(0.0f);

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    // warp leaders store the data to shared memory.
    if (lane == 0) {
        reduce_smem[warp] = prod;
    }

    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (tid == 0) {
        atomicAdd(y, prod);
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void dot_prod_f16x2_f32_kernel(half *a, half *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    // keep the data in register is enough for warp operaion.
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);

    half prod_f16 = (idx < N) ? __hadd(__hmul(reg_a.x, reg_b.x), __hmul(reg_a.y, reg_b.y)) : __float2half(0.0f);

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // perform warp sync reduce.
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    // warp leaders store the data to shared memory.
    if (lane == 0) {
        reduce_smem[warp] = prod;
    }

    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (tid == 0) {
        atomicAdd(y, prod);
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half *a, half *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half pack_a[8], pack_b[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

    const half z = __float2half(0.0f);

    half prod_f16 = z;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        prod_f16 += (((idx + i) < N) ? (v.x + v.y) : z);
    }

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);

    if (lane == 0) {
        reduce_smem[warp] = prod;
    }
    __syncthreads();

    prod = (lane < NUM_THREADS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }
    if (tid == 0) {
        atomicAdd(y, prod);
    }
}


// 上面的版本都对超过1024长度的情况支持不好，所以需要编写支持 Grid-Stride Loops的Kernel
template <const int NUM_THREADS = 256>
__global__ void dot_prod_f16_f32_kernel_grid(half *a, half *b, float *y, int N) {
    // 1. 定义 Block 私有的共享内存
    constexpr int NUM_WARPS = (NUM_THREADS + 31) / 32;
    __shared__ float reduce_smem[NUM_WARPS];

    // 2. Grid-Stride Loop 累加
    float local_sum = 0.0f;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_tid; i < N; i += stride) {
        // 核心：在循环中持续累加到 float 类型的寄存器 local_sum 中
        local_sum += __half2float(__hmul(a[i], b[i]));
    }

    // 3. 计算 Block 内的局部索引
    int tid = threadIdx.x;
    int warp = tid / 32; // 注意：必须用 threadIdx.x，因为 smem 是 Block 级的
    int lane = tid % 32;

    // 4. 第一级：Warp 级规约 (在寄存器中完成)
    local_sum = warp_reduce_sum_f32<32>(local_sum);

    // 5. 第二级：Block 级规约 (通过 Shared Memory 汇总)
    if (lane == 0) {
        reduce_smem[warp] = local_sum;
    }
    __syncthreads();

    // 只有第一个 Warp 负责汇总 Block 内所有 Warp 的结果
    float block_sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        block_sum = warp_reduce_sum_f32<32>(block_sum); 
    }

    // 6. 第三级：Global 级规约 (通过 Atomic 写回全局内存)
    if (tid == 0) {
        atomicAdd(y, block_sum);
    }
}

// 将函数名转换为字符串，用于 Pybind11 绑定
#define STRINGFY(str) #str

// 简化 Pybind11 的导出过程
// m.def("函数名", &函数指针, "文档字符串")
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

// 运行时类型检查：确保传入的 Python Tensor 类型与 Kernel 要求的一致
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type))) {                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

#define LAUNCH_DOT_PROD_1D(NT, packed_type, acc_type, element_type, stream) \
    do {                                                                   \
        dim3 block(NT);                                                     \
        dim3 grid((N + NT - 1) / NT);                                       \
        dot_prod_##packed_type##_##acc_type##_kernel<(NT)>                 \
            <<<grid, block, 0, stream>>>(                                  \
                reinterpret_cast<element_type *>(a.data_ptr()),            \
                reinterpret_cast<element_type *>(b.data_ptr()),            \
                prod.data_ptr<float>(), N);                                 \
    } while (0)

#define DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,  \
                                 n_elements, stream)                      \
    do {                                                                   \
        const int NT = (K) / (n_elements);                                 \
        dim3 block(NT);                                                    \
        dim3 grid(S);                                                      \
        switch (NT) {                                                      \
        case 32:   dot_prod_##packed_type##_##acc_type##_kernel<32>   <<<grid, block, 0, stream>>>(reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), prod.data_ptr<float>(), N); break; \
        case 64:   dot_prod_##packed_type##_##acc_type##_kernel<64>   <<<grid, block, 0, stream>>>(reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), prod.data_ptr<float>(), N); break; \
        case 128:  dot_prod_##packed_type##_##acc_type##_kernel<128>  <<<grid, block, 0, stream>>>(reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), prod.data_ptr<float>(), N); break; \
        case 256:  dot_prod_##packed_type##_##acc_type##_kernel<256>  <<<grid, block, 0, stream>>>(reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), prod.data_ptr<float>(), N); break; \
        case 512:  dot_prod_##packed_type##_##acc_type##_kernel<512>  <<<grid, block, 0, stream>>>(reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), prod.data_ptr<float>(), N); break; \
        case 1024: dot_prod_##packed_type##_##acc_type##_kernel<1024> <<<grid, block, 0, stream>>>(reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), prod.data_ptr<float>(), N); break; \
        default: throw std::runtime_error("Unsupported threads");          \
        }                                                                  \
    } while (0)

#define TORCH_BINDING_DOT_PROD(packed_type, acc_type, th_type, element_type, \
                               n_elements)                                   \
torch::Tensor dot_prod_##packed_type##_##acc_type(                            \
    torch::Tensor a, torch::Tensor b,                                         \
    at::optional<at::cuda::CUDAStream> stream) {                              \
                                                                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                    \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                    \
                                                                              \
    at::cuda::CUDAStream my_stream = stream.has_value()                       \
        ? stream.value()                                                      \
        : at::cuda::getCurrentCUDAStream(a.get_device());                     \
    cudaStream_t cuda_stream = my_stream.stream();                            \
                                                                              \
    auto options = torch::TensorOptions()                                     \
        .dtype(torch::kFloat32)                                               \
        .device(torch::kCUDA, a.get_device());                                \
    auto prod = torch::zeros({1}, options);                                   \
                                                                              \
    const int N = a.numel();                                                  \
                                                                              \
    if (a.dim() != 2) {                                                       \
        LAUNCH_DOT_PROD_1D(256, packed_type, acc_type, element_type, cuda_stream); \
    } else {                                                                  \
        const int S = a.size(0);                                              \
        const int K = a.size(1);                                              \
        if ((K / (n_elements)) <= 1024) {                                     \
            DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type, n_elements, cuda_stream); \
        } else {                                                              \
            LAUNCH_DOT_PROD_1D(256, packed_type, acc_type, element_type, cuda_stream); \
        }                                                                     \
    }                                                                         \
    return prod;                                                              \
}


// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_DOT_PROD(f32, f32, torch::kFloat32, float, 1)
TORCH_BINDING_DOT_PROD(f32x4, f32, torch::kFloat32, float, 4)
TORCH_BINDING_DOT_PROD(f16, f32, torch::kHalf, half, 1)
TORCH_BINDING_DOT_PROD(f16x2, f32, torch::kHalf, half, 2)
TORCH_BINDING_DOT_PROD(f16x8_pack, f32, torch::kHalf, half, 8)

// 定义一个新的导出宏，支持默认参数 stream=None
/*
py::arg("stream") = py::none() 的作用是让 Python 知道：如果用户调用 dot_prod(a, b)，第三个参数会自动填充为 None。

在 C++ 侧，None 会被自动转换成 at::optional 的 nullopt。
*/
#define TORCH_BINDING_STREAM_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func), \
          py::arg("a"), py::arg("b"), py::arg("stream") = py::none());

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_STREAM_EXTENSION(dot_prod_f32_f32)
    TORCH_BINDING_STREAM_EXTENSION(dot_prod_f32x4_f32)
    TORCH_BINDING_STREAM_EXTENSION(dot_prod_f16_f32)
    TORCH_BINDING_STREAM_EXTENSION(dot_prod_f16x2_f32)
    TORCH_BINDING_STREAM_EXTENSION(dot_prod_f16x8_pack_f32)
}
