#ifndef FLASHINFER_SAMPLING_CUH_
#define FLASHINFER_SAMPLING_CUH_
// 这个文件实现了 FlashInfer 中与采样相关的核心 CUDA 逻辑，主要包括：
// 1. Softmax 与带温度的在线 softmax
// 2. 从 logits 或概率分布中直接采样
// 3. Top-K / Top-P / Min-P / Top-K+Top-P 采样
// 4. Top-P 重归一化
// 5. Chain speculative sampling（链式 speculative decoding）
//
// 使用方式上，一般不直接调用底层 __global__ kernel，
// 而是优先使用文件后半部分提供的 host 侧封装函数，例如：
// - OnlineSoftmax(...)
// - SamplingFromLogits(...)
// - SamplingFromProb(...)
// - TopKSamplingFromProb(...)
// - TopPSamplingFromProb(...)
// - MinPSamplingFromProb(...)
// - TopKTopPSamplingFromProb(...)
// - TopPRenormProb(...)
// - ChainSpeculativeSampling(...)

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cstdlib>
#include <cstring>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <limits>
#include <numeric>
#include <tuple>

#include "allocator.h"
#include "math.cuh"
#include "topk.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

// 根据 CUDA 版本选择归约算子类型。
// CUDA 13（12.9+）开始逐步弃用 cub::Max/Min，改为 cuda::maximum/minimum。
#if CUDA_VERSION >= 12090
using MaxReduceOp = cuda::maximum<>;
using MinReduceOp = cuda::minimum<>;
#else
using MaxReduceOp = cub::Max;
using MinReduceOp = cub::Min;
#endif

namespace flashinfer {

namespace sampling {

using namespace cub;

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
    if (deterministic) {                                          \
        constexpr bool DETERMINISTIC = true;                      \
        __VA_ARGS__                                               \
    } else {                                                      \
        constexpr bool DETERMINISTIC = false;                     \
        __VA_ARGS__                                               \
    }

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
    if (compute_capacity.first >= 8) {                                         \
        constexpr uint32_t BLOCK_THREADS = 1024;                               \
        __VA_ARGS__                                                            \
    } else {                                                                   \
        constexpr uint32_t BLOCK_THREADS = 512;                                \
        __VA_ARGS__                                                            \
    }

#define DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, ...) \
    if (cache_input) {                                              \
        constexpr bool CACHE_INPUT = true;                          \
        __VA_ARGS__                                                 \
    } else {                                                        \
        constexpr bool CACHE_INPUT = false;                         \
        __VA_ARGS__                                                 \
    }

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
#define FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
#endif

template <typename T>
struct ValueCount {
    T value;
    int count;

    __device__ ValueCount operator+(const ValueCount &other) const {
        return {value + other.value, count + other.count};
    }
    __device__ ValueCount &operator+=(const ValueCount &other) {
        value += other.value;
        count += other.count;
        return *this;
    }
};

struct BoolDiffOp {
    __device__ __forceinline__ bool operator()(const bool &lhs, const bool &rhs) const {
        return lhs != rhs;
    }
};

struct Float2SoftmaxReduceOp {
    __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const {
        if (isinf(a.x)) return b;
        if (isinf(b.x)) return a;

        float new_max = max(a.x, b.x);
        float new_denom = a.y * __expf(a.x - new_max) + b.y * __expf(b.x - new_max);
        __expf(b.x - new_max);
    }
};

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
    union {
        float deterministic_scan[BLOCK_THREADS / 32];
        typename BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
        typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
        typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
        typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
            reduce_value_count;
        typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
    } block_prim;
    struct {
        int32_t sampled_id;
        int32_t last_valid_id;
        float max_val;
        union {
            float value;
            ValueCount<float> pair;
        } block_aggregate;
    };
};

template <uint32_t BLOCK_THREADS>
struct OnlineSoftmaxTempStorage {
    union {
        typename cub::BlockReduce<float, BLOCK_THREADS>::TempStorage reduce;
        typename cub::BlockReduce<float2, BLOCK_THREADS>::TempStorage reduce_pair;
    } block_prim;

    struct {
        float max_val;
        float denominator;
    } shared_state;
};

struct PartialSoftmaxResult {
    float max_val;
    float denominator;
};

/*!
 * \brief 确定性的 inclusive scan 实现，使用 Belloch scan 算法。
 * \note 这个实现通常比 cub::BlockScan 更慢，但输出顺序是确定的，
 *       适合 deterministic 采样路径。
 */
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const float *in_data, float *out_data,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> *temp_storage) {
    float *smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
    float thread_data[VEC_SIZE];
    float thread_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        thread_sum += in_data[i];
        thread_data[i] = thread_sum;
    }

    float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
        float tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
        if ((threadIdx.x + 1) % (offset * 2) == 0) {
            thread_exclusive_prefix_sum += tmp;
        }
    }

    float warp_sum = __shfl_sync(0xffffffff, thread_exclusive_prefix_sum, threadIdx.x | 0xffffffff);
    if (threadIdx.x % 32 == 31) {
        thread_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset /= 2) {
        float tmp = __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
        if ((threadIdx.x + 1) % (offset * 2) == 0) {
            thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
        }
        if ((threadIdx.x + 1) % (offset * 2) == offset) {
            thread_exclusive_prefix_sum = tmp;
        }
    }

    smem_prefix_sum[threadIdx.x / 32] = warp_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        float warp_exclusive_prefix_sum =
            (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            float tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
            if ((threadIdx.x + 1) % (offset * 2) == 0) {
                warp_exclusive_prefix_sum += tmp;
            }
        }

        if (threadIdx.x % 32 == 31) {
            warp_exclusive_prefix_sum = 0;
        }

#pragma unroll
        for (uint32_t offset = 16; offset >= 1; offset /= 2) {
            float tmp = __shfl_xor_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
            if ((threadIdx.x + 1) % (offset * 2) == 0) {
                warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
            }
            if ((threadIdx.x + 1) % (offset * 2) == offset) {
                warp_exclusive_prefix_sum = tmp;
            }
        }
        if (threadIdx.x < BLOCK_THREADS / 32) {
            smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
        }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        out_data[i] = smem_prefix_sum[threadIdx.x / 32] + thread_exclusive_prefix_sum + thread_data[i];
    }
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename TempStorage>
__device__ __forceinline__ float GetMaxValue(float *in_data, uint32_t row_idx, uint32_t d,
                                             TempStorage &temp_storage) {
    const uint32_t tx = threadIdx.x;
    vec_t<float, VEC_SIZE> in_data_vec;

    // 每个线程先在自己负责的数据片段上求局部最大值，暂不立刻做块内归约。
    float thread_max = 0.0f;
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        in_data_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            thread_max = max(thread_max, static_cast<float>(in_data_vec[j]));
        }
    }

    // 循环结束后再做一次块级归约，得到整行的最大值。
    float max_val =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Reduce(thread_max, MaxReduceOp{});
    if (tx == 0) {
        temp_storage.max_val = max_val;
    }
    __syncthreads();
    return temp_storage.max_val;
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, bool CACHE_INPUT>
__global__ void OnlineSoftmaxFusedKernel(DType *logits, DType *output, DType *temperature_arr,
                                         DType temperature_val, uint32_t d) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
    const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

    using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
    extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
    auto &temp_storage = reinterpret_cast<TempStorage &>(smem);

    DType *smem_vec_base = nullptr;
    if constexpr (CACHE_INPUT) {
        constexpr size_t vec_alignment = alignof(vec_t<DType, VEC_SIZE>);
        size_t aligned_offset = round_up(sizeof(TempStorage), vec_alignment);
        smem_vec_base = reinterpret_cast<DType *>(smem + aligned_offset);
    }

    vec_t<DType, VEC_SIZE> logits_vec;

    float running_max = -cuda::std::numeric_limits<float>::infinity();
    float running_denominator = 0.0f;
    float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    // 第一遍：在线计算全局最大值和 softmax 分母。
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);

#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                logits_vec[j] *= inv_temp;
            }

            if constexpr (CACHE_INPUT) {
                logits_vec.store(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }
        }

        float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            thread_max = max(thread_max, logits_vec[j]);
        }
        float block_max = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                              .Reduce(thread_max, MaxReduceOp{});

        if (tx == 0) {
            temp_storage.shared_state.max_val = block_max;
        }
        __syncthreads();
        block_max = temp_storage.shared_state.max_val;
        // 如果 block_max 是 -inf，说明当前这块全是 -inf，可以直接跳过更新。
        if (!isinf(block_max)) {
            float threadlocal_sum = 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                threadlocal_sum += __expf(logits_vec[j] - block_max);
            }
            float new_max = max(running_max, block_max);
            threadlocal_running_denominator =
                threadlocal_running_denominator * __expf(running_max - new_max) + threadlocal_sum * __expf(block_max - new_max);
            running_max = new_max;
        }
    }

    running_denominator = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                              .Sum(threadlocal_running_denominator);
    if (tx == 0) {
        temp_storage.shared_state.denominator = running_denominator;
    }
    __syncthreads();
    running_denominator = temp_storage.shared_state.denominator;

    const float final_max = running_max;
    const float inv_denominator = 1.0f / running_denominator;

    // 第二遍：根据最终 max 和 denominator 做归一化，写出 softmax 概率。
    vec_t<DType, VEC_SIZE> prob_vec;
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        if constexpr (CACHE_INPUT) {
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                logits_vec.load(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }
        } else {
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);

#pragma unroll
                for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                    logits_vec[j] *= inv_temp;
                }
            }
        }

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
            prob_vec[j] = static_cast<DType>(p);
        }

        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType>
__global__ void OnlineSoftmaxMapKernel(DType *logits, PartialSoftmaxResult *partial_results,
                                       DType *temperature_arr, float temperature_val, uint32_t d,
                                       uint32_t num_slices) {
    const uint32_t bx = blockIdx.x;
    const uint32_t by = blockIdx.y; // slice index
    const uint32_t tx = threadIdx.x;
    float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
    const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

    const uint32_t vec_alignment_elems = alignof(vec_t<DType, VEC_SIZE>) / sizeof(DType);
    const uint32_t slice_stride = round_up(ceil_div(d, num_slices), vec_alignment_elems);
    const uint32_t slice_start = by * slice_stride;
    const uint32_t slice_size = min((by + 1) * slice_stride, d) - slice_start;

    if (slice_start >= d) return;

    using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
    extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
    auto &temp_storage = reinterpret_cast<TempStorage &>(smem);

    vec_t<DType, VEC_SIZE> logits_vec;
    float running_max = -cuda::std::numeric_limits<float>::infinity();
    float running_denominator = 0.0f;
    float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(slice_size, BLOCK_THREADS * VEC_SIZE); ++i) {
        logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < slice_size) {
            logits_vec.cast_load(logits + bx * d + slice_start + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

        float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            logits_vec[j] *= inv_temp;
            thread_max = max(thread_max, logits_vec[j]);
        }

        float block_max = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                              .Reduce(thread_max, MaxReduceOp{});

        if (tx == 0) {
            temp_storage.shared_state.max_val = block_max;
        }
        __syncthreads();
        block_max = temp_storage.shared_state.max_val;

        // 如果 block_max 是 -inf，说明当前 slice 全是 -inf，可以直接跳过更新。
        if (!isinf(block_max)) {
            float threadlocal_sum = 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                threadlocal_sum += __expf(logits_vec[j] - block_max);
            }
            float new_max = max(running_max, block_max);
            threadlocal_running_denominator =
                threadlocal_running_denominator * __expf(running_max - new_max) + threadlocal_sum * __expf(block_max - new_max);
            running_max = new_max;
        }
    }

    running_denominator = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                              .Sum(threadlocal_running_denominator);
    if (tx == 0) {
        temp_storage.shared_state.denominator = running_denominator;
    }
    __syncthreads();
    running_denominator = temp_storage.shared_state.denominator;

    if (tx == 0) {
        partial_results[bx * num_slices + by] = {running_max, running_denominator};
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType>
__global__ void OnlineSoftmaxReduceKernel(DType *logits, DType *output,
                                          PartialSoftmaxResult *partial_results,
                                          DType *temperature_arr, float temperature_val, uint32_t d,
                                          uint32_t num_slices) {
    const uint32_t bx = blockIdx.x;
    const uint32_t tx = threadIdx.x;
    float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
    const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

    // 把各个 slice 的部分 softmax 结果归并成整行结果。
    using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
    extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
    auto &temp_storage = reinterpret_cast<TempStorage &>(smem);

    const Float2SoftmaxReduceOp reduce_op;

    float2 thread_aggregate = make_float2(-cuda::std::numeric_limits<float>::infinity(), 0.0f);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    for (uint32_t i = tx; i < num_slices; i += BLOCK_THREADS) {
        PartialSoftmaxResult partial = partial_results[bx * num_slices + i];
        float2 partial_pair = make_float2(partial.max_val, partial.denominator);
        thread_aggregate = reduce_op(thread_aggregate, partial_pair);
    }

    float2 block_result = cub::BlockReduce<float2, BLOCK_THREADS>(temp_storage.block_prim.reduce_pair)
                              .Reduce(thread_aggregate, reduce_op);

    if (tx == 0) {
        temp_storage.shared_state.max_val = block_result.x;
        temp_storage.shared_state.denominator = block_result.y;
    }
    __syncthreads();

    block_result =
        make_float2(temp_storage.shared_state.max_val, temp_storage.shared_state.denominator);

    const float final_max = temp_storage.shared_state.max_val;
    const float inv_denominator = 1.0f / temp_storage.shared_state.denominator;

    // 用归并后的最终 max / denominator 对整行做归一化。
    vec_t<DType, VEC_SIZE> logits_vec;
    vec_t<DType, VEC_SIZE> prob_vec;

    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            logits_vec[j] *= inv_temp;
            float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
            prob_vec[j] = static_cast<DType>(p);
        }

        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float &aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> *temp_storage) {
    const uint32_t tx = threadIdx.x;
    float prob_greater_than_threshold[VEC_SIZE];
    float inclusive_cdf[VEC_SIZE];
    bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
        valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
    }
    float aggregate_local =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
            .template Sum<VEC_SIZE>(prob_greater_than_threshold);
    if (tx == 0) {
        temp_storage->block_aggregate.value = aggregate_local;
    }
    __syncthreads();
    aggregate_local = temp_storage->block_aggregate.value;

    if (aggregate + aggregate_local > u) {
        if constexpr (DETERMINISTIC) {
            DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
                prob_greater_than_threshold, inclusive_cdf, temp_storage);
        } else {
            BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
                .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

            __syncthreads();
        }

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
        }

        bool greater_than_u_diff[VEC_SIZE];
#ifdef FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
        BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
            .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u_diff, BoolDiffOp());
#else
        BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
            .template FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
#endif
        __syncthreads();

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            if (greater_than_u_diff[j]) {
                atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
            }
        }
        __syncthreads();
    }

    // 更新当前扫描范围内最后一个有效下标。
    int valid_index[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        if (valid[j]) {
            valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
        } else {
            valid_index[j] = -1;
        }
    }
    int max_valid_index =
        BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
            .Reduce(valid_index, MaxReduceOp{});
    if (tx == 0 && max_valid_index != -1 && max_valid_index < (int)d) {
        temp_storage->last_valid_id = max_valid_index;
    }
    __syncthreads();
    aggregate += aggregate_local;
}

template <typename DType, typename IdType>
struct DataAndIndex {
    DType data;
    IdType index;

    __device__ DataAndIndex operator+(const DataAndIndex &other) const {
        if (data > other.data) {
            return {data, index};
        } else {
            return {other.data, other.index};
        }
    }
    __device__ DataAndIndex &operator+=(const DataAndIndex &other) {
        if (data > other.data) {
            return *this;
        } else {
            data = other.data;
            index = other.index;
            return *this;
        }
    }
};

template <typename DType, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<DType, VEC_SIZE> GenerateGumbelNoise(uint64_t philox_seed,
                                                                      uint64_t philox_offset,
                                                                      uint64_t subsequence) {
    curandStatePhilox4_32_10_t state;
    vec_t<float, VEC_SIZE> noise;
    constexpr float kSCALE = 1.0f - cuda::std::numeric_limits<float>::epsilon();
    constexpr float kLOG2 = 0.6931471806f;
    auto uniform2gumbel = [](float x) {
        // 说明：
        // 1. cuRAND 返回的是严格大于 0 的正规浮点数，最大可能取到 1.0。
        // 2. 这里用 kSCALE 把 1.0 压到 1.0 - epsilon，避免后续出现 log(0) 或数值边界问题。
        // 3. 这样可以保证：
        //      1.18e-38 <= x * kSCALE <= 1.0f - epsilon
        //   从而：
        //      -4.47 <= -log(-log(...)) <= 15.9
        // 4. 在 NVIDIA GPU 上，log2f 会直接映射到一条 PTX 的 LG2 指令；
        //   而 logf 内部通常还要再乘 ln(2)。因此这里用 log2f 再乘一个常数更高效。
        return -kLOG2 * log2f(-log2f((x * kSCALE)));
    };
#pragma unroll
    for (uint32_t i = 0; i + 4 <= VEC_SIZE; i += 4) {
        curand_init(philox_seed, subsequence + i, philox_offset, &state);
        float4 noise_vec = curand_uniform4(&state);
        noise[i] = uniform2gumbel(noise_vec.x);
        noise[i + 1] = uniform2gumbel(noise_vec.y);
        noise[i + 2] = uniform2gumbel(noise_vec.z);
        noise[i + 3] = uniform2gumbel(noise_vec.w);
    }
    if constexpr (VEC_SIZE % 4 != 0) {
        curand_init(philox_seed, subsequence + VEC_SIZE / 4 * 4, philox_offset, &state);
        float4 noise_vec = curand_uniform4(&state);
        if constexpr (VEC_SIZE % 4 == 1) {
            noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.x);
        } else if constexpr (VEC_SIZE % 4 == 2) {
            noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.x);
            noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.y);
        } else if constexpr (VEC_SIZE % 4 == 3) {
            noise[VEC_SIZE - 3] = uniform2gumbel(noise_vec.x);
            noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.y);
            noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.z);
        }
    }

    if constexpr (std::is_same_v<DType, float>) {
        return noise;
    } else {
        vec_t<DType, VEC_SIZE> ret;
#pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            ret[i] = static_cast<DType>(noise[i]);
        }
        return ret;
    }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void SamplingFromLogitsKernel(DType *logits, IdType *output, IdType *indices, uint32_t d,
                                         uint64_t *seed_arr, uint64_t seed_val,
                                         uint64_t *offset_arr, uint64_t offset_val) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
    using SharedMem = typename BlockReduce<DataAndIndex<DType, IdType>, BLOCK_THREADS,
                                           REDUCE_ALGORITHM>::TempStorage;
    extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
    auto &temp_storage = reinterpret_cast<SharedMem &>(smem_sampling_logit);

    vec_t<DType, VEC_SIZE> logits_vec;
    DataAndIndex<DType, IdType> max_data = {-cuda::std::numeric_limits<DType>::infinity(), 0};
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            logits_vec.cast_load(logits + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }

        vec_t<DType, VEC_SIZE> gumbel_noise = GenerateGumbelNoise<DType, VEC_SIZE>(
            philox_seed, philox_offset,
            static_cast<uint64_t>(bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE));
        DataAndIndex<DType, IdType> cur_data[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            cur_data[j].data = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d ? logits_vec[j] + gumbel_noise[j] : -cuda::std::numeric_limits<DType>::infinity();
            cur_data[j].index = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
        }

        max_data +=
            BlockReduce<DataAndIndex<DType, IdType>, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage)
                .template Sum<VEC_SIZE>(cur_data);
    }
    if (tx == 0) {
        output[bx] = max_data.index;
    }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void SamplingFromProbKernel(DType *probs, IdType *output, bool *valid, IdType *indices,
                                       uint32_t d, uint64_t *seed_arr, uint64_t seed_val,
                                       uint64_t *offset_arr, uint64_t offset_val) {
    curandStatePhilox4_32_10_t state;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    curand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto &temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> &>(
            smem_sampling);
    temp_storage.sampled_id = d;
    temp_storage.last_valid_id = -1;
    __syncthreads();

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate(0);
    float u = curand_uniform(&state);

#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }

        DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                               DETERMINISTIC>(
            i, d, [](float x) { return x > 0; }, u, probs_vec, aggregate, &temp_storage);
        if (float(aggregate) > u) {
            break;
        }
    }
    int sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
        // 这个情况通常发生在：
        // 1. 随机数 u 非常接近 1
        // 2. 当前可采样概率总和略小于 u（例如数值误差或输入分布本身未完全归一）
        // 这时退化为使用最后一个有效下标作为采样结果。
        if (temp_storage.last_valid_id == -1) {
            if (tx == 0) {
                output[bx] = 0;
                valid[bx] = false;
            }
            return;
        }
        sampled_id = temp_storage.last_valid_id;
    }
    if (tx == 0) {
        output[bx] = sampled_id;
        valid[bx] = true;
    }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKSamplingFromProbKernel(DType *probs, IdType *output, bool *valid,
                                           IdType *indices, IdType *top_k_arr, uint32_t top_k_val,
                                           uint32_t d, uint64_t *seed_arr, uint64_t seed_val,
                                           uint64_t *offset_arr, uint64_t offset_val) {
    const uint32_t batch_size = gridDim.x;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    curandStatePhilox4_32_10_t state;
    curand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[bx];
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto &temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> &>(
            smem_sampling);

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate;
    float q = 1;
    double low = 0, high = 1.f;
    int sampled_id;
    int round = 0;
    do {
        round += 1;
        temp_storage.sampled_id = d;
        temp_storage.last_valid_id = -1;
        __syncthreads();
        float u = curand_uniform(&state) * q;
        aggregate = 0;
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                                   DETERMINISTIC>(
                i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
            if (aggregate > u) {
                break;
            }
        }
        __syncthreads();
        sampled_id = temp_storage.sampled_id;
        if (sampled_id == d) {
            // 这个情况通常发生在随机数非常接近 1，且累计概率和略小于 u 时。
            // 这里退化为使用最后一个有效下标作为采样结果。
            if (temp_storage.last_valid_id == -1) {
                if (tx == 0) {
                    output[bx] = 0;
                    valid[bx] = false;
                }
                return;
            }
            sampled_id = temp_storage.last_valid_id;
        }
        double pivot_0 = probs[row_idx * d + sampled_id];
        double pivot_1 = (pivot_0 + high) / 2;

        ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
        ValueCount<float> threadlocal_gt_pivot_0{0, 0}, threadlocal_gt_pivot_1{0, 0};
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                probs_gt_pivot_0[j] = {
                    (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
                    (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
                probs_gt_pivot_1[j] = {
                    (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
                    (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
                threadlocal_gt_pivot_0 += probs_gt_pivot_0[j];
                threadlocal_gt_pivot_1 += probs_gt_pivot_1[j];
            }
        }
        aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .Sum(threadlocal_gt_pivot_0);
        if (tx == 0) {
            temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
        }
        __syncthreads();
        aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

        aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .Sum(threadlocal_gt_pivot_1);
        if (tx == 0) {
            temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
        }
        __syncthreads();
        aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
        if (aggregate_gt_pivot_0.count < k) {
            // 情况 1：pivot_0 已经满足约束，直接接受。
            break;
        }
        if (aggregate_gt_pivot_1.count < k) {
            // 情况 2：pivot_0 不满足，但 pivot_1 满足，继续缩小到 [pivot_0, pivot_1]。
            low = pivot_0;
            high = pivot_1;
            q = aggregate_gt_pivot_0.value;
        } else {
            // 情况 3：pivot_0 和 pivot_1 都不满足，说明阈值还要更高。
            low = pivot_1;
            q = aggregate_gt_pivot_1.value;
        }
    } while (low < high);
    __syncthreads();
    if (tx == 0) {
        output[bx] = sampled_id;
        valid[bx] = true;
    }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType *probs, IdType *output, bool *valid,
                                           IdType *indices, float *top_p_arr, float top_p_val,
                                           uint32_t d, uint64_t *seed_arr, uint64_t seed_val,
                                           uint64_t *offset_arr, uint64_t offset_val) {
    const uint32_t batch_size = gridDim.x;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    curandStatePhilox4_32_10_t state;
    curand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
    float top_p = (top_p_arr == nullptr) ? top_p_val : top_p_arr[row_idx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto &temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> &>(
            smem_sampling);

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate;
    float q = 1;
    double low = 0, high = 1.f;
    int sampled_id;
    do {
        temp_storage.sampled_id = d;
        temp_storage.last_valid_id = -1;
        __syncthreads();
        float u = curand_uniform(&state) * q;
        aggregate = 0;
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                                   DETERMINISTIC>(
                i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
            if (aggregate > u) {
                break;
            }
        }
        __syncthreads();
        sampled_id = temp_storage.sampled_id;
        if (sampled_id == d) {
            // 这个情况通常发生在随机数非常接近 1，且累计概率和略小于 u 时。
            // 这里退化为使用最后一个有效下标作为采样结果。
            if (temp_storage.last_valid_id == -1) {
                if (tx == 0) {
                    output[bx] = 0;
                    valid[bx] = false;
                }
                return;
            }
            sampled_id = temp_storage.last_valid_id;
        }
        double pivot_0 = probs[row_idx * d + sampled_id];
        double pivot_1 = (pivot_0 + high) / 2;

        float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
        float threadlocal_aggregate_gt_pivot_0 = 0;
        float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
                probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;
                threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
                threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
            }
        }
        aggregate_gt_pivot_0 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                    .Sum(threadlocal_aggregate_gt_pivot_0);
        if (tx == 0) {
            temp_storage.block_aggregate.value = aggregate_gt_pivot_0;
        }
        __syncthreads();
        aggregate_gt_pivot_0 = temp_storage.block_aggregate.value;

        aggregate_gt_pivot_1 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                    .Sum(threadlocal_aggregate_gt_pivot_1);
        if (tx == 0) {
            temp_storage.block_aggregate.value = aggregate_gt_pivot_1;
        }
        __syncthreads();
        aggregate_gt_pivot_1 = temp_storage.block_aggregate.value;

        if (aggregate_gt_pivot_0 < top_p) {
            // 情况 1：pivot_0 已经满足 top-p 约束，直接接受。
            break;
        }
        if (aggregate_gt_pivot_1 < top_p) {
            // 情况 2：pivot_0 不满足，但 pivot_1 满足，继续缩小阈值区间。
            low = pivot_0;
            high = pivot_1;
            q = aggregate_gt_pivot_0;
        } else {
            // 情况 3：两个候选阈值都不满足，需要继续提高阈值。
            low = pivot_1;
            q = aggregate_gt_pivot_1;
        }
    } while (low < high);
    __syncthreads();
    if (tx == 0) {
        output[bx] = sampled_id;
        valid[bx] = true;
    }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void MinPSamplingFromProbKernel(DType *probs, float *min_p_arr, IdType *output,
                                           bool *valid, IdType *indices, float min_p_val,
                                           uint32_t d, uint64_t *seed_arr, uint64_t seed_val,
                                           uint64_t *offset_arr, uint64_t offset_val) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    float p = (min_p_arr == nullptr) ? min_p_val : min_p_arr[bx];
    curandStatePhilox4_32_10_t state;
    curand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto &temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> &>(
            smem_sampling);

    float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                                SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>>(
        probs, row_idx, d, temp_storage);
    float pivot = max_val * p;

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate_gt_pivot = 0;
    float threadlocal_aggregate_gt_pivot = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

        float probs_gt_pivot[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            probs_gt_pivot[j] = (probs_vec[j] >= pivot) ? probs_vec[j] : 0;
            threadlocal_aggregate_gt_pivot += probs_gt_pivot[j];
        }
    }

    aggregate_gt_pivot += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                              .Sum(threadlocal_aggregate_gt_pivot);
    if (tx == 0) {
        temp_storage.block_aggregate.value = aggregate_gt_pivot;
    }
    __syncthreads();

    float aggregate = 0;
    float q = temp_storage.block_aggregate.value;

    int sampled_id;
    temp_storage.sampled_id = d;
    temp_storage.last_valid_id = -1;
    __syncthreads();
    float u = curand_uniform(&state) * q;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

        DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                               DETERMINISTIC>(
            i, d, [&](float x) { return x >= pivot; }, u, probs_vec, aggregate, &temp_storage);
        if (aggregate > u) {
            break;
        }
    }
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
        // 这个情况通常发生在随机数非常接近 1，且累计概率和略小于 u 时。
        // 这里退化为使用最后一个有效下标作为采样结果。
        if (temp_storage.last_valid_id == -1) {
            if (tx == 0) {
                output[bx] = 0;
                valid[bx] = false;
            }
            return;
        }
        sampled_id = temp_storage.last_valid_id;
    }
    output[bx] = sampled_id;
    valid[bx] = true;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType *probs, IdType *top_k_arr, float *top_p_arr,
                                               IdType *output, bool *valid, IdType *indices,
                                               IdType top_k_val, float top_p_val, uint32_t d,
                                               uint64_t *seed_arr, uint64_t seed_val,
                                               uint64_t *offset_arr, uint64_t offset_val) {
    const uint32_t batch_size = gridDim.x;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    curandStatePhilox4_32_10_t state;
    curand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
    const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
    const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto &temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> &>(
            smem_sampling);

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate;
    float q = 1;
    double low = 0, high = 1.f;
    int sampled_id;
    do {
        temp_storage.sampled_id = d;
        temp_storage.last_valid_id = -1;
        __syncthreads();
        float u = curand_uniform(&state) * q;
        aggregate = 0;
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                                   DETERMINISTIC>(
                i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
            if (aggregate > u) {
                break;
            }
        }
        __syncthreads();
        sampled_id = temp_storage.sampled_id;
        if (sampled_id == d) {
            // 这个情况通常发生在随机数非常接近 1，且累计概率和略小于 u 时。
            // 这里退化为使用最后一个有效下标作为采样结果。
            sampled_id = temp_storage.last_valid_id;
            if (temp_storage.last_valid_id == -1) {
                if (tx == 0) {
                    output[bx] = 0;
                    valid[bx] = false;
                }
                return;
            }
        }
        double pivot_0 = probs[row_idx * d + sampled_id];
        double pivot_1 = (pivot_0 + high) / 2;

        ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
        ValueCount<float> threadlocal_aggregate_gt_pivot_0{0, 0};
        ValueCount<float> threadlocal_aggregate_gt_pivot_1{0, 0};
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                probs_gt_pivot_0[j] = {
                    (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
                    (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
                probs_gt_pivot_1[j] = {
                    (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
                    (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
                threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
                threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
            }
        }
        aggregate_gt_pivot_0 +=
            BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
                .Sum(threadlocal_aggregate_gt_pivot_0);
        if (tx == 0) {
            temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
        }
        __syncthreads();
        aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

        aggregate_gt_pivot_1 +=
            BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
                .Sum(threadlocal_aggregate_gt_pivot_1);
        if (tx == 0) {
            temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
        }
        __syncthreads();
        aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
        if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
            // 情况 1：pivot_0 同时满足 top-k 与 top-p 约束，直接接受。
            break;
        }
        if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
            // 情况 2：pivot_0 不满足，但 pivot_1 满足，继续缩小阈值区间。
            low = pivot_0;
            high = pivot_1;
            q = aggregate_gt_pivot_0.value;
        } else {
            // 情况 3：两个候选阈值都不满足，需要继续提高阈值。
            low = pivot_1;
            q = aggregate_gt_pivot_1.value;
        }
    } while (low < high);
    __syncthreads();
    if (tx == 0) {
        output[bx] = sampled_id;
        valid[bx] = true;
    }
}

template <typename DType>
// Host 侧的在线 softmax 封装入口。
// 使用方式：
// 1. 输入 logits 形状为 [batch_size, d]
// 2. output 用于写出 softmax 结果，形状同 logits
// 3. temperature_arr 为 nullptr 时，整批共用 temperature_val；
//    否则按行从 temperature_arr 中读取温度
// 4. workspace_buffer 只在“小 batch + 大词表”的分片 softmax 路径中会用到
// 5. enable_pdl 打开后会尝试启用 programmatic dependent launch
cudaError_t OnlineSoftmax(DType *logits, DType *output, uint32_t batch_size, uint32_t d,
                          DType *temperature_arr, DType temperature_val, void *workspace_buffer,
                          size_t workspace_buffer_size_in_bytes, bool enable_pdl,
                          cudaStream_t stream = 0) {
    constexpr uint32_t SMALL_BATCH_THRESHOLD = 128;
    constexpr uint32_t LARGE_VOCAB_THRESHOLD = 24576;
    constexpr uint32_t DEFAULT_SLICE_SIZE = 8192;

    const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);
    auto compute_capacity = GetCudaComputeCapability();

    DISPATCH_COMPUTE_CAP_NUM_THREADS(
        compute_capacity, BLOCK_THREADS, {DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
            if (batch_size <= SMALL_BATCH_THRESHOLD && d >= LARGE_VOCAB_THRESHOLD) {
                // 路径 A：小 batch、大词表时，采用按词表切片的 Map-Reduce 策略。
                uint32_t num_slices = ceil_div(d, DEFAULT_SLICE_SIZE);

                const size_t partial_buffer_size = batch_size * num_slices * sizeof(PartialSoftmaxResult);
                if (workspace_buffer_size_in_bytes < partial_buffer_size) {
                    return cudaErrorInvalidValue;
                }

                AlignedAllocator allocator(workspace_buffer, workspace_buffer_size_in_bytes);
                auto partial_results = allocator.aligned_alloc<PartialSoftmaxResult>(
                    partial_buffer_size, alignof(PartialSoftmaxResult), "softmax_workspace");

                // 阶段 1：对词表切片做 map-reduce，得到每个 slice 的部分 softmax 结果。
                dim3 phase1_nblks(batch_size, num_slices);
                dim3 phase1_nthrs(BLOCK_THREADS);
                size_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>);

                auto phase1_kernel = OnlineSoftmaxMapKernel<BLOCK_THREADS, VEC_SIZE, DType>;
                void *phase1_args[] = {&logits, &partial_results, &temperature_arr, &temperature_val,
                                       &d, &num_slices};

                FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                    phase1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

                if (enable_pdl) {
                    cudaLaunchAttribute attribute[1];
                    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    attribute[0].val.programmaticStreamSerializationAllowed = 1;

                    cudaLaunchConfig_t config;
                    config.gridDim = phase1_nblks;
                    config.blockDim = phase1_nthrs;
                    config.dynamicSmemBytes = smem_size;
                    config.stream = stream;
                    config.attrs = attribute;
                    config.numAttrs = 1;

                    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, phase1_kernel, logits, partial_results,
                                                            temperature_arr, temperature_val, d,
                                                            num_slices));
                } else {
                    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)phase1_kernel, phase1_nblks, phase1_nthrs,
                                                          phase1_args, smem_size, stream));
                }

                // 阶段 2：把所有 slice 的结果合并，再对整行应用归一化。
                dim3 phase2_nblks(batch_size);
                dim3 phase2_nthrs(BLOCK_THREADS);

                auto phase2_kernel = OnlineSoftmaxReduceKernel<BLOCK_THREADS, VEC_SIZE, DType>;
                void *phase2_args[] = {&logits, &output, &partial_results, &temperature_arr,
                                       &temperature_val, &d, &num_slices};

                FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                    phase2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

                if (enable_pdl) {
                    cudaLaunchAttribute attribute[1];
                    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    attribute[0].val.programmaticStreamSerializationAllowed = 1;

                    cudaLaunchConfig_t config;
                    config.gridDim = phase2_nblks;
                    config.blockDim = phase2_nthrs;
                    config.dynamicSmemBytes = smem_size;
                    config.stream = stream;
                    config.attrs = attribute;
                    config.numAttrs = 1;

                    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, phase2_kernel, logits, output,
                                                            partial_results, temperature_arr,
                                                            temperature_val, d, num_slices));
                } else {
                    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)phase2_kernel, phase2_nblks, phase2_nthrs,
                                                          phase2_args, smem_size, stream));
                }
            } else {
                // 路径 B：单 block 处理整行。
                // 根据行长度判断是否把输入缓存到 shared memory 里，以减少二次读取开销。
                uint32_t cache_threshold;
                if (batch_size <= 16) {
                    cache_threshold = 4096;
                } else if (batch_size <= 32) {
                    cache_threshold = 2048;
                } else {
                    cache_threshold = 0;
                }
                const bool cache_input = d <= cache_threshold;

                dim3 nblks(batch_size);
                dim3 nthrs(BLOCK_THREADS);
                void *args[] = {&logits, &output, &temperature_arr, &temperature_val, &d};

                const size_t smem_logits_bytes = (round_up(d, VEC_SIZE) + VEC_SIZE) * sizeof(DType);

                uint32_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>) + (cache_input ? smem_logits_bytes : 0);

                DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, {
                    auto kernel = OnlineSoftmaxFusedKernel<BLOCK_THREADS, VEC_SIZE, DType, CACHE_INPUT>;
                    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

                    if (enable_pdl) {
                        cudaLaunchAttribute attribute[1];
                        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                        attribute[0].val.programmaticStreamSerializationAllowed = 1;

                        cudaLaunchConfig_t config;
                        config.gridDim = nblks;
                        config.blockDim = nthrs;
                        config.dynamicSmemBytes = smem_size;
                        config.stream = stream;
                        config.attrs = attribute;
                        config.numAttrs = 1;

                        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, logits, output,
                                                                temperature_arr, temperature_val, d));
                    } else {
                        FLASHINFER_CUDA_CALL(
                            cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
                    }
                });
            }
        })});
    return cudaSuccess;
}

template <typename T, typename IdType>
// 从 logits 中直接采样。
// 实现上通过“logits + Gumbel 噪声”转成 argmax，等价于按 softmax 分布采样。
// 常用于不显式先做 softmax 的场景。
cudaError_t SamplingFromLogits(T *logits, IdType *output, IdType *indices, uint32_t batch_size,
                               uint32_t d, bool deterministic, uint64_t *seed_arr,
                               uint64_t seed_val, uint64_t *offset_arr, uint64_t offset_val,
                               cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&logits, &output, &indices, &d, &seed_arr, &seed_val, &offset_arr, &offset_val};
        const uint32_t smem_size = sizeof(
            typename BlockReduce<DataAndIndex<T, IdType>, BLOCK_THREADS, REDUCE_ALGO>::TempStorage);

        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = SamplingFromLogitsKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                       DETERMINISTIC, T, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

template <typename T, typename IdType>
// 从概率分布 probs 中直接采样。
// 输入 probs 通常是 [batch_size, d]，每行表示一个离散分布。
// output 写出采样结果；valid 标记本次采样是否有效。
cudaError_t SamplingFromProb(T *probs, IdType *output, bool *valid, IdType *indices,
                             uint32_t batch_size, uint32_t d, bool deterministic,
                             uint64_t *seed_arr, uint64_t seed_val, uint64_t *offset_arr,
                             uint64_t offset_val, cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&probs, &output, &valid, &indices, &d,
                        &seed_arr, &seed_val, &offset_arr, &offset_val};
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);

        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = SamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                     DETERMINISTIC, T, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

template <typename T, typename IdType>
// Top-K 采样：
// 先把候选集合限制在 top-k 范围内，再在该范围内进行采样。
// top_k_arr 为空时整批共用 top_k_val，否则按行读取各自的 k。
cudaError_t TopKSamplingFromProb(T *probs, IdType *output, bool *valid, IdType *indices,
                                 T *top_k_arr, uint32_t batch_size, uint32_t top_k_val, uint32_t d,
                                 bool deterministic, uint64_t *seed_arr, uint64_t seed_val,
                                 uint64_t *offset_arr, uint64_t offset_val,
                                 cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&probs, &output, &valid, &indices, &top_k_arr, &top_k_val,
                        &d, &seed_arr, &seed_val, &offset_arr, &offset_val};

        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = TopKSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                         DETERMINISTIC, T, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

template <typename T, typename IdType>
// Top-P 采样：
// 动态寻找一个概率阈值，使得“大于该阈值”的概率质量刚好覆盖 top-p，
// 然后只在该集合中采样。
cudaError_t TopPSamplingFromProb(T *probs, IdType *output, bool *valid, IdType *indices,
                                 T *top_p_arr, uint32_t batch_size, T top_p_val, uint32_t d,
                                 bool deterministic, uint64_t *seed_arr, uint64_t seed_val,
                                 uint64_t *offset_arr, uint64_t offset_val,
                                 cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&probs, &output, &valid, &indices, &top_p_arr, &top_p_val,
                        &d, &seed_arr, &seed_val, &offset_arr, &offset_val};

        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = TopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                         DETERMINISTIC, T, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

template <typename T, typename IdType>
// Min-P 采样：
// 先找到当前行最大概率 max_val，再令阈值为 max_val * min_p，
// 只在不小于该阈值的候选上采样。
cudaError_t MinPSamplingFromProb(T *probs, T *min_p_arr, IdType *output, bool *valid,
                                 IdType *indices, uint32_t batch_size, float min_p_val, uint32_t d,
                                 bool deterministic, uint64_t *seed_arr, uint64_t seed_val,
                                 uint64_t *offset_arr, uint64_t offset_val,
                                 cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&probs, &min_p_arr, &output, &valid, &indices, &min_p_val,
                        &d, &seed_arr, &seed_val, &offset_arr, &offset_val};

        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = MinPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                         DETERMINISTIC, T, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

template <typename T, typename IdType>
// 同时满足 Top-K 与 Top-P 约束的采样。
// 只有同时满足“排名在前 k 个以内”且“位于 nucleus 集合内”的候选才会被保留。
cudaError_t TopKTopPSamplingFromProb(T *probs, IdType *top_k_arr, T *top_p_arr, IdType *output,
                                     bool *valid, IdType *indices, uint32_t batch_size,
                                     IdType top_k_val, T top_p_val, uint32_t d, bool deterministic,
                                     uint64_t *seed_arr, uint64_t seed_val, uint64_t *offset_arr,
                                     uint64_t offset_val, cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&probs, &top_k_arr, &top_p_arr, &output, &valid,
                        &indices, &top_k_val, &top_p_val, &d, &seed_arr,
                        &seed_val, &offset_arr, &offset_val};

        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = TopKTopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                             VEC_SIZE, DETERMINISTIC, T, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage {
    union {
        typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
        typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
        typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
            reduce_value_count;
    } block_prim;
    struct {
        float max_val;
        float min_val;
        float row_sum;
        union {
            struct {
                float values[2];
            };
            struct {
                int counts[2];
            };
            struct {
                ValueCount<float> pairs[2];
            };
        } block_aggregate;
    };
};

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType>
__global__ void TopPRenormProbKernel(DType *probs, DType *renormed_prob, float *top_p_arr,
                                     float top_p_val, uint32_t d) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    const uint32_t row_idx = bx;
    float p = top_p_arr == nullptr ? top_p_val : top_p_arr[bx];

    extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
        uint8_t smem_renorm[];
    auto &temp_storage =
        reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO> &>(smem_renorm);
    vec_t<float, VEC_SIZE> probs_vec;

    // 快路径：当 p >= 1.0（例如 p == 1.0）时，不需要做 top-p 截断，
    // 直接求和并归一化即可。
    if (p >= 1.0f) {
        // 阶段 A：每个线程在自己负责的向量化片段上累计局部和。
        float thread_sum = 0.0f;
        const uint32_t num_iters = ceil_div(d, BLOCK_THREADS * VEC_SIZE);
        for (uint32_t i = 0; i < num_iters; ++i) {
            probs_vec.fill(0.0f);
            const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
            if (base_idx < d) {
                probs_vec.cast_load(probs + row_idx * d + base_idx);
            }
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                const uint32_t idx = base_idx + j;
                if (idx < d) thread_sum += probs_vec[j];
            }
        }

        // 做一次块级归约，得到整行总和。
        float row_sum =
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                .Sum(thread_sum);
        // 通过 shared memory 把总和广播给整个线程块。
        if (tx == 0) temp_storage.row_sum = row_sum;
        __syncthreads();
        row_sum = temp_storage.row_sum;

        // 防止总和为 0 导致除零。
        const float denom = (row_sum <= 1e-8f) ? 1.0f : row_sum;
        const float normalizer = math::ptx_rcp(denom);

        // 阶段 B：按总和归一化并写出结果。
        for (uint32_t i = 0; i < num_iters; ++i) {
            probs_vec.fill(0.0f);
            const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
            if (base_idx < d) {
                probs_vec.cast_load(probs + row_idx * d + base_idx);
            }
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                const uint32_t idx = base_idx + j;
                float v = probs_vec[j];
                probs_vec[j] = (idx < d) ? (v * normalizer) : 0.0f;
            }
            if (base_idx < d) {
                probs_vec.cast_store(renormed_prob + row_idx * d + base_idx);
            }
        }
        return; // 快路径完成后直接返回。
    }

    // 常规 Top-P 重归一化逻辑。
    temp_storage.max_val = 0;
    float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                                RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(probs, row_idx, d,
                                                                                    temp_storage);

    double low = 0, high = max_val;
    float min_gt_low, max_le_high;
    float sum_low = 1;
    // 记 f(x) = sum(probs[probs > x])，它是关于 x 的单调不增函数。
    // min_gt_low 表示所有大于 low 的概率中的最小值；
    // max_le_high 表示所有小于等于 high 的概率中的最大值。
    // 循环不变量：
    // 1. f(low) >= p，f(high) < p
    // 2. f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
    // 停止条件：
    // 1. f(low) >= p
    // 2. f(min_gt_low) == f(max_le_high) == f(high) < p
    do {
        double pivot_0 = (high + 2 * low) / 3;
        double pivot_1 = (2 * high + low) / 3;

        float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
        min_gt_low = high;
        max_le_high = low;
        float threadlocal_aggregate_gt_pivot_0 = 0;
        float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
        for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
            probs_vec.fill(0);
            if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
                probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }

            float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
            for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
                probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;

                if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
                    min_gt_low = min(min_gt_low, probs_vec[j]);
                }
                if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
                    max_le_high = max(max_le_high, probs_vec[j]);
                }
                threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
                threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
            }
        }
        aggregate_gt_pivot_0 =
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                .Sum(threadlocal_aggregate_gt_pivot_0);
        __syncthreads();
        aggregate_gt_pivot_1 =
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                .Sum(threadlocal_aggregate_gt_pivot_1);
        __syncthreads();

        min_gt_low = BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                         .Reduce(min_gt_low, MinReduceOp{});
        __syncthreads();
        max_le_high =
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                .Reduce(max_le_high, MaxReduceOp{});
        if (tx == 0) {
            temp_storage.block_aggregate.values[0] = aggregate_gt_pivot_0;
            temp_storage.block_aggregate.values[1] = aggregate_gt_pivot_1;
            temp_storage.min_val = min_gt_low;
            temp_storage.max_val = max_le_high;
        }
        __syncthreads();
        aggregate_gt_pivot_0 = temp_storage.block_aggregate.values[0];
        aggregate_gt_pivot_1 = temp_storage.block_aggregate.values[1];
        min_gt_low = temp_storage.min_val;
        max_le_high = temp_storage.max_val;

        if (aggregate_gt_pivot_1 >= p) {
            low = pivot_1;
            sum_low = aggregate_gt_pivot_1;
        } else if (aggregate_gt_pivot_0 >= p) {
            low = pivot_0;
            high = min(pivot_1, max_le_high);
            sum_low = aggregate_gt_pivot_0;
        } else {
            high = min(pivot_0, max_le_high);
        }
    } while (min_gt_low != max_le_high);

    float normalizer = math::ptx_rcp(max(sum_low, 1e-8));

    // 根据最终阈值做归一化，只保留大于 low 的项。
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            probs_vec[j] = (probs_vec[j] > low) ? probs_vec[j] * normalizer : 0;
        }
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            probs_vec.cast_store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
    }
}

template <typename DType>
// Top-P 重归一化入口。
// 使用方式：
// 1. probs 是输入概率张量 [batch_size, d]
// 2. renormed_prob 是输出张量，形状同 probs
// 3. top_p_arr 为空时整批共用 top_p_val，否则按行读取各自的 top-p
// 4. 输出会把 nucleus 集合之外的元素置 0，并把保留部分重新归一化到和为 1
cudaError_t TopPRenormProb(DType *probs, DType *renormed_prob, float *top_p_arr,
                           uint32_t batch_size, float top_p_val, uint32_t d,
                           cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&probs, &renormed_prob, &top_p_arr, &top_p_val, &d};
        DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
            auto kernel = TopPRenormProbKernel<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType>;
            FLASHINFER_CUDA_CALL(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
        });
        return cudaSuccess;
    });
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void ChainSpeculativeSampling(DType *draft_probs, IdType *draft_token_ids,
                                         DType *target_probs, IdType *output_token_ids,
                                         IdType *output_accepted_token_num,
                                         IdType *output_emitted_draft_token_num,
                                         uint32_t num_speculative_tokens, uint32_t d,
                                         uint64_t *seed_arr, uint64_t seed_val,
                                         uint64_t *offset_arr, uint64_t offset_val) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    const uint32_t row_idx = bx;

    // 从张量参数或标量参数中解析 Philox 的 seed / offset。
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    curandStatePhilox4_32_10_t curand_state;
    curand_init(philox_seed, bx, philox_offset, &curand_state);

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto &temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> &>(
            smem_sampling);

    uint32_t pos = num_speculative_tokens;
    for (uint32_t i = 0; i < num_speculative_tokens; ++i) {
        IdType draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
        float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
              p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
        float u = curand_uniform(&curand_state);
        if (u * p < q) {
            // 接受 draft 模型给出的 token。
            output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
        } else {
            pos = i;
            break;
        }
    }

    uint32_t emitted_token_num = pos;
    uint32_t accepted_token_num = pos;
    for (uint32_t i = pos; i < num_speculative_tokens; ++i) {
        int draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
        float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
              p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
        float u = curand_uniform(&curand_state);
        if (u * p < q) {
            ++accepted_token_num;
        }
    }

    if (tx == 0) {
        output_accepted_token_num[row_idx] += accepted_token_num;
        output_emitted_draft_token_num[row_idx] += emitted_token_num;
    }

    // 在 relu(target_probs - draft_probs) 这个修正分布上做一次采样。
    float sum_relu_q_minus_p = 0;
    vec_t<float, VEC_SIZE> q_vec, p_vec;
    float relu_q_minus_p[VEC_SIZE];
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        q_vec.fill(0);
        p_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            if (pos != num_speculative_tokens) {
                // bonus token 没有对应的 draft_probs。
                p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], 0.0f);
        }
        sum_relu_q_minus_p +=
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                .Sum<VEC_SIZE>(relu_q_minus_p);
        __syncthreads();
    }
    if (tx == 0) {
        temp_storage.block_aggregate.value = sum_relu_q_minus_p;
    }
    // 先把“第一个被拒绝位置”的输出初始化成 d，表示尚未找到。
    temp_storage.sampled_id = d;
    __syncthreads();
    sum_relu_q_minus_p = temp_storage.block_aggregate.value;
    float u = curand_uniform(&curand_state) * sum_relu_q_minus_p;

    float aggregate_relu_q_minus_p(0);
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        q_vec.fill(0);
        p_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            if (pos != num_speculative_tokens) {
                // bonus token 没有对应的 draft_probs。
                p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }
        }

        vec_t<float, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], 0.0f);
        }

        DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                               DETERMINISTIC>(
            i, d, [&](float x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p,
            &temp_storage);
        if (aggregate_relu_q_minus_p > u) {
            break;
        }
    }
    __syncthreads();
    int sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
        // 这个情况通常发生在随机数非常接近 1，且累计概率和略小于 u 时。
        // 这里退化为使用最后一个有效下标作为采样结果。
        sampled_id = temp_storage.last_valid_id;
    }
    // 写入第一个被拒绝位置的新采样 token。
    output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = sampled_id;
    // 移动到下一个位置。
    pos++;

    // 其余位置全部填成 -1，表示无效输出。
    for (; pos < num_speculative_tokens + 1; ++pos) {
        output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = -1;
    }
}

template <typename DType, typename IdType>
// 链式 speculative sampling 的 host 侧入口。
// 使用方式：
// 1. draft_probs 是草稿模型对每一步的概率分布
// 2. draft_token_ids 是草稿模型生成的 token 序列
// 3. target_probs 是目标模型在每一步上的概率分布
// 4. output_token_ids 会写出最终接受/修正后的 token 序列
// 5. output_accepted_token_num 与 output_emitted_draft_token_num 分别统计接受数与直接发射数
cudaError_t ChainSpeculativeSampling(
    DType *draft_probs, IdType *draft_token_ids, DType *target_probs, IdType *output_token_ids,
    IdType *output_accepted_token_num, IdType *output_emitted_draft_token_num, uint32_t batch_size,
    uint32_t num_speculative_tokens, uint32_t d, bool deterministic, uint64_t *seed_arr,
    uint64_t seed_val, uint64_t *offset_arr, uint64_t offset_val, cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void *args[] = {&draft_probs,
                        &draft_token_ids,
                        &target_probs,
                        &output_token_ids,
                        &output_accepted_token_num,
                        &output_emitted_draft_token_num,
                        &num_speculative_tokens,
                        &d,
                        &seed_arr,
                        &seed_val,
                        &offset_arr,
                        &offset_val};
        DISPATCH_ALIGNED_VEC_SIZE(
            vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
                auto kernel = ChainSpeculativeSampling<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                       DETERMINISTIC, DType, IdType>;
                FLASHINFER_CUDA_CALL(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                FLASHINFER_CUDA_CALL(
                    cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
            })});
        return cudaSuccess;
    });
}

}

} // namespace flashinfer::sampling

#endif // FLASHINFER_SAMPLING_CUH_
