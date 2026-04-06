#ifndef FLASHINFER_TOPK_CUH_
#define FLASHINFER_TOPK_CUH_
#include <cuda.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <numeric>
#include <type_traits>

#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace sampling {

// CUDA内核中计算可用共享内存的模板函数
template <uint32_t BLOCK_THREADS>
inline size_t GetRadixTopKAvailableOrderedSmemBytes() {
    // BLOCK_THREADS 是 CUDA block中的线程数
    using RadixTopKDetBlockScanT = cub::BlockScan<uint32_t, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    constexpr size_t RADIX_TOPK_DETERMINISTIC_BLOCK_SCAN_SMEM = sizeof(typename RadixTopKDetBlockScanT::TempStorage);
    // BlockScan算法，，BLOCK_SCAN_RAKING_MEMOIZE 是一种内存优化策略。这行计算了该 BlockScan 需要的共享内存大小
    constexpr size_t RADIX_TOPK_LAUNCH_SMEM_HEADROOM = 2 * RADIX_TOPK_DETERMINISTIC_BLOCK_SCAN_SMEM;
    // launch 时需要预留 2 倍的 BlockScan 共享内存作为 headroom
    const size_t launch_headroom = reserve_launch_headroom ? RADIX_TOPK_LAUNCH_SMEM_HEADROOM : size_t(0);
    // 如果需要预留，则使用 headroom，否则为 0
    if (max_smem_per_block <= fixed_smem_aligned + launch_headroom) {
        return 0;
    }
    // 为确定性radix内核预留足够的启动时headroom
    // 这些内核会实例化额外的静态共享scratch，如BlockScan temp storage。
    // 最后返回：可用共享内存 = 最大可用 - 固定占用 - launch headroom。如果不足以分配，则返回 0
    return max_smem_per_block - fixed_smem_aligned - launch_headroom;
}

// ============================================================================
// RadixTopK 类型 Traits - 支持 float, half, bfloat16
// OrderedType: float -> uint32_t, half/bf16 -> uint16_t
// NUM_ROUNDS = sizeof(OrderedType) * 8 / RADIX_BITS
// ============================================================================
template <typename DType>
struct RadixTopKTraits;

// float (32位) 类型特化
// Radix TopK 的 traits 特化，用于将 float 类型转换为可进行基数排序（radix sort）的 uint32_t 格式
/*
核心作用：
解决 float 的符号位问题：IEEE 754 标准中，负数的最高位是 1，正数是 0。但 radix sort 按位比较时，0 会排在 1 前面，导致负数排在正数后面。
*/
template <>
struct RadixTopKTraits<float> {
    using OrderedType = uint32_t;  // 有序类型为32位无符号整数

    // 计算基数排序所需的轮数（32位 / 8位 = 4轮）
    template <uint32_t RADIX_BITS>
    static __host__ __device__ constexpr uint32_t num_rounds() {
        return sizeof(OrderedType) * 8 / RADIX_BITS;
    }

    // 将 float 转换为有序整数表示（用于radix排序）
    // 处理符号位：正数最高位由0变1，负数最高位由1变0，使radix sort能正确处理负数
    __device__ __forceinline__ static OrderedType ToOrdered(float val) {
        uint32_t bits = __float_as_uint(val);  // 获取float的位表示
        // 符号位为1(负数)则按位取反，否则异或0x80000000
        return (bits & 0x80000000) ? !bits : (bits ^ 0x80000000);
    }

    // 将有序整数还原为float
    __device__ __forceinline__ static float FromOrdered(OrderedType ordered) {
        uint32_t bits = (ordered & 0x80000000) ? (ordered ^ 0x80000000) : ~ordered;
        return __uint_as_float(bits);
    }

    // 返回负无穷大作为哨兵值
    __device__ __forceinline__ static float NegInf() {
        return -cuda::std::numeric_limits<float>::infinity();
    }
};

// half (16位) 类型特化
template <>
struct RadixTopKTraits<half> {
    using OrderedType = uint16_t;

    template <uint32_t RADIX_BITS>
    static __host__ __device__ constexpr uint32_t num_rounds() {
        return sizeof(OrderedType) * 8 / RADIX_BITS;
    }

    __device__ __forceinline__ static OrderedType ToOrdered(half val) {
        uint16_t bits = __half_as_ushort(val);
        return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
    }

    __device__ __forceinline__ static half FromOrdered(OrderedType ordered) {
        uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000) : static_cast<uint16_t>(~ordered);
        return __ushort_as_half(bits);
    }

    __device__ __forceinline__ static half NegInf() {
        return __ushort_as_half(static_cast<uint16_t>(0xFC00)); // -inf in fp16
    }
};

// nv_bfloat16 (16位) 类型特化
template <>
struct RadixTopKTraits<nv_bfloat16> {
    using OrderedType = uint16_t;

    template <uint32_t RADIX_BITS>
    static __host__ __device__ constexpr uint32_t num_rounds() {
        return sizeof(OrderedType) * 8 / RADIX_BITS;
    }

    __device__ __forceinline__ static OrderedType ToOrdered(nv_bfloat16 val) {
        uint16_t bits = __bfloat16_as_ushort(val);
        return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
    }

    __device__ __forceinline__ static nv_bfloat16 FromOrdered(OrderedType ordered) {
        uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000) : static_cast<uint16_t>(~ordered);
        return __ushort_as_bfloat16(bits);
    }

    __device__ __forceinline__ static nv_bfloat16 NegInf() {
        return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80)); // -inf in bf16
    }
};

// ==================== Multi-CTA Top-K Implementation ====================

// Acquire/Release primitives for inter-CTA synchronization
/*
CUDA 编程中的线程块间同步机制，用于实现跨 CTA（Cooperative Thread Array，即 block）的内存同步

Acquire	获取权限，确保看到其他 CTA 之前的写入
Release	释放权限，让其他 CTA 能看到当前 CTA 的写入

应用场景：
多 CTA 协作时（比如多个 block 协同完成 TopK 排序），需要确保：
1. CTA A 写入的数据对 CTA B 可见
2. 避免因编译优化导致读写顺序错乱

实现方式：
通常使用 memory fence 或 barrier：
- __threadfence() - 线程级内存屏障
- __syncthreads() - 仅限同一 CTA 内
- __grid_sync() - 多 CTA 全局同步（CUDA 11+）
在 Radix TopK 中，多个 CTA 分别处理不同数据分片，需要通过这种机制确保全局有序
*/
__device__ __forceinline__ int ld_acquire(int *ptr) {
    int state = 0;

#if (__CUDA_ARCH__ >= 700)
    // SM70及更新版本使用内存一致性修饰符
    // 使用acquire修饰符的Acquire模式
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#else
    asm volatile("ld.cg.global.b32 %0,[%1];\n" : "=r"(state) : "l"(ptr));
#endif
    return state;
}

__device__ __forceinline__ void red_release(int *ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
    // SM70及更新版本使用内存一致性修饰符
    // Release模式：acq_rel fence + relaxed修饰符
    // (fence还会释放之前通过弱写入其他线程的数据，在最后一个syncthreads之前)
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
    __threadfence();
    atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int *ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
    // SM70及更新版本使用内存一致性修饰符
    // Release模式：fence + release store
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
    __threadfence();
    atomicExch(ptr, val);
#endif
}

// 使用acquire语义等待ptr处的值达到target_val
// 只有thread 0自旋，然后所有线程同步
__device__ __forceinline__ void wait_ge(int *ptr, int target_val, int thread_idx) {
    if (thread_idx == 0) {
#pragma unroll 1
        while (ld_acquire(ptr) < target_val) {
        }
    }
    __syncthreads();
}

// ==================== Multi-CTA Radix Top-K Mask Logits ====================

// 多CTA radix归约的全局状态（每组一个）
struct RadixRowState {
    uint32_t histogram[3][256]; // 三重缓冲的直方图数组。256 对应 8 位 radix（2^8），3 用于 ping-pong 交替，避免 barrier 同步开销
    uint32_t remaining_k;       // 当前轮次还剩多少个元素需要选出
    uint32_t prefix;            // 累积的前缀（已确定的 radix 高位），用于最终重建完整的 key
    int arrival_counter;        // 到达计数器，用于多 CTA 间的同步（如 barrier）
    int output_counter;         // 输出计数器，用于收集最终的 TopK 结果索引
    float sum_topk;             // TopK 元素概率之和，用于 RenormProb（概率归一化）
};

constexpr uint32_t RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP = 256;

struct RadixDeterministicCollectScratch {
    uint32_t gt_count[RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP];
    uint32_t eq_count[RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP];
};

inline RadixDeterministicCollectScratch *MaybeGetRadixDeterministicCollectScratchBuffer(
    RadixRowState *row_states_buffer, uint32_t num_groups, bool single_cta, bool deterministic) {
    return (single_cta || !deterministic || row_states_buffer == nullptr) ? nullptr : reinterpret_cast<RadixDeterministicCollectScratch *>(row_states_buffer + num_groups);
}

// ==================== Common Device Functions for Radix Top-K ====================
/*!
 * \brief 跨 CTA（线程块）的软件屏障同步。
 *
 * 每个 CTA 通过 tx==0 贡献一次到达，然后等待全局到达计数器达到当前阶段的目标值。
 *
 * \param state 每组的 radix 行状态，包含到达计数器
 * \param barrier_phase 当前 CTA 组的软件屏障阶段
 * \param ctas_per_group 参与组屏障的 CTA 数量
 * \param tx 线程块内的线程索引
 */
__device__ __forceinline__ void AdvanceRadixGroupBarrier(RadixRowState *state, int &barrier_phase,
                                                         uint32_t ctas_per_group, uint32_t tx) {
    if (tx == 0) {
        // 线程 0 报告到达
        red_release(&state->arrival_counter, 1);
    }
    // 目标值 = (当前阶段 + 1) * CTA 数量
    int target = (barrier_phase + 1) * ctas_per_group;
    // 等待到达计数器达到目标值
    wait_ge(&state->arrival_counter, target, tx);
    // 阶段 +1
    barrier_phase++;
    // 同步当前 CTA 内的所有线程
    __syncthreads();
}

/*!
 * \brief 使用全CTA扫描确定性收集线程步幅匹配。
 *
 * 线程按固定顺序 `tx, tx + BLOCK_THREADS, ...` 遍历索引，
 * 计算整个步幅链上的每线程匹配计数，对这些计数执行exclusive-scan，
 * 然后按相同的确定性线程步幅顺序输出匹配。
 *
 * \tparam BLOCK_THREADS CTA中的线程数
 * \param tx CTA内的线程索引
 * \param length 要扫描的元素数
 * \param scan_temp_storage CUB BlockScan临时存储，被调用者复用
 * \param is_selected 线程步幅索引上的谓词
 * \param emit_limit 要输出的最大选中元素数
 * \param emit_selected 回调函数，调用形式 emit_selected(index, local_pos)
 */
template <uint32_t BLOCK_THREADS, typename TempStorage, typename Predicate, typename EmitFn>
__device__ __forceinline__ void DeterministicThreadStridedCollect(uint32_t tx, uint32_t length,
                                                                  TempStorage &scan_temp_storage,
                                                                  Predicate is_selected,
                                                                  uint32_t emit_limit,
                                                                  EmitFn emit_selected) {
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>;

    uint32_t thread_local_selected_count = 0;
    for (uint32_t i = tx; i < length; i += BLOCK_THREADS) {
        thread_local_selected_count += static_cast<uint32_t>(is_selected(i));
    }

    uint32_t thread_local_selected_prefix = 0;
    BlockScan(scan_temp_storage)
        .ExclusiveSum(thread_local_selected_count, thread_local_selected_prefix);

    if (thread_local_selected_count > 0 && thread_local_selected_prefix < emit_limit) {
        uint32_t thread_local_emit_pos = thread_local_selected_prefix;
        const uint32_t thread_local_emit_end =
            min(thread_local_selected_prefix + thread_local_selected_count, emit_limit);
        for (uint32_t i = tx; i < length; i += BLOCK_THREADS) {
            if (is_selected(i)) {
                emit_selected(i, thread_local_emit_pos);
                if (++thread_local_emit_pos == thread_local_emit_end) {
                    break;
                }
            }
        }
    }
    __syncthreads();
}

/*!
 * \brief 使用并行归约在共享内存中计算后缀和。
 *
 * 此函数后，suffix_sum[i] 包含 >= bucket i 的元素数量。
 * 通过对 bucket i 到 255 的所有直方图值求和计算得出。
 *
 * \param suffix_sum 大小为 RADIX (256) 的共享内存数组
 * \param tx Block内的线程索引
 */
template <uint32_t BLOCK_THREADS>
__device__ __forceinline__ void RadixSuffixSum(uint32_t* suffix_sum,uint32_t tx){
    constexpr uint32_t RADIX = 256;

    // 并行后缀和：计算 >= 每个bucket的元素数量
    for (uint32_t stride = 1; stride < RADIX; stride*=2)
    {
        uint32_t val = 0;
        if (tx < RADIX){
            val = suffix_sum[tx];
            if (tx + stride < RADIX){
                val += suffix_sum[tx + stride];
            }
        }

        __syncthreads();
        if (tx < RADIX){
            suffix_sum[tx] = val;
        }
        __syncthreads();
    }
}

/*!
 * \brief 找到包含第k大元素的阈值bucket。
 *
 * 阈值bucket满足：count_ge >= k && count_gt < k
 * 其中 count_ge = suffix_sum[bucket]，count_gt = suffix_sum[bucket+1]。
 *
 * \param suffix_sum 包含后缀和的共享内存数组
 * \param remaining_k 仍需找到的top-k元素数
 * \param found_bucket 输出：找到的阈值bucket
 * \param found_remaining_k 输出：remaining_k减去大于阈值的元素数量
 * \param tx Block内的线程索引
 */
__device__ __forceinline__ void RadixFindThresholdBucket(uint32_t* suffix_sum, uint32_t remaining_k,
                                                         uint32_t* found_bucket,
                                                         uint32_t* found_remaining_k, uint32_t tx) {
    constexpr uint32_t RADIX = 256;

    // 初始化（仅线程0）
    if (tx == 0){
        *found_bucket = 0;
        *found_remaining_k = remaining_k;
    }

    __syncthreads();

    // RADIX范围内的所有线程检查各自的bucket
      if (tx < RADIX) {
    uint32_t count_ge = suffix_sum[tx];
    uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
    if (count_ge >= remaining_k && count_gt < remaining_k) {
      *found_bucket = tx;
      *found_remaining_k = remaining_k - count_gt;
    }
  }
  __syncthreads();
    




}


/*!
 * \brief 为一轮radix select构建本地直方图。
 *
 * 统计 shared_ordered 中与当前前缀匹配的元素，并按当前移位位置的字节值将它们分桶。
 *
 * \tparam OrderedType 有序整数类型 (uint16_t 或 uint32_t)
 * \param shared_ordered 包含有序值的共享内存
 * \param actual_chunk_size 此CTA分块的元素数
 * \param local_histogram 输出共享内存直方图
 * \param prefix 当前前缀（到目前为止确定的高位）
 * \param shift 用于提取当前字节的位移
 * \param round 当前轮次 (0 到 NUM_ROUNDS-1)
 * \param tx 线程索引
 */
template <uint32_t BLOCK_THREADS, typename OrderedType>
__device__ __forceinline__ void RadixBuildLocalHistogram(const OrderedType* shared_ordered,
                                                         uint32_t actual_chunk_size,
                                                         uint32_t* local_histogram, uint32_t prefix,
                                                         uint32_t shift, uint32_t round,
                                                         uint32_t tx) {
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t RADIX_BITS = 8;

  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered = shared_ordered[i];

    // 检查此元素是否匹配前缀（到目前为止确定的高位）
    OrderedType mask =
        (round == 0)
            ? OrderedType(0)
            : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
    if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
      uint32_t bucket = (ordered >> shift) & 0xFF;
      atomicAdd(&local_histogram[bucket], 1);
    }
  }
}

/*!
 * \brief 执行一轮radix select，可选支持多CTA同步。
 *
 * 这是所有TopK内核的核心radix select逻辑。
 * 它构建直方图，跨CTA聚合（如果是multi-CTA），计算后缀和，并找到阈值bucket。
 *
 * \tparam BLOCK_THREADS 每block的线程数
 * \tparam SINGLE_CTA 如果是单CTA模式（不需要CTA间同步）为true
 * \tparam OrderedType 有序整数类型
 *
 * \param shared_ordered 包含有序值的共享内存
 * \param actual_chunk_size 此CTA分块的元素数
 * \param local_histogram 本地直方图共享内存（大小RADIX）
 * \param suffix_sum 后缀和计算共享内存（大小RADIX）
 * \param state 指向RadixRowState的指针，用于多CTA同步（SINGLE_CTA时为nullptr）
 * \param prefix 当前前缀值
 * \param remaining_k 当前剩余k值
 * \param round 当前轮次 (0 到 NUM_ROUNDS-1)
 * \param barrier_phase 屏障阶段计数器引用
 * \param ctas_per_group 每组的CTA数
 * \param tx 线程索引
 * \param out_new_prefix 输出：此轮后更新的前缀
 * \param out_new_remaining_k 输出：此轮后更新的remaining_k
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType>
__device__ __forceinline__ void RadixSelectOneRound(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t* local_histogram,
    uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state, uint32_t prefix,
    uint32_t remaining_k, uint32_t round, uint32_t iter, int& barrier_phase,
    uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t* out_new_prefix,
    uint32_t* out_new_remaining_k) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t NUM_ROUNDS = ORDERED_BITS / RADIX_BITS;
  uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
  uint32_t global_round = iter * NUM_ROUNDS + round;

  // 多CTA: 全局直方图指针（三重缓冲）
  uint32_t* current_hist = nullptr;
  uint32_t* next_hist = nullptr;
  if constexpr (!SINGLE_CTA) {
    current_hist = state->histogram[global_round % 3];
    next_hist = state->histogram[(global_round + 1) % 3];
  }

  // 仅清理本地直方图
  for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
    local_histogram[i] = 0;
  }
  __syncthreads();

  // 从共享内存构建本地直方图
  RadixBuildLocalHistogram<BLOCK_THREADS, OrderedType>(shared_ordered, actual_chunk_size,
                                                       local_histogram, prefix, shift, round, tx);
  __syncthreads();

  // 多CTA: 写入 -> (领先CTA清理下一个) -> barrier -> 读取
  // 单CTA: local_histogram 已经是完整的直方图
  if constexpr (!SINGLE_CTA) {
    // 将本地直方图累加到全局
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    // 仅领先CTA在barrier之前清理下一轮的直方图
    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        next_hist[i] = 0;
      }
    }

    // Barrier: 等待所有CTA完成atomicAdd和清理
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

    // 读取当前直方图（barrier之后，所有atomicAdd已完成）
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      suffix_sum[i] = current_hist[i];
    }
  } else {
    // 单CTA: 直接将本地直方图复制到suffix_sum
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      suffix_sum[i] = local_histogram[i];
    }
  }
  __syncthreads();

  // 计算后缀和
  RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);

  // 使用shared_scalars找到阈值bucket
  // shared_scalars[0] = found_bucket, shared_scalars[1] = found_remaining_k
  RadixFindThresholdBucket(suffix_sum, remaining_k, &shared_scalars[0], &shared_scalars[1], tx);

  // 输出新的prefix和remaining_k
  *out_new_prefix = prefix | (shared_scalars[0] << shift);
  *out_new_remaining_k = shared_scalars[1];
}

/*!
 * \brief Load data from global memory to shared memory and convert to ordered representation.
 *
 * This is the common Stage 1 for all TopK kernels. It loads data using vectorized
 * memory access and converts to ordered representation for radix select.
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam VEC_SIZE Vector size for memory access
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam Traits Type traits for DType
 *
 * \param input Pointer to input data row start (already offset by row)
 * \param shared_ordered Shared memory for ordered values
 * \param chunk_start Start index within the row for this CTA's chunk
 * \param actual_chunk_size Number of elements in this CTA's chunk
 * \param tx Thread index
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, typename Traits>
__device__ __forceinline__ void LoadToSharedOrdered(const DType* input,
                                                    typename Traits::OrderedType* shared_ordered,
                                                    uint32_t chunk_start,
                                                    uint32_t actual_chunk_size, uint32_t tx) {
  using OrderedType = typename Traits::OrderedType;
  vec_t<DType, VEC_SIZE> input_vec;
  const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
  for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
    input_vec.cast_load(input + chunk_start + i);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      shared_ordered[i + j] = Traits::ToOrdered(input_vec[j]);
    }
  }
  // 处理尾部（不对齐部分）
  for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    shared_ordered[i] = Traits::ToOrdered(input[chunk_start + i]);
  }
  __syncthreads();
}

/*!
 * \brief 使用预加载的共享内存通过radix select找到第k大的元素。
 *
 * 此函数假设数据已加载到 shared_ordered 中。
 * 它执行完整的radix select算法（初始barrier + NUM_ROUNDS）
 * 并返回有序表示的pivot值。
 *
 * \tparam BLOCK_THREADS 每block的线程数
 * \tparam SINGLE_CTA 如果是单CTA模式为true
 * \tparam OrderedType 有序整数类型
 *
 * \param shared_ordered 包含有序值的共享内存（已预加载）
 * \param actual_chunk_size 此CTA分块的元素数
 * \param k 要选择的top元素数
 * \param local_histogram 本地直方图共享内存（大小RADIX）
 * \param suffix_sum 后缀和共享内存（大小RADIX）
 * \param shared_scalars 标量共享内存 [prefix_cache, remaining_k_cache, found_bucket,
 * found_remaining_k, output_counter]
 * \param state RadixRowState指针，用于多CTA同步（SINGLE_CTA时为nullptr）
 * \param barrier_phase 屏障阶段计数器引用
 * \param ctas_per_group 每组的CTA数
 * \param cta_in_group 组内CTA索引
 * \param tx 线程索引
 * \param iter 当前迭代（用于三重缓冲索引）
 * \return 有序表示的pivot值
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, bool TRACK_EQ_COUNT>
__device__ __forceinline__ OrderedType RadixSelectFromSharedMemory(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t k,
    uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
    int& barrier_phase, uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t iter,
    uint32_t& out_local_gt_count, uint32_t& out_local_eq_count) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t NUM_ROUNDS = ORDERED_BITS / RADIX_BITS;

// 标量共享变量的别名
#define prefix_cache shared_scalars[0]
#define remaining_k_cache shared_scalars[1]
#define found_bucket shared_scalars[2]
#define found_remaining_k shared_scalars[3]
#define shared_output_counter shared_scalars[4]

  // 初始化本地缓存
  if (tx == 0) {
    prefix_cache = 0;
    remaining_k_cache = k;
    if constexpr (SINGLE_CTA) {
      shared_output_counter = 0;
    }
  }
  __syncthreads();

  // 初始barrier（单CTA跳过）
  if constexpr (!SINGLE_CTA) {
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

    // CTA 0 在barrier之后清理output counter
    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->output_counter, 0);
    }
  }

  // NUM_ROUNDS轮radix select
  for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
    uint32_t global_round = iter * NUM_ROUNDS + round;
    uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
    uint32_t prefix = prefix_cache;
    uint32_t remaining_k = remaining_k_cache;

    // 多CTA: 全局直方图指针（三重缓冲）
    uint32_t* current_hist = nullptr;
    uint32_t* next_hist = nullptr;
    if constexpr (!SINGLE_CTA) {
      current_hist = state->histogram[global_round % 3];
      next_hist = state->histogram[(global_round + 1) % 3];
    }

    // 清理本地直方图
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    // 构建本地直方图
#pragma unroll 2
    for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      OrderedType ordered = shared_ordered[i];
      OrderedType mask =
          (round == 0)
              ? OrderedType(0)
              : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
      if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
        uint32_t bucket = (ordered >> shift) & 0xFF;
        atomicAdd(&local_histogram[bucket], 1);
      }
    }
    __syncthreads();

    // 多CTA: 累加到全局，barrier，读取回来
    if constexpr (!SINGLE_CTA) {
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        if (local_histogram[i] > 0) {
          atomicAdd(&current_hist[i], local_histogram[i]);
        }
      }
      if (cta_in_group == 0) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          next_hist[i] = 0;
        }
      }
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        suffix_sum[i] = current_hist[i];
      }
    } else {
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        suffix_sum[i] = local_histogram[i];
      }
    }

    // 计算后缀和
    RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);

    // 找到阈值bucket
    if (tx == 0) {
      found_bucket = 0;
      found_remaining_k = remaining_k;
    }
    __syncthreads();

    if (tx < RADIX) {
      uint32_t count_ge = suffix_sum[tx];
      uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
      if (count_ge >= remaining_k && count_gt < remaining_k) {
        found_bucket = tx;
        found_remaining_k = remaining_k - count_gt;
      }
    }
    __syncthreads();

    // 更新缓存
    if (tx == 0) {
      prefix_cache = prefix | (found_bucket << shift);
      remaining_k_cache = found_remaining_k;
    }
    __syncthreads();
  }

  OrderedType ordered_pivot = static_cast<OrderedType>(prefix_cache);

  // 扫描shared_ordered统计>pivot（以及可选的==pivot）元素数量。
  // 这是必要的，因为suffix_sum只跟踪匹配当前prefix的元素，
  // 而不是所有>pivot的元素（包括高位大于pivot的元素）
  if (tx == 0) {
    suffix_sum[0] = 0;
    if constexpr (TRACK_EQ_COUNT) {
      suffix_sum[1] = 0;
    }
  }
  __syncthreads();

  uint32_t my_gt_count = 0;
  uint32_t my_eq_count = 0;
#pragma unroll 2
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    const OrderedType ordered = shared_ordered[i];
    if (ordered > ordered_pivot) {
      my_gt_count++;
    }
    if constexpr (TRACK_EQ_COUNT) {
      if (ordered == ordered_pivot) {
        my_eq_count++;
      }
    }
  }

  // Warp级归约
  for (int offset = 16; offset > 0; offset /= 2) {
    my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
    if constexpr (TRACK_EQ_COUNT) {
      my_eq_count += __shfl_down_sync(0xffffffff, my_eq_count, offset);
    }
  }

  // 每个warp的第一个线程将结果原子写入共享内存
  int lane = tx % 32;
  if (lane == 0 && my_gt_count > 0) {
    atomicAdd(&suffix_sum[0], my_gt_count);
  }
  if constexpr (TRACK_EQ_COUNT) {
    if (lane == 0 && my_eq_count > 0) {
      atomicAdd(&suffix_sum[1], my_eq_count);
    }
  }
  __syncthreads();

  out_local_gt_count = suffix_sum[0];
  if constexpr (TRACK_EQ_COUNT) {
    out_local_eq_count = suffix_sum[1];
  } else {
    out_local_eq_count = 0;
  }

#undef prefix_cache
#undef remaining_k_cache
#undef found_bucket
#undef found_remaining_k
#undef shared_output_counter

  return ordered_pivot;
}

/*!
 * \brief Load one CTA chunk into ordered shared memory, then find the pivot with radix select.
 *
 * This helper centralizes the shared-memory load and the exact k-th-element radix
 * select. It returns the pivot in ordered representation. Callers can optionally request the
 * CTA-local counts of elements
 * `> pivot` and `== pivot`, which are needed by deterministic collect paths.
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, bool TRACK_EQ_COUNT,
          typename DType>
__device__ __forceinline__ typename RadixTopKTraits<DType>::OrderedType RadixSelectFindPivot(
    const DType* input, typename RadixTopKTraits<DType>::OrderedType* shared_ordered,
    uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
    uint32_t chunk_start, uint32_t actual_chunk_size, uint32_t k, int& barrier_phase,
    uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t iter,
    uint32_t& out_local_gt_count, uint32_t& out_local_eq_count) {
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  LoadToSharedOrdered<BLOCK_THREADS, VEC_SIZE, DType, Traits>(input, shared_ordered, chunk_start,
                                                              actual_chunk_size, tx);
  return RadixSelectFromSharedMemory<BLOCK_THREADS, SINGLE_CTA, OrderedType, TRACK_EQ_COUNT>(
      shared_ordered, actual_chunk_size, k, local_histogram, suffix_sum, shared_scalars, state,
      barrier_phase, ctas_per_group, cta_in_group, tx, iter, out_local_gt_count,
      out_local_eq_count);
}

/*!
 * \brief Collect top-k indices based on pivot value with custom output transform (Single Pass).
 *
 * This optimized version uses a single pass to write all elements:
 * - > pivot: use shared memory atomic for local offset within CTA's allocation
 * - == pivot: use global memory atomic, check if pos < k before writing
 *
 * The local_gt_count is computed during the last round of radix select, so we know
 * exactly how many > pivot elements each CTA has. This allows batched global atomic
 * (one per CTA) for > pivot elements.
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam SINGLE_CTA True if single-CTA mode
 * \tparam OrderedType The ordered integer type
 * \tparam OutputFunc Functor type: void(uint32_t original_idx, OrderedType ordered_val, int
 * output_pos)
 *
 * \param shared_ordered Shared memory containing ordered values
 * \param actual_chunk_size Number of elements in this CTA's chunk
 * \param chunk_start Start index in input for this chunk
 * \param k Number of top elements to select
 * \param ordered_pivot The pivot value in ordered representation
 * \param local_gt_count Number of > pivot elements in this CTA (from radix select)
 * \param local_histogram Shared memory for counters
 * \param shared_output_counter Pointer to shared output counter (SINGLE_CTA mode)
 * \param state RadixRowState pointer for multi-CTA sync (nullptr if SINGLE_CTA)
 * \param barrier_phase Reference to barrier phase counter (unused in new implementation)
 * \param ctas_per_group Number of CTAs per group
 * \param tx Thread index
 * \param output_func Functor called as output_func(original_idx, ordered_val, output_pos) for each
 * element
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, typename OutputFunc>
__device__ __forceinline__ void RadixCollectIndices(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t chunk_start, uint32_t k,
    OrderedType ordered_pivot, uint32_t local_gt_count, uint32_t* local_histogram,
    uint32_t* shared_output_counter, RadixRowState* state, int& barrier_phase,
    uint32_t ctas_per_group, uint32_t tx, OutputFunc output_func) {
// 使用 local_histogram 作为计数器：
// [0]: local_offset_gt (CTA分配内 > pivot 元素的本地偏移)
// [1]: global_base_gt (> pivot 元素的全局基位置)
#define local_offset_gt local_histogram[0]
#define global_base_gt local_histogram[1]

  // 获取此CTA的>pivot元素的全局基位置（每个CTA一个atomic）
  if (tx == 0) {
    local_offset_gt = 0;
    if (local_gt_count > 0) {
      if constexpr (SINGLE_CTA) {
        global_base_gt = atomicAdd(shared_output_counter, local_gt_count);
      } else {
        global_base_gt = atomicAdd(&state->output_counter, local_gt_count);
      }
    }
  }
  __syncthreads();

  // 第一遍：写入 > pivot 的元素
  // 这些元素保证在top-k中，使用CTA分配内的本地偏移
#pragma unroll 2
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered_val = shared_ordered[i];
    if (ordered_val > ordered_pivot) {
      uint32_t local_pos = atomicAdd(&local_offset_gt, 1);
      int pos = global_base_gt + local_pos;
      output_func(chunk_start + i, ordered_val, pos);
    }
  }

  // Barrier确保所有>pivot元素先被收集（仅多CTA）
  // 这很关键：没有这个barrier，CTAs可能在其他CTAs还在写入>pivot元素时写入==pivot元素，导致位置错误。
  if constexpr (!SINGLE_CTA) {
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
  } else {
    __syncthreads();
  }

  // 第二遍：写入 == pivot 的元素
  // 直接使用全局原子，因为我们需要跨CTA协调来遵守k限制（一些==pivot元素可能被截断）。
#pragma unroll 2
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered_val = shared_ordered[i];
    if (ordered_val == ordered_pivot) {
      int pos;
      if constexpr (SINGLE_CTA) {
        pos = atomicAdd(shared_output_counter, 1);
      } else {
        pos = atomicAdd(&state->output_counter, 1);
      }
      if (pos < static_cast<int>(k)) {
        output_func(chunk_start + i, ordered_pivot, pos);
      }
    }
  }

#undef local_offset_gt
#undef global_base_gt
}

struct DeterministicCollectCountPair {
  uint32_t gt;
  uint32_t eq;
};

struct DeterministicCollectCountPairSum {
  __device__ __forceinline__ DeterministicCollectCountPair operator()(
      const DeterministicCollectCountPair& lhs, const DeterministicCollectCountPair& rhs) const {
    return {lhs.gt + rhs.gt, lhs.eq + rhs.eq};
  }
};

/*!
 * \brief Collect top-k indices with deterministic cross-CTA ordering.
 *
 * This variant preserves repeatable output by replacing cross-CTA atomic tie
 * claiming with a fixed allocation scheme:
 * - All > pivot elements are assigned output ranges in CTA order.
 * - == pivot elements are then assigned deterministic prefixes from
 *   per-CTA gt/eq counts stored in \p det_scratch.
 *
 * Single-CTA mode degenerates to a block-local deterministic collect without
 * using \p det_scratch.
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam SINGLE_CTA True if single-CTA mode
 * \tparam OrderedType The ordered integer type
 * \tparam OutputFunc Functor type: void(uint32_t original_idx, OrderedType ordered_val, int
 * output_pos)
 *
 * \param shared_ordered Shared memory containing ordered values
 * \param actual_chunk_size Number of elements in this CTA's chunk
 * \param chunk_start Start index in input for this chunk
 * \param k Number of top elements to select
 * \param ordered_pivot The pivot value in ordered representation
 * \param cta_local_gt_count Number of > pivot elements in this CTA (from radix select)
 * \param cta_local_eq_count Number of == pivot elements in this CTA (from radix select)
 * \param local_histogram Shared memory scratch reused for deterministic collect state
 * \param state RadixRowState pointer for multi-CTA sync (nullptr if SINGLE_CTA)
 * \param det_scratch Per-group scratch for multi-CTA gt/eq counts (nullptr if SINGLE_CTA)
 * \param barrier_phase Reference to barrier phase counter
 * \param ctas_per_group Number of CTAs per group
 * \param cta_in_group CTA index within the current group
 * \param tx Thread index
 * \param output_func Functor called as output_func(original_idx, ordered_val, output_pos) for each
 * selected element
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, typename OutputFunc>
__device__ __forceinline__ void RadixCollectIndicesDeterministic(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t chunk_start, uint32_t k,
    OrderedType ordered_pivot, uint32_t cta_local_gt_count, uint32_t cta_local_eq_count,
    uint32_t* local_histogram, RadixRowState* state, RadixDeterministicCollectScratch* det_scratch,
    int& barrier_phase, uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx,
    OutputFunc output_func) {
// 使用 local_histogram 作为计数器：
// [0]: s_cta_local_gt_prefix   - 之前CTA的>pivot总数
// [1]: s_cta_local_eq_prefix   - 之前CTA的==pivot总数
// [2]: s_row_total_gt_count    - 跨所有CTA的行级>pivot总数
// [3]: s_row_eq_needed         - >pivot写入后仍需要的==pivot条目数
// [4]: s_cta_local_eq_take     - 此CTA分配的==pivot配额
#define s_cta_local_gt_prefix local_histogram[0]
#define s_cta_local_eq_prefix local_histogram[1]
#define s_row_total_gt_count local_histogram[2]
#define s_row_eq_needed local_histogram[3]
#define s_cta_local_eq_take local_histogram[4]
  uint32_t cta_local_eq_emit_limit = 0;
  uint32_t cta_local_eq_output_base = 0;
  if constexpr (SINGLE_CTA) {
    if (tx == 0) {
      s_cta_local_gt_prefix = 0;
      s_cta_local_eq_prefix = 0;
      s_row_total_gt_count = cta_local_gt_count;
      s_row_eq_needed = (k > cta_local_gt_count) ? (k - cta_local_gt_count) : 0;
      s_cta_local_eq_take = 0;
    }
    __syncthreads();
    // 单CTA: 在所有>pivot条目之后保持完整的==pivot后缀连续
    cta_local_eq_emit_limit = s_row_eq_needed;
    cta_local_eq_output_base = s_row_total_gt_count;
  } else {
    // 每个CTA写入其本地的>pivot / ==pivot计数
    if (tx == 0) {
      s_cta_local_eq_prefix = 0;
      s_cta_local_eq_take = 0;
      det_scratch->gt_count[cta_in_group] = cta_local_gt_count;
      det_scratch->eq_count[cta_in_group] = cta_local_eq_count;
    }
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
    // 每个CTA读取所有的>pivot / ==pivot计数
    if (tx == 0) {
      uint32_t cta_local_gt_prefix_accum = 0;
      uint32_t row_total_gt = 0;
      uint32_t cta_local_eq_prefix_accum = 0;
      for (uint32_t c = 0; c < ctas_per_group; ++c) {
        const uint32_t c_gt = det_scratch->gt_count[c];
        const uint32_t c_eq = det_scratch->eq_count[c];
        if (c < cta_in_group) {
          cta_local_gt_prefix_accum += c_gt;
          cta_local_eq_prefix_accum += c_eq;
        }
        row_total_gt += c_gt;
      }
      s_cta_local_gt_prefix = cta_local_gt_prefix_accum;
      s_row_total_gt_count = row_total_gt;
      s_row_eq_needed = (k > row_total_gt) ? (k - row_total_gt) : 0;
      s_cta_local_eq_prefix = cta_local_eq_prefix_accum;
      s_cta_local_eq_take = 0;
      if (s_row_eq_needed > cta_local_eq_prefix_accum) {
        s_cta_local_eq_take = min(cta_local_eq_count, s_row_eq_needed - cta_local_eq_prefix_accum);
      }
    }
    __syncthreads();
    // 多CTA: 仅在此CTA的确定性输出基位置输出其分配的==pivot配额
    cta_local_eq_emit_limit = s_cta_local_eq_take;
    cta_local_eq_output_base = s_row_total_gt_count + s_cta_local_eq_prefix;
  }
  const uint32_t cta_local_gt_output_base = s_cta_local_gt_prefix;
  const uint32_t cta_local_gt_emit_limit =
      (k > cta_local_gt_output_base) ? (k - cta_local_gt_output_base) : 0;

#undef s_cta_local_gt_prefix
#undef s_cta_local_eq_prefix
#undef s_row_total_gt_count
#undef s_row_eq_needed
#undef s_cta_local_eq_take

  using ScalarBlockScan = cub::BlockScan<uint32_t, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
  using PairBlockScan =
      cub::BlockScan<DeterministicCollectCountPair, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
  union DeterministicCollectScanTempStorage {
    typename ScalarBlockScan::TempStorage scalar;
    typename PairBlockScan::TempStorage pair;
  };
  __shared__ DeterministicCollectScanTempStorage scan_temp_storage;

  // 当此CTA的 ==pivot 配额为0时，只需收集 >pivot 的元素
  // 这是一种快速路径优化：对于大多数情况，pivot值唯一确定，不需要处理等于pivot的情况
  if (cta_local_eq_emit_limit == 0) {
    DeterministicThreadStridedCollect<BLOCK_THREADS>(
        tx, actual_chunk_size, scan_temp_storage.scalar,
        [&](uint32_t i) { return shared_ordered[i] > ordered_pivot; }, cta_local_gt_emit_limit,
        [&](uint32_t i, uint32_t local_pos) {
          output_func(chunk_start + i, shared_ordered[i], cta_local_gt_output_base + local_pos);
        });
    return;
  }

  // 收集gt和eq元素
  DeterministicCollectCountPair thread_local_counts = {0, 0};
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    const OrderedType ordered = shared_ordered[i];
    thread_local_counts.gt += static_cast<uint32_t>(ordered > ordered_pivot);
    thread_local_counts.eq += static_cast<uint32_t>(ordered == ordered_pivot);
  }

  DeterministicCollectCountPair thread_local_prefix = {0, 0};
  PairBlockScan(scan_temp_storage.pair)
      .ExclusiveScan(thread_local_counts, thread_local_prefix, DeterministicCollectCountPair{0, 0},
                     DeterministicCollectCountPairSum{});

  DeterministicCollectCountPair thread_local_pos = thread_local_prefix;
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    const OrderedType ordered = shared_ordered[i];
    if (ordered > ordered_pivot && thread_local_pos.gt < cta_local_gt_emit_limit) {
      output_func(chunk_start + i, ordered, cta_local_gt_output_base + thread_local_pos.gt);
      ++thread_local_pos.gt;
    } else if (ordered == ordered_pivot && thread_local_pos.eq < cta_local_eq_emit_limit) {
      output_func(chunk_start + i, ordered, cta_local_eq_output_base + thread_local_pos.eq);
      ++thread_local_pos.eq;
    }
  }
  __syncthreads();
}

// ==================== Unified Radix Top-K Kernel with Epilogue Modes ====================

/*!
 * \brief Epilogue mode for unified RadixTopK kernel.
 */
enum class RadixTopKMode {
  Basic,               ///< Returns (indices, values) pairs
  PageTableTransform,  ///< Gathers indices through page table
  RaggedTransform,     ///< Adds offset to indices
};

/*!
 * \brief Unified Multi-CTA Radix Top-K kernel with mode-specific epilogues.
 *
 * This kernel unifies three top-k variants:
 * - Basic: Returns top-k indices and values
 * - PageTableTransform: Gathers top-k indices through a page table
 * - RaggedTransform: Adds per-row offset to top-k indices
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam VEC_SIZE Vector size for memory access
 * \tparam SINGLE_CTA True if single-CTA mode
 * \tparam DETERMINISTIC True to use deterministic collect path
 * \tparam MODE Epilogue mode (Basic, PageTableTransform, or RaggedTransform)
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, bool DETERMINISTIC,
          RadixTopKMode MODE, typename DType, typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKKernel_Unified(
    DType* input,            // [num_rows, stride]
    IdType* output_indices,  // [num_rows, top_k] - indices or page table entries
    DType* output_values,    // [num_rows, top_k] - 仅在Basic模式使用，否则为nullptr
    const IdType*
        aux_data,  // 模式相关：top_k_arr (Basic), src_page_table (PageTable), offsets (Ragged)
    IdType* lengths,             // [num_rows] 每行长度，Basic模式为nullptr（使用stride）
    const IdType* row_to_batch,  // [num_rows] PageTable的batch映射，否则为nullptr
    // aux_stride: 辅助数据的步长，用于PageTable模式下的页表条目寻址
    // - Basic模式: 0 (不使用)
    // - PageTable模式: src_page_table 的行跨度，用于定位每行的页表入口
    // - Ragged模式: 0 (使用 offsets 数组)
    int64_t aux_stride,
    uint32_t top_k_val, uint32_t stride, uint32_t num_rows, RadixRowState* row_states,
    RadixDeterministicCollectScratch* det_scratches, uint32_t chunk_size, uint32_t ctas_per_group) {
  // 获取全局CTA ID和组内CTA索引
  // global_cta_id: 当前CTA在整个grid中的索引 (0 到 gridDim.x - 1)
  // group_id: CTA所属的组ID，每组包含 ctas_per_group 个CTA，处理同一行数据
  // cta_in_group: CTA在组内的索引 (0 到 ctas_per_group - 1)，用于计算chunk边界
  // tx: 当前线程在CTA内的索引 (0 到 BLOCK_THREADS - 1)
  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // 声明外部共享内存数组，由运行时按kernel launch参数分配
  extern __shared__ uint8_t smem[];

  // 共享内存布局：
  // | local_histogram[256] | suffix_sum[256] | shared_scalars[4或5] | ... | ordered data ... |
  // 单CTA模式需要额外的 shared_output_counter，故为5个标量；多CTA模式只需4个
  // local_histogram: 本地直方图，用于radix select过程中的计数
  // suffix_sum: 后缀和数组，用于计算 >= 每个bucket的元素总数
  // shared_scalars: 标量缓存 [prefix_cache, remaining_k_cache, found_bucket, found_remaining_k, output_counter]
  constexpr size_t num_scalars = SINGLE_CTA ? 5 : 4;
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + num_scalars);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;

  // ordered data 区域需要16字节对齐，以确保矢量加载的正确性
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

#define shared_output_counter shared_scalars[4]

  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }
  RadixDeterministicCollectScratch* det_scratch = nullptr;
  // 多CTA模式：获取当前组的状态指针，用于跨CTA同步
  // 每个group_id对应一个RadixRowState结构，包含直方图、计数器等全局状态
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }
  // 确定性收集模式：获取当前组的scratch buffer，用于存储各CTA的gt/eq计数
  // 确保多次运行结果一致，避免因CTA执行顺序不同导致结果不确定
  RadixDeterministicCollectScratch* det_scratch = nullptr;
  if constexpr (!SINGLE_CTA && DETERMINISTIC) {
    det_scratch = &det_scratches[group_id];
  }

  // 计算总组数和每组处理的迭代次数
  // num_groups: grid中CTA组的数量 = gridDim.x / ctas_per_group
  // total_iterations: 每组需要处理的行数，由于行数可能大于组数，需要多轮迭代
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (num_rows + num_groups - 1) / num_groups;

  // barrier_phase: 软件屏障阶段计数器，用于多CTA间的同步
  // 每次调用AdvanceRadixGroupBarrier时递增，确保各CTA按正确的顺序执行
  int barrier_phase = 0;

  // 主循环：迭代处理所有行
  // 每轮迭代中，group_id对应的组处理一行数据
  // 这样可以实现负载均衡：即使行数远大于组数，也能分配到各组
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    // 计算当前迭代处理的行索引
    // 例如：num_groups=4, iter=0时处理行0,1,2,3; iter=1时处理行4,5,6,7
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= num_rows) break;

    // 模式相关：获取行长度和k值
    // length: 该行实际元素个数（可能小于stride）
    // k: 需要选择的top-k元素个数
    uint32_t length, k;
    if constexpr (MODE == RadixTopKMode::Basic) {
      // Basic模式：所有行使用相同的stride，k值来自aux_data（每行可选的top_k数组）或全局top_k_val
      length = stride;                                            // 所有行的固定长度
      k = (aux_data != nullptr) ? aux_data[row_idx] : top_k_val;  // aux_data = top_k_arr
    } else {
      // PageTable/Ragged模式：每行长度可能不同，从lengths数组获取
      length = lengths[row_idx];  // 每行长度
      k = top_k_val;              // 固定k
    }

    // 模式相关：输出指针和辅助数据
    // row_output: 当前行输出数组的起始位置
    IdType* row_output = output_indices + row_idx * top_k_val;

    // 处理边界情况
    if constexpr (MODE == RadixTopKMode::Basic) {
      if (k >= length) {
    // 如果 k >= length，表示需要返回所有元素（ vocab size 内所有token的概率都已计算）
    // 此时无需进行TopK排序，直接按顺序返回所有索引即可
    // 这种边界情况在语言模型中很常见：当vocab_size较小时，通常直接返回完整词汇表
    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, length);
    const uint32_t actual_chunk_size = ((chunk_start < length) ? (chunk_end - chunk_start) : 0);

        for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
          if (chunk_start + i < k) {
            row_output[chunk_start + i] = static_cast<IdType>(chunk_start + i);
            output_values[row_idx * top_k_val + chunk_start + i] =
                input[static_cast<size_t>(row_idx) * stride + chunk_start + i];
          }
        }
        // 为下一次迭代清理直方图（以防k < length）
        if constexpr (!SINGLE_CTA) {
          constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
          uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
          if (cta_in_group == 0) {
            for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
              state->histogram[next_first_hist_idx][i] = 0;
            }
          }
        }
        continue;
      }
    } else if constexpr (MODE == RadixTopKMode::PageTableTransform) {
      // PageTableTransform模式：用于Page-Decoding场景，通过页表查找实际的token ID
      // row_to_batch: 行到batch的映射，用于支持多batch场景
      // src_page_entry: 当前行的页表入口地址 = aux_data + batch_idx * aux_stride
      uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row_idx] : row_idx;
      const IdType* src_page_entry = aux_data + batch_idx * aux_stride;
      // 如果有效元素数 <= top_k_val，直接返回所有元素（无需排序）
      // 超出部分用-1填充（表示无效位置）
      if (length <= top_k_val) {
        for (uint32_t i = tx; i < top_k_val; i += BLOCK_THREADS) {
          row_output[i] = (i < length) ? src_page_entry[i] : static_cast<IdType>(-1);
        }
        // 为下一次迭代清理直方图
        if constexpr (!SINGLE_CTA) {
          constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
          uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
          if (cta_in_group == 0) {
            for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
              state->histogram[next_first_hist_idx][i] = 0;
            }
          }
        }
        continue;
      }
    } else {  // RaggedTransform模式
      // RaggedTransform模式：用于处理可变长度序列，通过offsets数组获取每行的起始偏移
      // offset: 当前行在连续内存中的起始位置（累积偏移）
      IdType offset = aux_data[row_idx];
      // 如果有效元素数 <= top_k_val，直接返回所有元素
      if (length <= top_k_val) {
        for (uint32_t i = tx; i < top_k_val; i += BLOCK_THREADS) {
          row_output[i] = (i < length) ? static_cast<IdType>(i) + offset : static_cast<IdType>(-1);
        }
        // 为下一次迭代清理直方图
        if constexpr (!SINGLE_CTA) {
          constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
          uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
          if (cta_in_group == 0) {
            for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
              state->histogram[next_first_hist_idx][i] = 0;
            }
          }
        }
        continue;
      }
    }

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, length);
    const uint32_t actual_chunk_size = ((chunk_start < length) ? (chunk_end - chunk_start) : 0);

    // 阶段1：将分块加载到共享内存，然后radix-select pivot。
    uint32_t cta_local_gt_count = 0;
    uint32_t cta_local_eq_count = 0;
    OrderedType ordered_pivot =
        RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, DETERMINISTIC, DType>(
            input + static_cast<size_t>(row_idx) * stride, shared_ordered, local_histogram,
            suffix_sum, shared_scalars, state, chunk_start, actual_chunk_size, k, barrier_phase,
            ctas_per_group, cta_in_group, tx, iter, cta_local_gt_count, cta_local_eq_count);

    auto collect_indices = [&](auto&& output_func) {
      if constexpr (DETERMINISTIC) {
        RadixCollectIndicesDeterministic<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
            shared_ordered, actual_chunk_size, chunk_start, k, ordered_pivot, cta_local_gt_count,
            cta_local_eq_count, local_histogram, state, det_scratch, barrier_phase, ctas_per_group,
            cta_in_group, tx, output_func);
      } else {
        RadixCollectIndices<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
            shared_ordered, actual_chunk_size, chunk_start, k, ordered_pivot, cta_local_gt_count,
            local_histogram, &shared_output_counter, state, barrier_phase, ctas_per_group, tx,
            output_func);
      }
    };

    // 阶段2：使用模式特定结尾收集索引（单次传递）
    if constexpr (MODE == RadixTopKMode::Basic) {
      DType* row_output_values = output_values + row_idx * top_k_val;
      collect_indices([&](uint32_t original_idx, OrderedType ordered_val, int pos) {
        row_output[pos] = static_cast<IdType>(original_idx);
        row_output_values[pos] = Traits::FromOrdered(ordered_val);
      });
    } else if constexpr (MODE == RadixTopKMode::PageTableTransform) {
      uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row_idx] : row_idx;
      const IdType* src_page_entry = aux_data + batch_idx * aux_stride;

      // 先收集原始索引
      collect_indices([&](uint32_t original_idx, OrderedType /*ordered_val*/, int pos) {
        row_output[pos] = static_cast<IdType>(original_idx);
      });

      if constexpr (SINGLE_CTA) {
        __syncthreads();
        // 通过页表转换，使用合并访问
        for (uint32_t i = tx; i < k; i += BLOCK_THREADS) {
          IdType idx = row_output[i];
          row_output[i] = src_page_entry[idx];
        }
      } else {
        // Barrier确保所有CTA完成索引写入
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

        // 所有CTA参与页表转换（合并访问）
        uint32_t elems_per_cta = (k + ctas_per_group - 1) / ctas_per_group;
        uint32_t my_start = cta_in_group * elems_per_cta;
        uint32_t my_end = min(my_start + elems_per_cta, k);
        for (uint32_t i = my_start + tx; i < my_end; i += BLOCK_THREADS) {
          IdType idx = row_output[i];
          row_output[i] = src_page_entry[idx];
        }
      }
    } else {  // RaggedTransform ( ragged转换模式)
      IdType offset = aux_data[row_idx];
      collect_indices([&](uint32_t original_idx, OrderedType /*ordered_val*/, int pos) {
        row_output[pos] = static_cast<IdType>(original_idx) + offset;
      });
    }
  }

  // 清理直方图缓冲区并重置到达计数器，为下一次内核启动（仅多CTA）
  if constexpr (!SINGLE_CTA) {
    if (cta_in_group == 0) {
      for (uint32_t buf = 0; buf < 3; ++buf) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          state->histogram[buf][i] = 0;
        }
      }
      if constexpr (DETERMINISTIC) {
        static_assert(sizeof(RadixDeterministicCollectScratch) % sizeof(uint32_t) == 0);
        uint32_t* det_words = reinterpret_cast<uint32_t*>(det_scratch);
        constexpr uint32_t DET_WORDS = sizeof(RadixDeterministicCollectScratch) / sizeof(uint32_t);
        for (uint32_t i = tx; i < DET_WORDS; i += BLOCK_THREADS) {
          det_words[i] = 0;
        }
      }
      if (tx == 0) {
        st_release(&state->arrival_counter, 0);
      }
    }
  }

#undef shared_output_counter
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKMaskLogitsKernel_MultiCTA(
    DType* logits,         // [batch, vocab_size] - 输入的logits值（未mask）
    DType* masked_logits,  // [batch, vocab_size] - 输出：mask后的logits（低于pivot的值设为-inf）
    IdType* top_k_arr,     // [batch] - 每行可选的top-k值数组，如果为nullptr则使用全局top_k_val
    uint32_t top_k_val,    // 默认的top-k值，当top_k_arr为nullptr时使用
    uint32_t vocab_size,   // 词汇表大小（每行的元素数）
    uint32_t batch_size,   // batch大小（行数）
    RadixRowState* row_states,  // [num_groups] - 多CTA状态数组，用于跨CTA同步（单CTA模式为nullptr）
    uint32_t chunk_size,        // 每个CTA处理的元素数（分块大小）
    uint32_t ctas_per_group)    // 每行使用的CTA数量（单CTA模式为1）
{
  // 获取数据类型对应的RadixTopK Traits
  // Traits负责将浮点数转换为可排序的整数表示（处理符号位问题）
  // FP16/BF16使用uint16_t，FP32使用uint32_t
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  // RADIX = 256 表示使用8位基数（2^8 = 256个bucket）
  // 每轮radix select处理8位，四轮处理完32位（FP32）或两轮处理完16位（FP16/BF16）
  constexpr uint32_t RADIX = 256;  // 8位radix

  // 获取当前CTA的全局ID和组内索引
  // global_cta_id: 当前CTA在grid中的索引
  // group_id: CTA所属的组（每组处理一行），等于 global_cta_id / ctas_per_group
  // cta_in_group: CTA在组内的索引，用于计算chunk边界
  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // 共享内存布局：
  // | local_histogram[256] | suffix_sum[256] | shared_scalars[5] | ... 对齐 ... | ordered data ... |
  // - local_histogram: 本地直方图，用于radix select统计每个bucket的元素个数
  // - suffix_sum: 后缀和数组，用于计算 >= 每个bucket值的元素总数
  // - shared_scalars: 标量缓存 [prefix_cache, remaining_k_cache, found_bucket, found_remaining_k, output_counter]
  // - ordered data: 有序值缓存，用于存储从global memory加载的数据
  extern __shared__ uint8_t smem[];

  // 固定共享内存大小计算：两个256元素的uint32_t数组 + 5个标量
  // histogram[256] + suffix[256] + 5个标量（用于RadixSelectFromSharedMemory）
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 5);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;

  // 有序值缓存需要16字节对齐，以确保矢量加载（vectorized load）能正确执行
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  // 多CTA模式：获取当前组的状态指针，用于跨CTA同步和全局直方图
  // 单CTA模式：state为nullptr，不需要跨CTA同步
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // 计算持久循环（persistent loop）的总迭代次数
  // 由于行数可能大于CTA组数，需要多轮迭代才能处理完所有行
  // num_groups: grid中CTA组的数量 = gridDim.x / ctas_per_group
  // total_iterations: 每组需要处理的行数（向上取整）
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  // barrier_phase: 软件屏障阶段计数器，用于多CTA同步
  // 每次调用AdvanceRadixGroupBarrier时递增
  int barrier_phase = 0;

  // 持久循环：每轮迭代处理一行
  // 同一组的多个CTA协同处理同一行数据
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    // 计算当前迭代处理的行索引
    // 例如：num_groups=4, iter=0时处理行0,1,2,3; iter=1时处理行4,5,6,7
    uint32_t row_idx = group_id + iter * num_groups;

    // 行索引超出batch范围时退出
    if (row_idx >= batch_size) break;

    // 计算当前CTA处理的元素范围（chunk）
    // chunk_start: 当前CTA处理的起始位置 = cta_in_group * chunk_size
    // chunk_end: 结束位置，受vocab_size边界限制
    // actual_chunk_size: 实际处理的元素数
    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    // 获取当前行的k值：优先使用top_k_arr中的值，否则使用全局top_k_val
    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // pivot: 阈值元素值，初始为负无穷
    // 在radix select过程中会被更新为第k大的元素值
    DType pivot = Traits::NegInf();

    // 边界情况：当 k >= vocab_size 时，无需masking，直接复制所有logits
    // 这种情况下所有元素都在top-k中，不需要过滤任何元素
    if (k >= vocab_size) {
      // k >= vocab_size: no masking needed, just copy
      vec_t<DType, VEC_SIZE> logits_vec_copy;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        logits_vec_copy.cast_load(logits + row_idx * vocab_size + chunk_start + i);
        logits_vec_copy.store(masked_logits + row_idx * vocab_size + chunk_start + i);
      }
      // 处理尾部
      // 使用矢量加载（vectorized load）高效复制对齐部分
      // vec_t<DType, VEC_SIZE> 支持4字节或8字节矢量加载，提高内存带宽利用率
      // VEC_SIZE 的计算：对于float为4，half/bf16为8
      vec_t<DType, VEC_SIZE> logits_vec_copy;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        // 从global memory加载logits到寄存器
        logits_vec_copy.cast_load(logits + row_idx * vocab_size + chunk_start + i);
        // 直接存储到输出，不做任何过滤
        logits_vec_copy.store(masked_logits + row_idx * vocab_size + chunk_start + i);
      }
      // 处理尾部（不对齐部分），使用标量访问
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        masked_logits[row_idx * vocab_size + chunk_start + i] =
            logits[row_idx * vocab_size + chunk_start + i];
      }

      // 为下一次迭代清理直方图（以防后续行的 k < vocab_size）
      // 使用三重缓冲：清理下一个迭代将使用的直方图缓冲区
      // 仅多CTA模式需要清理；单CTA模式每次迭代都会重新初始化共享内存
      if constexpr (!SINGLE_CTA) {
        // 计算下一个迭代对应的直方图缓冲区索引（使用三重缓冲避免数据冲突）
        constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
        uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
        // 仅领先CTA（cta_in_group == 0）负责清理，避免多CTA同时写入
        if (cta_in_group == 0) {
          for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
            state->histogram[next_first_hist_idx][i] = 0;
          }
        }
        // 无需显式同步 - 下一次迭代的barrier将确保直方图清理的可见性
      }
      continue;
    }

    // 阶段1：查找pivot值（第k大的元素）
    // 使用radix select算法高效地找到第k大的元素值
    // 此阶段会：
    // 1. 将当前CTA处理的logits加载到共享内存
    // 2. 转换为有序整数表示（处理符号位）
    // 3. 执行多轮radix select找到pivot
    // 4. 统计 >pivot 和 ==pivot 的元素个数
    // 注意：MaskLogits内核不使用local_gt_count和local_eq_count，因为只需要pivot值
    uint32_t local_gt_count = 0;  // 此内核不使用
    uint32_t local_eq_count = 0;  // 此内核不使用
    OrderedType ordered_pivot =
        RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, false, DType>(
            logits + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum,
            shared_scalars, state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group,
            cta_in_group, tx, iter, local_gt_count, local_eq_count);

    // 将pivot从有序整数表示转换回原始浮点数格式
    pivot = Traits::FromOrdered(ordered_pivot);

    // 阶段2：最终masking通道
    // 将所有低于pivot的logits值设为负无穷（-inf）
    // 这样softmax时这些元素的值趋近于0，实现Top-K过滤效果
    // 使用矢量加载提高效率，对每个元素进行阈值比较
    const DType neg_inf = Traits::NegInf();
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
    vec_t<DType, VEC_SIZE> logits_vec;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      // 加载logits
      logits_vec.cast_load(logits + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        // 如果元素值 >= pivot，保留原值；否则设为-neg_inf
        logits_vec[j] = (logits_vec[j] >= pivot) ? logits_vec[j] : neg_inf;
      }
      // 存储mask后的结果
      logits_vec.store(masked_logits + row_idx * vocab_size + chunk_start + i);
    }

    // 处理尾部（不对齐部分）
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = logits[row_idx * vocab_size + chunk_start + i];
      masked_logits[row_idx * vocab_size + chunk_start + i] = (val >= pivot) ? val : neg_inf;
    }
  }

  // 内核结束时的清理工作
  // 重置直方图缓冲区和到达计数器，为下一次内核启动做准备
  // 仅多CTA模式需要清理；单CTA模式由运行时自动管理
  if constexpr (!SINGLE_CTA) {
    // 仅领先CTA（cta_in_group == 0）执行清理，避免数据竞争
    // 使用release语义确保清理操作对其他CTA可见
    if (cta_in_group == 0) {
      // 清理所有三个直方图缓冲区
      for (uint32_t buf = 0; buf < 3; ++buf) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          state->histogram[buf][i] = 0;
        }
      }

      // 重置到达计数器为0，确保下一次内核启动时同步正确
      if (tx == 0) {
        st_release(&state->arrival_counter, 0);
      }
    }
  }
}

template <typename DType, typename IdType>
cudaError_t RadixTopKMaskLogitsMultiCTA(DType* logits, DType* masked_logits, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // 获取设备属性
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // 固定共享内存开销：histogram[256] + suffix_sum[256] + 5个标量
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);

  // 计算适合共享内存的最大分块大小
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, false);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
  uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
  const bool single_cta = (ctas_per_group == 1);

  // 计算组数（同时处理多少行）
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      auto kernel =
          RadixTopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      auto kernel =
          RadixTopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });

  return cudaSuccess;
}

// ==================== Multi-CTA Radix Top-K Renorm Probs ====================

/*!
 * \brief Multi-CTA Radix Top-K RenormProb kernel（概率归一化内核）
 *
 * 此内核用于Top-K概率归一化：
 * 1. 找到第k大的概率值作为pivot
 * 2. 将所有 >= pivot 的概率归一化，使它们的和为1
 * 3. 将所有 < pivot 的概率设为0
 *
 * 应用场景：Top-K采样中，需要将概率分布归一化到top-k个最可能的token
 *
 * \tparam BLOCK_THREADS 每block的线程数
 * \tparam VEC_SIZE 矢量加载大小
 * \tparam SINGLE_CTA 是否为单CTA模式
 * \tparam DType 数据类型（float, half, nv_bfloat16）
 * \tparam IdType 索引类型
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKRenormProbKernel_MultiCTA(
    DType* probs,          // [batch, vocab_size] - 输入的原始概率分布
    DType* renormed_prob,  // [batch, vocab_size] - 输出：归一化后的概率分布
    IdType* top_k_arr,     // [batch] - 每行可选的top-k值数组，如果为nullptr则使用全局top_k_val
    uint32_t top_k_val,    // 默认的top-k值
    uint32_t vocab_size,   // 词汇表大小（每行的元素数）
    uint32_t batch_size,   // batch大小（行数）
    RadixRowState* row_states,  // [num_groups] - 多CTA状态数组（单CTA模式为nullptr）
    uint32_t chunk_size,        // 每个CTA处理的元素数
    uint32_t ctas_per_group)    // 每行使用的CTA数量（单CTA模式为1）
{
  // 获取数据类型对应的RadixTopK Traits
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  // RADIX = 256 表示使用8位基数
  constexpr uint32_t RADIX = 256;  // 8-bit radix

  // 获取当前CTA的全局ID和组内索引
  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // 共享内存布局：
  // | local_histogram[256] | suffix_sum[256] | shared_scalars[4] | shared_sum[1] | ... 对齐 ... | ordered data ... |
  // - local_histogram: 本地直方图
  // - suffix_sum: 后缀和数组
  // - shared_scalars: 标量缓存
  // - shared_sum: 单CTA模式下的局部求和结果
  extern __shared__ uint8_t smem[];

  // 固定共享内存大小：两个256元素uint32_t数组 + 4个标量 + 1个float
  // histogram[256] + suffix[256] + scalars[4] + sum_local[1]
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  float* shared_sum = reinterpret_cast<float*>(shared_scalars + 4);

  // 有序值缓存需要16字节对齐
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  // 多CTA模式：获取当前组的状态指针
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // 计算持久循环的总迭代次数
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  // barrier_phase: 软件屏障阶段计数器
  int barrier_phase = 0;

  // 持久循环遍历行
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    // 计算当前CTA处理的chunk边界
    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    // 获取当前行的k值
    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // pivot: 阈值概率值（用于归一化）
    // normalizer: 归一化因子（= 1 / sum_of_topk_probabilities）
    DType pivot = DType(0);
    float normalizer = 1.0f;

    // 边界情况：当 k >= vocab_size 时，无需过滤
    // 只需要计算所有概率的总和，然后重新归一化
    if (k >= vocab_size) {
      // k >= vocab_size: 无需过滤，只需计算总和并重新归一化
      // 阶段1：计算当前CTA chunk中所有概率的总和
      // 使用thread本地累加器收集每个线程处理的元素
      float thread_sum = 0.0f;
      vec_t<DType, VEC_SIZE> data_vec;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          // 累加每个元素的概率值
          thread_sum += float(data_vec[j]);
        }
      }
      // 处理尾部（不对齐部分）
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        thread_sum += float(probs[row_idx * vocab_size + chunk_start + i]);
      }

      // 使用CUB的BlockReduce进行块内归约求和
      // 将所有线程的thread_sum汇总为block_sum
      typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
      __syncthreads();

      // 多CTA模式：跨CTA累加总和
      if constexpr (!SINGLE_CTA) {
        // 多CTA: 使用原子加将每个CTA的block_sum累加到全局state->sum_topk
        if (tx == 0) {
          // 仅领先CTA（cta_in_group == 0）初始化全局和为0
          if (cta_in_group == 0) {
            state->sum_topk = 0.0f;  // 第一个CTA初始化
          }
        }
        // 初始化barrier：确保所有CTA都已准备好累加
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

        // 每个CTA将其block_sum原子加到全局和
        if (tx == 0 && block_sum > 0) {
          atomicAdd(&state->sum_topk, block_sum);
        }

        // Barrier确保所有CTA都完成了累加操作
        // 然后计算归一化因子：使用rcp（倒数）指令加速除法
        // 添加1e-8f避免除零
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
        normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
      } else {
        // 单CTA模式：直接使用block_sum
        if (tx == 0) {
          *shared_sum = block_sum;
        }
        __syncthreads();
        normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
      }

      // 阶段2：归一化并存储结果
      // 将每个概率值乘以归一化因子后输出
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          data_vec[j] = DType(float(data_vec[j]) * normalizer);
        }
        data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
      }
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        renormed_prob[row_idx * vocab_size + chunk_start + i] =
            DType(float(probs[row_idx * vocab_size + chunk_start + i]) * normalizer);
      }

      // 为下一次迭代清理直方图（以防k < vocab_size）
      // 仅多CTA模式需要；单CTA使用每次迭代清理的共享内存
      // 下一次迭代(iter+1)将使用histogram[((iter+1)*NUM_ROUNDS) % 3]作为第一轮
      // 为下一次迭代清理直方图（以防后续行的 k < vocab_size）
      // 仅多CTA模式需要清理
      if constexpr (!SINGLE_CTA) {
        constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
        uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
        if (cta_in_group == 0) {
          for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
            state->histogram[next_first_hist_idx][i] = 0;
          }
        }
        // 无需sync - 下一次迭代的barrier将确保可见性
      }
      continue;
    }

    // ========== 阶段1：使用radix select找pivot（阈值概率值）==========
    // 通过radix select算法找到第k大的概率值
    // 这个值将作为归一化的分界点：>= pivot的概率将被保留并归一化
    uint32_t local_gt_count = 0;  // 此内核不使用
    uint32_t local_eq_count = 0;  // 此内核不使用
    auto ordered_pivot = RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, false, DType>(
        probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
        state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, cta_in_group, tx,
        iter, local_gt_count, local_eq_count);
    // 将pivot从有序整数转换回原始浮点数格式
    pivot = Traits::FromOrdered(ordered_pivot);

    // ========== 阶段2：计算 >= pivot 的所有元素的和（normalizer）==========
    // 统计当前CTA chunk中所有 >= pivot 的概率值之和
    // 这个和将作为归一化因子，使top-k概率之和为1
    float thread_sum = 0.0f;
    vec_t<DType, VEC_SIZE> data_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        // 只有 >= pivot 的元素才被累加
        if (data_vec[j] >= pivot) {
          thread_sum += float(data_vec[j]);
        }
      }
    }
    // 处理尾部
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      if (val >= pivot) {
        thread_sum += float(val);
      }
    }

    // 使用CUB的BlockReduce进行块内归约求和
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    __syncthreads();

    // 多CTA模式：跨CTA累加求和
    if constexpr (!SINGLE_CTA) {
      // 使用原子加累加每个CTA的block_sum到全局state->sum_topk
      if (tx == 0) {
        if (cta_in_group == 0) {
          state->sum_topk = 0.0f;  // 第一个CTA初始化
        }
      }
      // 初始化barrier
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

      if (tx == 0 && block_sum > 0) {
        atomicAdd(&state->sum_topk, block_sum);
      }

      // Barrier确保所有CTA都贡献了
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
      // 计算归一化因子：1 / sum_topk，使用rcp指令加速
      normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
    } else {
      // 单CTA模式：直接使用block_sum
      if (tx == 0) {
        *shared_sum = block_sum;
      }
      __syncthreads();
      normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
    }

    // 阶段3：使用矢量存储进行归一化输出
    // 对齐部分：使用矢量加载+处理+存储
#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        // 如果 >= pivot，乘以归一化因子；否则设为0
        data_vec[j] = (data_vec[j] >= pivot) ? DType(float(data_vec[j]) * normalizer) : DType(0);
      }
      data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
    }
    // 处理尾部（不对齐部分）
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      renormed_prob[row_idx * vocab_size + chunk_start + i] =
          (val >= pivot) ? DType(float(val) * normalizer) : DType(0);
    }
  }

  // 内核结束时的清理工作
  // 重置直方图缓冲区和到达计数器，为下一次内核启动做准备
  if constexpr (!SINGLE_CTA) {
    // 仅领先CTA执行清理，使用release语义
    if (cta_in_group == 0) {
      // 清理所有三个直方图缓冲区
      for (uint32_t buf = 0; buf < 3; ++buf) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          state->histogram[buf][i] = 0;
        }
      }

      // 重置到达计数器为0
      if (tx == 0) {
        st_release(&state->arrival_counter, 0);
      }
    }
  }
}

template <typename DType, typename IdType>
cudaError_t RadixTopKRenormProbMultiCTA(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // 获取设备属性
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // 固定共享内存开销：histogram[256] + suffix_sum[256] + 4个标量 + 1个float
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 4) + sizeof(float);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);

  // 计算适合共享内存的最大分块大小
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, false);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
  uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
  const bool single_cta = (ctas_per_group == 1);

  // 计算组数（同时处理多少行）
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      auto kernel =
          RadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      auto kernel =
          RadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });

  return cudaSuccess;
}

/*!
 * \brief Launch multi-CTA Radix Top-K with Page Table Transform kernel.
 *
 * Performs top-k selection and gathers indices through a page table.
 * Used for sparse attention's second stage in prefill mode.
 *
 * \param input Input scores tensor [num_rows, max_len]
 * \param output_page_table Output page table entries [num_rows, top_k]
 * \param src_page_table Source page table [batch_size, max_len]
 * \param src_stride Stride of source page table (typically max_len)
 * \param row_to_batch Mapping from row index to batch index [num_rows], or nullptr if 1:1
 * \param lengths Sequence lengths per row [num_rows]
 * \param num_rows Number of rows to process
 * \param top_k_val Number of top elements to select
 * \param max_len Maximum sequence length (input stride)
 * \param row_states_buffer Buffer for inter-CTA synchronization
 * \param stream CUDA stream
 */
template <typename DType, typename IdType>
cudaError_t RadixTopKPageTableTransformMultiCTA(DType* input, IdType* output_page_table,
                                                const IdType* src_page_table, int64_t src_stride,
                                                const IdType* row_to_batch, IdType* lengths,
                                                uint32_t num_rows, uint32_t top_k_val,
                                                uint32_t max_len, RadixRowState* row_states_buffer,
                                                bool deterministic, cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), max_len);

  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, deterministic);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }

  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = ceil_div(max_len, max_chunk_elements);
  if (deterministic && ctas_per_group > RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP) {
    return cudaErrorInvalidConfiguration;
  }
  uint32_t chunk_size = ceil_div(max_len, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const bool single_cta = (ctas_per_group == 1);
  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);

  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, num_rows);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;
  RadixDeterministicCollectScratch* det_scratch_buffer =
      MaybeGetRadixDeterministicCollectScratchBuffer(row_states_buffer, num_groups, single_cta,
                                                     deterministic);

  // 统一内核参数
  DType* output_values = nullptr;  // PageTableTransform模式不使用
  dim3 nblks(total_ctas);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&input,      &output_page_table, &output_values,     &src_page_table,
                  &lengths,    &row_to_batch,      &src_stride,        &top_k_val,
                  &max_len,    &num_rows,          &row_states_buffer, &det_scratch_buffer,
                  &chunk_size, &ctas_per_group};

#define LAUNCH_PAGE_TABLE_KERNEL(THREADS, SINGLE_CTA_FLAG, DET_FLAG)                              \
  do {                                                                                            \
    auto kernel = RadixTopKKernel_Unified<THREADS, VEC_SIZE, SINGLE_CTA_FLAG, DET_FLAG,           \
                                          RadixTopKMode::PageTableTransform, DType, IdType>;      \
    FLASHINFER_CUDA_CALL(                                                                         \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));    \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)); \
  } while (0)

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      if (!deterministic) {
        LAUNCH_PAGE_TABLE_KERNEL(BLOCK_THREADS, true, false);
      } else {
        LAUNCH_PAGE_TABLE_KERNEL(BLOCK_THREADS, true, true);
      }
    } else {
      if (!deterministic) {
        LAUNCH_PAGE_TABLE_KERNEL(BLOCK_THREADS, false, false);
      } else {
        LAUNCH_PAGE_TABLE_KERNEL(BLOCK_THREADS, false, true);
      }
    }
  });

#undef LAUNCH_PAGE_TABLE_KERNEL

  return cudaSuccess;
}

/*!
 * \brief Launch multi-CTA Radix Top-K with Ragged Index Transform kernel.
 *
 * Performs top-k selection and adds an offset to each index.
 * Used for sparse attention's second stage with ragged KV cache.
 *
 * \param input Input scores tensor [num_rows, max_len]
 * \param output_indices Output indices [num_rows, top_k]
 * \param offsets Offset to add per row [num_rows]
 * \param lengths Sequence lengths per row [num_rows]
 * \param num_rows Number of rows to process
 * \param top_k_val Number of top elements to select
 * \param max_len Maximum sequence length (input stride)
 * \param row_states_buffer Buffer for inter-CTA synchronization
 * \param stream CUDA stream
 */
template <typename DType, typename IdType>
cudaError_t RadixTopKRaggedTransformMultiCTA(DType* input, IdType* output_indices,
                                             const IdType* offsets, IdType* lengths,
                                             uint32_t num_rows, uint32_t top_k_val,
                                             uint32_t max_len, RadixRowState* row_states_buffer,
                                             bool deterministic, cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), max_len);

  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, deterministic);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }

  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = ceil_div(max_len, max_chunk_elements);
  if (deterministic && ctas_per_group > RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP) {
    return cudaErrorInvalidConfiguration;
  }
  uint32_t chunk_size = ceil_div(max_len, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const bool single_cta = (ctas_per_group == 1);
  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);

  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, num_rows);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;
  RadixDeterministicCollectScratch* det_scratch_buffer =
      MaybeGetRadixDeterministicCollectScratchBuffer(row_states_buffer, num_groups, single_cta,
                                                     deterministic);

  // 统一内核参数
  DType* output_values = nullptr;        // RaggedTransform模式不使用
  const IdType* row_to_batch = nullptr;  // RaggedTransform模式不使用
  int64_t aux_stride = 0;                // RaggedTransform模式不使用
  dim3 nblks(total_ctas);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&input,      &output_indices, &output_values,     &offsets,
                  &lengths,    &row_to_batch,   &aux_stride,        &top_k_val,
                  &max_len,    &num_rows,       &row_states_buffer, &det_scratch_buffer,
                  &chunk_size, &ctas_per_group};

#define LAUNCH_RAGGED_KERNEL(THREADS, SINGLE_CTA_FLAG, DET_FLAG)                                  \
  do {                                                                                            \
    auto kernel = RadixTopKKernel_Unified<THREADS, VEC_SIZE, SINGLE_CTA_FLAG, DET_FLAG,           \
                                          RadixTopKMode::RaggedTransform, DType, IdType>;         \
    FLASHINFER_CUDA_CALL(                                                                         \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));    \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)); \
  } while (0)

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      if (!deterministic) {
        LAUNCH_RAGGED_KERNEL(BLOCK_THREADS, true, false);
      } else {
        LAUNCH_RAGGED_KERNEL(BLOCK_THREADS, true, true);
      }
    } else {
      if (!deterministic) {
        LAUNCH_RAGGED_KERNEL(BLOCK_THREADS, false, false);
      } else {
        LAUNCH_RAGGED_KERNEL(BLOCK_THREADS, false, true);
      }
    }
  });

#undef LAUNCH_RAGGED_KERNEL

  return cudaSuccess;
}

/*!
 * \brief Launch multi-CTA Radix Top-K kernel (returns indices and values)
 *
 * \param input Input tensor [batch_size, vocab_size]
 * \param output_indices Output indices tensor [batch_size, top_k]
 * \param output_values Output values tensor [batch_size, top_k]
 * \param top_k_arr Per-row top-k values or nullptr for uniform top_k
 * \param batch_size Number of rows
 * \param top_k_val Default top-k value (used when top_k_arr is nullptr)
 * \param vocab_size Number of elements per row
 * \param row_states_buffer Buffer for inter-CTA synchronization
 * \param stream CUDA stream
 */
template <typename DType, typename IdType>
cudaError_t RadixTopKMultiCTA(DType* input, IdType* output_indices, DType* output_values,
                              IdType* top_k_arr, uint32_t batch_size, uint32_t top_k_val,
                              uint32_t vocab_size, RadixRowState* row_states_buffer,
                              bool deterministic, cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // 固定共享内存：histogram[256] + suffix_sum[256] + 标量
  // 标量：单CTA用5个，多CTA用4个
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, deterministic);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }

  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
  if (deterministic && ctas_per_group > RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP) {
    return cudaErrorInvalidConfiguration;
  }
  uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  // 判断是否使用单CTA路径
  const bool single_cta = (ctas_per_group == 1);

  // 计算smem_size：固定 + 有序值
  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);

  // 计算组数（同时处理多少行）
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;
  RadixDeterministicCollectScratch* det_scratch_buffer =
      MaybeGetRadixDeterministicCollectScratchBuffer(row_states_buffer, num_groups, single_cta,
                                                     deterministic);

  // 统一内核参数
  IdType* lengths = nullptr;             // Basic模式不使用
  const IdType* row_to_batch = nullptr;  // Basic模式不使用
  int64_t aux_stride = 0;                // Basic模式不使用
  dim3 nblks(total_ctas);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&input,      &output_indices, &output_values,     &top_k_arr,
                  &lengths,    &row_to_batch,   &aux_stride,        &top_k_val,
                  &vocab_size, &batch_size,     &row_states_buffer, &det_scratch_buffer,
                  &chunk_size, &ctas_per_group};

#define LAUNCH_BASIC_KERNEL(THREADS, SINGLE_CTA_FLAG, DET_FLAG)                                   \
  do {                                                                                            \
    auto kernel = RadixTopKKernel_Unified<THREADS, VEC_SIZE, SINGLE_CTA_FLAG, DET_FLAG,           \
                                          RadixTopKMode::Basic, DType, IdType>;                   \
    FLASHINFER_CUDA_CALL(                                                                         \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));    \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)); \
  } while (0)

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      if (!deterministic) {
        LAUNCH_BASIC_KERNEL(BLOCK_THREADS, true, false);
      } else {
        LAUNCH_BASIC_KERNEL(BLOCK_THREADS, true, true);
      }
    } else {
      if (!deterministic) {
        LAUNCH_BASIC_KERNEL(BLOCK_THREADS, false, false);
      } else {
        LAUNCH_BASIC_KERNEL(BLOCK_THREADS, false, true);
      }
    }
  });

#undef LAUNCH_BASIC_KERNEL

  return cudaSuccess;
}
// ==================== FilteredTopK Implementation ====================
// 基于 sgl-kernel 的过滤算法，支持多数据类型

// 不同数据类型的 FilteredTopK traits
template <typename DType>
struct FilteredTopKTraits;

// float (32位) 特化：粗直方图使用FP16高8位，4轮细化和
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // 转换为FP16表示并提取高8位
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key =
        (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

// half (16位) 特化：粗直方图使用高8位，只需低8位进行细化
// 因为粗key = 高8位，细化只需看低8位（如果粗粒度能确定topk，则不需要额外轮次）
template <>
struct FilteredTopKTraits<half> {
  using OrderedType = uint16_t;
  static constexpr int NUM_REFINE_ROUNDS = 1;   // 仅需1轮处理低8位
  static constexpr int FIRST_REFINE_SHIFT = 0;  // 从bit 0开始（低8位）

  __device__ __forceinline__ static uint8_t ToCoarseKey(half x) {
    uint16_t bits = __half_as_ushort(x);
    uint16_t key =
        (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(half x) {
    uint16_t bits = __half_as_ushort(x);
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  }
};

// nv_bfloat16 (16位) 特化：与half相同
template <>
struct FilteredTopKTraits<nv_bfloat16> {
  using OrderedType = uint16_t;
  static constexpr int NUM_REFINE_ROUNDS = 1;
  static constexpr int FIRST_REFINE_SHIFT = 0;

  __device__ __forceinline__ static uint8_t ToCoarseKey(nv_bfloat16 x) {
    uint16_t bits = __bfloat16_as_ushort(x);
    uint16_t key =
        (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(nv_bfloat16 x) {
    uint16_t bits = __bfloat16_as_ushort(x);
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  }
};

// FilteredTopK 常量
constexpr uint32_t FILTERED_TOPK_MAX_K = 2048;
constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE = 16 * 1024;  // 每个缓冲区16K个索引
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

// 统一 FilteredTopK 内核的输出模式
enum class FilteredTopKMode { Plain, PageTable, Ragged };

/*!
 * \brief Unified Filtered Top-K kernel supporting multiple output modes.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 * \tparam MODE Output mode (Plain, PageTable, Ragged)
 *
  * Parameters vary by mode:
  * - Plain: output = indices, aux_output = values, aux_input/aux_stride/row_to_batch unused
  *   （普通模式）：直接输出索引和对应的值
  * - PageTable: output = dst_page_table, aux_input = src_page_table, aux_stride = src_stride
  *   （页表模式）：通过页表转换获取实际token ID
  * - Ragged: output = indices, aux_input = offsets, aux_output/aux_stride/row_to_batch unused
  *   （Ragged模式）：通过偏移量转换获取实际索引
  */
template <typename DType, typename IdType, int VEC_SIZE, bool DETERMINISTIC, FilteredTopKMode MODE>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input, IdType* __restrict__ output,
                              DType* __restrict__ aux_output,           // values for Plain mode
                              const IdType* __restrict__ aux_input,     // page_table or offsets
                              int64_t aux_stride,                       // src_stride for PageTable
                              const IdType* __restrict__ row_to_batch,  // for PageTable
                              const IdType* __restrict__ lengths, uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  // FILTERED_TOPK_BLOCK_THREADS: 过滤后TopK的block线程数（通常是256或512）
  // RADIX = 256: 8位基数，用于直方图统计
  // SMEM_INPUT_SIZE: 共享内存中输入索引数组的大小
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");

  // bid: 当前处理的行索引（batch index）
  // tx: 线程索引
  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  // 如果行索引超出范围，直接返回
  if (bid >= num_rows) return;

  // 获取当前行的长度和输入/输出指针
  // length: 当前行的实际元素数（可能小于max_len）
  // score: 当前行的输入分数数组
  // dst: 当前行的输出索引数组
  const int length = (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + static_cast<size_t>(bid) * max_len;
  IdType* dst = output + bid * top_k;

  // 模式相关设置：根据MODE设置不同的辅助数据指针
  // src_page_entry: 页表模式下，指向源页表条目
  // offset_val: Ragged模式下，当前行的偏移量
  // dst_values: Plain模式下，指向输出值数组
  [[maybe_unused]] const IdType* src_page_entry = nullptr;
  [[maybe_unused]] IdType offset_val = 0;
  [[maybe_unused]] DType* dst_values = nullptr;

  if constexpr (MODE == FilteredTopKMode::PageTable) {
    // PageTable模式：从row_to_batch获取batch索引，然后定位页表条目
    const uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[bid] : bid;
    src_page_entry = aux_input + batch_idx * aux_stride;
  } else if constexpr (MODE == FilteredTopKMode::Ragged) {
    // Ragged模式：从aux_input获取当前行的偏移量
    offset_val = aux_input[bid];
  } else {  // Plain (普通模式)
    // Plain模式：输出值存储在aux_output中
    dst_values = aux_output + bid * top_k;
  }

  // 简单情况：当有效元素数 <= top_k 时，无需排序
  // 直接按顺序输出所有元素，不足部分用-1填充，值为0
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      if constexpr (MODE == FilteredTopKMode::Plain) {
        // Plain模式：输出索引和对应的分数值
        if (i < length) {
          dst[i] = static_cast<IdType>(i);
          dst_values[i] = score[i];
        } else {
          dst[i] = static_cast<IdType>(-1);
          dst_values[i] = DType(0);
        }
      } else if constexpr (DETERMINISTIC) {
        // 确定性模式：page-table/ragged转换发生在SortTopKByIndexKernel中
        dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
      } else if constexpr (MODE == FilteredTopKMode::PageTable) {
        // PageTable模式：直接查表获取实际token ID
        dst[i] = (i < length) ? src_page_entry[i] : static_cast<IdType>(-1);
      } else {  // Ragged (Ragged模式)
        // Ragged模式：添加偏移量得到最终索引
        dst[i] = (i < length) ? static_cast<IdType>(i) + offset_val : static_cast<IdType>(-1);
      }
    }
    return;
  }

  // 静态共享内存声明（编译时确定大小）
  // s_histogram_buf[2][RADIX + 128]: 双缓冲直方图（+128用于对齐）
  // s_counter: 输出计数器
  // s_threshold_bin_id: 阈值bucket ID
  // s_refine_thresholds[4]: 每轮阈值副本（用于确定性pivot重建）
  // s_num_input[2]: 输入计数双缓冲
  // s_indices[FILTERED_TOPK_MAX_K]: 排序后的索引数组
  // s_refine_overflow: 溢出标志
  // s_last_remain: 最后一轮剩余元素数
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  __shared__ int s_counter;
  __shared__ int s_threshold_bin_id;
  // 每轮s_threshold_bin_id的副本，用于确定性pivot重建
  __shared__ int s_refine_thresholds[4];
  __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[FILTERED_TOPK_MAX_K];
  // 当s_input_idx在重 workload 中溢出时设为1
  __shared__ int s_refine_overflow;
  __shared__ int s_last_remain;

  auto& s_histogram = s_histogram_buf[0];

  // 用于输入双缓冲的动态共享内存（运行时确定大小）
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  // 获取FilteredTopK的类型traits
  using Traits = FilteredTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;
  int topk = top_k;
  // 初始化共享内存
  if (tx == 0) s_refine_overflow = 0;
  if constexpr (DETERMINISTIC) {
    if (tx < 4) {
      s_refine_thresholds[tx] = 0xFF;
    }
  }
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  // 阶段1：构建粗直方图并识别阈值bucket
  // 此阶段在确定性和非确定性模式中是共享的
  // 使用粗直方图（高8位）快速找到包含第k大元素的bucket
  // 模式差异只在后续收集 == pivot 元素时才会出现
  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
  // 全行扫描辅助函数（向量化主体 + 尾部）
  // 用于遍历当前行的所有分数值
  // 溢出回退时复用此遍历逻辑
  auto for_each_score_full = [&](auto&& fn) {
  // 向量化主体：使用矢量加载提高效率
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length; base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        fn(score_vec[j], base + j);
      }
    }
    // 尾部：处理不对齐的剩余元素
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      fn(score[i], i);
    }
  };
  // 累加到粗直方图的回调函数
  // 将每个分数转换为其粗key（高8位），然后原子加到直方图
  auto accumulate_coarse_hist = [&](auto raw_input, int /*index*/) {
    const auto bin = Traits::ToCoarseKey(raw_input);
    atomicAdd(&s_histogram[bin], 1);
  };
  // 执行全行扫描，构建粗直方图
  for_each_score_full(accumulate_coarse_hist);
  __syncthreads();

  // 后缀和计算（Hillis-Steele Scan算法）
  // 用于计算 >= 每个bucket的元素总数
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };
  // 更新细化阈值的回调函数
  // 找到第一个满足：s_histogram[tx] > topk 且 s_histogram[tx+1] <= topk 的bucket
  auto update_refine_threshold = [&](int next_input_idx, auto reset_next_input_tag) {
    constexpr bool RESET_NEXT_INPUT = decltype(reset_next_input_tag)::value;
    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      if constexpr (RESET_NEXT_INPUT) {
        s_num_input[next_input_idx] = 0;
      }
      // 计算剩余需要收集的元素数
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();
  };

  // 第一轮：运行后缀和计算，找到阈值bucket
  // 阈值bucket满足：s_histogram[tx] > topk 且 s_histogram[tx+1] <= topk
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  // 计算需要从阈值bucket中收集的元素数
  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];
  [[maybe_unused]] const int topk_after_coarse = topk;

  // 获取细化的轮数和移位值
  // fp16/bf16: NUM_ROUNDS = 1, FIRST_SHIFT = 8
  // fp32: NUM_ROUNDS = 4, FIRST_SHIFT = 24
  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  // fp16/bf16: stop_round = 0（只需1轮细化）
  // fp32: stop_round = 0,1,2,3（需要4轮细化）
  // 构建确定性pivot值：根据每轮的阈值构建完整的pivot
  auto build_det_pivot = [&](int stop_round) -> OrderedType {
    if constexpr (sizeof(OrderedType) == 2) {
      // fp16/bf16: 16位，pivot = (threshold_bin << 8) | refine_thresholds[0]
      return static_cast<OrderedType>((static_cast<uint32_t>(threshold_bin) << 8) |
                                      static_cast<uint32_t>(s_refine_thresholds[0]));
    } else {  // fp32
      // fp32: 32位，从每轮细化中累积构建pivot
      uint32_t pivot = 0;
      for (int round = 0; round < NUM_ROUNDS; ++round) {
        // 如果该轮已完成，使用细化阈值；否则使用0xFF（最小值）
        uint32_t byte =
            (round <= stop_round) ? static_cast<uint32_t>(s_refine_thresholds[round]) : 0xFFu;
        pivot |= (byte << (FIRST_SHIFT - round * 8));
      }
      return static_cast<OrderedType>(pivot);
    }
  };

  // 特殊情况：如果topk已经为0，只需收集 > threshold的元素
  if (topk == 0) {
    // 收集 bin > threshold 的索引
    auto collect_coarse_gt = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      }
    };
    for_each_score_full(collect_coarse_gt);
    __syncthreads();
  } else {
    // 继续处理：需要进一步细化阈值bucket
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // 非确定性和确定性模式都使用atomicAdd追加 > threshold winners；
    // 只有 == threshold 的处理在两种模式之间有所不同。
    // 非确定性模式：使用atomic从后向前分配==threshold元素
    // 确定性模式：使用block scan保证顺序
    auto collect_gt_and_nondet_eq_threshold = [&](auto value, auto threshold, int idx,
                                                  bool collect_eq) {
      if (value > threshold) {
        // 大于阈值：直接添加到输出
        const int pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = idx;
      } else if constexpr (!DETERMINISTIC) {
        // 非确定性模式：等于阈值时从后向前分配
        if (collect_eq && value == threshold) {
          const int pos = atomicAdd(&s_last_remain, -1);
          if (pos > 0) {
            s_indices[static_cast<int>(top_k) - pos] = idx;
          }
        }
      }
    };

    // 确定性模式：使用block scan收集==pivot元素
    auto collect_det_eq_pivot = [&](OrderedType pivot, int eq_needed) {
      if (eq_needed > 0) {
        // 使用CUB的BlockScan进行确定性收集
        using DetCollectBlockScan =
            cub::BlockScan<uint32_t, BLOCK_SIZE, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
        __shared__ typename DetCollectBlockScan::TempStorage temp_storage;
        DeterministicThreadStridedCollect<BLOCK_SIZE>(
            tx, length, temp_storage,
            // 谓词：只选择 == pivot 的元素
            [&](uint32_t idx) { return Traits::ToOrdered(score[idx]) == pivot; }, eq_needed,
            // 输出：从top_k - eq_needed位置开始写入
            [&](uint32_t idx, uint32_t local_pos) {
              s_indices[static_cast<int>(top_k) - eq_needed + static_cast<int>(local_pos)] =
                  static_cast<int>(idx);
            });
      }
    };

    // 过滤 + 细化直方图
    // 遍历所有分数值：
    // - bin > threshold_bin: 直接添加到输出
    // - bin == threshold_bin: 添加到细化输入缓冲区，构建细化直方图
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        // 大于阈值bucket：直接输出
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        // 等于阈值bucket：添加到细化缓冲区
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          // 缓冲区未溢出：存储索引并构建细化直方图
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          // 提取细化key（低8位）
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        } else {
          // 缓冲区溢出：标记溢出标志
          atomicOr(&s_refine_overflow, 1);
        }
      }
    };
    for_each_score_full(filter_and_add_to_histogram);
    __syncthreads();

    // 阶段2：使用8位radix通道进行细化
    // 细化过程：对于阈值bucket内的元素，进一步按低8位分类
    // 如果阈值bucket候选缓冲区在1轮细化模式（fp16/bf16）中溢出，
    // 切换到慢路径，重新对完整阈值bucket进行直方图统计以保证正确性。
    
    // 最后 round 的收集函数：直接收集，添加到输出末尾
    auto collect_with_threshold_last_round = [&](int r_idx, int num_input, int offset,
                                                 int threshold) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = score[idx];
        const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
        collect_gt_and_nondet_eq_threshold(static_cast<int>(bin), threshold, idx,
                                           /*allow_eq_claim=*/true);
      }
      __syncthreads();
    };
    
    // 非最后 round 的收集函数：准备下一轮细化
    auto collect_with_threshold_non_last_round = [&](int r_idx, int num_input, int offset,
                                                     int threshold) {
      const auto next_r_idx = r_idx ^ 1;
      __syncthreads();
      // 清理直方图准备下一轮
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = score[idx];
        const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
        if (static_cast<int>(bin) > threshold) {
          // 大于阈值：直接输出
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = idx;
        } else if (static_cast<int>(bin) == threshold) {
          // 等于阈值：添加到下一轮细化缓冲区
          const auto pos = atomicAdd(&s_num_input[next_r_idx], 1);
          if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
            s_input_idx[next_r_idx][pos] = idx;
            const auto bin32 = Traits::ToOrdered(raw_input);
            // 提取下一轮的sub_bin
            const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
            atomicAdd(&s_histogram[sub_bin], 1);
          } else {
            // 溢出：标记溢出标志
            atomicOr(&s_refine_overflow, 1);
          }
        }
      }
      __syncthreads();
    };
    
    // 运行一轮细化
    // 如果这一轮完全解决了pivot，即没有==threshold元素需要进入另一轮细化，则返回true
    auto run_refine_round = [&](int r_idx, int offset, auto is_last_round_tag) {
      constexpr bool IS_LAST_ROUND = decltype(is_last_round_tag)::value;
      const auto raw_num_input = s_num_input[r_idx];
      const auto num_input = (raw_num_input < SMEM_INPUT_SIZE) ? raw_num_input : SMEM_INPUT_SIZE;

      // 更新细化阈值：找到新的阈值bucket
      update_refine_threshold(r_idx ^ 1, std::true_type{});

      const auto threshold = s_threshold_bin_id;
      // 确定性模式：保存阈值用于后续pivot重建
      if constexpr (DETERMINISTIC) {
        if (tx == 0) {
          s_refine_thresholds[(FIRST_SHIFT - offset) / 8] = threshold;
        }
      }
      // 更新需要的元素数
      topk -= s_histogram[threshold + 1];
      if (topk == 0) {
        // 最后一轮：仅收集严格大于threshold的bucket
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        return true;
      }

      // 继续细化：根据是否最后一轮选择不同的收集策略
      if constexpr (IS_LAST_ROUND) {
        collect_with_threshold_last_round(r_idx, num_input, offset, threshold);
      } else {
        collect_with_threshold_non_last_round(r_idx, num_input, offset, threshold);
      }
      return false;
    };
    // fp16/bf16: 只需1轮细化（快速路径）
    // fp32: 需要4轮细化
    if constexpr (NUM_ROUNDS == 1) {  // 快速路径：1轮细化（fp16/bf16）
      // 溢出检测：如果细化缓冲区溢出，需要使用慢路径
      // 慢路径：对整个阈值bucket重新构建直方图（保证正确性）
      if (s_refine_overflow) {
        // 慢路径：重新扫描整个阈值bucket
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();

        // 重新构建阈值bucket的完整直方图
        auto build_full_threshold_hist = [&](auto raw_input, int /*index*/) {
          const auto coarse_bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
          if (coarse_bin == threshold_bin) {
            const auto ordered = Traits::ToOrdered(raw_input);
            const auto sub_bin = ordered & 0xFF;
            atomicAdd(&s_histogram[sub_bin], 1);
          }
        };

        for_each_score_full(build_full_threshold_hist);
        __syncthreads();

        // 重新找到阈值
        if (tx == 0) {
          s_threshold_bin_id = 0;
          s_last_remain = 0;
        }
        __syncthreads();

        update_refine_threshold(/*next_input_idx=*/0, std::false_type{});

        const auto threshold = s_threshold_bin_id;

        // 保持s_counter连续性：它已经在filter_and_add_to_histogram中统计了coarse_bin > threshold_bin的元素
        // 这里我们在该前缀之后追加threshold-bin的细化winners
        auto collect_from_full_threshold_bin = [&](auto raw_input, int index) {
          const auto coarse_bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
          if (coarse_bin != threshold_bin) {
            return;
          }
          const auto sub_bin = Traits::ToOrdered(raw_input) & 0xFF;
          collect_gt_and_nondet_eq_threshold(static_cast<int>(sub_bin), threshold, index,
                                             /*allow_eq_claim=*/true);
        };

        for_each_score_full(collect_from_full_threshold_bin);
        __syncthreads();
        // 确定性模式：收集==pivot的元素
        if constexpr (DETERMINISTIC) {
          int eq_needed = s_last_remain;
          collect_det_eq_pivot(static_cast<OrderedType>((static_cast<int>(threshold_bin) << 8) |
                                                        static_cast<int>(threshold)),
                               eq_needed);
        }
      } else {
        // 正常路径：无溢出，直接运行一轮细化
        const int round = 0;
        const auto r_idx = round % 2;
        const int offset = FIRST_SHIFT;
        run_refine_round(r_idx, offset, std::true_type{});
        // 确定性模式：收集==pivot的元素
        if constexpr (DETERMINISTIC) {
          collect_det_eq_pivot(build_det_pivot(/*stop_round=*/0), topk);
        }
      }
    } else {
      // 多轮细化路径（float32）
      // 如果检测到任何细化缓冲区溢出，切换到正确性优先的完整重建阈值bucket选择
      // 这个回退可能比快速路径慢，但避免了部分状态损坏
      int det_stop_round = NUM_ROUNDS - 1;
      if (!s_refine_overflow) {
#pragma unroll
        for (int round = 0; round < NUM_ROUNDS; ++round) {
          const auto r_idx = round % 2;
          const int offset = FIRST_SHIFT - round * 8;
          if (round == NUM_ROUNDS - 1) {
            if (run_refine_round(r_idx, offset, std::true_type{})) {
              det_stop_round = round;
              break;
            }
          } else {
            if (run_refine_round(r_idx, offset, std::false_type{})) {
              det_stop_round = round;
              break;
            }
          }
          if (s_refine_overflow) {
            break;
          }
        }
      }
      if constexpr (DETERMINISTIC) {
        if (!s_refine_overflow) {
          collect_det_eq_pivot(build_det_pivot(det_stop_round), topk);
        }
      }
      // run_refine_round can set s_refine_overflow during the loop above, so this
      // check is intentionally separate from the first if (!s_refine_overflow).
      if (s_refine_overflow) {
        static_assert(sizeof(OrderedType) == 4,
                      "Multi-round overflow fallback expects 32-bit ordered keys.");

        uint32_t topk_remain = static_cast<uint32_t>(topk_after_coarse);
        uint8_t threshold_bytes[NUM_ROUNDS];
#pragma unroll
        for (int i = 0; i < NUM_ROUNDS; ++i) {
          threshold_bytes[i] = 0xFF;
        }
        int stop_round = NUM_ROUNDS - 1;

#pragma unroll
        for (int round = 0; round < NUM_ROUNDS; ++round) {
          const int offset = FIRST_SHIFT - round * 8;

          if (tx < RADIX + 1) s_histogram[tx] = 0;
          __syncthreads();

          auto build_hist = [&](auto raw_input, int /*index*/) {
            const auto coarse_bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
            if (coarse_bin != threshold_bin) {
              return;
            }
            const auto ordered = static_cast<uint32_t>(Traits::ToOrdered(raw_input));
            bool prefix_match = true;
#pragma unroll
            for (int prev = 0; prev < round; ++prev) {
              const int prev_offset = FIRST_SHIFT - prev * 8;
              if (static_cast<uint8_t>((ordered >> prev_offset) & 0xFF) != threshold_bytes[prev]) {
                prefix_match = false;
              }
            }
            if (prefix_match) {
              const auto sub_bin = (ordered >> offset) & 0xFF;
              atomicAdd(&s_histogram[sub_bin], 1);
            }
          };
          for_each_score_full(build_hist);
          __syncthreads();

          run_cumsum();
          if (tx < RADIX && s_histogram[tx] > static_cast<int>(topk_remain) &&
              s_histogram[tx + 1] <= static_cast<int>(topk_remain)) {
            s_threshold_bin_id = tx;
          }
          __syncthreads();

          const int threshold = s_threshold_bin_id;
          threshold_bytes[round] = static_cast<uint8_t>(threshold);
          topk_remain -= static_cast<uint32_t>(s_histogram[threshold + 1]);

          if (topk_remain == 0) {
            stop_round = round;
            break;
          }
        }

        uint32_t pivot = 0;
#pragma unroll
        for (int round = 0; round < NUM_ROUNDS; ++round) {
          const int offset = FIRST_SHIFT - round * 8;
          uint32_t byte = static_cast<uint32_t>(threshold_bytes[round]);
          if (topk_remain == 0 && round > stop_round) {
            byte = 0xFFu;
          }
          pivot |= (byte << offset);
        }
        const int eq_needed = static_cast<int>(topk_remain);

        // 溢出可能发生在前几轮对s_indices/s_counter的部分写入之后。
        // 从完整扫描重置和重建以避免混合陈旧的局部状态。
        if (tx == 0) {
          s_counter = 0;
          s_last_remain = eq_needed;
        }
        __syncthreads();

        // 从头重新收集所有获胜者：
        //   1) coarse_bin > threshold_bin
        //   2) ordered > pivot 的 threshold_bin 条目
        //   3) ordered == pivot 的前 eq_needed 个条目
        auto collect_by_pivot = [&](auto raw_input, int index) {
          const auto coarse_bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
          if (coarse_bin > threshold_bin) {
            collect_gt_and_nondet_eq_threshold(coarse_bin, threshold_bin, index,
                                               /*allow_eq_claim=*/false);
            return;
          }
          if (coarse_bin != threshold_bin) {
            return;
          }
          const auto ordered = static_cast<uint32_t>(Traits::ToOrdered(raw_input));
          collect_gt_and_nondet_eq_threshold(ordered, pivot, index, eq_needed > 0);
        };
        for_each_score_full(collect_by_pivot);
        __syncthreads();
        if constexpr (DETERMINISTIC) {
          collect_det_eq_pivot(static_cast<OrderedType>(pivot), eq_needed);
        }
      }
    }
  }

  // 输出阶段 - 模式相关
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    if constexpr (MODE == FilteredTopKMode::Plain) {
      dst[base] = static_cast<IdType>(idx);
      dst_values[base] = score[idx];
    } else if constexpr (DETERMINISTIC) {  // 在SortTopKByIndexKernel中转换
      dst[base] = static_cast<IdType>(idx);
    } else if constexpr (MODE == FilteredTopKMode::PageTable) {
      dst[base] = src_page_entry[idx];
    } else {  // Ragged (Ragged模式)
      dst[base] = static_cast<IdType>(idx) + offset_val;
    }
  }
}

// 用于 VEC_SIZE 选择的 GCD 计算辅助函数
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// 根据 max_len 和 dtype 计算最优 VEC_SIZE
// 返回 1, 2, 4, 或 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 16 / sizeof(DType);  // float32 为 4，fp16/bf16 为 8
  // 使用 GCD 找最大 2 的幂因子
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <bool WITH_VALUES, uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD, typename DType>
struct SortTopKByIndexBlockRadixSort;

template <uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD, typename DType>
struct SortTopKByIndexBlockRadixSort<true, BLOCK_THREADS, ITEMS_PER_THREAD, DType> {
  using Type = cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, DType>;
};

template <uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD, typename DType>
struct SortTopKByIndexBlockRadixSort<false, BLOCK_THREADS, ITEMS_PER_THREAD, DType> {
  using Type = cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
};

/*!
 * \brief 按索引排序Kernel（用于确定性模式的页表/Ragged转换）
 *
 * 此内核在FilteredTopK确定性子模式中使用：
 * - 在完成TopK筛选后，对索引进行排序
 * - 同时执行页表或Ragged偏移转换
 *
 * 工作原理：
 * 1. 加载当前行的TopK索引作为排序keys
 * 2. 使用CUB的BlockRadixSort对索引进行排序
 * 3. 根据MODE执行模式特定的索引转换
 *
 * \tparam MODE 模式（Plain/PageTable/Ragged）
 * \tparam BLOCK_THREADS 线程数
 * \tparam ITEMS_PER_THREAD 每线程处理元素数
 * \tparam DType 数据类型
 * \tparam IdType 索引类型
 */
template <FilteredTopKMode MODE, uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS)
    SortTopKByIndexKernel(IdType* output_indices, DType* output_values, const IdType* aux_input,
                          int64_t aux_stride, const IdType* row_to_batch, uint32_t top_k,
                          uint32_t max_len) {
  // WITH_VALUES: 是否同时排序值（仅Plain模式需要）
  constexpr bool WITH_VALUES = (MODE == FilteredTopKMode::Plain);
  // 使用CUB的BlockRadixSort进行warp级排序
  using BlockRadixSortT = typename SortTopKByIndexBlockRadixSort<WITH_VALUES, BLOCK_THREADS,
                                                                 ITEMS_PER_THREAD, DType>::Type;
  // 排序用的共享内存temp storage
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  // row: 当前处理的行索引
  // tx: 线程索引
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  // 当前行输出索引的起始位置
  IdType* row_output = output_indices + static_cast<size_t>(row) * top_k;

  // keys: 排序用的索引数组
  // values: 对应的值数组（仅Plain模式使用）
  uint32_t keys[ITEMS_PER_THREAD];
  DType values[ITEMS_PER_THREAD];

  // 阶段1：加载数据到寄存器
  // 每个线程加载ITEMS_PER_THREAD个元素
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    uint32_t pos = tx * ITEMS_PER_THREAD + i;
    if (pos < top_k) {
      IdType idx = row_output[pos];
      // 将索引转换为无符号整数用于排序
      // 负值（无效）转换为全1（~0u），这样会排在最后
      keys[i] = (idx >= 0) ? static_cast<uint32_t>(idx) : ~0u;
      if constexpr (MODE == FilteredTopKMode::Plain) {
        // 加载对应的值
        values[i] = output_values[static_cast<size_t>(row) * top_k + pos];
      }
    } else {
      // 填充无效值
      keys[i] = ~0u;
      if constexpr (MODE == FilteredTopKMode::Plain) {
        values[i] = DType(0);
      }
    }
  }

  // 阶段2：执行Radix Sort排序
  // 计算排序的结束位：只需排序 max_len 位即可
  // end_bit = 32 - __clz(max_len) 表示需要保留的位数
  int end_bit = 32 - __clz(max_len);
  if constexpr (MODE == FilteredTopKMode::Plain) {
    // 同时排序keys和values
    BlockRadixSortT(temp_storage).Sort(keys, values, 0, end_bit);
  } else {
    // 只排序keys
    BlockRadixSortT(temp_storage).Sort(keys, 0, end_bit);
  }

  // 阶段3：根据模式执行索引转换
  // 准备页表条目指针或偏移量
  const IdType* src_page_entry = nullptr;
  IdType offset = 0;
  if constexpr (MODE == FilteredTopKMode::PageTable) {
    // PageTable模式：从row_to_batch获取batch索引，然后定位页表
    const uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row] : row;
    src_page_entry = aux_input + static_cast<int64_t>(batch_idx) * aux_stride;
  } else if constexpr (MODE == FilteredTopKMode::Ragged) {
    // Ragged模式：从aux_input获取当前行的偏移量
    offset = aux_input[row];
  }

  // 写回排序后的结果，同时执行模式转换
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    uint32_t pos = tx * ITEMS_PER_THREAD + i;
    if (pos < top_k) {
      uint32_t idx = keys[i];
      if constexpr (MODE == FilteredTopKMode::Plain) {
        // Plain模式：直接写回索引和值
        row_output[pos] = static_cast<IdType>(idx);
        output_values[static_cast<size_t>(row) * top_k + pos] = values[i];
      } else if constexpr (MODE == FilteredTopKMode::PageTable) {
        // PageTable模式：通过页表查表获取实际token ID
        // 无效索引（~0u）转换为-1
        row_output[pos] = (idx != ~0u) ? src_page_entry[idx] : static_cast<IdType>(-1);
      } else {  // Ragged (Ragged模式)
        // Ragged模式：添加偏移量得到最终索引
        row_output[pos] =
            (idx != ~0u) ? static_cast<IdType>(idx) + offset : static_cast<IdType>(-1);
      }
    }
  }
}

template <FilteredTopKMode MODE, typename DType, typename IdType>
cudaError_t LaunchSortTopKByIndex(IdType* output_indices, DType* output_values,
                                  const IdType* aux_input, int64_t aux_stride,
                                  const IdType* row_to_batch, uint32_t num_rows, uint32_t top_k_val,
                                  uint32_t max_len, cudaStream_t stream = 0) {
  // 局部排序变体最多覆盖 256 * 8 = 2048 个元素
  if (top_k_val > 2048) {
    return cudaErrorInvalidValue;
  }
  if constexpr (MODE == FilteredTopKMode::Plain) {
    if (top_k_val <= 1) {
      return cudaSuccess;
    }
  }
  if (top_k_val == 0) {
    return cudaSuccess;
  }

  dim3 grid(num_rows);
  void* args[] = {&output_indices, &output_values, &aux_input, &aux_stride,
                  &row_to_batch,   &top_k_val,     &max_len};
  auto launch_sort = [&](auto kernel, uint32_t threads) -> cudaError_t {
    dim3 block(threads);
    return cudaLaunchKernel((void*)kernel, grid, block, args, 0, stream);
  };

  cudaError_t status;
  if (top_k_val <= 128) {
    status = launch_sort(SortTopKByIndexKernel<MODE, 32, 4, DType, IdType>, 32);
  } else if (top_k_val <= 256) {
    status = launch_sort(SortTopKByIndexKernel<MODE, 32, 8, DType, IdType>, 32);
  } else if (top_k_val <= 512) {
    status = launch_sort(SortTopKByIndexKernel<MODE, 64, 8, DType, IdType>, 64);
  } else if (top_k_val <= 576) {
    status = launch_sort(SortTopKByIndexKernel<MODE, 64, 9, DType, IdType>, 64);
  } else if (top_k_val <= 1024) {
    status = launch_sort(SortTopKByIndexKernel<MODE, 128, 8, DType, IdType>, 128);
  } else {
    status = launch_sort(SortTopKByIndexKernel<MODE, 256, 8, DType, IdType>, 256);
  }
  return status;
}

/*!
 * \brief CUB stable radix sort: sorts top-k by value descending, carrying indices.
 *
 * Uses 32-bit flipped ordered value as key and 32-bit index as satellite data.
 * Since radix sort is stable, equal values preserve their prior relative order.
 * When preceded by an index sort, this yields (value desc, index asc) ordering.
 */
template <uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD, typename IdType, typename DType>
__global__ void __launch_bounds__(BLOCK_THREADS)
    StableSortTopKByValueKernel(IdType* output_indices, DType* output_values, uint32_t k,
                                uint32_t /*max_len*/) {
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;
  using BlockRadixSortT = cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, uint32_t>;
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  IdType* row_indices = output_indices + static_cast<size_t>(row) * k;
  DType* row_values = output_values + static_cast<size_t>(row) * k;

  uint32_t keys[ITEMS_PER_THREAD];
  uint32_t indices[ITEMS_PER_THREAD];

#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    uint32_t pos = tx * ITEMS_PER_THREAD + i;
    if (pos < k) {
      OrderedType ordered = Traits::ToOrdered(row_values[pos]);
      keys[i] = static_cast<uint32_t>(static_cast<OrderedType>(~ordered));
      indices[i] = static_cast<uint32_t>(row_indices[pos]);
    } else {
      keys[i] = ~0u;
      indices[i] = ~0u;
    }
  }

  constexpr int end_bit = sizeof(OrderedType) * 8;
  BlockRadixSortT(temp_storage).Sort(keys, indices, 0, end_bit);

#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    uint32_t pos = tx * ITEMS_PER_THREAD + i;
    if (pos < k) {
      row_indices[pos] = static_cast<IdType>(indices[i]);
      OrderedType ordered = static_cast<OrderedType>(~static_cast<OrderedType>(keys[i]));
      row_values[pos] = Traits::FromOrdered(ordered);
    }
  }
}

template <typename DType, typename IdType>
cudaError_t StableSortTopKByValue(IdType* output_indices, DType* output_values, uint32_t num_rows,
                                  uint32_t top_k_val, uint32_t max_len, cudaStream_t stream = 0) {
  // 局部排序变体最多覆盖 256 * 8 = 2048 个元素
  if (top_k_val > 2048) {
    return cudaErrorInvalidValue;
  }
  if (top_k_val <= 1) {
    return cudaSuccess;
  }

  dim3 grid(num_rows);
  void* args[] = {&output_indices, &output_values, &top_k_val, &max_len};
  auto launch_sort = [&](auto kernel, uint32_t threads) -> cudaError_t {
    dim3 block(threads);
    return cudaLaunchKernel((void*)kernel, grid, block, args, 0, stream);
  };

  cudaError_t status;
  if (top_k_val <= 128) {
    status = launch_sort(StableSortTopKByValueKernel<32, 4, IdType, DType>, 32);
  } else if (top_k_val <= 256) {
    status = launch_sort(StableSortTopKByValueKernel<32, 8, IdType, DType>, 32);
  } else if (top_k_val <= 512) {
    status = launch_sort(StableSortTopKByValueKernel<64, 8, IdType, DType>, 64);
  } else if (top_k_val <= 576) {
    status = launch_sort(StableSortTopKByValueKernel<64, 9, IdType, DType>, 64);
  } else if (top_k_val <= 1024) {
    status = launch_sort(StableSortTopKByValueKernel<128, 8, IdType, DType>, 128);
  } else {
    status = launch_sort(StableSortTopKByValueKernel<256, 8, IdType, DType>, 256);
  }
  return status;
}

template <FilteredTopKMode MODE, typename DType, typename IdType>
cudaError_t LaunchFilteredTopKUnified(DType* input, IdType* output, DType* aux_output,
                                      const IdType* aux_input, int64_t aux_stride,
                                      const IdType* row_to_batch, const IdType* lengths,
                                      uint32_t num_rows, uint32_t top_k_val, uint32_t max_len,
                                      bool deterministic = false, cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,        &output,  &aux_output, &aux_input, &aux_stride,
                  &row_to_batch, &lengths, &num_rows,   &top_k_val, &max_len};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                                                      \
  if (vec_size == VS) {                                                                            \
    if (!deterministic) {                                                                          \
      auto kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, false, MODE>;                     \
      FLASHINFER_CUDA_CALL(                                                                        \
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));   \
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, smem_size, stream)); \
    } else {                                                                                       \
      auto kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, true, MODE>;                      \
      FLASHINFER_CUDA_CALL(                                                                        \
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));   \
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, smem_size, stream)); \
    }                                                                                              \
    return cudaSuccess;                                                                            \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

// 使用统一内核的 VEC_SIZE 和 BLOCK_THREADS 分发启动函数
template <typename DType, typename IdType>
cudaError_t FilteredTopKPageTableTransform(DType* input, IdType* output_page_table,
                                           const IdType* src_page_table, int64_t src_stride,
                                           const IdType* row_to_batch, IdType* lengths,
                                           uint32_t num_rows, uint32_t top_k_val, uint32_t max_len,
                                           bool deterministic = false, cudaStream_t stream = 0) {
  DType* aux_output = nullptr;  // PageTable模式不使用
  return LaunchFilteredTopKUnified<FilteredTopKMode::PageTable, DType, IdType>(
      input, output_page_table, aux_output, src_page_table, src_stride, row_to_batch, lengths,
      num_rows, top_k_val, max_len, deterministic, stream);
}

template <typename DType, typename IdType>
cudaError_t FilteredTopKRaggedTransform(DType* input, IdType* output_indices, const IdType* offsets,
                                        IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len, bool deterministic = false,
                                        cudaStream_t stream = 0) {
  DType* aux_output = nullptr;           // Ragged模式不使用
  int64_t aux_stride = 0;                // Ragged模式不使用
  const IdType* row_to_batch = nullptr;  // Ragged模式不使用
  return LaunchFilteredTopKUnified<FilteredTopKMode::Ragged, DType, IdType>(
      input, output_indices, aux_output, offsets, aux_stride, row_to_batch, lengths, num_rows,
      top_k_val, max_len, deterministic, stream);
}

template <typename DType, typename IdType>
cudaError_t FilteredTopK(DType* input, IdType* output_indices, DType* output_values,
                         const IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                         uint32_t max_len, bool deterministic = false, cudaStream_t stream = 0) {
  const IdType* aux_input = nullptr;     // Plain模式不使用
  int64_t aux_stride = 0;                // Plain模式不使用
  const IdType* row_to_batch = nullptr;  // Plain模式不使用
  return LaunchFilteredTopKUnified<FilteredTopKMode::Plain, DType, IdType>(
      input, output_indices, output_values, aux_input, aux_stride, row_to_batch, lengths, num_rows,
      top_k_val, max_len, deterministic, stream);
}

/*!
 * \brief Check if the GPU supports enough shared memory for FilteredTopK algorithm.
 *
 * FilteredTopK requires 128KB dynamic shared memory. This function checks if the
 * current GPU's max shared memory per SM is sufficient.
 *
 * \return true if GPU supports FilteredTopK, false otherwise
 */
inline bool CanImplementFilteredTopK() {
  int device_id;
  if (cudaGetDevice(&device_id) != cudaSuccess) return false;
  int max_smem_per_sm;
  if (cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                             device_id) != cudaSuccess) {
    return false;
  }
  return static_cast<size_t>(max_smem_per_sm) >= FILTERED_TOPK_SMEM_DYNAMIC;
}

// 用于基准测试的算法覆盖（由 FLASHINFER_TOPK_ALGO 环境变量控制）
enum class TopKAlgoOverride { AUTO, FILTERED, MULTI_CTA };

inline TopKAlgoOverride GetTopKAlgoOverride() {
  const char* env = std::getenv("FLASHINFER_TOPK_ALGO");
  if (env == nullptr) return TopKAlgoOverride::AUTO;
  if (std::strcmp(env, "filtered") == 0) return TopKAlgoOverride::FILTERED;
  if (std::strcmp(env, "multi_cta") == 0) return TopKAlgoOverride::MULTI_CTA;
  return TopKAlgoOverride::AUTO;
}

/*!
 * \brief Unified heuristic to decide whether to use FilteredTopK or Multi-CTA RadixTopK.
 *
 * \tparam DType Data type (affects threshold due to memory bandwidth considerations)
 * \param num_rows Number of rows (batch size)
 * \param top_k_val Number of top elements to select
 * \param max_len Maximum sequence length
 * \param deterministic Whether deterministic top-k path is requested
 * \return true if FilteredTopK should be used, false for Multi-CTA RadixTopK
 */
template <typename DType>
inline bool ShouldUseFilteredTopK(uint32_t num_rows, uint32_t top_k_val, uint32_t max_len,
                                  bool deterministic) {
  // 检查GPU是否支持足够的共享内存用于FilteredTopK
  const bool gpu_supports_filtered = CanImplementFilteredTopK();
  const bool k_fits_filtered = (top_k_val <= FILTERED_TOPK_MAX_K) && (max_len > top_k_val);

  if (!gpu_supports_filtered || !k_fits_filtered) {
    return false;
  }

  // 检查算法覆盖
  const TopKAlgoOverride algo_override = GetTopKAlgoOverride();
  if (algo_override == TopKAlgoOverride::FILTERED) return true;
  if (algo_override == TopKAlgoOverride::MULTI_CTA) return false;

  // 16位类型：更简单的阈值
  // 32位类型：更细致的启发式方法
  if (deterministic) {
    if constexpr (sizeof(DType) <= 2) {
      return num_rows > (max_len / 256);
    } else {
      if (max_len <= 16384) {
        return true;
      } else {
        const uint32_t batch_threshold = std::min(64u, std::max(16u, max_len / 4096));
        return num_rows >= batch_threshold;
      }
    }
  }

  if constexpr (sizeof(DType) <= 2) {
    return (max_len <= 16384);
  } else {
    if (max_len <= 32768) {
      return true;
    } else {
      const uint32_t batch_threshold = max_len / 16384;
      return (num_rows > batch_threshold);
    }
  }
}

// 使用启发式的分发函数
template <typename DType, typename IdType>
cudaError_t TopKPageTableTransformDispatch(DType* input, IdType* output_page_table,
                                           const IdType* src_page_table, int64_t src_stride,
                                           const IdType* row_to_batch, IdType* lengths,
                                           uint32_t num_rows, uint32_t top_k_val, uint32_t max_len,
                                           RadixRowState* row_states_buffer, bool deterministic,
                                           cudaStream_t stream = 0) {
  if (ShouldUseFilteredTopK<DType>(num_rows, top_k_val, max_len, deterministic)) {
    FLASHINFER_CUDA_CALL((FilteredTopKPageTableTransform<DType, IdType>(
        input, output_page_table, src_page_table, src_stride, row_to_batch, lengths, num_rows,
        top_k_val, max_len, deterministic, stream)));
    if (deterministic) {
      FLASHINFER_CUDA_CALL((LaunchSortTopKByIndex<FilteredTopKMode::PageTable, uint8_t, IdType>(
          output_page_table, static_cast<uint8_t*>(nullptr), src_page_table, src_stride,
          row_to_batch, num_rows, top_k_val, max_len, stream)));
    }
    return cudaSuccess;
  }
  return RadixTopKPageTableTransformMultiCTA<DType, IdType>(
      input, output_page_table, src_page_table, src_stride, row_to_batch, lengths, num_rows,
      top_k_val, max_len, row_states_buffer, deterministic, stream);
}

template <typename DType, typename IdType>
cudaError_t TopKRaggedTransformDispatch(DType* input, IdType* output_indices, const IdType* offsets,
                                        IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len, RadixRowState* row_states_buffer,
                                        bool deterministic, cudaStream_t stream = 0) {
  if (ShouldUseFilteredTopK<DType>(num_rows, top_k_val, max_len, deterministic)) {
    FLASHINFER_CUDA_CALL((FilteredTopKRaggedTransform<DType, IdType>(
        input, output_indices, offsets, lengths, num_rows, top_k_val, max_len, deterministic,
        stream)));
    if (deterministic) {
      FLASHINFER_CUDA_CALL((LaunchSortTopKByIndex<FilteredTopKMode::Ragged, uint8_t, IdType>(
          output_indices, static_cast<uint8_t*>(nullptr), offsets, 0, nullptr, num_rows, top_k_val,
          max_len, stream)));
    }
    return cudaSuccess;
  }
  return RadixTopKRaggedTransformMultiCTA<DType, IdType>(input, output_indices, offsets, lengths,
                                                         num_rows, top_k_val, max_len,
                                                         row_states_buffer, deterministic, stream);
}

template <typename DType, typename IdType>
cudaError_t TopKDispatch(DType* input, IdType* output_indices, DType* output_values,
                         uint32_t num_rows, uint32_t top_k_val, uint32_t max_len,
                         RadixRowState* row_states_buffer, bool sorted_output = false,
                         bool deterministic = false, cudaStream_t stream = 0) {
  if (ShouldUseFilteredTopK<DType>(num_rows, top_k_val, max_len, deterministic)) {
    FLASHINFER_CUDA_CALL(
        (FilteredTopK<DType, IdType>(input, output_indices, output_values, nullptr, num_rows,
                                     top_k_val, max_len, deterministic, stream)));
    if (deterministic) {
      FLASHINFER_CUDA_CALL((LaunchSortTopKByIndex<FilteredTopKMode::Plain, DType, IdType>(
          output_indices, output_values, nullptr, 0, nullptr, num_rows, top_k_val, max_len,
          stream)));
    }
  } else {
    FLASHINFER_CUDA_CALL((RadixTopKMultiCTA<DType, IdType>(
        input, output_indices, output_values, nullptr, num_rows, top_k_val, max_len,
        row_states_buffer, deterministic, stream)));
  }
  if (sorted_output) {
    FLASHINFER_CUDA_CALL((StableSortTopKByValue<DType, IdType>(
        output_indices, output_values, num_rows, top_k_val, max_len, stream)));
  }
  return cudaSuccess;
}













}
} // namespace flashinfer::sampling

#endif
