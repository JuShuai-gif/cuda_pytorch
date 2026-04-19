/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_TOPK_CUH_
#define FLASHINFER_TOPK_CUH_
// 这个文件实现了 FlashInfer 中与 Top-K 采样相关的 CUDA 核心逻辑，主要包括：
// 1. 基于 radix select 的 Multi-CTA Top-K
// 2. Top-K 后的 logits mask / 概率重归一化
// 3. PageTable / Ragged 两类索引变换
// 4. 另一套 FilteredTopK 实现

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <numeric>
#include <type_traits>

#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace sampling {

template <uint32_t BLOCK_THREADS>
// 根据 block 线程数和固定共享内存开销，估算还能给 ordered cache 留多少共享内存。
// 这个结果会直接影响每个 CTA 一次能处理多少元素（chunk_size）。
inline size_t GetRadixTopKAvailableOrderedSmemBytes(size_t max_smem_per_block,
                                                    size_t fixed_smem_aligned,
                                                    bool reserve_launch_headroom) {
  // CUB 的 BlockScan 会在编译期决定一块临时共享内存 TempStorage。
  // 确定性收集路径里会复用/额外实例化这类 scratch，因此这里先估算它的体积。
  using RadixTopKDetBlockScanT =
      cub::BlockScan<uint32_t, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
  constexpr size_t RADIX_TOPK_DETERMINISTIC_BLOCK_SCAN_SMEM =
      sizeof(typename RadixTopKDetBlockScanT::TempStorage);
  // 这里保守地预留两份 BlockScan scratch 的空间，
  // 防止 kernel launch 时动态共享内存把静态共享内存“挤爆”。
  constexpr size_t RADIX_TOPK_LAUNCH_SMEM_HEADROOM = 2 * RADIX_TOPK_DETERMINISTIC_BLOCK_SCAN_SMEM;
  const size_t launch_headroom =
      reserve_launch_headroom ? RADIX_TOPK_LAUNCH_SMEM_HEADROOM : size_t(0);
  // 如果“固定开销 + 预留空间”已经把整块 shared memory 用完了，就没有空间存 ordered cache 了。
  if (max_smem_per_block <= fixed_smem_aligned + launch_headroom) {
    return 0;
  }
  // 为确定性 radix kernel 预留足够的启动期余量，
  // 因为这类 kernel 可能会额外实例化静态共享内存 scratch，例如 BlockScan 的 TempStorage。
  return max_smem_per_block - fixed_smem_aligned - launch_headroom;
}

// ============================================================================
// RadixTopK 的类型 traits：
// 1. 支持 float、half、bfloat16 三种输入类型
// 2. 为每种类型定义对应的 OrderedType
// 3. 提供“原始浮点值 <-> ordered 编码”的双向转换
// 4. 提供负无穷表示，便于 mask / 初始化 / 填充值场景使用
// ============================================================================
// 这里的 traits 用来把浮点值映射成“可按整数比较顺序做 radix 排序/选择”的 ordered 表示。
// 核心思想是：把浮点数的 bit 模式改造成“数值越大，ordered 值也越大”。
/*
也就是告诉这套 TopK 算法：
- float 对应的整数表示类型是什么
- 需要做多少轮 radix 选择
- 怎么把 float 转成适合按整数比较的编码
- 怎么再转回来
- -inf 怎么表示
*/
template <typename DType>
struct RadixTopKTraits;

// float 的特化版本（32 位）
// 给 float 类型定义一套专门供 Radix TopK 使用的“类型适配规则”。
// 本质上就是一个 float 版本的 traits。
template <>
struct RadixTopKTraits<float> {
  /*
  float 在这套 radix topk 里，最终要转成 uint32_t 来处理。

  因为 float 本身不适合直接拿位模式做无符号整数排序，但转成某种“有序整数编码”后，就可以按整数分桶、做 radix select
  */
  using OrderedType = uint32_t;

  // 返回总共要做多少轮 radix 选择。
  /*
  假如每次处理8bit,那就要做 32/8 = 4轮
  也就是会依次看：
  - 高 8 位
  - 次高 8 位
  - 次低 8 位
  - 低 8 位
  */
  template <uint32_t RADIX_BITS>
  static __host__ __device__ constexpr uint32_t num_rounds() {
    // 每轮处理 RADIX_BITS 个 bit，因此总轮数 = 总位宽 / 每轮位宽。
    return sizeof(OrderedType) * 8 / RADIX_BITS;
  }

  /*
  该函数作用：
  把 float 的原始 bit 模式，变成一个“按无符号整数比较时，顺序和 float 大小一致”的编码。

  为什么要这样做？
  因为 radix select 擅长处理整数 bit，而不是直接处理浮点数大小关系。

  普通 float bit 不能直接比较
  IEEE754 的 float 位布局是：

  - 1 位符号位
  - 8 位指数
  - 23 位尾数
  它的 bit 模式直接当作 uint32_t 比较时，负数区域的顺序会乱掉，不等于真实数值大小规律。

  所以要重新编码。

  经过 ToOrdered 后，会得到这样一种性质：
  如果 a > b,那么就有 ToOrdered(a) > ToOrdered(b)

  于是后面的 TopK 就可以不再按浮点数比较，而是直接：

  - 看 bit
  - 按字节分桶
  - 做 radix select
  这对 GPU 很友好。
  */
  __device__ __forceinline__ static OrderedType ToOrdered(float val) {
    uint32_t bits = __float_as_uint(val);
    // 这一步把 IEEE754 的 float bit 模式映射成“可直接做无符号整数比较”的顺序编码：
    // 1. 正数：翻转符号位，让所有正数落在有序空间的高半区
    // 2. 负数：按位取反，让“数值更大（更接近 0）”的负数 ordered 也更大
    // 这样后面做 radix select 时就不需要真的比较浮点数，只比较整数位模式即可。
    return (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000);
  }

  // ToOrdered 的逆操作。
  __device__ __forceinline__ static float FromOrdered(OrderedType ordered) {
    // ToOrdered 的逆变换：把 ordered 整数编码恢复回原始 float bit 模式。
    uint32_t bits = (ordered & 0x80000000) ? (ordered ^ 0x80000000) : ~ordered;
    return __uint_as_float(bits);
  }

  /*
  返回负无穷。

  这个值常用于：

  - 初始化最小值
  - mask logits
  - 表示“不可能被选中”的位置
  在 TopK / sampling 里很常见
  */
  __device__ __forceinline__ static float NegInf() {
    return -cuda::std::numeric_limits<float>::infinity();
  }
};

// half 的特化版本（16 位）
template <>
struct RadixTopKTraits<half> {
  using OrderedType = uint16_t;

  template <uint32_t RADIX_BITS>
  static __host__ __device__ constexpr uint32_t num_rounds() {
    return sizeof(OrderedType) * 8 / RADIX_BITS;
  }

  __device__ __forceinline__ static OrderedType ToOrdered(half val) {
    uint16_t bits = __half_as_ushort(val);
    // fp16/bf16 的处理思路和 float 相同，只是位宽从 32 bit 变成 16 bit。
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
  }

  __device__ __forceinline__ static half FromOrdered(OrderedType ordered) {
    // 恢复原始 fp16 位模式。
    uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                       : static_cast<uint16_t>(~ordered);
    return __ushort_as_half(bits);
  }

  __device__ __forceinline__ static half NegInf() {
    return __ushort_as_half(static_cast<uint16_t>(0xFC00));  // -inf in fp16
  }
};

// nv_bfloat16 的特化版本（16 位）
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
    uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                       : static_cast<uint16_t>(~ordered);
    return __ushort_as_bfloat16(bits);
  }

  __device__ __forceinline__ static nv_bfloat16 NegInf() {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80));  // -inf in bf16
  }
};
// ==================== Multi-CTA Top-K Implementation ====================

// 用于跨 CTA 同步的 acquire/release 原语
// 下面这些函数是跨 CTA 同步的底层原语。
// 因为一个 row 可能被多个 CTA 分块处理，所以需要软件 barrier 和发布/获取语义。
__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;

#if (__CUDA_ARCH__ >= 700)
  // SM70 及以上架构支持显式内存一致性限定符。
  // 这里使用 acquire 语义的 load。
  // acquire load 保证：看到这个计数器更新之后，也能看到更新之前发布出去的数据。
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#else
  // 老架构退化成 cache-global 的 load，配合同步原语一起使用。
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif

  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  // SM70 及以上架构支持显式内存一致性限定符。
  // 这里使用“fence + relaxed reduction”组合出 release 语义。
  // 其中 fence 还能保证在最近一次 syncthreads 之前由其他线程弱写入的数据也被正确发布。
  // 先 fence，再做 reduction atomic add，表示“我这一 CTA 在 barrier 前的写已经可见”。
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  // SM70 及以上架构支持显式内存一致性限定符。
  // 这里使用“fence + release store”实现发布语义。
  // 常用于把计数器/状态位重置为某个值，并保证之前写入的数据先于该状态对外可见。
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

// 使用 acquire 语义等待 *ptr 达到 target_val。
// 只有线程 0 负责自旋，随后再同步整个 CTA。
__device__ __forceinline__ void wait_ge(int* ptr, int target_val, int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    // 只让 thread 0 自旋，避免整个 CTA 的所有线程都去忙等浪费资源。
    while (ld_acquire(ptr) < target_val) {
    }
  }
  // 等 thread 0 观察到条件成立之后，再把结果同步给整个 CTA。
  __syncthreads();
}

// ==================== 多 CTA 的 Radix Top-K / Mask Logits 公共状态 ====================

// 多 CTA radix 归约的全局状态（每个 group 一份）
// 每个 group（即一行数据对应的一组 CTA）共享一个状态结构。
// 里面保存了 histogram、剩余 k、prefix 以及跨 CTA 协调用的计数器。
struct RadixRowState {
  uint32_t histogram[3][256];  // 三重缓冲 histogram，用于做到“每轮只需一次 barrier”
  uint32_t remaining_k;        // 当前轮之后还剩多少个名次需要继续找
  uint32_t prefix;             // 已经确定好的高位前缀（第 k 大元素的高位）
  int arrival_counter;         // 跨 CTA 软件 barrier 使用的到达计数器
  int output_counter;          // RadixTopK 收集输出时使用的全局位置计数器
  float sum_topk;              // RenormProb 路径中 top-k 元素总和
};

constexpr uint32_t RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP = 256;

// 确定性收集模式下的额外 scratch：
// 用于记录每个 CTA 的 >pivot / ==pivot 数量，从而按固定顺序分配输出区间。
struct RadixDeterministicCollectScratch {
  uint32_t gt_count[RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP];
  uint32_t eq_count[RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP];
};

inline RadixDeterministicCollectScratch* MaybeGetRadixDeterministicCollectScratchBuffer(
    RadixRowState* row_states_buffer, uint32_t num_groups, bool single_cta, bool deterministic) {
  return (single_cta || !deterministic || row_states_buffer == nullptr)
             ? nullptr
             : reinterpret_cast<RadixDeterministicCollectScratch*>(row_states_buffer + num_groups);
}

// ==================== Radix Top-K 共用的设备端辅助函数 ====================
/*!
 * \brief 在同一个 radix group 内实现软件 barrier。
 *
 * 每个 CTA 只由 tx==0 的线程上报一次“已到达”，随后所有 CTA 一起等待，
 * 直到 group 内的到达计数器达到当前阶段所需的目标值。
 *
 * \param state 当前 group 对应的共享状态，内部包含 arrival_counter
 * \param barrier_phase 当前软件 barrier 的阶段编号
 * \param ctas_per_group 参与本组 barrier 的 CTA 数量
 * \param tx 当前线程在线程块内的索引
 */
__device__ __forceinline__ void AdvanceRadixGroupBarrier(RadixRowState* state, int& barrier_phase,
                                                         uint32_t ctas_per_group, uint32_t tx) {
  if (tx == 0) {
    // 每个 CTA 只由 thread 0 上报一次“到达 barrier”。
    red_release(&state->arrival_counter, 1);
  }
  // 第 barrier_phase 轮需要等到一整组 ctas_per_group 个 CTA 都到齐。
  int target = (barrier_phase + 1) * ctas_per_group;
  wait_ge(&state->arrival_counter, target, tx);
  // phase 自增后，后面的 barrier 就会等待下一个目标值。
  barrier_phase++;
  __syncthreads();
}

/*!
 * \brief 以确定性的线程步长顺序收集命中元素，并通过完整的 CTA scan 计算输出位置。
 *
 * 每个线程都会按固定顺序 `tx, tx + BLOCK_THREADS, ...` 遍历元素，
 * 先统计自己负责的整条步长链上的命中数，再在整个 CTA 范围内做 exclusive scan，
 * 最后按照同样的固定顺序把命中元素写出。
 *
 * \tparam BLOCK_THREADS CTA 中的线程数量
 * \param tx 当前线程在线程块内的索引
 * \param length 需要扫描的元素数量
 * \param scan_temp_storage 调用者复用的 CUB BlockScan 临时存储
 * \param is_selected 判断某个线程步长位置是否命中的谓词
 * \param emit_limit 最多允许写出的命中元素个数
 * \param emit_selected 写出回调，调用形式为 emit_selected(index, local_pos)
 */
template <uint32_t BLOCK_THREADS, typename TempStorage, typename Predicate, typename EmitFn>
// 以“线程步长顺序”稳定收集满足条件的元素。
// 常用于 deterministic 模式，保证不同线程/CTA 组合下输出顺序可复现。
__device__ __forceinline__ void DeterministicThreadStridedCollect(uint32_t tx, uint32_t length,
                                                                  TempStorage& scan_temp_storage,
                                                                  Predicate is_selected,
                                                                  uint32_t emit_limit,
                                                                  EmitFn emit_selected) {
  using BlockScan = cub::BlockScan<uint32_t, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>;

  uint32_t thread_local_selected_count = 0;
  // 第一次遍历：每个线程统计“自己负责的步长链”里一共命中了多少个元素。
  for (uint32_t i = tx; i < length; i += BLOCK_THREADS) {
    thread_local_selected_count += static_cast<uint32_t>(is_selected(i));
  }

  uint32_t thread_local_selected_prefix = 0;
  // 对每个线程的命中数做 CTA 级 exclusive scan。
  // 得到的是“当前线程负责输出的元素在整个 CTA 输出序列中的起始偏移”。
  BlockScan(scan_temp_storage)
      .ExclusiveSum(thread_local_selected_count, thread_local_selected_prefix);

  if (thread_local_selected_count > 0 && thread_local_selected_prefix < emit_limit) {
    uint32_t thread_local_emit_pos = thread_local_selected_prefix;
    const uint32_t thread_local_emit_end =
        min(thread_local_selected_prefix + thread_local_selected_count, emit_limit);
    for (uint32_t i = tx; i < length; i += BLOCK_THREADS) {
      if (is_selected(i)) {
        // 第二次遍历：按同样的线程步长顺序真正写出结果。
        // 因为线程访问顺序固定，所以 deterministic 模式下输出顺序稳定。
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
 * \brief 在共享内存中并行计算后缀和。
 *
 * 调用结束后，suffix_sum[i] 表示“bucket i 及以上一共有多少个元素”。
 * 也就是把 histogram 中 bucket i 到 255 的计数全部累加起来。
 *
 * \param suffix_sum 大小为 RADIX（256）的共享内存数组
 * \param tx 当前线程在线程块内的索引
 */
template <uint32_t BLOCK_THREADS>
// 对 histogram 做后缀和，得到“bucket i 及以上一共有多少元素”。
// 这样就可以判断第 k 大元素落在哪个 bucket。
__device__ __forceinline__ void RadixSuffixSum(uint32_t* suffix_sum, uint32_t tx) {
  constexpr uint32_t RADIX = 256;
  // 并行计算后缀和，得到“每个 bucket 及以上的元素总数”。
  for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
    uint32_t val = 0;
    if (tx < RADIX) {
      // 这里不是传统的 prefix sum，而是“从当前 bucket 往更大 bucket 聚合”的 suffix sum。
      val = suffix_sum[tx];
      if (tx + stride < RADIX) {
        val += suffix_sum[tx + stride];
      }
    }
    __syncthreads();
    if (tx < RADIX) {
      suffix_sum[tx] = val;
    }
    __syncthreads();
  }
}

/*!
 * \brief 找到包含第 k 大元素的阈值 bucket。
 *
 * 满足条件的阈值 bucket 需要同时满足：
 * 1. count_ge >= k
 * 2. count_gt < k
 * 其中 count_ge = suffix_sum[bucket]，
 *      count_gt = suffix_sum[bucket + 1]。
 *
 * \param suffix_sum 保存后缀和的共享内存数组
 * \param remaining_k 当前还需要在候选集合里找第几名
 * \param found_bucket 输出：找到的阈值 bucket
 * \param found_remaining_k 输出：扣除所有“严格大于阈值”的元素后，下一轮还需找的名次
 * \param tx 当前线程在线程块内的索引
 */
__device__ __forceinline__ void RadixFindThresholdBucket(uint32_t* suffix_sum, uint32_t remaining_k,
                                                         uint32_t* found_bucket,
                                                         uint32_t* found_remaining_k, uint32_t tx) {
  constexpr uint32_t RADIX = 256;
  // 初始化，只让线程 0 负责。
  if (tx == 0) {
    *found_bucket = 0;
    *found_remaining_k = remaining_k;
  }
  __syncthreads();

  // 前 256 个线程并行检查自己对应的 bucket。
  if (tx < RADIX) {
    uint32_t count_ge = suffix_sum[tx];
    uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
    // 条件解释：
    // 1. count_ge >= remaining_k：说明第 remaining_k 大元素至少不小于当前 bucket
    // 2. count_gt < remaining_k：说明严格大于当前 bucket 的元素还不够多
    // 两个条件同时满足时，说明“第 remaining_k 大”恰好落在当前 bucket。
    if (count_ge >= remaining_k && count_gt < remaining_k) {
      *found_bucket = tx;
      // 进入下一轮时，只需要在当前 bucket 内继续找第几名即可，
      // 所以要扣掉所有“严格大于当前 bucket”的元素数量。
      *found_remaining_k = remaining_k - count_gt;
    }
  }
  __syncthreads();
}

/*!
 * \brief 为 radix select 的某一轮构建本地 histogram。
 *
 * 只统计那些“高位前缀已经匹配当前 prefix”的元素，
 * 再根据当前轮负责的那一个字节，把它们分配到 0~255 的 bucket 中。
 *
 * \tparam OrderedType ordered 整数类型（uint16_t 或 uint32_t）
 * \param shared_ordered 存放 ordered 值的共享内存缓存
 * \param actual_chunk_size 当前 CTA 负责的 chunk 实际元素数
 * \param local_histogram 输出用的本地 histogram（共享内存）
 * \param prefix 当前已经确定好的高位前缀
 * \param shift 为了取出当前轮目标字节所需要右移的位数
 * \param round 当前处于第几轮（0 到 NUM_ROUNDS-1）
 * \param tx 当前线程在线程块内的索引
 */
template <uint32_t BLOCK_THREADS, typename OrderedType>
// 当前 round 的局部 histogram 构建：
// 只统计“高位前缀已经匹配 prefix”的元素，然后按当前 8 bit 落桶。
__device__ __forceinline__ void RadixBuildLocalHistogram(const OrderedType* shared_ordered,
                                                         uint32_t actual_chunk_size,
                                                         uint32_t* local_histogram, uint32_t prefix,
                                                         uint32_t shift, uint32_t round,
                                                         uint32_t tx) {
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t RADIX_BITS = 8;

  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered = shared_ordered[i];

    // 检查当前元素是否匹配已经确定好的高位前缀。
    // round=0 时还没有任何高位前缀，因此 mask=0，等价于所有元素都参与第一轮统计。
    OrderedType mask =
        (round == 0)
            ? OrderedType(0)
            : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
    if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
      // 只看当前轮负责的 8 个 bit，把元素丢进 0~255 的某个 bucket。
      uint32_t bucket = (ordered >> shift) & 0xFF;
      atomicAdd(&local_histogram[bucket], 1);
    }
  }
}

/*!
 * \brief 执行一轮 radix select，可选地包含多 CTA 同步与聚合。
 *
 * 这是所有 TopK kernel 共用的 radix select 单轮核心逻辑。
 * 它会构建 histogram，在多 CTA 模式下聚合所有 CTA 的统计结果，
 * 然后计算后缀和并找出本轮命中的阈值 bucket。
 *
 * \tparam BLOCK_THREADS 每个线程块的线程数
 * \tparam SINGLE_CTA 若为 true，表示单 CTA 模式，不需要跨 CTA 同步
 * \tparam OrderedType ordered 整数类型
 *
 * \param shared_ordered 保存 ordered 值的共享内存缓存
 * \param actual_chunk_size 当前 CTA 负责的 chunk 实际元素数
 * \param local_histogram 本地 histogram 的共享内存空间（大小为 RADIX）
 * \param suffix_sum 计算后缀和所用的共享内存空间（大小为 RADIX）
 * \param state 多 CTA 模式下指向 RadixRowState 的指针，单 CTA 时为 nullptr
 * \param prefix 当前已经确定的高位前缀
 * \param remaining_k 当前剩余要找的名次
 * \param round 当前轮数（0 到 NUM_ROUNDS-1）
 * \param barrier_phase 软件 barrier 的阶段计数器引用
 * \param ctas_per_group 每个 group 里包含的 CTA 数量
 * \param tx 当前线程在线程块内的索引
 * \param out_new_prefix 输出：本轮更新后的 prefix
 * \param out_new_remaining_k 输出：本轮更新后的 remaining_k
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType>
// radix select 的单轮核心逻辑：
// 1. 构局部 histogram
// 2. 多 CTA 时聚合到全局 histogram
// 3. 做 suffix sum
// 4. 找到本轮命中的 bucket，更新 prefix 和 remaining_k
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
  // shift 表示“当前轮要看哪一段 8 bit”。
  // 例如 32-bit 数据会依次看 bit[31:24]、[23:16]、[15:8]、[7:0]。

  // For multi-CTA: pointers to global histograms (triple buffer)
  uint32_t* current_hist = nullptr;
  uint32_t* next_hist = nullptr;
  if constexpr (!SINGLE_CTA) {
    current_hist = state->histogram[global_round % 3];
    next_hist = state->histogram[(global_round + 1) % 3];
  }

  // Clear local histogram only
  for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
    local_histogram[i] = 0;
  }
  __syncthreads();

  // Build local histogram from shared memory
  RadixBuildLocalHistogram<BLOCK_THREADS, OrderedType>(shared_ordered, actual_chunk_size,
                                                       local_histogram, prefix, shift, round, tx);
  __syncthreads();

  // For multi-CTA: write -> (leading CTA clears next) -> barrier -> read
  // For single-CTA: local_histogram is already the complete histogram
  if constexpr (!SINGLE_CTA) {
    // 多 CTA 时，每个 CTA 先把自己的局部 histogram 原子累加到 group 共享 histogram。
    // Accumulate local histogram to global
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    // Only leading CTA clears next round's histogram BEFORE barrier
    // triple buffer 的目的是让“本轮读取 current_hist”和“下一轮写入 next_hist”不互相踩内存。
    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        next_hist[i] = 0;
      }
    }

    // Barrier: wait for all CTAs to finish atomicAdd and clearing
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

    // Read current histogram (after barrier, all atomicAdds are complete)
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      suffix_sum[i] = current_hist[i];
    }
  } else {
    // Single-CTA: copy local histogram directly to suffix_sum
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      suffix_sum[i] = local_histogram[i];
    }
  }
  __syncthreads();

  // Compute suffix sum
  RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);

  // Find threshold bucket using shared_scalars for found_bucket and found_remaining_k
  // shared_scalars[0] = found_bucket, shared_scalars[1] = found_remaining_k
  RadixFindThresholdBucket(suffix_sum, remaining_k, &shared_scalars[0], &shared_scalars[1], tx);

  // Output new prefix and remaining_k
  // 把当前轮找到的 bucket 拼接到 prefix 后面，
  // 表示“第 k 大元素的高位前缀目前已经确定到了这里”。
  *out_new_prefix = prefix | (shared_scalars[0] << shift);
  *out_new_remaining_k = shared_scalars[1];
}

/*!
 * \brief 从全局内存加载数据到共享内存，并转换成 ordered 表示。
 *
 * 这是所有 TopK kernel 共用的第一阶段：
 * 先尽量使用向量化访存把数据搬到共享内存，再把每个元素转成适合 radix select 的 ordered 编码。
 *
 * \tparam BLOCK_THREADS 每个线程块的线程数
 * \tparam VEC_SIZE 每次向量化访存处理的元素个数
 * \tparam DType 输入数据类型（float、half、nv_bfloat16）
 * \tparam Traits DType 对应的 traits 类型
 *
 * \param input 指向当前行起始地址的输入指针（外部已经完成按行偏移）
 * \param shared_ordered 保存 ordered 值的共享内存缓存
 * \param chunk_start 当前 CTA 在本行中负责的起始位置
 * \param actual_chunk_size 当前 CTA 负责的 chunk 实际元素数
 * \param tx 当前线程在线程块内的索引
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, typename Traits>
// 将全局内存中的一个 chunk 搬到 shared memory，并顺手转成 ordered 表示。
// 后面的 radix select 全部在 shared memory 上进行。
__device__ __forceinline__ void LoadToSharedOrdered(const DType* input,
                                                    typename Traits::OrderedType* shared_ordered,
                                                    uint32_t chunk_start,
                                                    uint32_t actual_chunk_size, uint32_t tx) {
  using OrderedType = typename Traits::OrderedType;
  vec_t<DType, VEC_SIZE> input_vec;
  // 先处理能被 VEC_SIZE 整除的前半段，尽量走向量化 load/store。
  const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
  for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
    input_vec.cast_load(input + chunk_start + i);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      // shared_ordered 里保存的不是原始值，而是适合整数比较/分桶的 ordered 编码。
      shared_ordered[i + j] = Traits::ToOrdered(input_vec[j]);
    }
  }
  // 处理无法被 VEC_SIZE 整除的尾部元素。
  for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    shared_ordered[i] = Traits::ToOrdered(input[chunk_start + i]);
  }
  __syncthreads();
}

/*!
 * \brief 基于已经加载到共享内存的数据，使用 radix select 找到第 k 大元素。
 *
 * 这个函数假设 shared_ordered 里已经放好了 ordered 编码。
 * 它会执行完整的 radix select 流程（初始 barrier + 多轮细化），
 * 并返回最终得到的 ordered pivot。
 *
 * \tparam BLOCK_THREADS 每个线程块的线程数
 * \tparam SINGLE_CTA 若为 true，表示单 CTA 模式
 * \tparam OrderedType ordered 整数类型
 *
 * \param shared_ordered 已经预加载好的 ordered 值共享内存
 * \param actual_chunk_size 当前 CTA 负责的 chunk 实际元素数
 * \param k 目标 top-k 中的 k
 * \param local_histogram 本地 histogram 使用的共享内存（大小为 RADIX）
 * \param suffix_sum 后缀和使用的共享内存（大小为 RADIX）
 * \param shared_scalars 若干标量缓存使用的共享内存
 * \param state 多 CTA 模式下使用的 RadixRowState 指针，单 CTA 时为 nullptr
 * \param barrier_phase 软件 barrier 阶段计数器引用
 * \param ctas_per_group 每个 group 里的 CTA 数量
 * \param cta_in_group 当前 CTA 在 group 内的编号
 * \param tx 当前线程在线程块内的索引
 * \param iter 当前持久化循环的轮次，用于三重缓冲索引
 * \return ordered 编码形式的 pivot
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, bool TRACK_EQ_COUNT>
// 在 shared memory 中完成完整的 radix select，最终找到第 k 大元素对应的 ordered pivot。
// 如果 TRACK_EQ_COUNT=true，还会额外统计当前 CTA 中 >pivot / ==pivot 的数量。
__device__ __forceinline__ OrderedType RadixSelectFromSharedMemory(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t k,
    uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
    int& barrier_phase, uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx, uint32_t iter,
    uint32_t& out_local_gt_count, uint32_t& out_local_eq_count) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t NUM_ROUNDS = ORDERED_BITS / RADIX_BITS;

// Aliases for scalar shared variables
// 这里把 shared memory 里几块小标量缓存取了别名，便于理解：
// prefix_cache / remaining_k_cache 是每轮递进更新的状态，
// found_bucket / found_remaining_k 是当前轮的中间输出。
#define prefix_cache shared_scalars[0]
#define remaining_k_cache shared_scalars[1]
#define found_bucket shared_scalars[2]
#define found_remaining_k shared_scalars[3]
#define shared_output_counter shared_scalars[4]

  // 初始化本轮 radix select 的共享状态缓存。
  if (tx == 0) {
    prefix_cache = 0;
    remaining_k_cache = k;
    if constexpr (SINGLE_CTA) {
      // 单 CTA 模式下，输出位置计数器直接放在 shared memory 里即可。
      shared_output_counter = 0;
    }
  }
  __syncthreads();

  // 初始 barrier：多 CTA 模式下需要先把所有 CTA 对齐到同一阶段。
  if constexpr (!SINGLE_CTA) {
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

    // barrier 之后由组内第 0 个 CTA 负责清空全局输出计数器。
    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->output_counter, 0);
    }
  }

  // 逐轮执行 radix select，直到把整个 ordered 位宽全部确定下来。
  for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
    uint32_t global_round = iter * NUM_ROUNDS + round;
    uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
    uint32_t prefix = prefix_cache;
    uint32_t remaining_k = remaining_k_cache;
    // prefix / remaining_k 是“上一轮已经缩小后的候选空间”。
    // 当前轮只需要在这个候选空间里继续细分。

    // 多 CTA 模式下，current_hist / next_hist 指向本轮和下一轮要用的全局 histogram 缓冲区。
    uint32_t* current_hist = nullptr;
    uint32_t* next_hist = nullptr;
    if constexpr (!SINGLE_CTA) {
      current_hist = state->histogram[global_round % 3];
      next_hist = state->histogram[(global_round + 1) % 3];
    }

    // 清空当前 CTA 的局部 histogram。
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    // 基于当前 prefix 构建本轮局部 histogram。
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

    // 多 CTA 模式下：先聚合到全局 histogram，再 barrier，最后再读回聚合结果。
    if constexpr (!SINGLE_CTA) {
      // 多 CTA 聚合：每个 CTA 提供自己 chunk 内的局部统计。
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
    __syncthreads();

    // 计算后缀和。
    RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);

    // 找到本轮包含目标名次的阈值 bucket。
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

    // Update caches
    if (tx == 0) {
      // 当前轮确定了一个更精细的 bucket，把它拼进 prefix，
      // 下一轮会基于这个更长的 prefix 继续筛。
      prefix_cache = prefix | (found_bucket << shift);
      remaining_k_cache = found_remaining_k;
    }
    __syncthreads();
  }

  OrderedType ordered_pivot = static_cast<OrderedType>(prefix_cache);
  // 到这里 prefix_cache 已经精确到完整位宽，因此它就是最终 pivot 的 ordered 编码。

  // 再扫一遍 shared_ordered，统计 > pivot（以及可选地 == pivot）的元素个数。
  // 原因是 suffix_sum 只覆盖“当前 prefix 命中的候选集合”，
  // 它并不能直接给出“整个 chunk 里所有 > pivot 的元素总数”。
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

  // 先在 warp 内做一次规约。
  // 先在 warp 内做规约，减少后面对 shared memory 的 atomic 压力。
  for (int offset = 16; offset > 0; offset /= 2) {
    my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
    if constexpr (TRACK_EQ_COUNT) {
      my_eq_count += __shfl_down_sync(0xffffffff, my_eq_count, offset);
    }
  }

  // 每个 warp 只让 lane 0 把结果原子加到共享内存里。
  // 每个 warp 只由 lane 0 把部分和写回 shared memory。
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
 * \brief 先把一个 CTA 的 chunk 加载到 ordered 共享内存，再执行 radix select 找 pivot。
 *
 * 这个辅助函数把“加载到共享内存”和“精确找到第 k 大元素的 pivot”两步合在一起。
 * 返回值是 ordered 编码形式的 pivot。
 * 调用方还可以选择是否回收当前 CTA 内 > pivot / == pivot 的计数，
 * 这些计数在确定性收集路径里会用到。
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, bool TRACK_EQ_COUNT,
          typename DType>
// 这是“加载 + 找 pivot”的组合辅助函数。
// 调用方不需要关心 shared load 和 radix select 的细节。
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
 * \brief 基于 pivot 收集 top-k 索引，并允许调用方自定义输出变换（单次遍历收集）。
 *
 * 这个优化版本用两次线性遍历完成收集：
 * 1. 对于 > pivot 的元素，先一次性为整个 CTA 申请全局输出区间，再在 CTA 内部分配局部位置
 * 2. 对于 == pivot 的元素，直接使用全局 atomic 竞争剩余位置，并在写出前检查 pos < k
 *
 * 其中 local_gt_count 在 radix select 的最后阶段已经算好，
 * 因此每个 CTA 都知道自己有多少个 > pivot 元素，
 * 可以把原本“每个元素一次全局 atomic”的开销降成“每个 CTA 一次全局 atomic”。
 *
 * \tparam BLOCK_THREADS 每个线程块的线程数
 * \tparam SINGLE_CTA 若为 true，表示单 CTA 模式
 * \tparam OrderedType ordered 整数类型
 * \tparam OutputFunc 输出回调类型，形式为
 *         void(uint32_t original_idx, OrderedType ordered_val, int output_pos)
 *
 * \param shared_ordered 保存 ordered 值的共享内存缓存
 * \param actual_chunk_size 当前 CTA 负责的 chunk 实际元素数
 * \param chunk_start 当前 chunk 在原始输入中的起始位置
 * \param k 目标 top-k 中的 k
 * \param ordered_pivot ordered 编码形式的 pivot
 * \param local_gt_count 当前 CTA 中 > pivot 元素的数量
 * \param local_histogram 这里被复用成若干计数器的共享内存
 * \param shared_output_counter 单 CTA 模式下使用的共享输出计数器
 * \param state 多 CTA 模式下使用的 RadixRowState 指针
 * \param barrier_phase 软件 barrier 阶段计数器引用
 * \param ctas_per_group 每个 group 里的 CTA 数量
 * \param tx 当前线程在线程块内的索引
 * \param output_func 对每个选中元素调用的输出回调
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, typename OutputFunc>
// 非确定性收集路径：
// 先收集所有 > pivot 的元素，再用全局 atomic 竞争式收集 == pivot 的元素直到填满 k。
__device__ __forceinline__ void RadixCollectIndices(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t chunk_start, uint32_t k,
    OrderedType ordered_pivot, uint32_t local_gt_count, uint32_t* local_histogram,
    uint32_t* shared_output_counter, RadixRowState* state, int& barrier_phase,
    uint32_t ctas_per_group, uint32_t tx, OutputFunc output_func) {
// 这里把 local_histogram 临时复用成两个计数器：
// [0] local_offset_gt：当前 CTA 内部 > pivot 元素的局部偏移
// [1] global_base_gt：当前 CTA 申请到的全局输出起始位置
#define local_offset_gt local_histogram[0]
#define global_base_gt local_histogram[1]

  // 为当前 CTA 的所有 > pivot 元素申请一段连续的全局输出区间（每个 CTA 只做一次 atomic）。
  if (tx == 0) {
    local_offset_gt = 0;
    if (local_gt_count > 0) {
      // 每个 CTA 只做一次全局 atomic，申请一整段连续输出区间给所有 > pivot 元素。
      // 这样比“每个元素都 atomic 一次”便宜很多。
      if constexpr (SINGLE_CTA) {
        global_base_gt = atomicAdd(shared_output_counter, local_gt_count);
      } else {
        global_base_gt = atomicAdd(&state->output_counter, local_gt_count);
      }
    }
  }
  __syncthreads();

  // 第一遍：写出所有 > pivot 的元素。
  // 它们一定属于 top-k，因此直接写入当前 CTA 的专属输出区间。
#pragma unroll 2
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered_val = shared_ordered[i];
    if (ordered_val > ordered_pivot) {
      // CTA 内再用 shared atomic 申请局部位置，拼成最终输出位置。
      uint32_t local_pos = atomicAdd(&local_offset_gt, 1);
      int pos = global_base_gt + local_pos;
      output_func(chunk_start + i, ordered_val, pos);
    }
  }

  // barrier：保证所有 > pivot 的元素都先写完。
  // 否则某些 CTA 可能提前写 == pivot，和其他 CTA 的 > pivot 写入交叉，导致输出位置错误。
  if constexpr (!SINGLE_CTA) {
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
  } else {
    __syncthreads();
  }

  // 第二遍：写出 == pivot 的元素。
  // 因为这些元素只需要“补足到 k”，所以必须跨 CTA 协同竞争剩余位置。
#pragma unroll 2
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered_val = shared_ordered[i];
    if (ordered_val == ordered_pivot) {
      int pos;
      // == pivot 的元素不能预先知道最终要取多少个，
      // 因为它们需要在所有 CTA 之间共同补足到 k，所以这里直接用全局计数器竞争。
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

// 用于 pair scan 的加法规约器，分别累加 gt / eq 两个计数。
struct DeterministicCollectCountPairSum {
  __device__ __forceinline__ DeterministicCollectCountPair operator()(
      const DeterministicCollectCountPair& lhs, const DeterministicCollectCountPair& rhs) const {
    return {lhs.gt + rhs.gt, lhs.eq + rhs.eq};
  }
};

/*!
 * \brief 以确定性的跨 CTA 顺序收集 top-k 索引。
 *
 * 这个版本为了保证结果可复现，不再依赖“谁先抢到 atomic 谁先写”的动态顺序，
 * 而是改成固定分配方案：
 * 1. 所有 > pivot 的元素按 CTA 顺序分配输出区间
 * 2. == pivot 的元素再基于 det_scratch 中记录的各 CTA gt/eq 计数，
 *    计算出稳定的前缀和与固定的输出范围
 *
 * 单 CTA 模式下会退化成块内的确定性收集，不需要额外使用 det_scratch。
 *
 * \tparam BLOCK_THREADS 每个线程块的线程数
 * \tparam SINGLE_CTA 若为 true，表示单 CTA 模式
 * \tparam OrderedType ordered 整数类型
 * \tparam OutputFunc 输出回调类型
 *
 * \param shared_ordered 保存 ordered 值的共享内存缓存
 * \param actual_chunk_size 当前 CTA 负责的 chunk 实际元素数
 * \param chunk_start 当前 chunk 在原始输入中的起始位置
 * \param k 目标 top-k 中的 k
 * \param ordered_pivot ordered 编码形式的 pivot
 * \param cta_local_gt_count 当前 CTA 中 > pivot 元素的数量
 * \param cta_local_eq_count 当前 CTA 中 == pivot 元素的数量
 * \param local_histogram 被复用成确定性收集状态的共享内存 scratch
 * \param state 多 CTA 模式下使用的 RadixRowState 指针
 * \param det_scratch 每个 group 使用的确定性收集 scratch
 * \param barrier_phase 软件 barrier 阶段计数器引用
 * \param ctas_per_group 每个 group 里的 CTA 数量
 * \param cta_in_group 当前 CTA 在 group 内的编号
 * \param tx 当前线程在线程块内的索引
 * \param output_func 对每个选中元素调用的输出回调
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType, typename OutputFunc>
// 确定性收集路径：
// 通过为每个 CTA 预先分配固定输出区间，避免不同 CTA 之间因 atomic 竞争造成结果顺序漂移。
__device__ __forceinline__ void RadixCollectIndicesDeterministic(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t chunk_start, uint32_t k,
    OrderedType ordered_pivot, uint32_t cta_local_gt_count, uint32_t cta_local_eq_count,
    uint32_t* local_histogram, RadixRowState* state, RadixDeterministicCollectScratch* det_scratch,
    int& barrier_phase, uint32_t ctas_per_group, uint32_t cta_in_group, uint32_t tx,
    OutputFunc output_func) {
// Use local_histogram for counters:
// [0]: s_cta_local_gt_prefix   - total >pivot count from earlier CTAs
// [1]: s_cta_local_eq_prefix   - total ==pivot count from earlier CTAs
// [2]: s_row_total_gt_count    - row-wide >pivot count across all CTAs
// [3]: s_row_eq_needed         - number of ==pivot entries still needed after >pivot writes
// [4]: s_cta_local_eq_take     - this CTA's assigned ==pivot quota
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
    // Single-CTA: keep the full ==pivot suffix contiguous after all >pivot entries.
    cta_local_eq_emit_limit = s_row_eq_needed;
    cta_local_eq_output_base = s_row_total_gt_count;
  } else {
    // Each CTA writes its local >pivot / ==pivot counts
    if (tx == 0) {
      s_cta_local_eq_prefix = 0;
      s_cta_local_eq_take = 0;
      det_scratch->gt_count[cta_in_group] = cta_local_gt_count;
      det_scratch->eq_count[cta_in_group] = cta_local_eq_count;
    }
    AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
    // Each CTA reads all >pivot / ==pivot counts
    if (tx == 0) {
      uint32_t cta_local_gt_prefix_accum = 0;
      uint32_t row_total_gt = 0;
      uint32_t cta_local_eq_prefix_accum = 0;
      // 这里显式按 CTA 顺序扫描所有计数。
      // 这样每个 CTA 的输出区间只依赖 CTA 编号，不依赖运行时谁先抢到 atomic。
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
      // 当前 CTA 最多只能拿到“全局还需要的 eq 数量”中的一部分。
      if (s_row_eq_needed > cta_local_eq_prefix_accum) {
        s_cta_local_eq_take = min(cta_local_eq_count, s_row_eq_needed - cta_local_eq_prefix_accum);
      }
    }
    __syncthreads();
    // Multi-CTA: only emit this CTA's assigned ==pivot quota at its deterministic output base.
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

  if (cta_local_eq_emit_limit == 0) {  // gt-only collect
    // 如果根本不需要 == pivot 元素，直接走更简单的 gt-only 收集。
    DeterministicThreadStridedCollect<BLOCK_THREADS>(
        tx, actual_chunk_size, scan_temp_storage.scalar,
        [&](uint32_t i) { return shared_ordered[i] > ordered_pivot; }, cta_local_gt_emit_limit,
        [&](uint32_t i, uint32_t local_pos) {
          output_func(chunk_start + i, shared_ordered[i], cta_local_gt_output_base + local_pos);
        });
    return;
  }

  // Collect gt and eq elements
  DeterministicCollectCountPair thread_local_counts = {0, 0};
  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    const OrderedType ordered = shared_ordered[i];
    thread_local_counts.gt += static_cast<uint32_t>(ordered > ordered_pivot);
    thread_local_counts.eq += static_cast<uint32_t>(ordered == ordered_pivot);
  }

  DeterministicCollectCountPair thread_local_prefix = {0, 0};
  // 对 (gt, eq) 二元计数做 CTA 级 scan，
  // 让每个线程都知道自己负责的 gt/eq 元素在稳定序列中的起始位置。
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

// ==================== 统一的 Radix Top-K Kernel 与尾处理模式 ====================

/*!
 * \brief 统一 RadixTopK kernel 的尾处理模式。
 */
// 同一个 kernel 主体支持三种尾处理方式：
// 1. Basic：直接输出 top-k 的原始索引和值
// 2. PageTableTransform：把选出的索引再映射到 page table
// 3. RaggedTransform：给索引加上每行对应的 offset
enum class RadixTopKMode {
  Basic,               ///< Returns (indices, values) pairs
  PageTableTransform,  ///< Gathers indices through page table
  RaggedTransform,     ///< Adds offset to indices
};

/*!
 * \brief 统一的多 CTA Radix Top-K kernel，支持多种尾处理方式。
 *
 * 这个 kernel 把三类 top-k 逻辑统一到了同一个主体里：
 * 1. Basic：输出 top-k 的原始索引和值
 * 2. PageTableTransform：先得到 top-k 索引，再通过页表映射成目标条目
 * 3. RaggedTransform：先得到 top-k 索引，再叠加每行自己的偏移量
 *
 * \tparam BLOCK_THREADS 每个线程块的线程数
 * \tparam VEC_SIZE 向量化访存宽度
 * \tparam SINGLE_CTA 若为 true，表示一行只由一个 CTA 处理
 * \tparam DETERMINISTIC 若为 true，使用确定性收集路径
 * \tparam MODE 尾处理模式
 * \tparam DType 数据类型
 * \tparam IdType 索引类型
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, bool DETERMINISTIC,
          RadixTopKMode MODE, typename DType, typename IdType>
// 统一的 RadixTopK kernel 主体。
// 一个 row 可能由多个 CTA 协同处理；每个 CTA 负责其中一个 chunk。
// 整体流程基本是：
// 1. 计算本行长度、k 和本 CTA 的 chunk 范围
// 2. 在 shared memory 中找到全局第 k 大的 pivot
// 3. 收集 top-k 对应索引
// 4. 按 MODE 做尾处理
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKKernel_Unified(
    DType* input,            // [num_rows, stride]
    IdType* output_indices,  // [num_rows, top_k] - indices or page table entries
    DType* output_values,    // [num_rows, top_k] - only used in Basic mode, nullptr otherwise
    const IdType*
        aux_data,  // Mode-specific: top_k_arr (Basic), src_page_table (PageTable), offsets (Ragged)
    IdType* lengths,             // [num_rows] per-row lengths, nullptr for Basic (uses stride)
    const IdType* row_to_batch,  // [num_rows] batch mapping for PageTable, nullptr otherwise
    int64_t aux_stride,          // src_page_table stride for PageTable mode, 0 otherwise
    uint32_t top_k_val, uint32_t stride, uint32_t num_rows, RadixRowState* row_states,
    RadixDeterministicCollectScratch* det_scratches, uint32_t chunk_size, uint32_t ctas_per_group) {
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;
  constexpr uint32_t RADIX = 256;

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  extern __shared__ uint8_t smem[];

  constexpr size_t num_scalars = SINGLE_CTA ? 5 : 4;
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + num_scalars);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;

  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

#define shared_output_counter shared_scalars[4]

  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }
  RadixDeterministicCollectScratch* det_scratch = nullptr;
  if constexpr (!SINGLE_CTA && DETERMINISTIC) {
    det_scratch = &det_scratches[group_id];
  }
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (num_rows + num_groups - 1) / num_groups;

  int barrier_phase = 0;

  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= num_rows) break;

    // 根据 MODE 决定本行的有效长度和 k 的来源。
    uint32_t length, k;
    if constexpr (MODE == RadixTopKMode::Basic) {
      length = stride;                                            // Fixed length for all rows
      k = (aux_data != nullptr) ? aux_data[row_idx] : top_k_val;  // aux_data = top_k_arr
    } else {
      length = lengths[row_idx];  // Per-row length
      k = top_k_val;              // Fixed k
    }

    // 当前行对应的输出起始位置。
    IdType* row_output = output_indices + row_idx * top_k_val;

    // 处理简单场景：当 k 已经不小于有效长度时，不需要再做 pivot 查找。
    if constexpr (MODE == RadixTopKMode::Basic) {
      if (k >= length) {
        // k >= 当前行长度：直接返回整行所有位置即可。
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
        // 为后续持久化循环预清理 histogram，避免下一行读到旧数据。
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
      uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row_idx] : row_idx;
      const IdType* src_page_entry = aux_data + batch_idx * aux_stride;
      if (length <= top_k_val) {
        for (uint32_t i = tx; i < top_k_val; i += BLOCK_THREADS) {
          row_output[i] = (i < length) ? src_page_entry[i] : static_cast<IdType>(-1);
        }
        // Clear histogram for next iteration
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
    } else {  // RaggedTransform
      IdType offset = aux_data[row_idx];
      if (length <= top_k_val) {
        for (uint32_t i = tx; i < top_k_val; i += BLOCK_THREADS) {
          row_output[i] = (i < length) ? static_cast<IdType>(i) + offset : static_cast<IdType>(-1);
        }
        // Clear histogram for next iteration
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

    // 阶段 1：把当前 CTA 的 chunk 加载到共享内存，然后做 radix select 找 pivot。
    uint32_t cta_local_gt_count = 0;
    uint32_t cta_local_eq_count = 0;
    OrderedType ordered_pivot =
        RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, DETERMINISTIC, DType>(
            input + static_cast<size_t>(row_idx) * stride, shared_ordered, local_histogram,
            suffix_sum, shared_scalars, state, chunk_start, actual_chunk_size, k, barrier_phase,
            ctas_per_group, cta_in_group, tx, iter, cta_local_gt_count, cta_local_eq_count);

    // collect_indices 封装了“非确定性 / 确定性”两种收集路径。
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

    // 阶段 2：收集索引，并根据 MODE 做不同的尾处理。
    if constexpr (MODE == RadixTopKMode::Basic) {
      DType* row_output_values = output_values + row_idx * top_k_val;
      collect_indices([&](uint32_t original_idx, OrderedType ordered_val, int pos) {
        row_output[pos] = static_cast<IdType>(original_idx);
        row_output_values[pos] = Traits::FromOrdered(ordered_val);
      });
    } else if constexpr (MODE == RadixTopKMode::PageTableTransform) {
      uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row_idx] : row_idx;
      const IdType* src_page_entry = aux_data + batch_idx * aux_stride;

      // Collect raw indices first
      collect_indices([&](uint32_t original_idx, OrderedType /*ordered_val*/, int pos) {
        row_output[pos] = static_cast<IdType>(original_idx);
      });

      if constexpr (SINGLE_CTA) {
        __syncthreads();
        // Transform through page table with coalesced access
        for (uint32_t i = tx; i < k; i += BLOCK_THREADS) {
          IdType idx = row_output[i];
          row_output[i] = src_page_entry[idx];
        }
      } else {
        // Barrier to ensure all CTAs finished writing indices
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

        // All CTAs participate in page table transform (coalesced access)
        uint32_t elems_per_cta = (k + ctas_per_group - 1) / ctas_per_group;
        uint32_t my_start = cta_in_group * elems_per_cta;
        uint32_t my_end = min(my_start + elems_per_cta, k);
        for (uint32_t i = my_start + tx; i < my_end; i += BLOCK_THREADS) {
          IdType idx = row_output[i];
          row_output[i] = src_page_entry[idx];
        }
      }
    } else {  // RaggedTransform
      IdType offset = aux_data[row_idx];
      collect_indices([&](uint32_t original_idx, OrderedType /*ordered_val*/, int pos) {
        row_output[pos] = static_cast<IdType>(original_idx) + offset;
      });
    }
  }

  // Clear histogram buffers and reset arrival counter for next kernel launch (only for multi-CTA)
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
// logits mask kernel：
// 先找 top-k 的阈值 pivot，再把小于 pivot 的位置写成 -inf，其余保持原值。
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKMaskLogitsKernel_MultiCTA(
    DType* logits,         // [batch, vocab_size]
    DType* masked_logits,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
    uint32_t chunk_size,        // elements per CTA
    uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
{
  // Type traits for FP16/BF16/FP32 support
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;  // 8-bit radix

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  // histogram[256] + suffix[256] + 5 scalars (for RadixSelectFromSharedMemory)
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 5);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;

  // Align ordered values cache to 16 bytes
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    DType pivot = Traits::NegInf();

    if (k >= vocab_size) {
      // k >= vocab_size: no masking needed, just copy
      vec_t<DType, VEC_SIZE> logits_vec_copy;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        logits_vec_copy.cast_load(logits + row_idx * vocab_size + chunk_start + i);
        logits_vec_copy.store(masked_logits + row_idx * vocab_size + chunk_start + i);
      }
      // Handle tail
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        masked_logits[row_idx * vocab_size + chunk_start + i] =
            logits[row_idx * vocab_size + chunk_start + i];
      }

      // Clear histogram for next iteration (in case it's k < vocab_size)
      // Only needed for multi-CTA mode; single-CTA uses shared memory cleared each iteration
      if constexpr (!SINGLE_CTA) {
        constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
        uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
        if (cta_in_group == 0) {
          for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
            state->histogram[next_first_hist_idx][i] = 0;
          }
        }
        // No sync needed - next iteration's barrier will ensure visibility
      }
      continue;
    }

    // Stage 1: Load the chunk into shared memory, then radix-select the pivot.
    uint32_t local_gt_count = 0;  // Not used in this kernel
    uint32_t local_eq_count = 0;  // Not used in this kernel
    OrderedType ordered_pivot =
        RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, false, DType>(
            logits + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum,
            shared_scalars, state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group,
            cta_in_group, tx, iter, local_gt_count, local_eq_count);

    pivot = Traits::FromOrdered(ordered_pivot);

    // Stage 2: Final masking pass
    const DType neg_inf = Traits::NegInf();
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
    vec_t<DType, VEC_SIZE> logits_vec;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      logits_vec.cast_load(logits + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        logits_vec[j] = (logits_vec[j] >= pivot) ? logits_vec[j] : neg_inf;
      }
      logits_vec.store(masked_logits + row_idx * vocab_size + chunk_start + i);
    }

    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = logits[row_idx * vocab_size + chunk_start + i];
      masked_logits[row_idx * vocab_size + chunk_start + i] = (val >= pivot) ? val : neg_inf;
    }
  }

  // Clear histogram buffers and reset arrival counter for next kernel launch (only for multi-CTA)
  if constexpr (!SINGLE_CTA) {
    // Only leading CTA clears the buffers using release semantics
    if (cta_in_group == 0) {
      for (uint32_t buf = 0; buf < 3; ++buf) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          state->histogram[buf][i] = 0;
        }
      }

      if (tx == 0) {
        st_release(&state->arrival_counter, 0);
      }
    }
  }
}

template <typename DType, typename IdType>
// Host 侧 launch 包装：
// 根据设备共享内存上限、向量化宽度、词表长度等信息，决定 chunk_size / ctas_per_group，
// 然后选择 single-CTA 或 multi-CTA kernel 实例。
cudaError_t RadixTopKMaskLogitsMultiCTA(DType* logits, DType* masked_logits, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  // vec_size 表示一次向量化 load/store 处理多少个元素。
  // 这里要求 16 字节对齐，因此取 gcd(16 / sizeof(DType), vocab_size)。
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // Get device properties
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 5 scalars
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 5);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);

  // Calculate max chunk size that fits in shared memory
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, false);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  // 至少保证每个线程有一个完整向量可处理，避免 CTA 太“瘦”。
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  // 先估算每个 CTA 最多能吃多少元素，再据此反推一行需要多少个 CTA 协作。
  uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
  uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
  const bool single_cta = (ctas_per_group == 1);

  // Calculate number of groups (how many rows to process concurrently)
  // 一组 group 处理一行；如果一行要占多个 CTA，那么同时能并发的 group 数就会减少。
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
 * \brief Multi-CTA Radix Top-K RenormProb kernel with unified single/multi-CTA paths.
 *
 * Finds the k-th largest probability, then normalizes all probs >= pivot to sum to 1,
 * setting all others to 0. Reuses the shared load+radix-select helper.
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
// renorm prob kernel：
// 先找第 k 大阈值，再只保留 >= pivot 的概率，并重新归一化到和为 1。
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKRenormProbKernel_MultiCTA(
    DType* probs,          // [batch, vocab_size]
    DType* renormed_prob,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
    uint32_t chunk_size,        // elements per CTA
    uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
{
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;  // 8-bit radix

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  // histogram[256] + suffix[256] + scalars[4] + sum_local[1]
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  float* shared_sum = reinterpret_cast<float*>(shared_scalars + 4);

  // Align ordered values cache to 16 bytes
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // For RenormProb, pivot is compared with probs (must be non-negative)
    DType pivot = DType(0);
    float normalizer = 1.0f;

    if (k >= vocab_size) {
      // k >= vocab_size: no filtering needed, just compute sum and renormalize
      // Stage 1: Compute sum
      float thread_sum = 0.0f;
      vec_t<DType, VEC_SIZE> data_vec;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          thread_sum += float(data_vec[j]);
        }
      }
      // Handle tail
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        thread_sum += float(probs[row_idx * vocab_size + chunk_start + i]);
      }

      // Block reduction for sum
      typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
      __syncthreads();

      if constexpr (!SINGLE_CTA) {
        // Multi-CTA: atomic add to global sum
        if (tx == 0) {
          if (cta_in_group == 0) {
            state->sum_topk = 0.0f;  // First CTA initializes
          }
        }
        // Barrier for initialization
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

        if (tx == 0 && block_sum > 0) {
          atomicAdd(&state->sum_topk, block_sum);
        }

        // Barrier to ensure all CTAs have contributed
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
        normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
      } else {
        // Single-CTA: use block_sum directly
        if (tx == 0) {
          *shared_sum = block_sum;
        }
        __syncthreads();
        normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
      }

      // Normalize and store
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

      // Clear histogram for next iteration (in case it's k < vocab_size)
      // Only needed for multi-CTA mode; single-CTA uses shared memory cleared each iteration
      // Next iteration (iter+1) will use histogram[((iter+1)*NUM_ROUNDS) % 3] for its first round
      if constexpr (!SINGLE_CTA) {
        constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
        uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
        if (cta_in_group == 0) {
          for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
            state->histogram[next_first_hist_idx][i] = 0;
          }
        }
        // No sync needed - next iteration's barrier will ensure visibility
      }
      continue;
    }

    // ========== Stage 1: Find pivot ==========
    uint32_t local_gt_count = 0;  // Not used in this kernel
    uint32_t local_eq_count = 0;  // Not used in this kernel
    auto ordered_pivot = RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, false, DType>(
        probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
        state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, cta_in_group, tx,
        iter, local_gt_count, local_eq_count);
    pivot = Traits::FromOrdered(ordered_pivot);

    // ========== Stage 2: Compute sum of elements >= pivot ==========
    float thread_sum = 0.0f;
    vec_t<DType, VEC_SIZE> data_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        if (data_vec[j] >= pivot) {
          thread_sum += float(data_vec[j]);
        }
      }
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      if (val >= pivot) {
        thread_sum += float(val);
      }
    }

    // Block reduction for sum
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    __syncthreads();

    if constexpr (!SINGLE_CTA) {
      // Multi-CTA: atomic add to global sum
      if (tx == 0) {
        if (cta_in_group == 0) {
          state->sum_topk = 0.0f;  // First CTA initializes
        }
      }
      // Barrier for initialization
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

      if (tx == 0 && block_sum > 0) {
        atomicAdd(&state->sum_topk, block_sum);
      }

      // Barrier to ensure all CTAs have contributed
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
      normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
    } else {
      // Single-CTA: use block_sum directly
      if (tx == 0) {
        *shared_sum = block_sum;
      }
      __syncthreads();
      normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
    }

    // ========== Stage 3: Normalize elements >= pivot, set others to 0 ==========
#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        data_vec[j] = (data_vec[j] >= pivot) ? DType(float(data_vec[j]) * normalizer) : DType(0);
      }
      data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      renormed_prob[row_idx * vocab_size + chunk_start + i] =
          (val >= pivot) ? DType(float(val) * normalizer) : DType(0);
    }
  }

  // Clear histogram buffers and reset arrival counter for next kernel launch (only for multi-CTA)
  if constexpr (!SINGLE_CTA) {
    // Only leading CTA clears the buffers using release semantics
    if (cta_in_group == 0) {
      for (uint32_t buf = 0; buf < 3; ++buf) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          state->histogram[buf][i] = 0;
        }
      }

      if (tx == 0) {
        st_release(&state->arrival_counter, 0);
      }
    }
  }
}

template <typename DType, typename IdType>
// Host 侧 launch 包装：负责为 RenormProb 版本计算运行配置并启动对应 kernel。
cudaError_t RadixTopKRenormProbMultiCTA(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // Get device properties
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 4 scalars + 1 float
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 4) + sizeof(float);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);

  // Calculate max chunk size that fits in shared memory
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, false);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  // 与 MaskLogits 版本相同：根据共享内存预算，选择“每行拆成多少个 chunk”。
  uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
  uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
  const bool single_cta = (ctas_per_group == 1);

  // Calculate number of groups (how many rows to process concurrently)
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
// PageTable 变换版 Top-K 的 host 入口。
// 在选出 top-k 后，把原始索引映射成 page table 中的条目。
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
  // 确定性模式需要给每个 CTA 记录 gt/eq 计数，scratch 数组长度有限，因此这里有限制。
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
  // det_scratch_buffer 紧跟在 row_states_buffer 后面复用一块连续大缓冲区。
  RadixDeterministicCollectScratch* det_scratch_buffer =
      MaybeGetRadixDeterministicCollectScratchBuffer(row_states_buffer, num_groups, single_cta,
                                                     deterministic);

  // Unified kernel parameters
  DType* output_values = nullptr;  // Not used in PageTableTransform mode
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
// Ragged 变换版 Top-K 的 host 入口。
// 在选出 top-k 后，对每个索引加上该行对应的 offset。
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

  // Unified kernel parameters
  DType* output_values = nullptr;        // Not used in RaggedTransform mode
  const IdType* row_to_batch = nullptr;  // Not used in RaggedTransform mode
  int64_t aux_stride = 0;                // Not used in RaggedTransform mode
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
// Basic Top-K 的 host 入口：
// 返回 top-k 的索引和对应值。
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

  // Fixed smem: histogram[256] + suffix_sum[256] + scalars
  // Scalars: 5 for single-CTA, 4 for multi-CTA
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

  // Determine if we use single-CTA path
  // 当一整行能被一个 CTA 装下时，single-CTA 路径能省掉所有跨 CTA 同步开销。
  const bool single_cta = (ctas_per_group == 1);

  // Calculate smem_size: fixed + ordered values
  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);

  // Calculate number of groups (how many rows to process concurrently)
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;
  RadixDeterministicCollectScratch* det_scratch_buffer =
      MaybeGetRadixDeterministicCollectScratchBuffer(row_states_buffer, num_groups, single_cta,
                                                     deterministic);

  // Unified kernel parameters
  IdType* lengths = nullptr;             // Not used in Basic mode
  const IdType* row_to_batch = nullptr;  // Not used in Basic mode
  int64_t aux_stride = 0;                // Not used in Basic mode
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
// Based on sgl-kernel's filter algorithm with multi-dtype support
// 下面是另一套 FilteredTopK 实现。
// 思路和上面的“完整 radix select”不同，它先做一个粗粒度筛选，再做细化。

// FilteredTopK traits for different data types
template <typename DType>
struct FilteredTopKTraits;

// Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4 refinement rounds
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // Convert to FP16 representation and extract high 8 bits
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

// Specialization for half (16-bit): coarse histogram uses high 8 bits, only need low 8 bits for
// refinement Since coarse key = high 8 bits, refinement only needs to look at low 8 bits (no
// additional rounds needed if we can determine topk from coarse pass alone)
template <>
struct FilteredTopKTraits<half> {
  using OrderedType = uint16_t;
  static constexpr int NUM_REFINE_ROUNDS = 1;   // Only 1 round for low 8 bits
  static constexpr int FIRST_REFINE_SHIFT = 0;  // Start from bit 0 (low 8 bits)

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

// Specialization for nv_bfloat16 (16-bit): same as half
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

// FilteredTopK constants
// FILTERED_TOPK_SMEM_INPUT_SIZE 表示共享内存里缓存的索引/候选规模。
constexpr uint32_t FILTERED_TOPK_MAX_K = 2048;
constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE = 16 * 1024;  // 16K indices per buffer
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

// Output modes for unified FilteredTopK kernel
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
 * - PageTable: output = dst_page_table, aux_input = src_page_table, aux_stride = src_stride
 * - Ragged: output = indices, aux_input = offsets, aux_output/aux_stride/row_to_batch unused
 */
template <typename DType, typename IdType, int VEC_SIZE, bool DETERMINISTIC, FilteredTopKMode MODE>
// FilteredTopK 统一 kernel。
// 与上面的 RadixTopKKernel_Unified 类似，也支持 Plain / PageTable / Ragged 三种输出形式。
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input, IdType* __restrict__ output,
                              DType* __restrict__ aux_output,           // values for Plain mode
                              const IdType* __restrict__ aux_input,     // page_table or offsets
                              int64_t aux_stride,                       // src_stride for PageTable
                              const IdType* __restrict__ row_to_batch,  // for PageTable
                              const IdType* __restrict__ lengths, uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length = (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + static_cast<size_t>(bid) * max_len;
  IdType* dst = output + bid * top_k;

  // Mode-specific setup
  [[maybe_unused]] const IdType* src_page_entry = nullptr;
  [[maybe_unused]] IdType offset_val = 0;
  [[maybe_unused]] DType* dst_values = nullptr;

  if constexpr (MODE == FilteredTopKMode::PageTable) {
    const uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[bid] : bid;
    src_page_entry = aux_input + batch_idx * aux_stride;
  } else if constexpr (MODE == FilteredTopKMode::Ragged) {
    offset_val = aux_input[bid];
  } else {  // Plain
    dst_values = aux_output + bid * top_k;
  }

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      if constexpr (MODE == FilteredTopKMode::Plain) {
        if (i < length) {
          dst[i] = static_cast<IdType>(i);
          dst_values[i] = score[i];
        } else {
          dst[i] = static_cast<IdType>(-1);
          dst_values[i] = DType(0);
        }
      } else if constexpr (DETERMINISTIC) {
        // In deterministic mode the page-table/ragged transform happens in SortTopKByIndexKernel
        dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
      } else if constexpr (MODE == FilteredTopKMode::PageTable) {
        dst[i] = (i < length) ? src_page_entry[i] : static_cast<IdType>(-1);
      } else {  // Ragged
        dst[i] = (i < length) ? static_cast<IdType>(i) + offset_val : static_cast<IdType>(-1);
      }
    }
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  __shared__ int s_counter;
  __shared__ int s_threshold_bin_id;
  // Per-round copies of s_threshold_bin_id for deterministic pivot rebuild.
  __shared__ int s_refine_thresholds[4];
  __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[FILTERED_TOPK_MAX_K];
  // Set 1 when s_input_idx overflows in tie-heavy workload
  __shared__ int s_refine_overflow;
  __shared__ int s_last_remain;

  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;
  int topk = top_k;
  if (tx == 0) s_refine_overflow = 0;
  if constexpr (DETERMINISTIC) {
    if (tx < 4) {
      s_refine_thresholds[tx] = 0xFF;
    }
  }
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  // Stage 1: (shared by deterministic and non-deterministic modes)
  // build a coarse histogram and identify the threshold bin.
  // The modes diverge later when collecting == pivot elements.
  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
  // Full-row scan helper (vectorized body + tail). Overflow fallback reuses this traversal.
  auto for_each_score_full = [&](auto&& fn) {
  // vectorized body
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length; base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        fn(score_vec[j], base + j);
      }
    }
    // tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      fn(score[i], i);
    }
  };
  auto accumulate_coarse_hist = [&](auto raw_input, int /*index*/) {
    const auto bin = Traits::ToCoarseKey(raw_input);
    atomicAdd(&s_histogram[bin], 1);
  };
  for_each_score_full(accumulate_coarse_hist);
  __syncthreads();

  // Suffix sum (Hillis Steele Scan)
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
  auto update_refine_threshold = [&](int next_input_idx, auto reset_next_input_tag) {
    constexpr bool RESET_NEXT_INPUT = decltype(reset_next_input_tag)::value;
    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      if constexpr (RESET_NEXT_INPUT) {
        s_num_input[next_input_idx] = 0;
      }
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];
  [[maybe_unused]] const int topk_after_coarse = topk;

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  // fp16/bf16: stop_round = 0; fp32: stop_round = 0,1,2,3
  auto build_det_pivot = [&](int stop_round) -> OrderedType {
    if constexpr (sizeof(OrderedType) == 2) {
      return static_cast<OrderedType>((static_cast<uint32_t>(threshold_bin) << 8) |
                                      static_cast<uint32_t>(s_refine_thresholds[0]));
    } else {  // fp32
      uint32_t pivot = 0;
      for (int round = 0; round < NUM_ROUNDS; ++round) {
        uint32_t byte =
            (round <= stop_round) ? static_cast<uint32_t>(s_refine_thresholds[round]) : 0xFFu;
        pivot |= (byte << (FIRST_SHIFT - round * 8));
      }
      return static_cast<OrderedType>(pivot);
    }
  };

  if (topk == 0) {
    // Collect indices where bin > threshold
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
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // Both non-det and det modes use atomicAdd to append >threshold winners here;
    // only ==threshold handling diverges between the two modes.
    auto collect_gt_and_nondet_eq_threshold = [&](auto value, auto threshold, int idx,
                                                  bool collect_eq) {
      if (value > threshold) {
        const int pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = idx;
      } else if constexpr (!DETERMINISTIC) {
        if (collect_eq && value == threshold) {
          const int pos = atomicAdd(&s_last_remain, -1);
          if (pos > 0) {
            s_indices[static_cast<int>(top_k) - pos] = idx;
          }
        }
      }
    };

    auto collect_det_eq_pivot = [&](OrderedType pivot, int eq_needed) {
      if (eq_needed > 0) {
        using DetCollectBlockScan =
            cub::BlockScan<uint32_t, BLOCK_SIZE, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
        __shared__ typename DetCollectBlockScan::TempStorage temp_storage;
        DeterministicThreadStridedCollect<BLOCK_SIZE>(
            tx, length, temp_storage,
            [&](uint32_t idx) { return Traits::ToOrdered(score[idx]) == pivot; }, eq_needed,
            [&](uint32_t idx, uint32_t local_pos) {
              s_indices[static_cast<int>(top_k) - eq_needed + static_cast<int>(local_pos)] =
                  static_cast<int>(idx);
            });
      }
    };

    // Filter + histogram for refinement
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        } else {
          atomicOr(&s_refine_overflow, 1);
        }
      }
    };
    for_each_score_full(filter_and_add_to_histogram);
    __syncthreads();

    // Stage 2: refine with 8bit radix passes.
    // If the threshold-bin candidate buffer overflows in 1-round refine mode
    // (fp16/bf16), switch to a slow path that re-histograms the full threshold
    // bin to preserve correctness.
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
    auto collect_with_threshold_non_last_round = [&](int r_idx, int num_input, int offset,
                                                     int threshold) {
      const auto next_r_idx = r_idx ^ 1;
      __syncthreads();
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = score[idx];
        const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
        if (static_cast<int>(bin) > threshold) {
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = idx;
        } else if (static_cast<int>(bin) == threshold) {
          const auto pos = atomicAdd(&s_num_input[next_r_idx], 1);
          if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
            s_input_idx[next_r_idx][pos] = idx;
            const auto bin32 = Traits::ToOrdered(raw_input);
            const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
            atomicAdd(&s_histogram[sub_bin], 1);
          } else {
            atomicOr(&s_refine_overflow, 1);
          }
        }
      }
      __syncthreads();
    };
    // Returns true if this round fully resolves the pivot, i.e. no ==threshold
    // elements need to be carried into another refine round.
    auto run_refine_round = [&](int r_idx, int offset, auto is_last_round_tag) {
      constexpr bool IS_LAST_ROUND = decltype(is_last_round_tag)::value;
      const auto raw_num_input = s_num_input[r_idx];
      const auto num_input = (raw_num_input < SMEM_INPUT_SIZE) ? raw_num_input : SMEM_INPUT_SIZE;

      update_refine_threshold(r_idx ^ 1, std::true_type{});

      const auto threshold = s_threshold_bin_id;
      if constexpr (DETERMINISTIC) {
        if (tx == 0) {
          s_refine_thresholds[(FIRST_SHIFT - offset) / 8] = threshold;
        }
      }
      topk -= s_histogram[threshold + 1];
      if (topk == 0) {
        // Final round reached: only collect bins strictly greater than threshold.
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

      if constexpr (IS_LAST_ROUND) {
        collect_with_threshold_last_round(r_idx, num_input, offset, threshold);
      } else {
        collect_with_threshold_non_last_round(r_idx, num_input, offset, threshold);
      }
      return false;
    };
    if constexpr (NUM_ROUNDS == 1) {  // fast path for 1-round refine.
      if (s_refine_overflow) {
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();

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

        if (tx == 0) {
          s_threshold_bin_id = 0;
          s_last_remain = 0;
        }
        __syncthreads();

        update_refine_threshold(/*next_input_idx=*/0, std::false_type{});

        const auto threshold = s_threshold_bin_id;

        // Keep s_counter continuity: it already counts coarse_bin > threshold_bin
        // elements collected in filter_and_add_to_histogram. Here we append
        // threshold-bin refined winners after that prefix.
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
        if constexpr (DETERMINISTIC) {
          int eq_needed = s_last_remain;
          collect_det_eq_pivot(static_cast<OrderedType>((static_cast<int>(threshold_bin) << 8) |
                                                        static_cast<int>(threshold)),
                               eq_needed);
        }
      } else {
        const int round = 0;
        const auto r_idx = round % 2;
        const int offset = FIRST_SHIFT;
        run_refine_round(r_idx, offset, std::true_type{});
        if constexpr (DETERMINISTIC) {
          collect_det_eq_pivot(build_det_pivot(/*stop_round=*/0), topk);
        }
      }
    } else {
      // Multi-round refine path (float32): if any refine-buffer overflow is detected,
      // switch to a correctness-first full rebuild of the threshold-bin selection.
      // This fallback may be slower than the fast path, but avoids partial-state corruption.
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

        // Overflow can happen after partial writes to s_indices/s_counter in earlier rounds.
        // Reset and rebuild from full scans to avoid mixing stale partial state.
        if (tx == 0) {
          s_counter = 0;
          s_last_remain = eq_needed;
        }
        __syncthreads();

        // Re-collect all winners from scratch:
        //   1) coarse_bin > threshold_bin
        //   2) threshold_bin entries with ordered > pivot
        //   3) first eq_needed entries where ordered == pivot
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

  // Output phase - mode-specific
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    if constexpr (MODE == FilteredTopKMode::Plain) {
      dst[base] = static_cast<IdType>(idx);
      dst_values[base] = score[idx];
    } else if constexpr (DETERMINISTIC) {  // transform in SortTopKByIndexKernel
      dst[base] = static_cast<IdType>(idx);
    } else if constexpr (MODE == FilteredTopKMode::PageTable) {
      dst[base] = src_page_entry[idx];
    } else {  // Ragged
      dst[base] = static_cast<IdType>(idx) + offset_val;
    }
  }
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE based on max_len and dtype
// Returns 1, 2, 4, or 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
  // Use GCD to find largest power-of-2 divisor
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

template <FilteredTopKMode MODE, uint32_t BLOCK_THREADS, uint32_t ITEMS_PER_THREAD, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS)
    SortTopKByIndexKernel(IdType* output_indices, DType* output_values, const IdType* aux_input,
                          int64_t aux_stride, const IdType* row_to_batch, uint32_t top_k,
                          uint32_t max_len) {
  constexpr bool WITH_VALUES = (MODE == FilteredTopKMode::Plain);
  using BlockRadixSortT = typename SortTopKByIndexBlockRadixSort<WITH_VALUES, BLOCK_THREADS,
                                                                 ITEMS_PER_THREAD, DType>::Type;
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  IdType* row_output = output_indices + static_cast<size_t>(row) * top_k;

  uint32_t keys[ITEMS_PER_THREAD];
  DType values[ITEMS_PER_THREAD];

#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    uint32_t pos = tx * ITEMS_PER_THREAD + i;
    if (pos < top_k) {
      IdType idx = row_output[pos];
      keys[i] = (idx >= 0) ? static_cast<uint32_t>(idx) : ~0u;
      if constexpr (MODE == FilteredTopKMode::Plain) {
        values[i] = output_values[static_cast<size_t>(row) * top_k + pos];
      }
    } else {
      keys[i] = ~0u;
      if constexpr (MODE == FilteredTopKMode::Plain) {
        values[i] = DType(0);
      }
    }
  }

  int end_bit = 32 - __clz(max_len);
  if constexpr (MODE == FilteredTopKMode::Plain) {
    BlockRadixSortT(temp_storage).Sort(keys, values, 0, end_bit);
  } else {
    BlockRadixSortT(temp_storage).Sort(keys, 0, end_bit);
  }

  const IdType* src_page_entry = nullptr;
  IdType offset = 0;
  if constexpr (MODE == FilteredTopKMode::PageTable) {
    const uint32_t batch_idx = (row_to_batch != nullptr) ? row_to_batch[row] : row;
    src_page_entry = aux_input + static_cast<int64_t>(batch_idx) * aux_stride;
  } else if constexpr (MODE == FilteredTopKMode::Ragged) {
    offset = aux_input[row];
  }

#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    uint32_t pos = tx * ITEMS_PER_THREAD + i;
    if (pos < top_k) {
      uint32_t idx = keys[i];
      if constexpr (MODE == FilteredTopKMode::Plain) {
        row_output[pos] = static_cast<IdType>(idx);
        output_values[static_cast<size_t>(row) * top_k + pos] = values[i];
      } else if constexpr (MODE == FilteredTopKMode::PageTable) {
        row_output[pos] = (idx != ~0u) ? src_page_entry[idx] : static_cast<IdType>(-1);
      } else {  // Ragged
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
  // Block-local sort variants cover at most 256 * 8 = 2048 elements.
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
  // Block-local sort variants cover at most 256 * 8 = 2048 elements.
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

// Launch functions with VEC_SIZE and BLOCK_THREADS dispatch - using unified kernel
template <typename DType, typename IdType>
cudaError_t FilteredTopKPageTableTransform(DType* input, IdType* output_page_table,
                                           const IdType* src_page_table, int64_t src_stride,
                                           const IdType* row_to_batch, IdType* lengths,
                                           uint32_t num_rows, uint32_t top_k_val, uint32_t max_len,
                                           bool deterministic = false, cudaStream_t stream = 0) {
  DType* aux_output = nullptr;  // Not used for PageTable mode
  return LaunchFilteredTopKUnified<FilteredTopKMode::PageTable, DType, IdType>(
      input, output_page_table, aux_output, src_page_table, src_stride, row_to_batch, lengths,
      num_rows, top_k_val, max_len, deterministic, stream);
}

template <typename DType, typename IdType>
cudaError_t FilteredTopKRaggedTransform(DType* input, IdType* output_indices, const IdType* offsets,
                                        IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len, bool deterministic = false,
                                        cudaStream_t stream = 0) {
  DType* aux_output = nullptr;           // Not used for Ragged mode
  int64_t aux_stride = 0;                // Not used for Ragged mode
  const IdType* row_to_batch = nullptr;  // Not used for Ragged mode
  return LaunchFilteredTopKUnified<FilteredTopKMode::Ragged, DType, IdType>(
      input, output_indices, aux_output, offsets, aux_stride, row_to_batch, lengths, num_rows,
      top_k_val, max_len, deterministic, stream);
}

template <typename DType, typename IdType>
cudaError_t FilteredTopK(DType* input, IdType* output_indices, DType* output_values,
                         const IdType* lengths, uint32_t num_rows, uint32_t top_k_val,
                         uint32_t max_len, bool deterministic = false, cudaStream_t stream = 0) {
  const IdType* aux_input = nullptr;     // Not used for Plain mode
  int64_t aux_stride = 0;                // Not used for Plain mode
  const IdType* row_to_batch = nullptr;  // Not used for Plain mode
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

// Algorithm override for benchmarking (controlled by FLASHINFER_TOPK_ALGO env var)
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
  // Check if GPU supports enough shared memory for FilteredTopK
  const bool gpu_supports_filtered = CanImplementFilteredTopK();
  const bool k_fits_filtered = (top_k_val <= FILTERED_TOPK_MAX_K) && (max_len > top_k_val);

  if (!gpu_supports_filtered || !k_fits_filtered) {
    return false;
  }

  // Check for algorithm override
  const TopKAlgoOverride algo_override = GetTopKAlgoOverride();
  if (algo_override == TopKAlgoOverride::FILTERED) return true;
  if (algo_override == TopKAlgoOverride::MULTI_CTA) return false;

  // 16-bit types: simpler threshold
  // 32-bit types: more nuanced heuristic
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

// Dispatch functions with heuristics
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

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_TOPK_CUH_
