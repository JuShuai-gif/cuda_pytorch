#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
// FlashInfer 通用工具头文件：
// 包含 CUDA 错误检查、模板分发宏、基础数学工具、设备属性查询和底层 PTX 辅助函数。
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "exception.h"

// 将宏参数展开为字符串。
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// 编译期开关：关闭 fp16 qk reduction 以减少二进制体积。
#ifndef FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
// Debug 版本下包装 CUDA 调用：
// 失败时打印详细错误信息并返回对应的 cudaError_t。
#define FLASHINFER_CUDA_CALL(func, ...)                                                           \
    {                                                                                             \
        cudaError_t e = (func);                                                                   \
        if (e != cudaSuccess) {                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                      << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
            return e;                                                                             \
        }                                                                                         \
    }
#else
// Release 版本下保持轻量：
// 失败时直接返回错误码，不额外打印日志。
#define FLASHINFER_CUDA_CALL(func, ...) \
    {                                   \
        cudaError_t e = (func);         \
        if (e != cudaSuccess) {         \
            return e;                   \
        }                               \
    }
#endif

// CUDA 调用检查宏：
// 如果调用失败，构造带文件名、行号和函数信息的异常消息并抛出。
#define FLASHINFER_CUDA_CHECK(func)                                                             \
    do {                                                                                        \
        cudaError_t e = (func);                                                                 \
        FLASHINFER_CHECK(e == cudaSuccess, "CUDA Error: ", cudaGetErrorString(e), " (", int(e), \
                         ") at ", __FILE__, ":", __LINE__, " in ", STR(func));                  \
    } while (0)

// 检查指针地址是否满足指定字节对齐要求。
#define FLASHINFER_CHECK_ALIGNMENT(ptr, size_bytes)                              \
    FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(ptr) % (size_bytes) == 0, #ptr, \
                     " must be aligned to ", (size_bytes), " bytes, got address ", (uintptr_t)(ptr))
// TMA 路径通常要求 128 字节对齐。
#define FLASHINFER_CHECK_TMA_ALIGNED(ptr) FLASHINFER_CHECK_ALIGNMENT(ptr, 128)

// 将运行时的 use_fp16_qk_reduction 分发为编译期常量。
// 当前这里只支持 false，传 true 会直接报错。
#define DISPATCH_USE_FP16_QK_REDUCTION(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, ...) \
    if (use_fp16_qk_reduction) {                                                          \
        FLASHINFER_ERROR("FP16_QK_REDUCTION disabled at compile time");                   \
    } else {                                                                              \
        constexpr bool USE_FP16_QK_REDUCTION = false;                                     \
        __VA_ARGS__                                                                       \
    }

// 将 num_mma_q 从运行时值映射为编译期常量，便于模板特化和静态优化。
#define DISPATCH_NUM_MMA_Q(num_mma_q, NUM_MMA_Q, ...)      \
    if (num_mma_q == 1) {                                  \
        constexpr size_t NUM_MMA_Q = 1;                    \
        __VA_ARGS__                                        \
    } else if (num_mma_q == 2) {                           \
        constexpr size_t NUM_MMA_Q = 2;                    \
        __VA_ARGS__                                        \
    } else {                                               \
        std::ostringstream err_msg;                        \
        err_msg << "Unsupported num_mma_q: " << num_mma_q; \
        FLASHINFER_ERROR(err_msg.str());                   \
    }

// 根据 max_mma_kv 选择不超过它的编译期常量 NUM_MMA_KV。
#define DISPATCH_NUM_MMA_KV(max_mma_kv, NUM_MMA_KV, ...)     \
    if (max_mma_kv >= 8) {                                   \
        constexpr size_t NUM_MMA_KV = 8;                     \
        __VA_ARGS__                                          \
    } else if (max_mma_kv >= 4) {                            \
        constexpr size_t NUM_MMA_KV = 4;                     \
        __VA_ARGS__                                          \
    } else if (max_mma_kv >= 2) {                            \
        constexpr size_t NUM_MMA_KV = 2;                     \
        __VA_ARGS__                                          \
    } else if (max_mma_kv >= 1) {                            \
        constexpr size_t NUM_MMA_KV = 1;                     \
        __VA_ARGS__                                          \
    } else {                                                 \
        std::ostringstream err_msg;                          \
        err_msg << "Unsupported max_mma_kv: " << max_mma_kv; \
        FLASHINFER_ERROR(err_msg.str());                     \
    }

// 将 cta_tile_q 分发成支持的固定 tile 大小。
#define DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)     \
    switch (cta_tile_q) {                                    \
    case 128: {                                              \
        constexpr uint32_t CTA_TILE_Q = 128;                 \
        __VA_ARGS__                                          \
        break;                                               \
    }                                                        \
    case 64: {                                               \
        constexpr uint32_t CTA_TILE_Q = 64;                  \
        __VA_ARGS__                                          \
        break;                                               \
    }                                                        \
    case 16: {                                               \
        constexpr uint32_t CTA_TILE_Q = 16;                  \
        __VA_ARGS__                                          \
        break;                                               \
    }                                                        \
    default: {                                               \
        std::ostringstream err_msg;                          \
        err_msg << "Unsupported cta_tile_q: " << cta_tile_q; \
        FLASHINFER_ERROR(err_msg.str());                     \
    }                                                        \
    }

// 将 GQA group size 分发成编译期常量。
#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
    if (group_size == 1) {                                   \
        constexpr size_t GROUP_SIZE = 1;                     \
        __VA_ARGS__                                          \
    } else if (group_size == 2) {                            \
        constexpr size_t GROUP_SIZE = 2;                     \
        __VA_ARGS__                                          \
    } else if (group_size == 3) {                            \
        constexpr size_t GROUP_SIZE = 3;                     \
        __VA_ARGS__                                          \
    } else if (group_size == 4) {                            \
        constexpr size_t GROUP_SIZE = 4;                     \
        __VA_ARGS__                                          \
    } else if (group_size == 8) {                            \
        constexpr size_t GROUP_SIZE = 8;                     \
        __VA_ARGS__                                          \
    } else {                                                 \
        std::ostringstream err_msg;                          \
        err_msg << "Unsupported group_size: " << group_size; \
        FLASHINFER_ERROR(err_msg.str());                     \
    }

// 将 mask mode 分发成编译期常量，减少后续模板代码中的运行时分支。
#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)               \
    switch (mask_mode) {                                            \
    case MaskMode::kNone: {                                         \
        constexpr MaskMode MASK_MODE = MaskMode::kNone;             \
        __VA_ARGS__                                                 \
        break;                                                      \
    }                                                               \
    case MaskMode::kCausal: {                                       \
        constexpr MaskMode MASK_MODE = MaskMode::kCausal;           \
        __VA_ARGS__                                                 \
        break;                                                      \
    }                                                               \
    case MaskMode::kCustom: {                                       \
        constexpr MaskMode MASK_MODE = MaskMode::kCustom;           \
        __VA_ARGS__                                                 \
        break;                                                      \
    }                                                               \
    case MaskMode::kMultiItemScoring: {                             \
        constexpr MaskMode MASK_MODE = MaskMode::kMultiItemScoring; \
        __VA_ARGS__                                                 \
        break;                                                      \
    }                                                               \
    default: {                                                      \
        std::ostringstream err_msg;                                 \
        err_msg << "Unsupported mask_mode: " << int(mask_mode);     \
        FLASHINFER_ERROR(err_msg.str());                            \
    }                                                               \
    }

// 将 head_dim 分发成编译期常量。
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)       \
    switch (head_dim) {                                  \
    case 64: {                                           \
        constexpr size_t HEAD_DIM = 64;                  \
        __VA_ARGS__                                      \
        break;                                           \
    }                                                    \
    case 128: {                                          \
        constexpr size_t HEAD_DIM = 128;                 \
        __VA_ARGS__                                      \
        break;                                           \
    }                                                    \
    case 256: {                                          \
        constexpr size_t HEAD_DIM = 256;                 \
        __VA_ARGS__                                      \
        break;                                           \
    }                                                    \
    case 512: {                                          \
        constexpr size_t HEAD_DIM = 512;                 \
        __VA_ARGS__                                      \
        break;                                           \
    }                                                    \
    default: {                                           \
        std::ostringstream err_msg;                      \
        err_msg << "Unsupported head_dim: " << head_dim; \
        FLASHINFER_ERROR(err_msg.str());                 \
    }                                                    \
    }

// 将 interleave 布尔值分发成编译期常量。
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
    if (interleave) {                                    \
        constexpr bool INTERLEAVE = true;                \
        __VA_ARGS__                                      \
    } else {                                             \
        constexpr bool INTERLEAVE = false;               \
        __VA_ARGS__                                      \
    }

// 将 rope 维度分发成支持的固定值。
#define DISPATCH_ROPE_DIM(rope_dim, ROPE_DIM, ...)             \
    switch (rope_dim) {                                        \
    case 16: {                                                 \
        constexpr uint32_t ROPE_DIM = 16;                      \
        __VA_ARGS__                                            \
        break;                                                 \
    }                                                          \
    case 32: {                                                 \
        constexpr uint32_t ROPE_DIM = 32;                      \
        __VA_ARGS__                                            \
        break;                                                 \
    }                                                          \
    case 64: {                                                 \
        constexpr uint32_t ROPE_DIM = 64;                      \
        __VA_ARGS__                                            \
        break;                                                 \
    }                                                          \
    case 128: {                                                \
        constexpr uint32_t ROPE_DIM = 128;                     \
        __VA_ARGS__                                            \
        break;                                                 \
    }                                                          \
    case 256: {                                                \
        constexpr uint32_t ROPE_DIM = 256;                     \
        __VA_ARGS__                                            \
        break;                                                 \
    }                                                          \
    default: {                                                 \
        std::ostringstream err_msg;                            \
        err_msg << "Unsupported ROPE_DIM: " << rope_dim;       \
        err_msg << ". Supported values: 16, 32, 64, 128, 256"; \
        err_msg << " in DISPATCH_ROPE_DIM";                    \
        FLASHINFER_ERROR(err_msg.str());                       \
    }                                                          \
    }

// 将位置编码模式分发成编译期常量。
#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)      \
    switch (pos_encoding_mode) {                                                   \
    case PosEncodingMode::kNone: {                                                 \
        constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;      \
        __VA_ARGS__                                                                \
        break;                                                                     \
    }                                                                              \
    case PosEncodingMode::kRoPELlama: {                                            \
        constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama; \
        __VA_ARGS__                                                                \
        break;                                                                     \
    }                                                                              \
    case PosEncodingMode::kALiBi: {                                                \
        constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi;     \
        __VA_ARGS__                                                                \
        break;                                                                     \
    }                                                                              \
    default: {                                                                     \
        std::ostringstream err_msg;                                                \
        err_msg << "Unsupported pos_encoding_mode: " << int(pos_encoding_mode);    \
        FLASHINFER_ERROR(err_msg.str());                                           \
    }                                                                              \
    }

// 将对齐向量大小分发成编译期常量。
// 常用于根据地址对齐情况选择不同宽度的向量化访存路径。
#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
    switch (aligned_vec_size) {                                            \
    case 16: {                                                             \
        constexpr size_t ALIGNED_VEC_SIZE = 16;                            \
        __VA_ARGS__                                                        \
        break;                                                             \
    }                                                                      \
    case 8: {                                                              \
        constexpr size_t ALIGNED_VEC_SIZE = 8;                             \
        __VA_ARGS__                                                        \
        break;                                                             \
    }                                                                      \
    case 4: {                                                              \
        constexpr size_t ALIGNED_VEC_SIZE = 4;                             \
        __VA_ARGS__                                                        \
        break;                                                             \
    }                                                                      \
    case 2: {                                                              \
        constexpr size_t ALIGNED_VEC_SIZE = 2;                             \
        __VA_ARGS__                                                        \
        break;                                                             \
    }                                                                      \
    case 1: {                                                              \
        constexpr size_t ALIGNED_VEC_SIZE = 1;                             \
        __VA_ARGS__                                                        \
        break;                                                             \
    }                                                                      \
    default: {                                                             \
        std::ostringstream err_msg;                                        \
        err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;   \
        FLASHINFER_ERROR(err_msg.str());                                   \
    }                                                                      \
    }

// 根据 GPU 计算能力决定 decode kernel 在 shared memory 上采用的 stage 数。
#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, ...) \
    if (compute_capacity.first >= 8) {                                                      \
        constexpr uint32_t NUM_STAGES_SMEM = 2;                                             \
        __VA_ARGS__                                                                         \
    } else {                                                                                \
        constexpr uint32_t NUM_STAGES_SMEM = 1;                                             \
        __VA_ARGS__                                                                         \
    }

namespace flashinfer {

template <typename T1, typename T2>
// 向上整除，返回 ceil(x / y)。
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
    return (x + y - 1) / y;
}

template <typename T1, typename T2>
// 将 x 向上对齐到 y 的整数倍。
__forceinline__ __device__ __host__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
    return ceil_div(x, y) * y;
}

template <typename T1, typename T2>
// 将 x 向下对齐到 y 的整数倍。
__forceinline__ __device__ __host__ constexpr T1 round_down(const T1 x, const T2 y) noexcept {
    return (x / y) * y;
}

// 获取当前 CUDA 设备的计算能力 (major, minor)。
inline std::pair<int, int> GetCudaComputeCapability() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    return std::make_pair(major, minor);
}

// 获取当前 CUDA 设备的 SM 数量。
// 结果会被缓存，并用原子变量保证并发访问安全。
// 注意：这里只缓存当前设备的值，因此默认假设一个进程主要服务单张 GPU。
inline int GetCudaMultiProcessorCount() {
    static std::atomic<int> sm_count{0};
    int cached = sm_count.load(std::memory_order_relaxed);
    if (cached == 0) {
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_id);
        cached = device_prop.multiProcessorCount;
        sm_count.store(cached, std::memory_order_relaxed);
    }
    return cached;
}

template <typename T>
// 调试辅助函数：
// 将设备数组拷回主机并打印，适合检查小规模中间结果。
inline void DebugPrintCUDAArray(T *device_ptr, size_t size, std::string prefix = "") {
    std::vector<T> host_array(size);
    std::cout << prefix;
    cudaMemcpy(host_array.data(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; ++i) {
        std::cout << host_array[i] << " ";
    }
    std::cout << std::endl;
}

// 根据平均 qo 长度和 head_dim 选择 decode 阶段使用的 CTA_TILE_Q。
// 这是一个启发式策略，用于平衡并行度与资源占用。
inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim) {
    if (avg_packed_qo_len > 64 && head_dim < 256) {
        return 128;
    } else {
        auto compute_capacity = GetCudaComputeCapability();
        if (compute_capacity.first >= 8) {
            // Ampere or newer
            if (avg_packed_qo_len > 16) {
                // avg_packed_qo_len <= 64
                return 64;
            } else {
                // avg_packed_qo_len <= 16
                return 16;
            }
        } else {
            // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
            return 64;
        }
    }
}

// 返回不小于 x 的最小 2 的幂。
inline int UpPowerOfTwo(int x) {
    // Returns the smallest power of two greater than or equal to x
    if (x <= 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// 将循环拆成两个阶段：
// 第一段 WITH_MASK=true，第二段 WITH_MASK=false。
// 常用于前几轮需要边界 mask，后几轮可以走无 mask 快路径的场景。
#define LOOP_SPLIT_MASK(iter, COND1, COND2, ...)           \
    {                                                      \
        _Pragma("unroll 1") for (; (COND1); (iter) -= 1) { \
            constexpr bool WITH_MASK = true;               \
            __VA_ARGS__                                    \
        }                                                  \
        _Pragma("unroll 1") for (; (COND2); (iter) -= 1) { \
            constexpr bool WITH_MASK = false;              \
            __VA_ARGS__                                    \
        }                                                  \
    }

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
    return (x > y) ? x - y : 0U;
}

// ======================= PTX Memory Utility Functions =======================
// 下面是一组基于 PTX 内联汇编的底层访存工具。
// 这里主要使用 non-atomic + cache streaming hint(cs) 的全局内存访问形式，
// 适合流式访存、数据只用一次或复用较低的场景。

/*!
 * \brief Get the lane ID within a warp (0-31)
 */
__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

/*!
 * \brief Non-atomic global load for int (4 bytes) with cache streaming hint
 */
__forceinline__ __device__ int ld_na_global_v1(const int *addr) {
    int val;
    asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

/*!
 * \brief Non-atomic global load for int2 (8 bytes) with cache streaming hint
 */
__forceinline__ __device__ int2 ld_na_global_v2(const int2 *addr) {
    int2 val;
    asm volatile("ld.global.cs.v2.b32 {%0, %1}, [%2];" : "=r"(val.x), "=r"(val.y) : "l"(addr));
    return val;
}

/*!
 * \brief Non-atomic global store for int (4 bytes) with cache streaming hint
 */
__forceinline__ __device__ void st_na_global_v1(int *addr, int val) {
    asm volatile("st.global.cs.b32 [%0], %1;" ::"l"(addr), "r"(val));
}

/*!
 * \brief Non-atomic global store for int2 (8 bytes) with cache streaming hint
 */
__forceinline__ __device__ void st_na_global_v2(int2 *addr, int2 val) {
    asm volatile("st.global.cs.v2.b32 [%0], {%1, %2};" ::"l"(addr), "r"(val.x), "r"(val.y));
}

/*!
 * \brief Prefetch data to L2 cache
 */
template <typename T>
__forceinline__ __device__ void prefetch_L2(const T *addr) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
}

// 交换两个 uint32_t 变量的值。
__device__ __forceinline__ void swap(uint32_t &a, uint32_t &b) {
    uint32_t tmp = a;
    a = b;
    b = tmp;
}

// 计算二维张量按行主序展开后的线性偏移。
__device__ __forceinline__ uint32_t dim2_offset(const uint32_t &dim_a, const uint32_t &idx_b,
                                                const uint32_t &idx_a) {
    return idx_b * dim_a + idx_a;
}

// 计算三维张量按行主序展开后的线性偏移。
__device__ __forceinline__ uint32_t dim3_offset(const uint32_t &dim_b, const uint32_t &dim_a,
                                                const uint32_t &idx_c, const uint32_t &idx_b,
                                                const uint32_t &idx_a) {
    return (idx_c * dim_b + idx_b) * dim_a + idx_a;
}

// 计算四维张量按行主序展开后的线性偏移。
__device__ __forceinline__ uint32_t dim4_offset(const uint32_t &dim_c, const uint32_t &dim_b,
                                                const uint32_t &dim_a, const uint32_t &idx_d,
                                                const uint32_t &idx_c, const uint32_t &idx_b,
                                                const uint32_t &idx_a) {
    return ((idx_d * dim_c + idx_c) * dim_b + idx_b) * dim_a + idx_a;
}

// 生成成员存在性检测模板。
// 例如 DEFINE_HAS_MEMBER(foo) 会生成 has_foo<T> 和 has_foo_v<T>，
// 用于在编译期判断类型 T 是否拥有成员 foo。
#define DEFINE_HAS_MEMBER(member)                                                                \
    template <typename T, typename = void>                                                       \
    struct has_##member : std::false_type {};                                                    \
    template <typename T>                                                                        \
    struct has_##member<T, std::void_t<decltype(std::declval<T>().member)>> : std::true_type {}; \
    template <typename T>                                                                        \
    inline constexpr bool has_##member##_v = has_##member<T>::value;

} // namespace flashinfer


#endif // FLASHINFER_UTILS_CUH_
