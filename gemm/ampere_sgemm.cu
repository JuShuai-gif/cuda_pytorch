#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j + p * n];
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}


__device__ __forceinline__ 
uint32_t smem_u32addr(const void* smem_ptr){
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
        "cvta.to.shared.u64 u64addr, %1;\n"
        "cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr)
        : "l"(smem_ptr)
    );
    return addr;
}

__device__ __forceinline__ 
void ldgsts32(const uint32_t &smem_addr,
            const void*gmem_ptr,
            bool guard){
    asm volatile(
        "{.reg .pred p;\n"
        " step.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4;}\n"
#else
        " @p cp.async.ca.shared.global [%0], [%1], 4;}\n"
#endif
        : : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldgsts32(const uint32_t &smem_addr,
              const void *gmem_ptr,
              const uint32_t &src_size,
              bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;}\n"
#else
        " @p cp.async.ca.shared.global [%0], [%1], 4, %2;}\n"
#endif
        : : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size), "r"((int)guard)
    );
}


__device__ __forceinline__ 
void ldgsts_commit(){
    asm volatile("cp.async.wait_all;\n"::);
}

__device__ __forceinline__ 
void stg32(const float &reg,void* ptr,bool guard){
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p,%2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr),"f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile(
        "ld.shared.v4.f64 {%0, %1, %2, %3},[%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag
{
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[16][8],int tile_x,int tile_y){
        #pragma unroll
        for (int i = 0; i < 4; ++i){
            #pragma unroll
            for (int j = 0; j < 4; ++j){
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            } 
        }
    }
};


__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
                float* C_stg_ptr,
                const float* C_lds_ptr,
                uint32_t C_sts_addr,
                uint32_t m,
                uint32_t n,
                uint32_t m_idx,
                uint32_t n_idx){
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 9 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 36],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n256k8
 * warp tile: m64n64k8
 * thread tile: m16n8k8
 * thread fragment:
 *     matrixA: 16x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                         256
 *                    --|---------------------------------------|
 *             B_tile  8|                                       |
 *                    --|---------------------------------------|
 *
 *  A_tile   | 8 |      |    64   |
 *         --|---|    --|---------|---------|---------|---------|
 *           |   |      |         |         |         |         |
 *           |   |    64|  warp0  |  warp1  |  warp2  |  warp3  |
 *           |   |      |         |         |         |         |
 *        128|   |    --|---------|---------|---------|---------|
 *           |   |      |         |         |         |         |
 *           |   |      |  warp4  |  warp5  |  warp6  |  warp7  |
 *           |   |      |         |         |         |         |
 *         --|---|      |---------|---------|---------|---------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */












































