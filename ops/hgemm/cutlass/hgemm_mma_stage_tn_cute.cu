#include <__clang_cuda_runtime_wrapper.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>


template<
    typename T,
    int BM,
    int BN,
    int BK,
    int kStage,
    typename TiledMMA,
    typename G2SCopyA,
    typename G2SCopyB,
    typename SmemLayoutA,
    typename SmemLayoutB,
    typename SmemLayoutC,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB,
    typename R2SCopyAtomC,
    typename S2GCopyAtomC,
    typename S2GCopyC,
    const bool BlockSwizeele>
    __global__ void hgemm_mma_stages_block_swizzle_tn_cute_kernel(
        T* Aptr,T* Bptr,T* Dptr,int m,int n,int k
    ){
        using namespace cute;

        extern __shared__ T shm_data[];

        


    }


















