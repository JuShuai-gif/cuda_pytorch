#include <cublasLt.h>

// ============================================================================
// 文件: cublaslt_matmul.cu
// 功能: 使用cuBLASLt库进行BF16矩阵乘法的示例代码
// 说明: 
//   - cuBLASLt是NVIDIA的轻量级cuBLAS库，提供更灵活的矩阵乘法配置
//   - 支持混合精度计算（如BF16输入，FP32累加）
//   - 提供自动算法选择和调优功能
//   - 需要CUDA 10.0+和计算能力7.0+（支持Tensor Core）
// ============================================================================

// Random code snippets to use while development
// cuBLASLt工作空间大小：32MB，用于存储中间结果和算法特定的数据
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;        // GPU工作空间指针
cublasLtHandle_t cublaslt_handle;       // cuBLASLt句柄

// ============================================================================
// 函数: initCublasLt
// 功能: 初始化cuBLASLt库和GPU工作空间
// 说明:
//   - 创建cuBLASLt句柄，用于后续所有cuBLASLt操作
//   - 分配GPU工作空间，用于算法执行和中间结果存储
//   - 必须在调用任何cuBLASLt函数之前调用此函数
// 示例:
//   initCublasLt(); // 程序开始时调用一次
// ============================================================================
void initCublasLt() {
    cublasLtCreate(&cublaslt_handle);  // 创建cuBLASLt句柄
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));    // 分配GPU工作空间
}

// ============================================================================
// 函数: runCublasMatmulBF16
// 功能: 使用cuBLASLt执行BF16矩阵乘法 C = A × B
// 参数:
//   - M: 矩阵A的行数，矩阵C的行数
//   - N: 矩阵B的列数，矩阵C的列数
//   - K: 矩阵A的列数，矩阵B的行数
//   - A: 输入矩阵A (M×K, BF16格式，行主序)
//   - B: 输入矩阵B (K×N, BF16格式，行主序)
//   - C: 输出矩阵C (M×N, BF16格式，行主序)
// 说明:
//   - 使用FP32累加精度（CUBLAS_COMPUTE_32F）进行BF16矩阵乘法
//   - 自动选择最优算法（heuristic search）
//   - 支持Tensor Core加速（需要Volta+架构）
//   - 要求输入输出指针16字节对齐以获得最佳性能
// 示例:
//   int M=1024, N=1024, K=1024;
//   bf16 *A, *B, *C;
//   cudaMalloc(&A, M*K*sizeof(bf16));
//   cudaMalloc(&B, K*N*sizeof(bf16));
//   cudaMalloc(&C, M*N*sizeof(bf16));
//   runCublasMatmulBF16(M, N, K, A, B, C);
// ============================================================================
void runCublasMatmulBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
      // 检查内存对齐（某些模式支持非对齐，但对齐可获得最佳性能）
      // cuBLASLt要求16字节对齐以充分利用Tensor Core和内存带宽
      if(((uintptr_t)A % 16) != 0 || ((uintptr_t)B % 16) != 0 || ((uintptr_t)C % 16) != 0) {
          printf("All cuBLASLt pointers must be aligned!\n");
          exit(EXIT_FAILURE);
      }
  
      // create the operation descriptor
      cublasLtMatmulDesc_t operationDesc;
      cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  
      int returnedResults = 0;
      cublasLtMatmulPreference_t preference;
      cublasLtMatmulHeuristicResult_t heuristic;
  
      cublasOperation_t opNoTranspose = CUBLAS_OP_N;
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opNoTranspose, sizeof(opNoTranspose));
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose));
  
      // define matrix layouts
      cublasLtMatrixLayout_t ALayout;
      cublasLtMatrixLayout_t BLayout;
      cublasLtMatrixLayout_t CLayout;
      cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_16BF, M, K, M);
      cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_16BF, K, N, K);
      
      // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
      cublasLtMatrixLayoutCreate(&CLayout, CUDA_R_16BF, M, N, M);
  
      // create a preference handle with specified max workspace
      cublasLtMatmulPreferenceCreate(&preference);
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTE,
                                                       &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));
  
      // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
      cublasDataType_t scale_type = CUDA_R_32F;
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
  
      // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
      cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, CLayout,
                                     preference, 1, &heuristic, &returnedResults);
      if (returnedResults == 0) {
          printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d", N, M, K);
          exit(EXIT_FAILURE);
      }
  
      // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
      float alpha = 1, beta = 0;
  
      // call the matmul
      cublasLtMatmul(cublaslt_handle, operationDes,
                                 &alpha, A, ALayout, B, BLayout, &beta, C, CLayout, C, CLayout,
                                 &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, 0));
  
      // cleanups
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatmulDescDestroy(operationDesc);
      cublasLtMatrixLayoutDestroy(ALayout);
      cublasLtMatrixLayoutDestroy(BLayout);
      cublasLtMatrixLayoutDestroy(CLayout);
      cudaCheck(cudaGetLastError());
  }