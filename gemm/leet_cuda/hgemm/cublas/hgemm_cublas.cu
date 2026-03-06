/**
 * @file hgemm_cublas.cu
 * @brief 使用NVIDIA cuBLAS库实现半精度矩阵乘法(HGEMM)
 * 
 * 本文件提供基于cuBLAS的半精度矩阵乘法实现，包括：
 * 1. cuBLAS handle管理（初始化/销毁）
 * 2. 多种矩阵布局支持的GEMM操作
 * 3. 性能基准测试
 * 4. PyTorch扩展绑定（可选）
 * 
 * 支持的精度:
 * - FP16 (half): 16位浮点数
 * - BF16 (bfloat16): 16位浮点数（brain float）
 * - FP8: 8位浮点数（可选）
 * 
 * 编译选项:
 * - NO_CUBLAS_HGEMM_BIN: 仅编译PyTorch绑定，不编译二进制可执行文件
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>      // FP16半精度浮点数支持
#include <cuda_bf16.h>      // BF16脑浮点数支持
#include <cuda_fp8.h>       // FP8 8位浮点数支持
#include <mma.h>            // WMMA (Warp Matrix Multiply Accumulate) 指令
#include "cublas_v2.h"      // cuBLAS v2 API

/**
 * @brief 全局cuBLAS句柄
 * 
 * cuBLAS库需要通过句柄(handle)来管理库状态和配置。
 * 使用全局句柄可以避免重复创建的开销。
 * 
 * 设计说明:
 * - 静态全局变量在整个程序生命周期内有效
 * - 延迟初始化（首次使用时创建）
 * - 支持多线程（每个线程应使用独立句柄，本例使用单句柄简化）
 */
static cublasHandle_t g_handle = nullptr;

/**
 * @brief 初始化cuBLAS句柄
 * 
 * cuBLAS使用前必须创建句柄，类似于打开文件或建立连接。
 * 句柄包含了库的状态信息、配置和内部数据结构。
 * 
 * 初始化流程:
 * 1. 创建cuBLAS上下文句柄
 * 2. 设置数学模式为Tensor OP模式（启用Tensor Core加速）
 * 
 * @note CUBLAS_TENSOR_OP_MATH 启用了Tensor Core加速
 *       对于FP16/BF16矩阵乘法，使用Tensor Core可获得显著加速
 */
void init_cublas_handle() {
  if (g_handle == nullptr) {
    // 创建cuBLAS句柄
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cuBLAS handle: %d", status);
      exit(EXIT_FAILURE);
    }
    
    // 设置数学模式为Tensor OP模式
    // 这允许cuBLAS使用Tensor Core硬件加速器进行矩阵运算
    // Tensor Core是 Volta+ GPU上的专用矩阵运算单元
    status = cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to set cuBLAS Math Mode: %d", status);
      exit(EXIT_FAILURE);
    }
  }
}

/**
 * @brief 销毁cuBLAS句柄，释放资源
 * 
 * 与init_cublas_handle配对使用，确保资源不泄漏。
 * 应在程序结束前调用。
 */
void destroy_cublas_handle() {
  if (g_handle != nullptr) {
    cublasStatus_t status = cublasDestroy(g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to destroy cuBLAS handle: %d", status);
    }
    g_handle = nullptr;
  }
}

// =============================================================================
// cuBLAS GEMM 函数
// =============================================================================

/**
 * @brief cuBLAS NN模式矩阵乘法 (Row-Major)
 * 
 * 计算 C = α * A * B + β * C
 * 
 * 矩阵布局:
 * - A: M × K (行主序)
 * - B: K × N (行主序)
 * - C: M × N (行主序)
 * 
 * 参数说明:
 * - op(A) = A (不转置)
 * - op(B) = B (不转置)
 * - 形状: C[M,N] = A[M,K] × B[K,N]
 * 
 * @param A 输入矩阵A (M×K)
 * @param B 输入矩阵B (K×N)
 * @param C 输出矩阵C (M×N)，原地修改
 * @param M 矩阵A和C的行数
 * @param N 矩阵B和C的列数
 * @param K 矩阵A的列数，矩阵B的行数
 * 
 * @note alpha=1.0, beta=0.0 表示 C = A * B (不读取原始C)
 */
void cublas_tensor_op_nn(half *A, half *B, half *C, size_t M, size_t N, size_t K) {

  // alpha和beta是GEMM的缩放因子: C = α * A*B + β * C
  // alpha=1.0: 完全使用A*B的结果
  // beta=0.0: 忽略输入C的原始值（相当于C=0然后加上A*B）
  static half alpha = 1.0;
  static half beta = 0.0;

  // 延迟初始化句柄（如果尚未初始化）
  if (g_handle == nullptr) {
    init_cublas_handle();
  }

  /**
   * cublasGemmEx: cuBLAS的高级GEMM接口
   * 
   * 参数详解:
   * @param g_handle    cuBLAS句柄
   * @param opA         A矩阵的操作: CUBLAS_OP_N=不转置, T=转置, C=共轭转置
   * @param opB         B矩阵的操作
   * @param N           输出矩阵C的行数（注意：cuBLAS是列主序，所以是N而不是M）
   * @param M           输出矩阵C的列数
   * @param K           A的列数/B的行数（必须匹配）
   * @param &alpha      乘法系数
   * @param B           矩阵B的设备指针
   * @param CUDA_R_16F  B的数据类型: CUDA半精度浮点(FP16)
   * @param N           B的leading dimension (列数，因为行主序)
   * @param A           矩阵A的设备指针
   * @param CUDA_R_16F  A的数据类型
   * @param K           A的leading dimension (列数，因为行主序)
   * @param &beta       加法系数
   * @param C           矩阵C的设备指针
   * @param CUDA_R_16F  C的数据类型
   * @param N           C的leading dimension
   * @param CUBLAS_COMPUTE_16F  计算精度: FP16
   * @param CUBLAS_GEMM_DEFAULT_TENSOR_OP  GEMM算法选项: 默认使用Tensor OP
   * 
   * @note 注意cuBLAS的列主序约定与参数顺序:
   *       CUBLAS内部按列主序处理，所以参数顺序是 N, M 而不是 M, N
   *       这是因为 cublasGemmEx 源自列主序BLAS API
   */
  cublasGemmEx(g_handle, 
               CUBLAS_OP_N,    // op(A): 不转置
               CUBLAS_OP_N,    // op(B): 不转置
               N, M, K,        // 注意: N在前，M在后（cuBLAS列主序约定）
               &alpha, 
               B, CUDA_R_16F, N,   // B: K×N, ldb = N
               A, CUDA_R_16F, K,   // A: M×K, lda = K
               &beta,  
               C, CUDA_R_16F, N,   // C: M×N, ldc = N
               CUBLAS_COMPUTE_16F,    // 使用FP16计算
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // 使用Tensor Core
}

/**
 * @brief cuBLAS TN模式矩阵乘法 (A转置，B不转置)
 * 
 * 计算 C = α * A^T * B + β * C
 * 
 * 矩阵布局:
 * - A: K × M (行主序，但按列主序解释时为 M×K)
 * - B: K × N (行主序)
 * - C: M × N (行主序)
 * 
 * TN模式特点:
 * - A矩阵按转置处理: 输入K×M，视为M×K参与计算
 * - B矩阵不转置: 输入K×N，视为K×N参与计算
 * - 结果为M×N
 * 
 * @note TN模式在某些硬件上可能有更好的性能，因为内存访问模式更友好
 */
void cublas_tensor_op_tn(half *A, half *B, half *C, size_t M, size_t N, size_t K) {

  static half alpha = 1.0;
  static half beta = 0.0;

  if (g_handle == nullptr) {
    init_cublas_handle();
  }

  cublasGemmEx(g_handle, }

  /**
  emmEx TN模式参数说明:
   * 
   * op(A) = CUBLAS_OP_T (转置)
   * - 实际输入: A指针指向的矩阵是 K×K (行主序)
   * - 运算时视为: A^T 是 K×M (列主序)
   * - 参数 ldb = K (A按列主序解释时的leading dimension)
   * 
   * op(B) = CUBLAS_OP_N (不转置)
   * - 输入: B 是 K×N (行主序)
   * - 参数 ldb = K
   * 
   * 结果: C[M,N] = A^T[K,M] × B[K,N]
   */
  cublasGemmEx(g_handle, 
               CUBLAS_OP_T,    // op(A): 转置 A(M×K) -> A^T(K×M)
               CUBLAS_OP_N,    // op(B): 不转置
               N, M, K, 
               &alpha, 
               B, CUDA_R_16F, K,   // B: K×N, ldb = K (因为op(B)=N, 仍按行主序理解)
               A, CUDA_R_16F, K,   // A: K×M, lda = K (因为op(A)=T, 视为K×M)
               &beta,  
               C, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// =============================================================================
// 二进制可执行文件编译（默认）
// =============================================================================

// 当未定义NO_CUBLAS_HGEMM_BIN时，编译以下代码（二进制可执行文件）
#ifndef NO_CUBLAS_HGEMM_BIN

/**
 * @brief TN模式版本2 - 接收外部cuBLAS句柄
 * 
 * 与cublas_tensor_op_tn类似，但接受外部传入的handle。
 * 
 * 用途:
 * - 允许多线程场景下共享句柄
 * - 避免重复创建句柄的开销
 * - 更灵活的资源管理
 * 
 * @param handle 外部cuBLAS句柄（调用者负责创建和销毁）
 * @param A 输入矩阵A (K×M 行主序)
 * @param B 输入矩阵B (K×N 行主序)
 * @param C 输出矩阵C (M×N 行主序)
 * @param M 输出矩阵行数
 * @param N 输出矩阵列数
 * @param K 内部维度
 */
void cublas_tensor_op_tn_v2(cublasHandle_t handle, 
                            half *A, half *B, half *C,  
                            size_t M, size_t N, size_t K) {
  half alpha = 1.0;
  half beta = 0.0;

  // 使用传入的handle而非全局句柄
  cublasGemmEx(handle, 
               CUBLAS_OP_T, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_16F, K, 
               A, CUDA_R_16F, K, 
               &beta,  
               C, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

/**
 * @brief cuBLAS TN模式性能测试函数
 * 
 * 测试流程:
 * 1. 分配GPU内存
 * 2. 创建cuBLAS句柄并设置数学模式
 * 3. 预热10次迭代（让CUDA编译器JIT编译kernel）
 * 4. 使用CUDA Event计时
 * 5. 执行repeat次迭代并计算平均时间
 * 6. 释放资源
 * 
 * @param M 矩阵A的行数，矩阵C的行数
 * @param N 矩阵B的列数，矩阵C的列数
 * @param K 矩阵A的列数，矩阵B的行数
 * @param repeat 测试迭代次数
 * @return float 每次迭代的平均时间（秒）
 */
float perf_cublas_tn(int M, int N, int K, int repeat) {
  // 计算各矩阵所需的内存大小（字节）
  size_t size_a = M * K * sizeof(half);  // A: M×K
  size_t size_b = K * N * sizeof(half);  // B: K×N
  size_t size_c = M * N * sizeof(half);  // C: M×N

  // 在GPU上分配设备内存
  half *d_a, *d_b;
  half *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  // 创建cuBLAS句柄并配置
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  // =============================================================================
  // 预热阶段
  // =============================================================================
  // 第一次调用CUDA/cuBLAS函数时会触发JIT编译
  // 预热可以消除首次调用的编译开销，使后续测量更准确
  for (int i = 0; i < 10; ++i) {
    cublas_tensor_op_tn_v2(handle, d_a, d_b, d_c, M, N, K);
  }
  
  // 确保预热完成
  cudaDeviceSynchronize();

  // =============================================================================
  // 计时测试阶段
  // =============================================================================
  // 使用CUDA Event进行高精度设备端计时
  // CUDA Event是GPU上的时间戳，可以精确测量GPU操作时间
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  // 记录开始时间
  cudaEventRecord(start);

  // 执行指定的迭代次数
  for (int i = 0; i < repeat; i++) {
    cublas_tensor_op_tn_v2(handle, d_a, d_b, d_c, M, N, K);
  }

  // 记录结束时间
  cudaEventRecord(end);
  
  // 同步设备确保所有操作完成
  cudaDeviceSynchronize();
  cudaEventSynchronize(end);

  // 计算耗时
  float msec, sec;
  cudaEventElapsedTime(&msec, start, end);  // 获取毫秒级精度
  sec = msec / 1000.0 / repeat;             // 转换为秒并计算平均值

  // =============================================================================
  // 资源释放
  // =============================================================================
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy(handle);

  return sec;
}

/**
 * @brief 主函数 - cuBLAS性能基准测试
 * 
 * 测试流程:
 * 1. 初始化测试矩阵尺寸列表（64个不同尺寸）
 * 2. 对每个尺寸进行outer_repeat次测试
 * 3. 计算平均性能（TFLOPS）
 * 4. 输出结果
 */
int main(int argc, char *argv[]) {
  // 定义测试矩阵尺寸数量
  const int test_num = 64;
  int M_list[test_num];
  int N_list[test_num];
  int K_list[test_num];

  // 生成测试尺寸: 256, 512, 768, ..., 16384 (64个尺寸)
  // M, N, K 从256以256为步长增长到 256*64 = 16384
  for (int i = 0; i < test_num; i++) {
    M_list[i] = (i + 1) * 256;
    N_list[i] = (i + 1) * 256;
    K_list[i] = (i + 1) * 256;
  }

  // outer_repeat: 外层重复次数，用于多次测试取平均
  // inner_repeat: 内层重复次数，每次测量执行多次
  const int outer_repeat = 10, inner_repeat = 1;

  // 打印算法信息头
  printf("ALGO = cuBLAS CUBLAS_GEMM_DEFAULT_TENSOR_OP TN\n");

  // 对每个测试尺寸进行性能测试
  for (int j = 0; j < test_num; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];

    // 初始化计时统计变量
    double max_sec = 0.0;      // 最大耗时
    double min_sec = DBL_MAX;  // 最小耗时
    double total_sec = 0.0;    // 累计耗时

    // 多次测试取统计值
    for (int k = 0; k < outer_repeat; k++) {
      double this_sec = perf_cublas_tn(M, N, K, inner_repeat);
      max_sec = max(max_sec, this_sec);
      min_sec = min(min_sec, this_sec);
      total_sec += this_sec;
    }

    // =============================================================================
    // 计算性能指标
    // =============================================================================
    // TFLOPS计算公式: (M × N × K × 2) / 时间 / 10^12
    // 乘以2的原因: 矩阵乘法的乘加(MAD)操作包含一次乘法和一次加法
    // 
    // 示例: M=N=K=4096
    // FLOPs = 4096 × 4096 × 4096 × 2 ≈ 137.4 GFLOPs = 0.137 TFLOPS
    // (注: 实际计算量很大，上述数值是每秒钟的基准)
    
    // 1 TFLOPS = 10^12 FLOPS (每秒一万亿次浮点运算)
    // 参考: https://imgtec.eetrend.com/blog/2021/100062210.html
    double avg_sec = total_sec / outer_repeat;
    double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;

    // 打印结果
    // 格式: M N K = xxx xxx xxx, Time = min avg max, AVG Performance = xxx Tflops
    printf("M N K = %6d %6d %6d, ", M, N, K);
    printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
    printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
  }

  return 0;
}

// =============================================================================
// PyTorch扩展绑定（当定义NO_CUBLAS_HGEMM_BIN时编译）
// =============================================================================
#else

// --------------------- PyTorch bindings for custom kernel -----------------------

// 引入PyTorch C++扩展API头文件
#include <torch/types.h>
#include <torch/extension.h>

/**
 * @brief 宏: 将字符串转换为C字符串字面量
 * 
 * STRINGFY(x) 展开后是 "x"
 * 用于在宏中创建字符串
 */
#define STRINGFY(str) #str

/**
 * @brief 宏: 生成PyTorch扩展绑定
 * 
 * 这是一个模板宏，用于自动生成PyTorch的Python绑定函数。
 * 将C++函数注册到PyTorch的python模块中。
 * 
 * 使用示例:
 *   TORCH_BINDING_COMMON_EXTENSION(my_function)
 *   展开为: m.def("my_function", &my_function, "my_function");
 */
#define TORCH_BINDING_COMMON_EXTENSION(func)   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

/**
 * @brief 宏: 检查PyTorch张量的数据类型
 * 
 * 确保输入张量是指定的dtype，否则抛出错误。
 * 
 * @param T PyTorch张量
 * @param th_type PyTorch类型常量 (如 torch::kHalf)
 */
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

/**
 * @brief 宏: 检查PyTorch张量的形状
 * 
 * 确保张量是二维的，且形状匹配指定值。
 * 
 * @param T PyTorch张量
 * @param S0 期望的行数
 * @param S1 期望的列数
 */
#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

// =============================================================================
// PyTorch扩展函数 - NN模式
// =============================================================================

/**
 * @brief PyTorch绑定的HGEMM NN模式
 * 
 * 这是一个PyTorch扩展函数，可以直接从Python调用。
 * 
 * Python调用示例:
 *   import torch
 *   a = torch.randn(M, K, dtype=torch.float16, device='cuda')
 *   b = torch.randn(K, N, dtype=torch.float16, device='cuda')
 *   c = torch.empty(M, N, dtype=torch.float16, device='cuda')
 *   hgemm_cublas_tensor_op_nn(a, b, c)  # c = a @ b
 * 
 * @param a 输入张量A，形状(M, K)，dtype=half/float16
 * @param b 输入张量B，形状(K, N)，dtype=half/float16
 * @param c 输出张量C，形状(M, N)，dtype=half/float16
 * 
 * @note 函数内部使用reinterpret_cast将PyTorch指针转换为half*指针
 *       这是安全的，因为已通过CHECK_TORCH_TENSOR_DTYPE验证了类型
 */
void hgemm_cublas_tensor_op_nn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  // 类型检查: 确保输入输出都是FP16
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  // 获取矩阵维度
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  // 形状检查
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  // 调用cuBLAS实现
  // data_ptr()返回原始设备指针，reinterpret_cast转换为half*
  cublas_tensor_op_nn(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// =============================================================================
// PyTorch扩展函数 - TN模式
// =============================================================================

/**
 * @brief PyTorch绑定的HGEMM TN模式
 * 
 * 计算 C = A^T × B
 * 其中A是(K×M)，B是(K×N)，结果是(M×N)
 * 
 * Python调用示例:
 *   hgemm_cublas_tensor_op_tn(a, b, c)  # c = a.T @ b
 * 
 * @param a 输入张量A，形状(K, M)，dtype=half/float16
 * @param b 输入张量B，形状(K, N)，dtype=half/float16
 * @param c 输出张量C，形状(M, N)，dtype=half/float16
 */
void hgemm_cublas_tensor_op_tn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  // 类型检查
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  // 获取维度
  const int M = a.size(0);   // 注意：这里和NN模式不同
  const int K = a.size(1);   // TN模式下A是K×M
  const int N = b.size(1); 
  // 形状检查
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  // 调用cuBLAS实现
  cublas_tensor_op_tn(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}
#endif
