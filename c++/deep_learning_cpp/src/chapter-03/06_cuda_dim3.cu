/*
 * ================================================================
 * 文件名: 06_cuda_dim3.cu
 * 第 3 章：CUDA GPU 加速深度学习
 * 主题: dim3 多维网格与线程块配置
 *
 * 演示内容:
 *   1. 查询并打印最大维度限制 (表 3.1)
 *   2. 2D 核函数: 16x16 矩阵加法
 *   3. 3D 核函数: 4x8x8 三维体元处理
 *   4. 核函数启动与结果验证
 * ================================================================
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>

// ================================================================
// CUDA 错误检查工具函数
// ================================================================

inline void checkCudaError(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA 错误: %s (%s:%d)\n",
                     cudaGetErrorString(result), file, line);
        std::exit(result);
    }
}
#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

// ================================================================
// 2D 核函数: 二维矩阵加法
//   每个线程处理矩阵中的一个元素
//   blockIdx.x/y 确定当前线程块的列/行位置
//   threadIdx.x/y 确定线程在块内的列/行位置
// ================================================================

__global__ void matrixAdd2D(const float *a, const float *b, float *c,
                            int width, int height) {
    // 计算线程的全局 2D 坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引 (x 方向)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引 (y 方向)

    // 边界检查，防止越界访问
    if (row < height && col < width) {
        int idx = row * width + col; // 展平为 1D 索引
        c[idx] = a[idx] + b[idx];    // 执行矩阵加法
    }
}

// ================================================================
// 3D 核函数: 三维体元数据复制（演示 3D 索引模式）
//   模拟三维数据块的处理
//   blockIdx/threadIdx 的 x、y、z 分别对应三个空间维度
// ================================================================

__global__ void volumeProcess3D(const float *input, float *output,
                                int dimX, int dimY, int dimZ) {
    // 计算线程的全局 3D 坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X 维度
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y 维度
    int z = blockIdx.z * blockDim.z + threadIdx.z; // Z 维度

    // 边界检查
    if (x < dimX && y < dimY && z < dimZ) {
        // 展平 3D 索引: idx = z * (dimY * dimX) + y * dimX + x
        int idx = (z * dimY + y) * dimX + x;
        // 模拟处理: 对每个体元做平方运算
        output[idx] = input[idx] * input[idx];
    }
}

// ================================================================
// 辅助函数: 打印设备维度限制 (对应书中表 3.1)
// ================================================================

void printDeviceLimits(const cudaDeviceProp &prop) {
    std::printf("// ================================================================\n");
    std::printf("// 表 3.1: CUDA 设备维度限制\n");
    std::printf("// ================================================================\n");

    // ---- 网格 (Grid) 最大维度 ----
    std::printf("网格最大维度 (每个方向的线程块数):\n");
    std::printf("  gridDim.x 最大值 = %u        (理论: 2^31-1 = 2147483647)\n",
                static_cast<unsigned int>(prop.maxGridSize[0]));
    std::printf("  gridDim.y 最大值 = %u        (理论: 65535)\n",
                static_cast<unsigned int>(prop.maxGridSize[1]));
    std::printf("  gridDim.z 最大值 = %u        (理论: 65535)\n",
                static_cast<unsigned int>(prop.maxGridSize[2]));

    std::printf("\n");

    // ---- 线程块 (Block) 最大维度 ----
    std::printf("线程块最大维度 (每个方向的线程数):\n");
    std::printf("  blockDim.x 最大值 = %u       (理论: 1024)\n",
                static_cast<unsigned int>(prop.maxThreadsDim[0]));
    std::printf("  blockDim.y 最大值 = %u       (理论: 1024)\n",
                static_cast<unsigned int>(prop.maxThreadsDim[1]));
    std::printf("  blockDim.z 最大值 = %u       (理论: 64)\n",
                static_cast<unsigned int>(prop.maxThreadsDim[2]));

    std::printf("\n");

    // ---- 线程块总线程数限制 ----
    std::printf("线程块内总线程数上限:\n");
    std::printf("  blockDim.x * blockDim.y * blockDim.z ≤ %u\n",
                prop.maxThreadsPerBlock);

    std::printf("// ================================================================\n\n");
}

// ================================================================
// main 函数
// ================================================================

int main() {
    std::printf("===== dim3 多维网格与线程块配置演示 =====\n\n");

    // ---------- 查询设备属性 ----------
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    std::printf("检测到 %d 个 CUDA 设备\n\n", dev_count);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("当前设备: %s\n\n", prop.name);

    // ================================================================
    // 第一部分: 打印维度限制
    // ================================================================
    printDeviceLimits(prop);

    // ================================================================
    // 第二部分: 2D 核函数 - 16x16 矩阵加法
    // ================================================================
    std::printf("// ================================================================\n");
    std::printf("// 第二部分: 2D 矩阵加法 - dim3(16, 16, 1) 线程/块\n");
    std::printf("// ================================================================\n\n");

    const int MAT_SIZE = 16;                      // 16x16 矩阵
    const int MAT_ELEMENTS = MAT_SIZE * MAT_SIZE; // 总元素数 = 256
    const size_t mat_bytes = MAT_ELEMENTS * sizeof(float);

    // ---- 主机端数据分配与初始化 ----
    float *h_mat_a = new float[MAT_ELEMENTS];
    float *h_mat_b = new float[MAT_ELEMENTS];
    float *h_mat_c = new float[MAT_ELEMENTS];

    for (int row = 0; row < MAT_SIZE; ++row) {
        for (int col = 0; col < MAT_SIZE; ++col) {
            int idx = row * MAT_SIZE + col;
            h_mat_a[idx] = static_cast<float>(row + col);         // 行+列
            h_mat_b[idx] = static_cast<float>(row * 10.0f + col); // 行*10+列
        }
    }

    // ---- GPU 内存分配 ----
    float *d_mat_a = nullptr, *d_mat_b = nullptr, *d_mat_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mat_a, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_mat_b, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_mat_c, mat_bytes));

    // ---- 数据拷贝到 GPU ----
    CUDA_CHECK(cudaMemcpy(d_mat_a, h_mat_a, mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_b, h_mat_b, mat_bytes, cudaMemcpyHostToDevice));

    // ---- 2D 核函数启动 ----
    // 线程块维度: 16x16 线程/块
    dim3 block2D(16, 16, 1);
    // 网格维度: 1x1 块（因为矩阵就是 16x16，一个块刚好覆盖）
    dim3 grid2D(1, 1, 1);

    std::printf("2D 核函数配置:\n");
    std::printf("  矩阵大小: %d x %d = %d 个元素\n", MAT_SIZE, MAT_SIZE, MAT_ELEMENTS);
    std::printf("  网格: dim3(%d, %d, %d) = %d 个线程块\n",
                grid2D.x, grid2D.y, grid2D.z,
                grid2D.x * grid2D.y * grid2D.z);
    std::printf("  线程块: dim3(%d, %d, %d) = %d 个线程/块\n",
                block2D.x, block2D.y, block2D.z,
                block2D.x * block2D.y * block2D.z);
    std::printf("  总线程数: %d\n\n",
                grid2D.x * grid2D.y * grid2D.z * block2D.x * block2D.y * block2D.z);

    // 启动核函数（网格、线程块均使用 dim3）
    matrixAdd2D<<<grid2D, block2D>>>(d_mat_a, d_mat_b, d_mat_c,
                                     MAT_SIZE, MAT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 取回结果 ----
    CUDA_CHECK(cudaMemcpy(h_mat_c, d_mat_c, mat_bytes, cudaMemcpyDeviceToHost));

    // ---- 验证 2D 结果 ----
    std::printf("2D 矩阵加法结果验证 (前 4 行, 前 4 列):\n");
    std::printf("  期望: A[row+col] + B[row*10+col]\n");
    int errors_2d = 0;
    for (int row = 0; row < MAT_SIZE; ++row) {
        for (int col = 0; col < MAT_SIZE; ++col) {
            int idx = row * MAT_SIZE + col;
            float expected = h_mat_a[idx] + h_mat_b[idx];
            float diff = std::fabs(h_mat_c[idx] - expected);
            if (diff > 1e-5f) {
                if (errors_2d < 5) { // 只打印前 5 个错误
                    std::printf("  ✗ [%d][%d]: 期望 %.1f, 实际 %.1f\n",
                                row, col, expected, h_mat_c[idx]);
                }
                ++errors_2d;
            } else if (row < 4 && col < 4) {
                // 打印前 4x4 的正确结果样例
                std::printf("  ✓ [%d][%d]: %.1f + %.1f = %.1f\n",
                            row, col, h_mat_a[idx], h_mat_b[idx], h_mat_c[idx]);
            }
        }
    }
    if (errors_2d == 0) {
        std::printf("  全部 %d 个元素验证通过 ✓\n", MAT_ELEMENTS);
    } else {
        std::printf("  发现 %d 个错误 ✗\n", errors_2d);
    }
    std::printf("\n");

    // ---- 清理 2D GPU 内存 ----
    CUDA_CHECK(cudaFree(d_mat_a));
    CUDA_CHECK(cudaFree(d_mat_b));
    CUDA_CHECK(cudaFree(d_mat_c));
    delete[] h_mat_a;
    delete[] h_mat_b;
    delete[] h_mat_c;

    // ================================================================
    // 第三部分: 3D 核函数 - 4x8x8 三维体元处理
    // ================================================================
    std::printf("// ================================================================\n");
    std::printf("// 第三部分: 3D 体元处理 - dim3(8, 8, 4) 线程/块\n");
    std::printf("// ================================================================\n\n");

    const int DIM_X = 8;                               // X 维度大小
    const int DIM_Y = 8;                               // Y 维度大小
    const int DIM_Z = 4;                               // Z 维度大小
    const int VOLUME_ELEMENTS = DIM_X * DIM_Y * DIM_Z; // 8*8*4 = 256
    const size_t vol_bytes = VOLUME_ELEMENTS * sizeof(float);

    // ---- 主机端数据分配与初始化 ----
    float *h_vol_in = new float[VOLUME_ELEMENTS];
    float *h_vol_out = new float[VOLUME_ELEMENTS];
    float *h_vol_ref = new float[VOLUME_ELEMENTS]; // CPU 参考结果

    for (int z = 0; z < DIM_Z; ++z) {
        for (int y = 0; y < DIM_Y; ++y) {
            for (int x = 0; x < DIM_X; ++x) {
                int idx = (z * DIM_Y + y) * DIM_X + x;
                // 为每个体元赋一个有意义的值: z*100 + y*10 + x
                float val = static_cast<float>(z * 100 + y * 10 + x);
                h_vol_in[idx] = val;
                h_vol_ref[idx] = val * val; // x^2
            }
        }
    }

    // ---- GPU 内存分配 ----
    float *d_vol_in = nullptr, *d_vol_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vol_in, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_vol_out, vol_bytes));

    // ---- 数据拷贝到 GPU ----
    CUDA_CHECK(cudaMemcpy(d_vol_in, h_vol_in, vol_bytes, cudaMemcpyHostToDevice));

    // ---- 3D 核函数启动 ----
    // 线程块维度: 8x8x4 线程/块 (= 256 线程 ≤ 1024 限制)
    dim3 block3D(8, 8, 4);
    // 网格维度: 每个维度一个块即可覆盖 8x8x4 的数据
    dim3 grid3D(1, 1, 1);

    std::printf("3D 核函数配置:\n");
    std::printf("  体元范围: %d x %d x %d = %d 个体元\n",
                DIM_X, DIM_Y, DIM_Z, VOLUME_ELEMENTS);
    std::printf("  网格: dim3(%d, %d, %d) = %d 个线程块\n",
                grid3D.x, grid3D.y, grid3D.z,
                grid3D.x * grid3D.y * grid3D.z);
    std::printf("  线程块: dim3(%d, %d, %d) = %d 个线程/块\n",
                block3D.x, block3D.y, block3D.z,
                block3D.x * block3D.y * block3D.z);
    std::printf("  总线程数: %d\n",
                grid3D.x * grid3D.y * grid3D.z * block3D.x * block3D.y * block3D.z);

    // 验证线程块不超限
    if (block3D.x * block3D.y * block3D.z > prop.maxThreadsPerBlock) {
        std::printf("  ✗ 线程块大小超限! 最大允许 %d\n", prop.maxThreadsPerBlock);
        std::exit(1);
    }
    std::printf("  ✓ 线程块大小在限制范围内 (≤%d)\n\n",
                prop.maxThreadsPerBlock);

    // 启动 3D 核函数
    volumeProcess3D<<<grid3D, block3D>>>(d_vol_in, d_vol_out,
                                         DIM_X, DIM_Y, DIM_Z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 取回结果 ----
    CUDA_CHECK(cudaMemcpy(h_vol_out, d_vol_out, vol_bytes, cudaMemcpyDeviceToHost));

    // ---- 验证 3D 结果 ----
    std::printf("3D 体元处理结果验证 (平方运算):\n");
    std::printf("  坐标格式: (z, y, x) - 输入值 -> 输出值 = 输入^2\n");
    int errors_3d = 0;
    for (int z = 0; z < DIM_Z; ++z) {
        for (int y = 0; y < DIM_Y; ++y) {
            for (int x = 0; x < DIM_X; ++x) {
                int idx = (z * DIM_Y + y) * DIM_X + x;
                float diff = std::fabs(h_vol_out[idx] - h_vol_ref[idx]);
                if (diff > 1e-5f) {
                    if (errors_3d < 5) {
                        std::printf("  ✗ (%d,%d,%d): %.0f -> %.0f, 期望 %.0f\n",
                                    z, y, x,
                                    h_vol_in[idx], h_vol_out[idx], h_vol_ref[idx]);
                    }
                    ++errors_3d;
                }
            }
        }
    }
    if (errors_3d == 0) {
        std::printf("  全部 %d 个体元验证通过 ✓\n", VOLUME_ELEMENTS);
    } else {
        std::printf("  发现 %d 个错误 ✗\n", errors_3d);
    }

    // ---- 打印 3D 结果样例 ----
    std::printf("\n3D 体元处理样例 (z=0 层的 8x8 结果):\n");
    for (int y = 0; y < DIM_Y; ++y) {
        std::printf("  y=%d |", y);
        for (int x = 0; x < DIM_X; ++x) {
            int idx = (0 * DIM_Y + y) * DIM_X + x; // z=0
            std::printf(" %4.0f", h_vol_out[idx]);
        }
        std::printf("\n");
    }

    std::printf("\n3D 体元处理样例 (z=3 层的 8x8 结果):\n");
    for (int y = 0; y < DIM_Y; ++y) {
        std::printf("  y=%d |", y);
        for (int x = 0; x < DIM_X; ++x) {
            int idx = (3 * DIM_Y + y) * DIM_X + x; // z=3
            std::printf(" %4.0f", h_vol_out[idx]);
        }
        std::printf("\n");
    }

    // 展示 3D 索引的计算模式
    std::printf("\n3D 到 1D 索引映射示例:\n");
    std::printf("  idx = z * (dimY * dimX) + y * dimX + x\n");
    std::printf("  (2, 3, 5) -> idx = 2 * (%d * %d) + 3 * %d + 5 = %d\n",
                DIM_Y, DIM_X, DIM_X,
                2 * DIM_Y * DIM_X + 3 * DIM_X + 5);

    // ---- 清理 3D GPU 内存 ----
    CUDA_CHECK(cudaFree(d_vol_in));
    CUDA_CHECK(cudaFree(d_vol_out));
    delete[] h_vol_in;
    delete[] h_vol_out;
    delete[] h_vol_ref;

    // ================================================================
    // 完成
    // ================================================================
    std::printf("\n===== 演示完成: dim3 2D/3D 核函数配置全部通过 =====\n");
    std::printf("  要点回顾:\n");
    std::printf("    - gridDim 决定网格中的线程块数量 (dim3)\n");
    std::printf("    - blockDim 决定每个线程块中的线程数量 (dim3)\n");
    std::printf("    - 2D: threadIdx.x/y + blockIdx.x/y -> 矩阵索引\n");
    std::printf("    - 3D: threadIdx.x/y/z + blockIdx.x/y/z -> 体元索引\n");

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
