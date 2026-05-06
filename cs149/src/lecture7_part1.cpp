// lecture7_part1.cpp
// Stanford CS149 第7讲：GPU 架构与 CUDA 编程
// 第一部分：线程层次结构模拟 — 在 CPU 上模拟 grids、blocks 和 threads
//
// 本程序使用 C++ std::thread 模拟 CUDA 的线程层次结构，
// 演示 thread block 和 grid 如何以多维方式将工作映射到数据上，
// 类似 CUDA 的 <<<numBlocks, threadsPerBlock>>> 启动语法。
//
// CUDA 线程层次结构核心概念：
//   - Grid（网格）：由多个 thread block 组成的一维/二维/三维排列，
//     代表整个计算任务的划分。Grid 维度决定总共启动多少 block。
//   - Block（线程块）：包含一组 thread 的集合（最多 1024 个），
//     所有 block 内的 thread 可以共享 shared memory 并能通过
//     __syncthreads() 进行同步。一个 block 被调度到一个 SM 上执行。
//   - Thread（线程）：最小执行单元，每个 thread 拥有独立的寄存器
//     和程序计数器（program counter）。
//   - 全局索引计算公式：index = blockIdx.x * blockDim.x + threadIdx.x
//     将 block 索引和 block 内 thread 索引映射为全局数据索引。
//   - 上取整除法（Ceiling Division）：当数据大小不能被 block 大小
//     整除时，需要多分配一个 block 覆盖剩余元素：numBlocks = ⌈N/BLOCK_SIZE⌉
//
// 编译：g++ -std=c++17 -pthread lecture7_part1.cpp -o lecture7_part1
// 运行：./lecture7_part1

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <iomanip>

// ============================================================================
// 模拟 CUDA 数据类型
// CUDA 中使用 dim3 结构体指定 grid 和 block 的维度
// dim3(x, y, z) 支持最多三维的线程组织方式
// ============================================================================

struct dim3 {
    int x, y, z;
    dim3(int _x = 1, int _y = 1, int _z = 1) : x(_x), y(_y), z(_z) {}
};

// ============================================================================
// 配置参数
// MATRIX_WIDTH/MATRIX_HEIGHT：待处理矩阵的尺寸
// BLOCK_DIM_X/BLOCK_DIM_Y：每个 thread block 中包含的线程数量（二维排列）
// ============================================================================

constexpr int MATRIX_WIDTH  = 16;
constexpr int MATRIX_HEIGHT = 12;
constexpr int BLOCK_DIM_X   = 4;
constexpr int BLOCK_DIM_Y   = 3;

// ============================================================================
// 矩阵加法核函数：每个 "CUDA thread" 计算结果矩阵的一个元素
// 这是 device kernel 的等价实现
//
// 参数说明：
//   A, B: 输入矩阵
//   C: 输出矩阵（A + B 的结果）
//   blockIdx_x/y: 当前 block 在 grid 中的索引
//   threadIdx_x/y: 当前 thread 在 block 中的索引
//   blockDim_x/y: 每个 block 中的线程数
// ============================================================================

void matrixAddKernel(float A[][MATRIX_WIDTH],
                     float B[][MATRIX_WIDTH],
                     float C[][MATRIX_WIDTH],
                     int blockIdx_x, int blockIdx_y,
                     int threadIdx_x, int threadIdx_y,
                     int blockDim_x, int blockDim_y)
{
    // 通过 block 索引 + thread 索引计算全局索引
    // 等价于 CUDA 中的: int i = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x = blockIdx_x * blockDim_x + threadIdx_x;
    int global_y = blockIdx_y * blockDim_y + threadIdx_y;

    // 边界保护：超出矩阵范围的线程不执行任何操作
    // 这对应于上取整除法导致的"多余线程"场景
    if (global_x < MATRIX_WIDTH && global_y < MATRIX_HEIGHT) {
        C[global_y][global_x] = A[global_y][global_x] + B[global_y][global_x];
    }
}

// ============================================================================
// 模拟单个 thread block 的执行过程
// block 中每个 thread 计算自己被分配到的矩阵元素
// 在真实 CUDA 中，block 内的所有 thread 并发执行；
// 这里使用 std::thread 模拟并发
// ============================================================================

void executeThreadBlock(float A[][MATRIX_WIDTH],
                        float B[][MATRIX_WIDTH],
                        float C[][MATRIX_WIDTH],
                        int blockIdx_x, int blockIdx_y,
                        dim3 blockDim)
{
    // 每个 block 包含 blockDim.x * blockDim.y 个线程
    // 在 CUDA 硬件中这些线程会同时运行；此处使用 std::thread 模拟
    std::vector<std::thread> blockThreads;

    for (int ty = 0; ty < blockDim.y; ty++) {
        for (int tx = 0; tx < blockDim.x; tx++) {
            blockThreads.emplace_back(
                matrixAddKernel,
                A, B, C,
                blockIdx_x, blockIdx_y,
                tx, ty,
                blockDim.x, blockDim.y
            );
        }
    }

    // 等待 block 内所有线程完成（等价于 CUDA 的 __syncthreads()）
    // __syncthreads() 是 CUDA 的 block 级同步屏障，
    // 确保 block 内所有 thread 都执行到此处后才能继续
    for (auto& t : blockThreads) {
        t.join();
    }
}

// ============================================================================
// 模拟 GPU grid 启动：<<<numBlocks, threadsPerBlock>>>
// 在整个 grid 上启动 thread block — 各个 block 之间并发执行
// 在真实 CUDA 中，kernel 启动 <<<M, N>>> 会创建 M 个并发 block，
// 每个 block 包含 N 个线程
// ============================================================================

void cudaKernelLaunch(float A[][MATRIX_WIDTH],
                      float B[][MATRIX_WIDTH],
                      float C[][MATRIX_WIDTH],
                      dim3 numBlocks, dim3 threadsPerBlock)
{
    std::vector<std::thread> blockThreads;

    for (int by = 0; by < numBlocks.y; by++) {
        for (int bx = 0; bx < numBlocks.x; bx++) {
            // 每个 thread block 与其他 block 并发执行
            // 这是 CUDA 的核心并行模式：多个 block 同时在不同 SM 上运行
            blockThreads.emplace_back(
                executeThreadBlock,
                A, B, C,
                bx, by,
                threadsPerBlock
            );
        }
    }

    // 隐式 barrier：当所有 block 完成后，kernel 才算执行完毕
    // 在真实 GPU 中，这是由 hardware scheduler 自动管理的
    for (auto& t : blockThreads) {
        t.join();
    }
}

// ============================================================================
// 一维卷积模拟（第7讲示例）
// 演示具有数据重叠访问模式的 thread-to-element 映射
//
// 一维卷积概念（窗口大小为 3）：
//   output[i] = (input[i] + input[i+1] + input[i+2]) / 3.0
// 即计算每个位置及其邻域三个元素的滑动平均。
// 注意相邻线程之间存在数据重叠：thread i 和 thread i+1 共享 input[i+1] 和 input[i+2]
// ============================================================================

void convolution1D(const std::vector<float>& input,
                   std::vector<float>& output,
                   int totalElements,
                   int threadsPerBlk)
{
    // 模拟 CUDA 调用: convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, input, output)
    // 上取整除法计算所需 block 数量
    int numBlocks = (totalElements + threadsPerBlk - 1) / threadsPerBlk;

    std::vector<std::thread> blockThreads;
    const float* inPtr  = input.data();
    float*       outPtr = output.data();

    for (int blk = 0; blk < numBlocks; blk++) {
        blockThreads.emplace_back([=]() {
            for (int t = 0; t < threadsPerBlk; t++) {
                // 全局索引: blockIdx.x * blockDim.x + threadIdx.x
                int index = blk * threadsPerBlk + t;
                // 边界保护：跳过超出范围的线程
                if (index >= totalElements || index + 2 >= totalElements) continue;

                // 卷积窗口大小为 3（每个线程读取 3 个元素）
                float result = 0.0f;
                for (int i = 0; i < 3; i++) {
                    result += inPtr[index + i];
                }
                outPtr[index] = result / 3.0f;
            }
        });
    }

    for (auto& bt : blockThreads) {
        bt.join();
    }
}

// ============================================================================
// 打印矩阵的辅助函数
// ============================================================================

void printMatrix(const std::string& name, float mat[][MATRIX_WIDTH],
                 int height, int width)
{
    std::cout << "\n" << name << "：\n";
    for (int y = 0; y < height; y++) {
        std::cout << "  ";
        for (int x = 0; x < width; x++) {
            std::cout << std::setw(5) << std::fixed
                      << std::setprecision(0) << mat[y][x];
        }
        std::cout << "\n";
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第7讲 第一部分：CUDA 线程层次结构模拟\n";
    std::cout << "==================================================\n\n";

    // ----- 矩阵加法演示 -----
    std::cout << "--- 矩阵加法（二维线程层次结构）---\n";
    std::cout << "矩阵大小：" << MATRIX_HEIGHT << "×" << MATRIX_WIDTH << "\n";
    std::cout << "Block 大小：" << BLOCK_DIM_Y << "×" << BLOCK_DIM_X << "\n";

    // 分配矩阵内存
    float A[MATRIX_HEIGHT][MATRIX_WIDTH] = {};
    float B[MATRIX_HEIGHT][MATRIX_WIDTH] = {};
    float C[MATRIX_HEIGHT][MATRIX_WIDTH] = {};

    // 初始化矩阵 A 和 B
    // A 中元素值为行*宽度+列，B 中的元素为 A 的 10 倍
    for (int y = 0; y < MATRIX_HEIGHT; y++) {
        for (int x = 0; x < MATRIX_WIDTH; x++) {
            A[y][x] = y * MATRIX_WIDTH + x;
            B[y][x] = (y * MATRIX_WIDTH + x) * 10;
        }
    }

    // 计算 grid 维度（使用上取整除法处理非整数倍的情况）
    // 上取整公式: ⌈a/b⌉ = (a + b - 1) / b （整数除法）
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks((MATRIX_WIDTH  + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                   (MATRIX_HEIGHT + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    std::cout << "Grid：" << numBlocks.y << "×" << numBlocks.x
              << " 个 blocks\n";
    std::cout << "总线程数："
              << numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y
              << "（其中只有 " << MATRIX_HEIGHT * MATRIX_WIDTH
              << " 个元素需要计算）\n";

    cudaKernelLaunch(A, B, C, numBlocks, threadsPerBlock);

    printMatrix("矩阵 A", A, MATRIX_HEIGHT, MATRIX_WIDTH);
    printMatrix("矩阵 B", B, MATRIX_HEIGHT, MATRIX_WIDTH);
    printMatrix("矩阵 C = A + B", C, MATRIX_HEIGHT, MATRIX_WIDTH);

    // 验证结果正确性：逐一检查 C[y][x] == A[y][x] + B[y][x]
    bool correct = true;
    for (int y = 0; y < MATRIX_HEIGHT && correct; y++) {
        for (int x = 0; x < MATRIX_WIDTH && correct; x++) {
            if (C[y][x] != A[y][x] + B[y][x]) correct = false;
        }
    }
    std::cout << "\n验证结果：" << (correct ? "通过（PASSED）" : "失败（FAILED）") << "\n";

    // ----- 一维卷积演示 -----
    std::cout << "\n\n--- 一维卷积模拟 ---\n";

    constexpr int N = 20;
    std::vector<float> signal(N);
    for (int i = 0; i < N; i++) {
        signal[i] = static_cast<float>(i + 1);
    }

    std::vector<float> result(N - 2);
    // 每个 block 使用 8 个线程
    convolution1D(signal, result, N - 2, 8);

    std::cout << "输入信号：    ";
    for (float v : signal) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "卷积结果（窗口=3，滑动平均）：";
    for (float v : result) std::cout << std::setprecision(2) << v << " ";
    std::cout << "\n";

    // 手动验证：output[0] = (1+2+3)/3 = 2, output[1] = (2+3+4)/3 = 3, ...
    std::cout << "期望值：                      ";
    for (int i = 0; i < N - 2; i++) {
        float expected = (signal[i] + signal[i+1] + signal[i+2]) / 3.0f;
        std::cout << std::setprecision(2) << expected << " ";
    }
    std::cout << "\n";

    // ---- 总结 ----
    std::cout << "\n==================================================\n";
    std::cout << "演示的核心概念：\n";
    std::cout << "  - 二维线程层次结构：线程块网格（grid of thread blocks）\n";
    std::cout << "  - 全局索引计算：blockIdx*blockDim + threadIdx\n";
    std::cout << "  - 非整数倍大小的上取整除法处理\n";
    std::cout << "  - 通过 C++ 线程实现 block 的并发执行\n";
    std::cout << "  - 一维卷积：数据重叠访问模式\n";
    std::cout << "==================================================\n";

    return 0;
}
