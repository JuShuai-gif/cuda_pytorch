/*
 * 01_convolution_eigen.cpp - 第 6 章：卷积神经网络
 * 基于 im2col + Eigen 矩阵乘法的优化卷积（对应原书第 176-178 页）
 *
 * 演示内容：
 *   1. im2col：将滑动窗口操作转换成矩阵乘法
 *   2. OptimizedConvolutionalLayer：一行矩阵乘法替代所有嵌套循环
 *   3. BLAS/SIMD 加速：filters * inputMatrix 利用硬件优化
 *   4. 与 CPU 嵌套循环的耗时对比
 *
 *   // 关键：下面这一行代码替代了 00 文件中的全部四层嵌套循环
 *   // 利用 Eigen 调用 BLAS 和 SIMD 指令获得 ~5-6 倍加速
 */

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>

using Matrix = Eigen::MatrixXd;

/* ======================== im2col 矩阵化展开 ============================= */
/*
 * 将 2D 输入矩阵按滑动窗口展开为列向量组成的矩阵。
 * 每个窗口 (kSize × kSize) 拉成一列，共 outputRows × outputCols 列。
 * 返回矩阵尺寸：(kSize²) × (outputRows × outputCols)
 */
Matrix im2col(const Matrix &input, int kSize, int stride) {
    int inputRows = static_cast<int>(input.rows());
    int inputCols = static_cast<int>(input.cols());

    // 输出特征图尺寸
    int outputRows = (inputRows - kSize) / stride + 1;
    int outputCols = (inputCols - kSize) / stride + 1;
    int patchSize = kSize * kSize;            // 每个窗口的像素数
    int numPatches = outputRows * outputCols; // 总窗口数

    // 结果矩阵：每列存储一个展开后的窗口
    Matrix result(patchSize, numPatches);

    int col = 0; // 列索引（窗口序号）
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            // 提取以 (i*stride, j*stride) 为起点的 kSize×kSize 窗口
            int row = 0;
            for (int k = 0; k < kSize; ++k) {
                for (int l = 0; l < kSize; ++l) {
                    result(row++, col) = input(i * stride + k,
                                               j * stride + l);
                }
            }
            ++col;
        }
    }
    return result;
}

/* ================= OptimizedConvolutionalLayer ========================== */
class OptimizedConvolutionalLayer {
private:
    Matrix filters; // 形状：[outChannels] × [inChannels × kSize × kSize]
    int kernelSize;
    int stride;
    int inChannels;
    int outChannels;

public:
    /* ------- 构造函数：随机初始化卷积核矩阵 ----------- */
    OptimizedConvolutionalLayer(int inCh, int outCh,
                                int kSize, int strideSize) : kernelSize(kSize), stride(strideSize),
                                                             inChannels(inCh), outChannels(outCh) {
        // 每个输出通道需要 inChannels × kSize² 个权重
        int filterCols = inChannels * kSize * kSize;
        filters = Matrix::Random(outChannels, filterCols) * 0.1;
    }

    /* ------- 前向传播：im2col → 矩阵乘法 → ReLU ------ */
    Matrix forward(const Matrix &input) {
        // 1. im2col 将卷积操作转换为矩阵乘法形式
        Matrix inputMatrix = im2col(input, kernelSize, stride);

        // 2. 矩阵乘法替代所有嵌套循环！
        //    单行代码完成 outChannels × numPatches 次乘加运算
        //    Eigen 内部调用 BLAS / SIMD 进行硬件加速
        Matrix output = filters * inputMatrix;

        // 3. ReLU 激活函数：逐元素取 max(0, x)
        output = output.cwiseMax(0.0);

        return output;
    }
};

/* ====================== CPU 嵌套循环卷积（对比用） ====================== */
/*
 * 纯 CPU 循环卷积——与 00_convolution_cpu.cpp 逻辑一致，
 * outCh 个卷积核各做一次滑动窗口，等价于 OptimizedConvolutionalLayer 的计算量。
 * 输出展平为行向量用于对齐比较。
 */
Matrix cpuConvolutionLoops(const Matrix &input,
                           const std::vector<Matrix> &filters,
                           int kernelSize, int stride) {
    int inputRows = static_cast<int>(input.rows());
    int inputCols = static_cast<int>(input.cols());
    int outputRows = (inputRows - kernelSize) / stride + 1;
    int outputCols = (inputCols - kernelSize) / stride + 1;
    int outCh = static_cast<int>(filters.size());

    // 返回值：(outCh) × (outputRows * outputCols)
    Matrix output(outCh, outputRows * outputCols);

    for (int f = 0; f < outCh; ++f) {
        int col = 0;
        for (int i = 0; i < outputRows; ++i) {
            for (int j = 0; j < outputCols; ++j) {
                double s = 0.0;
                for (int k = 0; k < kernelSize; ++k) {
                    for (int l = 0; l < kernelSize; ++l) {
                        s += input(i + k, j + l)
                             * filters[f](k, l); // 单通道，逐滤波器
                    }
                }
                output(f, col++) = std::max(0.0, s); // ReLU
            }
        }
    }
    return output;
}

/* ======================== 耗时基准测试 ================================= */
/*
 * 对 CPU 循环和 im2col+矩阵乘法两种方式重复运行 rounds 次，
 * 分别记录总耗时并返回平均每次耗时（微秒）。
 */
void benchmark() {
    const int rounds = 1000;   // 重复次数以稳定测量
    const int kSize = 3;       // 3×3 卷积核
    const int stride = 1;      // 步长 1
    const int inChannels = 1;  // 单通道输入
    const int outChannels = 2; // 2 个输出通道

    // 准备测试数据
    Matrix input(5, 5);
    double val = 1.0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            input(i, j) = val++;
        }
    }

    // 准备 outChannels 个 3×3 卷积核
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    std::vector<Matrix> cpuFilters(outChannels);
    for (int f = 0; f < outChannels; ++f) {
        cpuFilters[f] = Matrix(3, 3);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                cpuFilters[f](i, j) = dist(rng);
            }
        }
    }

    // ---- 测量 CPU 嵌套循环耗时 (outChannels 个卷积核) ----
    volatile double sink = 0.0; // 防止编译器优化
    auto startCpu = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; ++r) {
        Matrix cpuOut = cpuConvolutionLoops(input, cpuFilters,
                                            kSize, stride);
        sink += cpuOut(0, 0); // 强制保留计算结果
    }
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> cpuElapsed = endCpu - startCpu;
    double cpuAvg = cpuElapsed.count() / rounds;
    (void)sink;

    // ---- 测量 im2col + 矩阵乘法耗时 ----
    OptimizedConvolutionalLayer optLayer(inChannels, outChannels,
                                         kSize, stride);
    volatile double sink2 = 0.0;
    auto startOpt = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; ++r) {
        Matrix result = optLayer.forward(input);
        sink2 += result(0, 0);
    }
    auto endOpt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> optElapsed = endOpt - startOpt;
    double optAvg = optElapsed.count() / rounds;
    (void)sink2;

    // ---- 输出比较结果 ----
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║             卷积实现性能对比（" << rounds << " 次平均）            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  CPU 嵌套循环(" << outChannels << "核): " << std::setw(8)
              << std::fixed << std::setprecision(2) << cpuAvg
              << " μs         ║\n";
    std::cout << "║  im2col + 矩阵    : " << std::setw(8) << std::fixed
              << std::setprecision(2) << optAvg << " μs         ║\n";

    double speedup = (cpuAvg > 0.0) ? optAvg / cpuAvg : 0.0;
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  加速比(矩阵/循环): " << std::setw(7) << std::fixed
              << std::setprecision(3) << speedup << "×                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::cout << "【小输入说明】5×5 时 im2col 展开与矩阵构造开销 > 计算收益\n\n";

    // 用更大的输入演示加速效果
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   大尺寸模拟（128×128 输入，3×3 核，" << rounds << " 次平均）      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    Matrix largeInput = Matrix::Random(128, 128);
    std::vector<Matrix> largeCpuFilters(outChannels);
    for (int f = 0; f < outChannels; ++f) {
        largeCpuFilters[f] = Matrix::Random(3, 3) * 0.1;
    }

    volatile double sink3 = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; ++r) {
        Matrix cpuOut = cpuConvolutionLoops(largeInput, largeCpuFilters,
                                            3, 1);
        sink3 += cpuOut(0, 0);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    (void)sink3;
    double cpuLarge = std::chrono::duration<double, std::micro>(t2 - t1).count()
                      / rounds;

    OptimizedConvolutionalLayer optLayerLarge(inChannels, outChannels, 3, 1);
    volatile double sink4 = 0.0;
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; ++r) {
        Matrix optOut = optLayerLarge.forward(largeInput);
        sink4 += optOut(0, 0);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    (void)sink4;
    double optTimeLarge = std::chrono::duration<double, std::micro>(t4 - t3).count()
                          / rounds;

    std::cout << "  CPU 嵌套循环(" << outChannels << "核): " << std::fixed
              << std::setprecision(2) << cpuLarge << " μs\n";
    std::cout << "  im2col + 矩阵    : " << std::fixed
              << std::setprecision(2) << optTimeLarge << " μs\n";
    double largeSpeedup = (optTimeLarge > 0.0) ? cpuLarge / optTimeLarge : 0.0;
    std::cout << "  加速比(循环/矩阵): " << std::fixed
              << std::setprecision(3) << largeSpeedup
              << "×（矩阵方法在原书实验中 ~5-6× 更快）\n";
}

/* ================================ main ================================= */
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   第 6 章：卷积神经网络  -  Eigen 矩阵优化卷积         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // 测试数据
    Matrix input(5, 5);
    double val = 1.0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            input(i, j) = val++;
        }
    }

    std::cout << "【输入矩阵】5×5:\n"
              << input << "\n\n";

    // im2col 转换演示
    Matrix patches = im2col(input, /*kSize=*/3, /*stride=*/1);
    std::cout << "【im2col 展开结果】(" << patches.rows()
              << " × " << patches.cols() << "):\n";
    std::cout << "  每列为一个 3×3 窗口的 9 个元素拉直 → 共 "
              << patches.cols() << " 列\n\n";

    // 优化卷积层：1 输入通道，2 输出通道，3×3 核，步长 1
    OptimizedConvolutionalLayer optConv(/*inCh=*/1, /*outCh=*/2,
                                        /*kSize=*/3, /*stride=*/1);

    Matrix output = optConv.forward(input);
    std::cout << "【优化卷积输出】矩阵乘法 + ReLU:\n";
    std::cout << "  形状：" << output.rows() << " × " << output.cols() << "\n";
    std::cout << "  (" << output.rows() << " 个输出通道，每行一个"
              << patches.cols() << " 维特征向量)\n";
    std::cout << output << "\n\n";

    /*
     * 关键点说明：
     * 下面这行在 OptimizedConvolutionalLayer::forward 中：
     *     output = filters * inputMatrix;
     * 一行矩阵乘法代替了 00_convolution_cpu.cpp 中的全部四层嵌套循环。
     * Eigen 内部自动调用 BLAS 和 SIMD 指令实现向量化加速。
     */

    // 运行性能基准测试
    benchmark();

    return 0;
}
