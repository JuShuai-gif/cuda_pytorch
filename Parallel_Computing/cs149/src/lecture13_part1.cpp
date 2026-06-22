// lecture13_part1.cpp — CS149 第13讲：图像模糊算法的演化
// ============================================================================
// 【课程核心概念】
// 本文件演示图像模糊算法的三种实现策略，展示从直观实现到硬件感知优化的演化过程。
//
// 核心思想：同一个算法（3x3 box blur）可以有完全不同的"调度"（schedule）方式，
// 而调度方式直接影响内存访问模式和缓存利用效率，最终决定性能。
//
// 三种策略：
//   策略一（Single-pass）：对每个像素执行 3x3 邻域卷积，每个像素做 9 次内存读取。
//         优点：直观、一次完成；缺点：O(9W*H) 工作量大，缓存局部性差。
//   策略二（Two-pass）：利用 3x3 box filter 的可分离性，先水平模糊再垂直模糊，
//         每个像素做 6 次操作。但需要中间缓冲区，引入额外内存流量。
//   策略三（Chunked/Tiled）：将图像分块处理，中间缓冲区只覆盖当前块，
//         使得整个工作集能放入 L1/L2 缓存，消除对主存的冗余访问。
//
// Halide 语言的核心哲学就是将"算法是什么"与"如何调度"分离，
// 让编译器/自动调度器自动生成最优的循环嵌套。
// ============================================================================
// 编译：g++ -std=c++17 -O2 lecture13_part1.cpp -o lecture13_part1
// 运行：./lecture13_part1

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>

// ============================================================================
// 配置参数
// ============================================================================
constexpr int WIDTH     = 2048;   // 图像宽度（像素）
constexpr int HEIGHT    = 2048;   // 图像高度（像素）
constexpr int PAD_W     = WIDTH + 2;  // 含边界填充的宽度（+2 用于 3x3 卷积的边界处理）
constexpr int PAD_H     = HEIGHT + 2; // 含边界填充的高度
constexpr int CHUNK_SIZE = 32;   // 分块模糊中每个 tile 的高度（行数），需匹配 L1 缓存大小

// ============================================================================
// 计时器工具类
// 使用 std::chrono::high_resolution_clock 进行高精度计时
// ============================================================================
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    void start() { t0 = Clock::now(); }                    // 开始计时
    double elapsed_ms() const {                             // 返回已过毫秒数
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
private:
    Clock::time_point t0;
};

// ============================================================================
// 用简单渐变填充图像，便于视觉验证算法的正确性
// 公式：pixel(x,y) = ((x*127 + y*63) % 256) / 255.0
// ============================================================================
void init_image(float* img, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img[y * w + x] = static_cast<float>((x * 127 + y * 63) % 256) / 255.0f;
}

// ============================================================================
// 策略一：单次遍历 3x3 box blur（基准实现）
// 每个输出像素直接读取 3x3 邻域的 9 个输入值进行加权平均
// 总计算量 = 9 * WIDTH * HEIGHT（每个像素 9 次乘加）
//
// 【性能瓶颈分析】
// 对于每个输出像素 (i,j)，算法读取 input[j..j+2][i..i+2] 共 9 个值。
// 相邻像素的 3x3 窗口有大量重叠——水平方向重叠 2 列，垂直方向重叠 2 行。
// 若不做任何优化，有效工作量为 9 次/像素，大量的数据被重复从内存读取。
// 缓存命中率取决于图像尺寸是否适合放入缓存（本示例中 2048x2048 太大，不命中）。
// ============================================================================
void blur_single_pass(const float* input, float* output) {
    // 3x3 box filter 的权重：每个像素等权重 1/9（平均值滤波）
    const float weights[9] = {1.f/9, 1.f/9, 1.f/9,
                               1.f/9, 1.f/9, 1.f/9,
                               1.f/9, 1.f/9, 1.f/9};
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            float sum = 0.0f;
            // 遍历 3x3 邻域窗口
            for (int jj = 0; jj < 3; ++jj)
                for (int ii = 0; ii < 3; ++ii)
                    sum += input[(j + jj) * PAD_W + (i + ii)] * weights[jj * 3 + ii];
            output[j * WIDTH + i] = sum;
        }
    }
}

// ============================================================================
// 策略二：可分离的两趟模糊（水平 + 垂直）
// 利用 3x3 box filter 的可分离性：blur = horizontal_3() ∘ vertical_3()
// 即先将图像按行做 1x3 的水平模糊，再将结果按列做 3x1 的垂直模糊。
//
// 【数学原理】
// 3x3 均值滤波可以分解为：[1 1 1] 与 [1 1 1]^T 的外积 = 3x3 全 1 矩阵。
// 因此执行顺序为：tmp(x,y) = Σ_{ii=0..2} in(x+ii, y)/3
//                 out(x,y) = Σ_{jj=0..2} tmp(x, y+jj)/3
//
// 总计算量 = 6 * WIDTH * HEIGHT（每个像素 3+3=6 次乘加，远少于单趟的 9 次）
//
// 【问题：额外的内存流量】
// 需要大小为 W × (H+2) 的中间缓冲区 tmp。假设图像 2048x2048：
//   tmp 大小 = 2048 × 2050 × 4 bytes ≈ 16 MB，远超 L3 缓存。
//   第一趟水平模糊写入 tmp，第二趟垂直模糊读取 tmp——
//   tmp 的写入与读取之间如果间隔了整幅图像，数据已从缓存中逐出，
//   必须重新从主存/LLC 读取，产生了额外的主存往返流量。
// ============================================================================
void blur_two_pass(const float* input, float* output) {
    const float weights[3] = {1.f/3, 1.f/3, 1.f/3};  // 1x3 水平滤波权重

    // 中间缓冲区：全宽 × 填充高度
    // 使用 make_unique 在堆上分配，避免栈溢出
    auto tmp = std::make_unique<float[]>(WIDTH * PAD_H);

    // 第一趟：水平模糊 —— 对每一行做 1x3 卷积
    for (int j = 0; j < PAD_H; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            float sum = 0.0f;
            for (int ii = 0; ii < 3; ++ii)
                sum += input[j * PAD_W + (i + ii)] * weights[ii];
            tmp[j * WIDTH + i] = sum;
        }
    }

    // 第二趟：垂直模糊 —— 对每一列做 3x1 卷积（读取 tmp 中缓存的数据）
    // 注意：此时 tmp 可能已经不在缓存中（如果图像大于缓存容量）
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            float sum = 0.0f;
            for (int jj = 0; jj < 3; ++jj)
                sum += tmp[(j + jj) * WIDTH + i] * weights[jj];
            output[j * WIDTH + i] = sum;
        }
    }
}

// ============================================================================
// 策略三：分块两趟模糊（融合的、缓存感知的优化）
// 将图像按垂直方向分为多个 CHUNK_SIZE 行高的块（tile）。
// 中间缓冲区大小被缩小为 WIDTH × (CHUNK_SIZE + 2)，使其能完全放入 L1/L2 缓存。
//
// 【核心优势：生产者-消费者局部性】
//   - 步骤 1：水平模糊产生 (CHUNK_SIZE+2) 行 tmp —— 热数据在缓存中
//   - 步骤 2：垂直模糊立即消费这些 tmp 行 —— 数据仍在缓存中！
//   没有中间数据被写出到主存再读回的开销。
//
// 【额外开销分析】
// 每个块的顶部和底部各需要额外的 1 行来满足 3x1 垂直卷积的边界：
//   块 0 产生 (CHUNK_SIZE+2) 行 → 垂直消费 CHUNK_SIZE 行
//   块 1 也需要顶部多 1 行（与块 0 的底部重叠）→ 多做了约 2 行的水平模糊
// 因此实际工作量为 6W*H + (2*num_tiles)*W ≈ 6W*H + (4/CHUNK_SIZE)*W*H
//
// CHUNK_SIZE → ∞ 时接近最优的 6×W×H，但受限于缓存大小。
// ============================================================================
void blur_chunked(const float* input, float* output) {
    const float weights[3] = {1.f/3, 1.f/3, 1.f/3};

    // 中间缓冲区大小 = (CHUNK_SIZE + 2) 行 × WIDTH 列
    // 对于 CHUNK_SIZE=32, WIDTH=2048：32*2048*4B ≈ 262KB，适合 L2 缓存
    // 对于 CHUNK_SIZE=16, WIDTH=2048：18*2048*4B ≈ 147KB，适合 L1 缓存
    auto tmp = std::make_unique<float[]>(WIDTH * (CHUNK_SIZE + 2));

    for (int j = 0; j < HEIGHT; j += CHUNK_SIZE) {
        // chunk_h = 当前块中的有效输出行数（最后一块可能不足 CHUNK_SIZE）
        int chunk_h = std::min(CHUNK_SIZE, HEIGHT - j);

        // 步骤 1：水平模糊 —— 产生 (chunk_h + 2) 行 tmp
        // +2 是因为垂直模糊需要访问第 chunk_h 和 chunk_h+1 行
        for (int j2 = 0; j2 < chunk_h + 2; ++j2) {
            for (int i = 0; i < WIDTH; ++i) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ++ii)
                    sum += input[(j + j2) * PAD_W + (i + ii)] * weights[ii];
                tmp[j2 * WIDTH + i] = sum;
            }
        }

        // 步骤 2：垂直模糊 —— 从 tmp 中消费（此时 tmp 数据在缓存中处于热状态）
        for (int j2 = 0; j2 < chunk_h; ++j2) {
            for (int i = 0; i < WIDTH; ++i) {
                float sum = 0.0f;
                for (int jj = 0; jj < 3; ++jj)
                    sum += tmp[(j2 + jj) * WIDTH + i] * weights[jj];
                output[(j + j2) * WIDTH + i] = sum;
            }
        }
    }
}

// ============================================================================
// 验证两个输出是否一致（在浮点容差范围内）
// ============================================================================
bool verify(const float* a, const float* b, int n, float eps = 1e-5f) {
    for (int i = 0; i < n; ++i)
        if (std::fabs(a[i] - b[i]) > eps)
            return false;
    return true;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::cout << "=== CS149 第13讲：图像模糊算法对比 ===" << std::endl;
    std::cout << "图像尺寸: " << WIDTH << "×" << HEIGHT << std::endl;
    std::cout << "分块大小: " << CHUNK_SIZE << " 行/块" << std::endl;
    std::cout << "中间缓冲区大小:" << std::endl;
    std::cout << "  两趟法:     " << WIDTH * PAD_H * sizeof(float) / 1024 << " KB ("
              << WIDTH << " × " << PAD_H << ")" << std::endl;
    std::cout << "  分块法:     " << WIDTH * (CHUNK_SIZE + 2) * sizeof(float) / 1024 << " KB ("
              << WIDTH << " × " << CHUNK_SIZE + 2 << ")" << std::endl;
    std::cout << std::endl;

    // 分配内存
    auto input  = std::make_unique<float[]>(PAD_W * PAD_H);
    auto ref    = std::make_unique<float[]>(WIDTH * HEIGHT);   // 基准输出（用策略一产生）
    auto result = std::make_unique<float[]>(WIDTH * HEIGHT);   // 被测输出

    init_image(input.get(), PAD_W, PAD_H);
    int n_pixels = WIDTH * HEIGHT;

    // --- 运行并计时每种策略 ---
    double t_single, t_two, t_chunk;
    Timer t;

    // 策略一：单趟模糊（基准参考）
    t.start();
    blur_single_pass(input.get(), ref.get());
    t_single = t.elapsed_ms();
    std::cout << "[单趟法]    计算量: 9 × W × H = " << 9LL * WIDTH * HEIGHT
              << " 次运算, 耗时: " << std::fixed << std::setprecision(2) << t_single << " ms" << std::endl;

    // 策略二：可分离两趟模糊
    t.start();
    blur_two_pass(input.get(), result.get());
    t_two = t.elapsed_ms();
    std::cout << "[两趟法]    计算量: 6 × W × H = " << 6LL * WIDTH * HEIGHT
              << " 次运算, 耗时: " << t_two << " ms" << std::endl;
    std::cout << "            与单趟法结果验证: "
              << (verify(ref.get(), result.get(), n_pixels) ? "通过" : "失败") << std::endl;

    // 策略三：分块模糊（tiled + fused）
    t.start();
    blur_chunked(input.get(), result.get());
    t_chunk = t.elapsed_ms();
    // 理论计算量：~6 * WIDTH * HEIGHT（随分块增大趋近此值）
    // 精确公式：6W*H + (4/CHUNK_SIZE)*W*H（每个块边界多 2 行的水平模糊 + 2 行的读取）
    double work_factor = 6.0 + (4.0 / CHUNK_SIZE);   // 精确: 6W*H + (4/CHUNK_SIZE)*W*H
    std::cout << "[分块法]    计算量: ~" << work_factor << " × W × H, 耗时: " << t_chunk << " ms" << std::endl;
    std::cout << "            与单趟法结果验证: "
              << (verify(ref.get(), result.get(), n_pixels) ? "通过" : "失败") << std::endl;

    std::cout << std::endl;
    std::cout << "=== 总结 ===" << std::endl;
    std::cout << "核心洞察：缓存感知的分块策略在减少中间缓冲区流量的同时，" << std::endl;
    std::cout << "趋近于可分离滤波器的最优 6×W×H 计算量。" << std::endl;
    std::cout << "Halide 通过以下调度原语自动做出这些调度决策：" << std::endl;
    std::cout << "  blurx.compute_at(out, x).vectorize(x, 8);" << std::endl;
    std::cout << "  out.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);" << std::endl;

    return 0;
}
