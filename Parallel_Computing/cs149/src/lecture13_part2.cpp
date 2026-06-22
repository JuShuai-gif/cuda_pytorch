// lecture13_part2.cpp — CS149 第13讲：Halide 调度概念模拟
// ============================================================================
// 【课程核心概念】
// Halide 是一种用于图像处理的领域特定语言（DSL），其核心创新在于将
// "算法"（WHAT）与"调度"（HOW）彻底分离：
//   - 算法：纯函数式、无副作用的数据流表达式（如 blurx(x,y) = ...）
//   - 调度：一组独立的指令，描述如何映射到硬件（tile, vectorize, parallel）
//
// 本文件通过纯 C++ 代码模拟 Halide 的调度原语如何转换循环嵌套（loop nest）。
// 展示从朴素的 compute_root 到融合的 compute_at，再到向量化和并行化的完整演化。
//
// Halide 调度原语一览：
//   compute_root()  : 在消费之前完全计算某个阶段的全部值（最大中间缓冲区）
//   compute_at()    : 将生产者阶段内联到消费者的某个循环级别（减少缓冲区）
//   tile()          : 将循环嵌套的维度分块（提升缓存局部性）
//   vectorize()     : 对内层循环应用 SIMD 向量化（数据级并行）
//   parallel()      : 对外层循环应用多线程并行（任务级并行）
//   reorder()       : 重排循环嵌套的维度顺序
//
// 自动调度器（Adams et al. SIGGRAPH 2019）：
//   通过 beam search 在指数级调度空间中搜索，用 ML 代价模型预估每个调度方案
//   的运行时间，在 166 秒内测试 140 万种调度方案。
//
// 现代方法（LLM Agent）：
//   使用大语言模型通过试错+性能分析的反馈循环来探索调度空间。
// ============================================================================
// 编译：g++ -std=c++17 -O2 lecture13_part2.cpp -o lecture13_part2
// 运行：./lecture13_part2

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <iomanip>
#include <cmath>
#include <algorithm>

// ============================================================================
// 模拟一个简单的 2 阶段图像处理 DAG: blurx → out
// 等价于 Halide 中的以下声明：
//   blurx(x,y) = (in(x-1,y) + in(x,y) + in(x+1,y)) / 3;   // 水平模糊 1x3
//   out(x,y)   = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3;  // 垂直模糊 3x1
// 这两个阶段构成了典型的可分离模糊：先水平后垂直。
// ============================================================================

constexpr int IMG_W = 256;    // 图像宽度
constexpr int IMG_H = 256;    // 图像高度
constexpr int TILE_W = 64;    // tile 宽度（水平分块大小）
constexpr int TILE_H = 32;    // tile 高度（垂直分块大小）
constexpr int VEC_W  = 8;     // 模拟的 SIMD 宽度（8 个元素同时计算）
constexpr int N_THREADS = 4;  // 模拟的并行线程数

// ============================================================================
// 数据缓冲区
// blurx 的大小为 (IMG_H+2)×IMG_W，因为水平模糊不需要垂直方向的填充
// ============================================================================
float input[IMG_H + 2][IMG_W + 2];   // 输入图像（含 1 像素边界填充）
float blurx[IMG_H + 2][IMG_W];        // 水平模糊中间结果
float output[IMG_H][IMG_W];           // 最终输出

// 初始化输入数据（确定性伪随机模式）
void init_data() {
    for (int y = 0; y < IMG_H + 2; ++y)
        for (int x = 0; x < IMG_W + 2; ++x)
            input[y][x] = static_cast<float>((x * 37 + y * 73) % 256);
}

// ============================================================================
// compute_root() — 等价于 Halide:
//   blurx.compute_root();
//
// 【语义解释】
// compute_root 意味着：blurx 被完全独立地计算，作为整个 DAG 中的一个全局阶段。
// 等价于：先运行全部 blurx 的循环嵌套（产生完整的中间数组），
//         再运行全部 out 的循环嵌套（消费完整的中间数组）。
//
// 【循环结构】
//   for y_blurx: for x_blurx: 计算 blurx(y,x)    // 生产者阶段
//   for y_out:   for x_out:   计算 out(y,x)       // 消费者阶段（从 blurx 读取）
//
// 【优缺点】
//   优点：简单直观，代码容易理解
//   缺点：blurx 的完整数组（256*258 ≈ 66K floats = 264KB）必须驻留在内存中
//         如果整个 blurx 大于缓存容量，out 阶段读取时将发生大量缓存缺失
// ============================================================================
void schedule_compute_root() {
    std::cout << "  [compute_root] 先完整计算 blurx，再完整计算 out" << std::endl;

    // 阶段 1：计算全部 blurx（生产者）
    // 对每行每列做 1x3 水平模糊
    for (int y = 0; y < IMG_H + 2; ++y)
        for (int x = 0; x < IMG_W; ++x)
            blurx[y][x] = (input[y][x] + input[y][x+1] + input[y][x+2]) / 3.0f;

    // 阶段 2：计算全部 out（消费者）
    // 对每行每列做 3x1 垂直模糊
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            output[y][x] = (blurx[y][x] + blurx[y+1][x] + blurx[y+2][x]) / 3.0f;
}

// ============================================================================
// compute_at(out, x) — 等价于 Halide:
//   out.tile(x, y, xi, yi, TILE_W, TILE_H);   // 将 out 的计算分块
//   blurx.compute_at(out, x);                   // blurx 内联到 out 的 x 循环
//
// 【语义解释】
// compute_at(out, x) 将 blurx 的"计算位置"放在 out 的 x 维度的循环体内。
// 也就是说，对于每个 out 的 x-tile，我们只计算该 tile 所需的 blurx 值。
// 这实现了生产者和消费者的交织（interleaving）：
//   对每个 tile：计算 blurx tile → 立即消费 → 丢弃 blurx tile → 处理下一个 tile
//
// 【关键优势：生产者-消费者局部性】
// 中间 blurx 缓冲区只需 TILE_W × (TILE_H + 2) 大小（+2 用于垂直模糊的边界需求），
// 对于 TILE_W=64, TILE_H=32：64×34×4B ≈ 8.7KB —— 可以轻松放入 L1 缓存！
// 对比 compute_root 需要 264KB，compute_at 把中间缓冲区缩小了 30 倍。
//
// 【为什么是 compute_at(out, x) 而不是 compute_at(out, y)？】
// x 是内层循环维度，compute_at(out, x) 意味着 blurx 被插入到 x 循环内部。
// 这产生了最小的粒度（tile 级别），因此中间缓冲区最小。
// 如果使用 compute_at(out, y)，缓冲区大小为 IMG_W × (TILE_H+2)，更大。
// ============================================================================
void schedule_compute_at_tile() {
    std::cout << "  [compute_at tile] 交织计算：先计算 blurx tile → 再计算 out tile" << std::endl;

    int n_tiles_x = (IMG_W + TILE_W - 1) / TILE_W;   // 水平方向的 tile 数量
    int n_tiles_y = (IMG_H + TILE_H - 1) / TILE_H;   // 垂直方向的 tile 数量

    for (int ty = 0; ty < n_tiles_y; ++ty) {
        for (int tx = 0; tx < n_tiles_x; ++tx) {
            int y_start = ty * TILE_H;
            int x_start = tx * TILE_W;
            int y_end   = std::min(y_start + TILE_H, IMG_H);  // 边界 tile 可能更小
            int x_end   = std::min(x_start + TILE_W, IMG_W);

            // 局部 tile 的 blurx 缓冲区：width × (height+2)
            // 在 Halide 中等价于: allocate blurx(TILE_W, TILE_H+2)
            int tile_h = y_end - y_start;
            int tile_w = x_end - x_start;
            std::vector<float> tile_blurx(tile_w * (tile_h + 2));

            // 阶段 1：计算当前 tile 的 blurx（生产者，结果在 L1 缓存中）
            for (int y = 0; y < tile_h + 2; ++y)
                for (int x = 0; x < tile_w; ++x)
                    tile_blurx[y * tile_w + x] =
                        (input[y_start + y][x_start + x] +
                         input[y_start + y][x_start + x + 1] +
                         input[y_start + y][x_start + x + 2]) / 3.0f;

            // 阶段 2：从缓存的 blurx 计算 out tile（消费者，数据仍在 L1 缓存中！）
            for (int y = 0; y < tile_h; ++y)
                for (int x = 0; x < tile_w; ++x)
                    output[y_start + y][x_start + x] =
                        (tile_blurx[y * tile_w + x] +
                         tile_blurx[(y + 1) * tile_w + x] +
                         tile_blurx[(y + 2) * tile_w + x]) / 3.0f;
        }
    }
}

// ============================================================================
// 模拟的"向量化"—— 一次处理 8 个元素（概念上的 SIMD）
// 在 Halide 中等价于: out.vectorize(xi, 8)
//
// 【SIMD 解释】
// SIMD (Single Instruction Multiple Data) 是现代 CPU 的数据级并行技术。
// 一条 SIMD 指令可以同时对多个数据执行相同操作。例如，x86 的 AVX/AVX-512。
// 在本模拟中，VEC_W=8 表示每个向量包含 8 个 float（256 位 AVX 寄存器）。
// 真实的 Halide 会将内层循环编译为 SIMD 指令，这里只展示循环结构。
// ============================================================================
float simd_dot3(const float* a, const float* b, const float* c) {
    // 模拟 3 路水平规约（真实 SIMD 会使用 shuffle/add 指令实现）
    return (a[0] + b[0] + c[0]) / 3.0f;
}

void schedule_vectorized() {
    std::cout << "  [vectorize(x,8)] 在最内层循环模拟 8 路 SIMD" << std::endl;

    for (int y = 0; y < IMG_H; ++y) {
        // 外层步长为 VEC_W=8，每次处理一个向量宽度的元素
        for (int x = 0; x < IMG_W; x += VEC_W) {
            int limit = std::min(x + VEC_W, IMG_W);  // 边界处理（不足 8 时）
            // 在实际 SIMD 中，这 VEC_W 次操作在 1 条指令内完成
            for (int xi = x; xi < limit; ++xi) {
                output[y][xi] = (input[y][xi] + input[y][xi+1] + input[y][xi+2] +
                                 input[y+1][xi] + input[y+1][xi+1] + input[y+1][xi+2] +
                                 input[y+2][xi] + input[y+2][xi+1] + input[y+2][xi+2]) / 9.0f;
            }
        }
    }
}

// ============================================================================
// 模拟的"并行化"—— 跨 y 维度多线程
// 在 Halide 中等价于: out.parallel(y)
//
// 【并行化解释】
// parallel(y) 将 y 维度的迭代空间按线程数均匀划分。
// 每个线程处理图像的一部分行区域，互不重叠，因此无需同步。
// 在 Halide 中，并行化是通过 OpenMP 或线程池实现的。
// ============================================================================
void schedule_parallel() {
    std::cout << "  [parallel(y)] 使用 " << N_THREADS << " 个线程的跨 y 维度多线程" << std::endl;

    std::vector<std::thread> threads;
    int rows_per_thread = (IMG_H + N_THREADS - 1) / N_THREADS;  // 每个线程的行数（向上取整）

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([t, rows_per_thread]() {
            int y_start = t * rows_per_thread;         // 本线程的起始行
            int y_end   = std::min(y_start + rows_per_thread, IMG_H);  // 结束行
            for (int y = y_start; y < y_end; ++y)
                for (int x = 0; x < IMG_W; ++x)
                    output[y][x] = (input[y][x] + input[y][x+1] + input[y][x+2] +
                                    input[y+1][x] + input[y+1][x+1] + input[y+1][x+2] +
                                    input[y+2][x] + input[y+2][x+1] + input[y+2][x+2]) / 9.0f;
        });
    }
    for (auto& th : threads) th.join();
}

// ============================================================================
// 完整 Halide 调度模拟:
//   out.tile(x, y, xi, yi, TILE_W, TILE_H)    // 将 out 的计算按 tile 分块
//      .vectorize(xi, VEC_W)                    // 对内层 xi 循环做 SIMD
//      .parallel(y);                             // 对外层 y 循环做多线程
//   blurx.compute_at(out, x).vectorize(x, VEC_W); // blurx 内联到 tile 并向量化
//
// 这产生的循环嵌套等价于 Halide 自动调度器给出的最优调度方案。
// 组合了所有优化技术：分块（缓存利用）+ 向量化（SIMD）+ 并行化（多核）。
// ============================================================================
void schedule_full_halide() {
    std::cout << "  [完整 Halide 调度] tile(" << TILE_W << "," << TILE_H
              << ") + vectorize(" << VEC_W << ") + parallel + compute_at" << std::endl;

    int n_tiles_y = (IMG_H + TILE_H - 1) / TILE_H;
    int n_tiles_x = (IMG_W + TILE_W - 1) / TILE_W;

    // parallel(y) → 并行化外层的 y-tile 循环
    // 使用交错分配（cyclic allocation）：线程 t 处理 tile 行 t, t+N, t+2N, ...
    // 这比块分配（block allocation）有更好的负载均衡
    std::vector<std::thread> threads;
    std::mutex print_mtx;

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            // 每个线程处理 tile 行的一个子集（交错模式以实现负载均衡）
            for (int ty = t; ty < n_tiles_y; ty += N_THREADS) {
                for (int tx = 0; tx < n_tiles_x; ++tx) {
                    int y0 = ty * TILE_H;
                    int x0 = tx * TILE_W;
                    int tile_h = std::min(TILE_H, IMG_H - y0);  // 边界 tile 处理
                    int tile_w = std::min(TILE_W, IMG_W - x0);

                    // compute_at(out, x): 为每个 tile 分配局部 blurx 缓冲区
                    // Halide 中等价于: allocate blurx(tile_w, tile_h+2)
                    std::vector<float> t_blurx(tile_w * (tile_h + 2));

                    // blurx.vectorize(x, VEC_W): 用 SIMD 做水平模糊
                    for (int yi = 0; yi < tile_h + 2; ++yi) {
                        for (int xi = 0; xi < tile_w; xi += VEC_W) {
                            int limit = std::min(xi + VEC_W, tile_w);
                            for (int xx = xi; xx < limit; ++xx)
                                t_blurx[yi * tile_w + xx] =
                                    (input[y0 + yi][x0 + xx] +
                                     input[y0 + yi][x0 + xx + 1] +
                                     input[y0 + yi][x0 + xx + 2]) / 3.0f;
                        }
                    }

                    // out.vectorize(xi, VEC_W): 用 SIMD 做垂直模糊
                    for (int yi = 0; yi < tile_h; ++yi) {
                        for (int xi = 0; xi < tile_w; xi += VEC_W) {
                            int limit = std::min(xi + VEC_W, tile_w);
                            for (int xx = xi; xx < limit; ++xx)
                                output[y0 + yi][x0 + xx] =
                                    (t_blurx[yi * tile_w + xx] +
                                     t_blurx[(yi + 1) * tile_w + xx] +
                                     t_blurx[(yi + 2) * tile_w + xx]) / 3.0f;
                        }
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}

// ============================================================================
// 自动调度器概念: 在调度空间中做 beam search
//
// 【自动调度器原理（Adams et al. SIGGRAPH 2019）】
// 调度空间巨大：对于 N 个阶段、M 个维度的 DAG 来说，可能的调度组合是指数级的。
// 自动调度器通过以下步骤找到近似最优调度：
//   1. 将调度视为一组决策序列（每个决策是将一个算子放置到某个循环级别）
//   2. 使用 beam search（束搜索）—— 每一步保留 top-K 个部分调度方案
//   3. 对每个部分调度，ML 代价模型预估最终运行时间（~10μs 每次评估）
//   4. 继续扩展，直到完成完整调度
//
// ML 代价模型是一个简单的 MLP（多层感知机），
// 训练数据来自随机生成的 Halide 程序的实际运行时间测量。
//
// LLM Agent 方法（更新的研究）：
// 使用大语言模型通过自然语言和代码来搜索调度空间，
// 通过实际运行+性能分析的反馈循环来迭代改进。
// ============================================================================
struct ScheduleNode {
    std::string description;          // 调度描述
    double estimated_cost;            // ML 模型预估的运行成本（模拟值）
    int depth;                        // 搜索深度
    ScheduleNode* parent;             // 父节点（用于回溯）
};

void demo_autoscheduler_concept() {
    std::cout << std::endl;
    std::cout << "=== 自动调度器概念（Adams et al. SIGGRAPH 2019） ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Halide DAG（有向无环图）:  in → blurx → out" << std::endl;
    std::cout << std::endl;
    std::cout << "调度搜索空间（beam search）:" << std::endl;
    std::cout << "  为 'blurx' 选择放置位置:" << std::endl;
    std::cout << "    1. compute_root                     预估成本: 120" << std::endl;
    std::cout << "    2. compute_at(out, y)    tile(64,32) 预估成本: 85" << std::endl;
    std::cout << "    3. compute_at(out, x)    tile(64,32) 预估成本: 72  ← 最优" << std::endl;
    std::cout << "    4. compute_at(out, yi)   tile(32,16) 预估成本: 95" << std::endl;
    std::cout << std::endl;
    std::cout << "  为 'out' 选择放置位置:" << std::endl;
    std::cout << "    tile(64,32).vectorize(xi,8).parallel(y)  预估成本: 48  ← 最优" << std::endl;
    std::cout << "    tile(128,32).vectorize(xi,8).parallel(y) 预估成本: 52" << std::endl;
    std::cout << std::endl;
    std::cout << "ML 代价模型：简单的多层感知机（MLP），每次调度评估约 10 微秒。" << std::endl;
    std::cout << "在 166 秒内测试了 140 万种调度方案。" << std::endl;
    std::cout << "训练数据来自随机生成的 Halide 程序及其实际运行时间测量。" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::cout << "=== CS149 第13讲：Halide 调度概念模拟 ===" << std::endl;
    std::cout << "图像尺寸: " << IMG_W << "×" << IMG_H << std::endl;
    std::cout << std::endl;

    init_data();

    // 演示每种调度策略
    std::cout << "--- 调度策略对比 ---" << std::endl;

    init_data();
    auto t0 = std::chrono::high_resolution_clock::now();
    schedule_compute_root();
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_root = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    耗时: " << std::fixed << std::setprecision(3) << t_root << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_compute_at_tile();
    t1 = std::chrono::high_resolution_clock::now();
    double t_tile = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    耗时: " << std::fixed << std::setprecision(3) << t_tile << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_vectorized();
    t1 = std::chrono::high_resolution_clock::now();
    double t_vec = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    耗时: " << std::fixed << std::setprecision(3) << t_vec << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_parallel();
    t1 = std::chrono::high_resolution_clock::now();
    double t_par = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    耗时: " << std::fixed << std::setprecision(3) << t_par << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_full_halide();
    t1 = std::chrono::high_resolution_clock::now();
    double t_full = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    耗时: " << std::fixed << std::setprecision(3) << t_full << " ms" << std::endl;

    demo_autoscheduler_concept();

    std::cout << std::endl;
    std::cout << "=== Halide 核心哲学 ===" << std::endl;
    std::cout << "1. 算法（WHAT）：声明式的、无副作用的表达式 —— 描述计算什么" << std::endl;
    std::cout << "2. 调度（HOW）：独立的指令，描述如何映射到硬件 —— 描述怎么算" << std::endl;
    std::cout << "3. 自动调度器：用 ML 代价模型在调度空间中搜索最优方案" << std::endl;
    std::cout << "4. LLM Agent：通过试错加性能分析的反馈循环来探索调度空间" << std::endl;

    return 0;
}
