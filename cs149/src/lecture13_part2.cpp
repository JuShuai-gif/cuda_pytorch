// lecture13_part2.cpp — CS149 Lecture 13: Halide Scheduling Concept Simulation
// Simulates how Halide scheduling primitives transform loop nests.
// Concepts: compute_root, compute_at, tile, vectorize, parallel, reorder
// Compile: g++ -std=c++17 -O2 lecture13_part2.cpp -o lecture13_part2
// Run:     ./lecture13_part2

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
// Simulation of a simple 2-stage image processing DAG: blurx -> out
// Equivalent to Halide:
//   blurx(x,y) = (in(x-1,y) + in(x,y) + in(x+1,y)) / 3;
//   out(x,y)   = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3;
// ============================================================================

constexpr int IMG_W = 256;
constexpr int IMG_H = 256;
constexpr int TILE_W = 64;
constexpr int TILE_H = 32;
constexpr int VEC_W  = 8;   // simulated SIMD width
constexpr int N_THREADS = 4;

// ============================================================================
// Data buffers
// ============================================================================
float input[IMG_H + 2][IMG_W + 2];
float blurx[IMG_H + 2][IMG_W];
float output[IMG_H][IMG_W];

void init_data() {
    for (int y = 0; y < IMG_H + 2; ++y)
        for (int x = 0; x < IMG_W + 2; ++x)
            input[y][x] = static_cast<float>((x * 37 + y * 73) % 256);
}

// ============================================================================
// compute_root() — Halide equivalent:
//   blurx.compute_root();
//
// Meaning: pre-compute ALL values of blurx before computing any output.
// Loop structure:
//   for y_blurx: for x_blurx: compute blurx(y,x)
//   for y_out:   for x_out:   compute out(y,x) from blurx
// ============================================================================
void schedule_compute_root() {
    std::cout << "  [compute_root] Pre-computing all of blurx, then all of out" << std::endl;

    // Stage 1: compute all blurx
    for (int y = 0; y < IMG_H + 2; ++y)
        for (int x = 0; x < IMG_W; ++x)
            blurx[y][x] = (input[y][x] + input[y][x+1] + input[y][x+2]) / 3.0f;

    // Stage 2: compute all out
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            output[y][x] = (blurx[y][x] + blurx[y+1][x] + blurx[y+2][x]) / 3.0f;
}

// ============================================================================
// compute_at(out, x) — Halide equivalent:
//   out.tile(x, y, xi, yi, TILE_W, TILE_H);
//   blurx.compute_at(out, x);
//
// Meaning: for each tile of out, compute only the blurx values needed.
// Interleaves producer and consumer: compute a tile of blurx, then a tile of out.
// Intermediate blurx buffer = TILE_W × (TILE_H + 2), fits in cache.
// ============================================================================
void schedule_compute_at_tile() {
    std::cout << "  [compute_at tile] Interleaved: blux tile -> out tile" << std::endl;

    int n_tiles_x = (IMG_W + TILE_W - 1) / TILE_W;
    int n_tiles_y = (IMG_H + TILE_H - 1) / TILE_H;

    for (int ty = 0; ty < n_tiles_y; ++ty) {
        for (int tx = 0; tx < n_tiles_x; ++tx) {
            int y_start = ty * TILE_H;
            int x_start = tx * TILE_W;
            int y_end   = std::min(y_start + TILE_H, IMG_H);
            int x_end   = std::min(x_start + TILE_W, IMG_W);

            // Local tile buffer for blurx: width × (height+2)
            // In Halide: allocate blurx(TILE_W, TILE_H+2)
            int tile_h = y_end - y_start;
            int tile_w = x_end - x_start;
            std::vector<float> tile_blurx(tile_w * (tile_h + 2));

            // Stage 1: compute blurx tile
            for (int y = 0; y < tile_h + 2; ++y)
                for (int x = 0; x < tile_w; ++x)
                    tile_blurx[y * tile_w + x] =
                        (input[y_start + y][x_start + x] +
                         input[y_start + y][x_start + x + 1] +
                         input[y_start + y][x_start + x + 2]) / 3.0f;

            // Stage 2: compute out tile from cached blurx
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
// Simulated "vectorization" — sum 8 elements at a time (conceptual SIMD)
// In Halide: out.vectorize(xi, 8)
// ============================================================================
float simd_dot3(const float* a, const float* b, const float* c) {
    // Simulate 3-wide horizontal reduction (actual SIMD would use shuffle/add)
    return (a[0] + b[0] + c[0]) / 3.0f;
}

void schedule_vectorized() {
    std::cout << "  [vectorize(x,8)] Simulated 8-wide SIMD on innermost loop" << std::endl;

    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; x += VEC_W) {
            int limit = std::min(x + VEC_W, IMG_W);
            // In real SIMD, these VEC_W operations happen in 1 instruction
            for (int xi = x; xi < limit; ++xi) {
                output[y][xi] = (input[y][xi] + input[y][xi+1] + input[y][xi+2] +
                                 input[y+1][xi] + input[y+1][xi+1] + input[y+1][xi+2] +
                                 input[y+2][xi] + input[y+2][xi+1] + input[y+2][xi+2]) / 9.0f;
            }
        }
    }
}

// ============================================================================
// Simulated "parallel" — multi-thread across y dimension
// In Halide: out.parallel(y)
// ============================================================================
void schedule_parallel() {
    std::cout << "  [parallel(y)] Multi-threaded across " << N_THREADS << " threads" << std::endl;

    std::vector<std::thread> threads;
    int rows_per_thread = (IMG_H + N_THREADS - 1) / N_THREADS;

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([t, rows_per_thread]() {
            int y_start = t * rows_per_thread;
            int y_end   = std::min(y_start + rows_per_thread, IMG_H);
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
// Full Halide schedule simulation:
//   out.tile(x, y, xi, yi, TILE_W, TILE_H)
//      .vectorize(xi, VEC_W)
//      .parallel(y);
//   blurx.compute_at(out, x).vectorize(x, VEC_W);
//
// This produces a loop nest equivalent to the optimal Halide schedule.
// ============================================================================
void schedule_full_halide() {
    std::cout << "  [Full Halide] tile(" << TILE_W << "," << TILE_H
              << ") + vectorize(" << VEC_W << ") + parallel + compute_at" << std::endl;

    int n_tiles_y = (IMG_H + TILE_H - 1) / TILE_H;
    int n_tiles_x = (IMG_W + TILE_W - 1) / TILE_W;

    // parallel(y) → parallelize outer y-tile loop
    std::vector<std::thread> threads;
    std::mutex print_mtx;

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            // Each thread processes a subset of tile rows
            for (int ty = t; ty < n_tiles_y; ty += N_THREADS) {
                for (int tx = 0; tx < n_tiles_x; ++tx) {
                    int y0 = ty * TILE_H;
                    int x0 = tx * TILE_W;
                    int tile_h = std::min(TILE_H, IMG_H - y0);
                    int tile_w = std::min(TILE_W, IMG_W - x0);

                    // compute_at(out, x): allocate blurx tile
                    // In Halide:: allocate blurx(tile_w, tile_h+2)
                    std::vector<float> t_blurx(tile_w * (tile_h + 2));

                    // blurx.vectorize(x, VEC_W): horizontal blur with SIMD
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

                    // out.vectorize(xi, VEC_W): vertical blur with SIMD
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
// Auto-scheduler concept: beam search over schedule space
// ============================================================================
struct ScheduleNode {
    std::string description;
    double estimated_cost;       // ML model prediction (simulated)
    int depth;
    ScheduleNode* parent;
};

void demo_autoscheduler_concept() {
    std::cout << std::endl;
    std::cout << "=== Auto-Scheduler Concept (Adams et al. SIGGRAPH 2019) ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Halide DAG:  in -> blurx -> out" << std::endl;
    std::cout << std::endl;
    std::cout << "Schedule search space (beam search):" << std::endl;
    std::cout << "  Node placement choices for 'blurx':" << std::endl;
    std::cout << "    1. compute_root                     cost: 120" << std::endl;
    std::cout << "    2. compute_at(out, y)    tile(64,32) cost: 85" << std::endl;
    std::cout << "    3. compute_at(out, x)    tile(64,32) cost: 72  ← best" << std::endl;
    std::cout << "    4. compute_at(out, yi)   tile(32,16) cost: 95" << std::endl;
    std::cout << std::endl;
    std::cout << "  Node placement choices for 'out':" << std::endl;
    std::cout << "    tile(64,32).vectorize(xi,8).parallel(y)  cost: 48  ← best" << std::endl;
    std::cout << "    tile(128,32).vectorize(xi,8).parallel(y) cost: 52" << std::endl;
    std::cout << std::endl;
    std::cout << "ML cost model: simple MLP, ~10 μs per schedule evaluation." << std::endl;
    std::cout << "1.4M schedules tested in 166 seconds." << std::endl;
    std::cout << "Trained on randomly generated Halide programs with measured runtimes." << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "=== CS149 Lecture 13: Halide Scheduling Concept Simulation ===" << std::endl;
    std::cout << "Image: " << IMG_W << "×" << IMG_H << std::endl;
    std::cout << std::endl;

    init_data();

    // Demonstrate each scheduling strategy
    std::cout << "--- Scheduling Strategies ---" << std::endl;

    init_data();
    auto t0 = std::chrono::high_resolution_clock::now();
    schedule_compute_root();
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_root = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << t_root << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_compute_at_tile();
    t1 = std::chrono::high_resolution_clock::now();
    double t_tile = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << t_tile << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_vectorized();
    t1 = std::chrono::high_resolution_clock::now();
    double t_vec = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << t_vec << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_parallel();
    t1 = std::chrono::high_resolution_clock::now();
    double t_par = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << t_par << " ms" << std::endl;

    init_data();
    t0 = std::chrono::high_resolution_clock::now();
    schedule_full_halide();
    t1 = std::chrono::high_resolution_clock::now();
    double t_full = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << t_full << " ms" << std::endl;

    demo_autoscheduler_concept();

    std::cout << std::endl;
    std::cout << "=== Key Halide Philosophy ===" << std::endl;
    std::cout << "1. Algorithm (WHAT): declarative, side-effect-free expressions" << std::endl;
    std::cout << "2. Schedule (HOW): separate directives for mapping to hardware" << std::endl;
    std::cout << "3. Auto-scheduler: search over schedule space with ML cost model" << std::endl;
    std::cout << "4. LLM agents: trial-and-error with profiling feedback loop" << std::endl;

    return 0;
}
