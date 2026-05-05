// lecture13_part1.cpp — CS149 Lecture 13: Image Blur Algorithms
// Demonstrates the evolution: single-pass → two-pass → chunked/tiled blur
// Compile: g++ -std=c++17 -O2 lecture13_part1.cpp -o lecture13_part1
// Run:     ./lecture13_part1

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>

// ============================================================================
// Configuration
// ============================================================================
constexpr int WIDTH     = 2048;
constexpr int HEIGHT    = 2048;
constexpr int PAD_W     = WIDTH + 2;
constexpr int PAD_H     = HEIGHT + 2;
constexpr int CHUNK_SIZE = 32;  // tile height for chunked blur

// ============================================================================
// Timer utility
// ============================================================================
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    void start() { t0 = Clock::now(); }
    double elapsed_ms() const {
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
private:
    Clock::time_point t0;
};

// ============================================================================
// Initialize image with a simple gradient for visual verification
// ============================================================================
void init_image(float* img, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img[y * w + x] = static_cast<float>((x * 127 + y * 63) % 256) / 255.0f;
}

// ============================================================================
// Strategy 1: Single-pass 3x3 box blur (N^2 work per pixel)
// Total work = 9 * WIDTH * HEIGHT
// ============================================================================
void blur_single_pass(const float* input, float* output) {
    const float weights[9] = {1.f/9, 1.f/9, 1.f/9,
                               1.f/9, 1.f/9, 1.f/9,
                               1.f/9, 1.f/9, 1.f/9};
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            float sum = 0.0f;
            for (int jj = 0; jj < 3; ++jj)
                for (int ii = 0; ii < 3; ++ii)
                    sum += input[(j + jj) * PAD_W + (i + ii)] * weights[jj * 3 + ii];
            output[j * WIDTH + i] = sum;
        }
    }
}

// ============================================================================
// Strategy 2: Two-pass separable blur (horizontal + vertical)
// Uses a separable 3x3 box filter: blur = horizontal_3() ∘ vertical_3()
// Total work = 6 * WIDTH * HEIGHT  (2N per pixel, N=3)
// BUT: requires an intermediate buffer (W x (H+2)) → extra memory traffic
// ============================================================================
void blur_two_pass(const float* input, float* output) {
    const float weights[3] = {1.f/3, 1.f/3, 1.f/3};

    // Intermediate buffer: full width × padded height
    auto tmp = std::make_unique<float[]>(WIDTH * PAD_H);

    // Pass 1: horizontal blur
    for (int j = 0; j < PAD_H; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            float sum = 0.0f;
            for (int ii = 0; ii < 3; ++ii)
                sum += input[j * PAD_W + (i + ii)] * weights[ii];
            tmp[j * WIDTH + i] = sum;
        }
    }

    // Pass 2: vertical blur
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
// Strategy 3: Chunked two-pass blur (fused, cache-aware)
// Processes the image in vertical chunks of CHUNK_SIZE rows.
// The intermediate buffer is sized so the entire working set fits in cache:
//   tmp_buf = WIDTH × (CHUNK_SIZE + 2)
// This captures all producer-consumer locality:
//   - Step 1: horizontal blur produces (CHUNK_SIZE+2) rows of tmp
//   - Step 2: vertical blur consumes from tmp (cached!)
// Total work → 6 × WIDTH × HEIGHT as CHUNK_SIZE → ∞
// ============================================================================
void blur_chunked(const float* input, float* output) {
    const float weights[3] = {1.f/3, 1.f/3, 1.f/3};

    // Intermediate buffer sized to fit in L1/L2 cache
    // (CHUNK_SIZE + 2) rows × WIDTH columns
    auto tmp = std::make_unique<float[]>(WIDTH * (CHUNK_SIZE + 2));

    for (int j = 0; j < HEIGHT; j += CHUNK_SIZE) {
        int chunk_h = std::min(CHUNK_SIZE, HEIGHT - j);

        // Step 1: horizontal blur — produce (chunk_h + 2) rows of tmp
        for (int j2 = 0; j2 < chunk_h + 2; ++j2) {
            for (int i = 0; i < WIDTH; ++i) {
                float sum = 0.0f;
                for (int ii = 0; ii < 3; ++ii)
                    sum += input[(j + j2) * PAD_W + (i + ii)] * weights[ii];
                tmp[j2 * WIDTH + i] = sum;
            }
        }

        // Step 2: vertical blur — consume from tmp (which is hot in cache)
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
// Verify two outputs match (within floating-point tolerance)
// ============================================================================
bool verify(const float* a, const float* b, int n, float eps = 1e-5f) {
    for (int i = 0; i < n; ++i)
        if (std::fabs(a[i] - b[i]) > eps)
            return false;
    return true;
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "=== CS149 Lecture 13: Image Blur Algorithm Comparison ===" << std::endl;
    std::cout << "Image size: " << WIDTH << "×" << HEIGHT << std::endl;
    std::cout << "Chunk size: " << CHUNK_SIZE << std::endl;
    std::cout << "Intermediate buffer sizes:" << std::endl;
    std::cout << "  Two-pass:   " << WIDTH * PAD_H * sizeof(float) / 1024 << " KB ("
              << WIDTH << " × " << PAD_H << ")" << std::endl;
    std::cout << "  Chunked:    " << WIDTH * (CHUNK_SIZE + 2) * sizeof(float) / 1024 << " KB ("
              << WIDTH << " × " << CHUNK_SIZE + 2 << ")" << std::endl;
    std::cout << std::endl;

    // Allocate
    auto input  = std::make_unique<float[]>(PAD_W * PAD_H);
    auto ref    = std::make_unique<float[]>(WIDTH * HEIGHT);
    auto result = std::make_unique<float[]>(WIDTH * HEIGHT);

    init_image(input.get(), PAD_W, PAD_H);
    int n_pixels = WIDTH * HEIGHT;

    // --- Run and time each strategy ---
    double t_single, t_two, t_chunk;
    Timer t;

    // Strategy 1: Single-pass (baseline)
    t.start();
    blur_single_pass(input.get(), ref.get());
    t_single = t.elapsed_ms();
    std::cout << "[Single-pass]  Work: 9 × W × H = " << 9LL * WIDTH * HEIGHT
              << " ops, Time: " << std::fixed << std::setprecision(2) << t_single << " ms" << std::endl;

    // Strategy 2: Two-pass separable
    t.start();
    blur_two_pass(input.get(), result.get());
    t_two = t.elapsed_ms();
    std::cout << "[Two-pass]     Work: 6 × W × H = " << 6LL * WIDTH * HEIGHT
              << " ops, Time: " << t_two << " ms" << std::endl;
    std::cout << "                Verify vs. single-pass: "
              << (verify(ref.get(), result.get(), n_pixels) ? "PASS" : "FAIL") << std::endl;

    // Strategy 3: Chunked (tiled, fused)
    t.start();
    blur_chunked(input.get(), result.get());
    t_chunk = t.elapsed_ms();
    // Theoretical work: ~6 * WIDTH * HEIGHT (approaches as chunk size grows)
    double work_factor = 6.0 + (4.0 / CHUNK_SIZE);   // exact: 6W*H + (4/CHUNK_SIZE)*W*H
    std::cout << "[Chunked]      Work: ~" << work_factor << " × W × H, Time: " << t_chunk << " ms" << std::endl;
    std::cout << "                Verify vs. single-pass: "
              << (verify(ref.get(), result.get(), n_pixels) ? "PASS" : "FAIL") << std::endl;

    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Key insight: cache-aware chunking reduces intermediate buffer" << std::endl;
    std::cout << "traffic while approaching the optimal 6×W×H work of separable filters." << std::endl;
    std::cout << "Halide automates these scheduling decisions via primitives like" << std::endl;
    std::cout << "  blurx.compute_at(out, x).vectorize(x, 8);" << std::endl;
    std::cout << "  out.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);" << std::endl;

    return 0;
}
