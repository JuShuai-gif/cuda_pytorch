// lecture9_part2.cpp
// Stanford CS149, Lecture 9: Efficiently Evaluating DNNs
// Part 2: Convolution Implementations
//
// Implements:
//   1. Direct 2D convolution (7-level nested loop)
//   2. Convolution as GEMM (im2col — explicit matrix construction)
//   3. Multi-channel convolution (batched)
//   4. ReLU activation and Max Pooling simulation
//   5. Arithmetic intensity analysis for each approach
//
// Compile: g++ -std=c++17 -O2 lecture9_part2.cpp -o lecture9_part2
// Run: ./lecture9_part2

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <algorithm>

// ============================================================================
// 4D Tensor: batch × height × width × channels
// Stores data in NHWC layout (batch, height, width, channels)
// ============================================================================

struct Tensor4D {
    std::vector<float> data;
    size_t N, H, W, C;  // batch, height, width, channels

    Tensor4D(size_t n, size_t h, size_t w, size_t c)
        : N(n), H(h), W(w), C(c), data(n * h * w * c, 0.0f) {}

    // Index into NHWC layout
    float& at(size_t n, size_t h, size_t w, size_t c) {
        return data[((n * H + h) * W + w) * C + c];
    }

    float at(size_t n, size_t h, size_t w, size_t c) const {
        return data[((n * H + h) * W + w) * C + c];
    }

    void randomize(float scale = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 17) * scale * 0.1f;
        }
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    bool equals(const Tensor4D& other, float tol = 0.001f) const {
        if (N != other.N || H != other.H || W != other.W || C != other.C) return false;
        for (size_t i = 0; i < data.size(); i++) {
            if (std::abs(data[i] - other.data[i]) > tol) return false;
        }
        return true;
    }
};

// ============================================================================
// Convolution weights: [num_filters][filter_h][filter_w][input_channels]
// ============================================================================

struct ConvWeights {
    std::vector<float> data;
    size_t F, H, W, C;  // numFilters, filterH, filterW, inputChannels

    ConvWeights(size_t f, size_t h, size_t w, size_t c)
        : F(f), H(h), W(w), C(c), data(f * h * w * c, 0.0f) {}

    float& at(size_t f, size_t h, size_t w, size_t c) {
        return data[((f * H + h) * W + w) * C + c];
    }

    float at(size_t f, size_t h, size_t w, size_t c) const {
        return data[((f * H + h) * W + w) * C + c];
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    void randomize(float scale = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 7 + 1) * scale * 0.05f;
        }
    }
};

// ============================================================================
// Print helper
// ============================================================================

void printChannel(const std::string& name, const Tensor4D& t,
                  size_t n, size_t c, size_t maxSize = 8)
{
    std::cout << name << " [batch=" << n << ", ch=" << c
              << "] " << t.H << "x" << t.W << ":\n";
    for (size_t h = 0; h < std::min(t.H, maxSize); h++) {
        std::cout << "  ";
        for (size_t w = 0; w < std::min(t.W, maxSize); w++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                      << t.at(n, h, w, c);
        }
        std::cout << "\n";
    }
}

// ============================================================================
// 1. Direct Convolution (7 nested loops)
// This is the "naive" implementation from the lecture.
//
// output[n][j][i][f] = bias[f] + Σ_kk Σ_jj Σ_ii
//   weights[f][jj][ii][kk] * input[n][j+jj][i+ii][kk]
//
// Key properties:
//   - Significant input data reuse: filter weights reused spatially
//   - Input values reused across different filters
//   - Low arithmetic intensity if implemented naively
// ============================================================================

Tensor4D convDirect(const Tensor4D& input,
                    const ConvWeights& weights,
                    const std::vector<float>& biases,
                    size_t stride = 1)
{
    size_t outH = (input.H - weights.H) / stride + 1;
    size_t outW = (input.W - weights.W) / stride + 1;

    Tensor4D output(input.N, outH, outW, weights.F);

    for (size_t n = 0; n < input.N; n++) {
        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                for (size_t f = 0; f < weights.F; f++) {
                    float sum = (f < biases.size()) ? biases[f] : 0.0f;

                    // Accumulate over input channels
                    for (size_t c = 0; c < input.C; c++) {
                        // Spatial convolution (kernel)
                        for (size_t kh = 0; kh < weights.H; kh++) {
                            for (size_t kw = 0; kw < weights.W; kw++) {
                                sum += weights.at(f, kh, kw, c)
                                     * input.at(n, oh * stride + kh, ow * stride + kw, c);
                            }
                        }
                    }

                    output.at(n, oh, ow, f) = sum;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// 2. Convolution via im2col + GEMM (Explicit Matrix Construction)
//
// This reshapes the input image into a "convolution matrix" where:
//   - Each row = filter-sized patch at a particular (oh, ow) position
//   - Matrix dimensions: (outH*outW) x (filterH*filterW*input.C)
// Then: C = W × X_col  (dense matrix multiply)
//
// Cost: O(R*S) storage overhead for im2col matrix
// ============================================================================

Tensor4D convIm2col(const Tensor4D& input,
                    const ConvWeights& weights,
                    const std::vector<float>& biases)
{
    size_t outH = input.H - weights.H + 1;
    size_t outW = input.W - weights.W + 1;
    size_t patchSize = weights.H * weights.W * input.C;  // R*S*C

    // Build im2col matrix X_col: (outH*outW) × patchSize
    // Build weight matrix W_mat:  numFilters × patchSize
    size_t X_rows = outH * outW;
    size_t X_cols = patchSize;

    std::vector<float> X_col(X_rows * X_cols, 0.0f);

    for (size_t oh = 0; oh < outH; oh++) {
        for (size_t ow = 0; ow < outW; ow++) {
            size_t row = oh * outW + ow;
            size_t col = 0;
            for (size_t c = 0; c < input.C; c++) {
                for (size_t kh = 0; kh < weights.H; kh++) {
                    for (size_t kw = 0; kw < weights.W; kw++) {
                        X_col[row * X_cols + col] = input.at(0, oh + kh, ow + kw, c);
                        col++;
                    }
                }
            }
        }
    }

    // W_mat is weights reshaped: F × patchSize
    std::vector<float> W_mat(weights.F * patchSize);
    for (size_t f = 0; f < weights.F; f++) {
        size_t col = 0;
        for (size_t c = 0; c < weights.C; c++) {
            for (size_t kh = 0; kh < weights.H; kh++) {
                for (size_t kw = 0; kw < weights.W; kw++) {
                    W_mat[f * patchSize + col] = weights.at(f, kh, kw, c);
                    col++;
                }
            }
        }
    }

    // GEMM: O_mat = W_mat × X_col^T
    // O_mat: F × (outH*outW)
    std::vector<float> O_mat(weights.F * X_rows, 0.0f);
    for (size_t f = 0; f < weights.F; f++) {
        for (size_t r = 0; r < X_rows; r++) {
            float sum = (f < biases.size()) ? biases[f] : 0.0f;
            for (size_t k = 0; k < patchSize; k++) {
                sum += W_mat[f * patchSize + k] * X_col[r * X_cols + k];
            }
            O_mat[f * X_rows + r] = sum;
        }
    }

    // Reshape O_mat back to Tensor4D
    Tensor4D output(input.N, outH, outW, weights.F);
    for (size_t f = 0; f < weights.F; f++) {
        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                size_t r = oh * outW + ow;
                output.at(0, oh, ow, f) = O_mat[f * X_rows + r];
            }
        }
    }

    return output;
}

// ============================================================================
// 3. ReLU Activation (element-wise)
// ReLU(x) = max(0, x)
// ============================================================================

void applyReLU(Tensor4D& tensor)
{
    for (float& v : tensor.data) {
        v = std::max(0.0f, v);
    }
}

// ============================================================================
// 4. Max Pooling (2x2)
// Reduces spatial dimensions by factor of 2
// ============================================================================

Tensor4D maxPool2x2(const Tensor4D& input)
{
    size_t outH = input.H / 2;
    size_t outW = input.W / 2;
    Tensor4D output(input.N, outH, outW, input.C);

    for (size_t n = 0; n < input.N; n++) {
        for (size_t c = 0; c < input.C; c++) {
            for (size_t oh = 0; oh < outH; oh++) {
                for (size_t ow = 0; ow < outW; ow++) {
                    float maxVal = input.at(n, oh * 2, ow * 2, c);
                    maxVal = std::max(maxVal, input.at(n, oh * 2,     ow * 2 + 1, c));
                    maxVal = std::max(maxVal, input.at(n, oh * 2 + 1, ow * 2,     c));
                    maxVal = std::max(maxVal, input.at(n, oh * 2 + 1, ow * 2 + 1, c));
                    output.at(n, oh, ow, c) = maxVal;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// 5. Single-channel 2D convolution (like the lecture's blur example)
// ============================================================================

std::vector<float> conv2DSingleChannel(const std::vector<float>& input,
                                       size_t W, size_t H,
                                       const std::vector<float>& kernel,
                                       size_t kW, size_t kH)
{
    size_t outW = W - kW + 1;
    size_t outH = H - kH + 1;
    std::vector<float> output(outW * outH, 0.0f);

    for (size_t j = 0; j < outH; j++) {
        for (size_t i = 0; i < outW; i++) {
            float sum = 0.0f;
            for (size_t jj = 0; jj < kH; jj++) {
                for (size_t ii = 0; ii < kW; ii++) {
                    sum += input[(j + jj) * W + (i + ii)]
                         * kernel[jj * kW + ii];
                }
            }
            output[j * outW + i] = sum;
        }
    }
    return output;
}

// ============================================================================
// Timing utility
// ============================================================================

template<typename Func>
double timeIt(Func f, const std::string& label) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  " << label << ": " << std::fixed << std::setprecision(2)
              << ms << " ms\n";
    return ms;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 9 Part 2: Convolution Implementations\n";
    std::cout << "==================================================\n\n";

    // ---- Single-channel convolution (blur example from lecture) ----
    {
        std::cout << "--- 1. Single-Channel 3x3 Convolution (Blur) ---\n";

        size_t W = 8, H = 8;
        std::vector<float> input(W * H);
        for (size_t i = 0; i < W * H; i++) input[i] = static_cast<float>(i);

        // 3x3 blur kernel (all 1/9)
        std::vector<float> blurKernel(9, 1.0f / 9.0f);

        auto output = conv2DSingleChannel(input, W, H, blurKernel, 3, 3);

        std::cout << "Input 8x8:\n";
        for (size_t j = 0; j < H; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W; i++)
                std::cout << std::setw(4) << static_cast<int>(input[j * W + i]);
            std::cout << "\n";
        }
        std::cout << "\nOutput after 3x3 blur (6x6):\n";
        for (size_t j = 0; j < H - 2; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W - 2; i++)
                std::cout << std::setw(7) << std::fixed << std::setprecision(2)
                         << output[j * (W - 2) + i];
            std::cout << "\n";
        }
    }

    // ---- Direct convolution vs im2col ----
    {
        std::cout << "\n--- 2. Direct vs im2col Convolution ---\n";

        size_t N = 1, H = 8, W = 8, C = 3;
        size_t numFilters = 4;
        size_t filterH = 3, filterW = 3;

        Tensor4D input(N, H, W, C);
        input.randomize(1.0f);

        ConvWeights weights(numFilters, filterH, filterW, C);
        weights.randomize(1.0f);

        std::vector<float> biases(numFilters, 0.1f);

        // Direct convolution
        auto outDirect = convDirect(input, weights, biases);

        // im2col convolution
        auto outIm2col = convIm2col(input, weights, biases);

        std::cout << "Input: " << N << "x" << H << "x" << W << "x" << C << "\n";
        std::cout << "Filters: " << numFilters << "x" << filterH << "x"
                  << filterW << "x" << C << "\n";
        std::cout << "Output: " << outDirect.N << "x" << outDirect.H
                  << "x" << outDirect.W << "x" << outDirect.C << "\n";

        // Print a channel for comparison
        printChannel("Input (channel 0)", input, 0, 0);
        printChannel("Direct output (filter 0)", outDirect, 0, 0);
        printChannel("im2col output (filter 0)", outIm2col, 0, 0);

        bool match = outDirect.equals(outIm2col, 0.01f);
        std::cout << "\nDirect == im2col: " << (match ? "PASSED" : "FAILED") << "\n";
    }

    // ---- ReLU + MaxPool pipeline ----
    {
        std::cout << "\n--- 3. Conv → ReLU → MaxPool Pipeline ---\n";

        size_t N = 1, H = 16, W = 16, C = 1;
        size_t numFilters = 2;
        size_t filterH = 3, filterW = 3;

        Tensor4D input(N, H, W, C);
        input.randomize();

        ConvWeights weights(numFilters, filterH, filterW, C);
        weights.randomize();

        std::vector<float> biases(numFilters, 0.0f);

        // Conv
        auto convOut = convDirect(input, weights, biases);
        std::cout << "After Conv: " << convOut.H << "x" << convOut.W << "x" << convOut.C << "\n";
        printChannel("  Filter 0 (before ReLU)", convOut, 0, 0, 8);

        // ReLU
        applyReLU(convOut);
        std::cout << "After ReLU:\n";
        printChannel("  Filter 0", convOut, 0, 0, 8);

        // MaxPool 2x2
        auto pooled = maxPool2x2(convOut);
        std::cout << "After MaxPool 2x2: " << pooled.H << "x" << pooled.W << "\n";
        printChannel("  Filter 0", pooled, 0, 0, 8);
    }

    // ---- Performance comparison ----
    {
        std::cout << "\n--- 4. Performance: Direct vs im2col (larger input) ---\n";

        size_t N = 1, H = 64, W = 64, C = 16;
        size_t numFilters = 32;
        size_t filterH = 3, filterW = 3;

        Tensor4D input(N, H, W, C);
        input.randomize();

        ConvWeights weights(numFilters, filterH, filterW, C);
        weights.randomize();

        std::vector<float> biases(numFilters, 0.1f);

        timeIt([&]() { convDirect(input, weights, biases); }, "Direct conv");
        timeIt([&]() { convIm2col(input, weights, biases); }, "im2col + GEMM");

        // im2col overhead analysis
        size_t outH = H - filterH + 1;
        size_t outW = W - filterW + 1;
        size_t patchSize = filterH * filterW * C;
        size_t im2colElements = outH * outW * patchSize;
        std::cout << "\n  im2col matrix size: " << im2colElements
                  << " elements (" << im2colElements * 4 / 1024 << " KB)\n";
        std::cout << "  Original input size: " << N * H * W * C * 4 / 1024 << " KB\n";
        std::cout << "  Overhead: " << std::fixed << std::setprecision(1)
                  << (static_cast<float>(im2colElements) / (N * H * W * C) - 1.0f) * 100.0f
                  << "% storage increase\n";
    }

    // ---- Convolution as a "pattern detector" ----
    {
        std::cout << "\n--- 5. Convolution as Pattern Detector ---\n";

        // Simple gradient detection kernels
        // Horizontal edge detector: [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        // (Sobel X operator)
        size_t W = 10, H = 10;

        // Create a simple test image with a vertical edge
        std::vector<float> image(W * H, 0.0f);
        // Left half = 0, right half = 10
        for (size_t j = 0; j < H; j++) {
            for (size_t i = 0; i < W; i++) {
                image[j * W + i] = (i >= W / 2) ? 10.0f : 0.0f;
            }
        }

        // Horizontal Sobel kernel (detects vertical edges)
        std::vector<float> sobelX = {1, 0, -1,
                                     2, 0, -2,
                                     1, 0, -1};

        auto edges = conv2DSingleChannel(image, W, H, sobelX, 3, 3);

        std::cout << "Input (vertical edge at center):\n";
        for (size_t j = 0; j < H; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W; i++)
                std::cout << std::setw(3) << static_cast<int>(image[j * W + i]);
            std::cout << "\n";
        }

        std::cout << "\nAfter Sobel-X filter (detects vertical edges):\n";
        for (size_t j = 0; j < H - 2; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W - 2; i++)
                std::cout << std::setw(6) << std::fixed << std::setprecision(0)
                         << edges[j * (W - 2) + i];
            std::cout << "\n";
        }
        std::cout << "  Note: high values at the vertical edge (center)\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - Direct convolution: 7-loop nest, reuse patterns\n";
    std::cout << "  - im2col: reshape input patches → matrix multiply\n";
    std::cout << "  - im2col overhead: O(R*S) storage increase\n";
    std::cout << "  - ReLU: element-wise max(0, x)\n";
    std::cout << "  - MaxPool: spatial downsampling (2x2)\n";
    std::cout << "  - Convolution as pattern detection\n";
    std::cout << "==================================================\n";

    return 0;
}
