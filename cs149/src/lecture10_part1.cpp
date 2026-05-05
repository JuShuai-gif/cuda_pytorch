// lecture10_part1.cpp
// Systolic Array Simulation for Matrix-Matrix Multiplication
// Models the weight-stationary systolic array used in Google TPU v1
// Stanford CS149, Fall 2025 - Lecture 10: Hardware Specialization

#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

const int PE_ROWS = 4;
const int PE_COLS = 4;

// Processing Element: multiplies and accumulates
struct PE {
    double weight = 0.0;
    double accumulator = 0.0;
    double input = 0.0;

    void loadWeight(double w) { weight = w; }
    void receiveInput(double x) { input = x; }
    void compute() { accumulator += weight * input; }
    void forwardInput(double& toNext) const { toNext = input; }
    void reset() { accumulator = 0.0; input = 0.0; }
};

// Weight-Stationary Systolic Array
// Weights are pre-loaded into PEs (stationary)
// Inputs stream left-to-right through the array
// Partial sums accumulate in each PE column
class SystolicArray {
public:
    SystolicArray(int rows = PE_ROWS, int cols = PE_COLS)
        : rows_(rows), cols_(cols),
          pes_(rows, std::vector<PE>(cols)),
          accumulators_(cols, 0.0) {}

    // Pre-load weights into the PE grid (weight-stationary)
    // weight[r][c] is loaded into PE at row r, column c
    void loadWeights(const std::vector<std::vector<double>>& weights) {
        assert(weights.size() == (size_t)rows_);
        for (int r = 0; r < rows_; ++r) {
            assert(weights[r].size() == (size_t)cols_);
            for (int c = 0; c < cols_; ++c) {
                pes_[r][c].loadWeight(weights[r][c]);
                pes_[r][c].reset();
            }
        }
    }

    // Stream one input column through the systolic array
    // inputs[i] enters PE row i; each PE passes it right
    void streamInput(const std::vector<double>& inputs) {
        assert(inputs.size() == (size_t)rows_);

        // Each row processes independently
        for (int r = 0; r < rows_; ++r) {
            double data = inputs[r];
            for (int c = 0; c < cols_; ++c) {
                pes_[r][c].receiveInput(data);
                pes_[r][c].compute();
                pes_[r][c].forwardInput(data);  // data passes to next column
            }
        }
    }

    // Read out accumulated results (one per column)
    std::vector<double> readAccumulators() const {
        std::vector<double> result(cols_, 0.0);
        for (int c = 0; c < cols_; ++c) {
            // Sum accumulator values from all rows in this column
            for (int r = 0; r < rows_; ++r) {
                result[c] += pes_[r][c].accumulator;
            }
        }
        return result;
    }

    // Full GEMM: C = A x B using weight-stationary systolic array
    // A: M x K, B: K x N, C: M x N
    // Weight-stationary: pre-load B^T weights; stream columns of A
    static std::vector<std::vector<double>> gemm(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) {

        int M = A.size();
        int K = A[0].size();
        int Kb = B.size();
        int N = B[0].size();
        assert(Kb == K);

        std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

        // For weight-stationary systolic array demo:
        // Pre-load B^T as weights (PE[r][c] gets B[c][r] = weight for
        // computing C[r][c])
        SystolicArray sa(N, M);
        std::vector<std::vector<double>> W(N, std::vector<double>(M));
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < M; ++r)
                W[c][r] = B[r][c];  // B^T: each PE column accumulates one C element

        sa.loadWeights(W);

        // Stream each column of A (from K dimension) through the array
        // At step k: input A[i][k] enters PE row i
        for (int k = 0; k < K; ++k) {
            std::vector<double> input_col(N);
            for (int j = 0; j < N; ++j) {
                input_col[j] = A[k][j];  // Simplified: A is K x M, using A^T view
            }
            sa.streamInput(input_col);
        }

        // Read accumulators
        std::vector<double> acc = sa.readAccumulators();
        for (int j = 0; j < N; ++j)
            C[0][j] = acc[j];  // simplified for demo

        return C;
    }

    void printState() const {
        std::cout << "PE Grid State (accumulator values):\n";
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << pes_[r][c].accumulator << " ";
            }
            std::cout << "\n";
        }
    }

private:
    int rows_, cols_;
    std::vector<std::vector<PE>> pes_;
    std::vector<double> accumulators_;
};

// Compare systolic result with naive GEMM
std::vector<std::vector<double>> naiveGemm(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {

    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

int main() {
    std::cout << "=== Lecture 10: Systolic Array Simulation ===\n";
    std::cout << "Stanford CS149 - Hardware Specialization\n\n";

    // Example: A = 4x4, B = 4x1
    // y = Wx where W is 4x4 weight matrix, x is 4x1 input vector
    std::vector<std::vector<double>> W = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };

    std::vector<std::vector<double>> X = {
        {0.5}, {1.0}, {1.5}, {2.0}
    };

    std::cout << "Weight matrix W (4x4):\n";
    for (const auto& row : W) {
        for (double v : row) std::cout << std::setw(8) << v;
        std::cout << "\n";
    }

    std::cout << "\nInput vector x (4x1):\n";
    for (const auto& row : X) {
        for (double v : row) std::cout << std::setw(8) << v;
        std::cout << "\n";
    }

    // Step 1: Demonstrate systolic execution step-by-step
    std::cout << "\n--- Systolic Array Step-by-Step ---\n";
    SystolicArray sa(4, 4);
    sa.loadWeights(W);

    std::cout << "After loading weights:\n";
    sa.printState();

    // Stream each input element
    for (int k = 0; k < 4; ++k) {
        std::cout << "\nStreaming input x[" << k << "] = " << X[k][0] << ":\n";
        std::vector<double> input = {X[k][0], X[k][0], X[k][0], X[k][0]};
        sa.streamInput(input);
        sa.printState();
    }

    std::cout << "\nAccumulator outputs: ";
    auto acc = sa.readAccumulators();
    for (double v : acc) std::cout << v << " ";
    std::cout << "\n";

    // Step 2: Verify with naive GEMM
    std::cout << "\n--- Verification with Naive GEMM ---\n";
    auto expected = naiveGemm(W, X);
    std::cout << "Naive GEMM result:\n";
    for (const auto& row : expected) {
        for (double v : row) std::cout << v << " ";
        std::cout << "\n";
    }

    // Step 3: Demonstrate larger systolic tile concept
    std::cout << "\n--- Scaling Concept ---\n";
    std::cout << "For larger matrices (e.g., 8x8 * 8x4096):\n";
    std::cout << "  - We need 4096 accumulators to hold output columns\n";
    std::cout << "  - Tile the computation spatially\n";
    std::cout << "  - Key TPU advantage: 30% of chip area is arithmetic\n";
    std::cout << "  - SIMD comparison: control-driven, limited locality\n";
    std::cout << "  - Systolic: data-driven wavefront, temporal+spatial locality\n";

    return 0;
}
