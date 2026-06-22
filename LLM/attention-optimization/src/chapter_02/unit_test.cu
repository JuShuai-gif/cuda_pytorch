/**
 * Unit tests for Chapter 02 CUDA attention kernels.
 *
 * The tests compare both GPU kernels against a CPU reference on small tensors.
 * They are intentionally small so they can run quickly in CI or during local
 * development while still covering numerical stability and boundary shapes.
 */

#define ATTENTION_CH02_NO_MAIN
#include "cuda_naive_attention.cu"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

void cpu_attention_reference(const std::vector<float>& Q,
                             const std::vector<float>& K,
                             const std::vector<float>& V,
                             std::vector<float>* O,
                             int N, int d_k, int d_v) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
  O->assign(N * d_v, 0.0f);

  for (int row = 0; row < N; ++row) {
    std::vector<float> scores(N);
    float max_score = -INFINITY;

    for (int j = 0; j < N; ++j) {
      float dot = 0.0f;
      for (int m = 0; m < d_k; ++m) {
        dot += Q[row * d_k + m] * K[j * d_k + m];
      }
      scores[j] = dot * scale;
      max_score = std::max(max_score, scores[j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < N; ++j) {
      scores[j] = std::exp(scores[j] - max_score);
      sum_exp += scores[j];
    }

    for (int col = 0; col < d_v; ++col) {
      float acc = 0.0f;
      for (int j = 0; j < N; ++j) {
        acc += (scores[j] / sum_exp) * V[j * d_v + col];
      }
      (*O)[row * d_v + col] = acc;
    }
  }
}

void expect_close(const std::vector<float>& actual,
                  const std::vector<float>& expected,
                  float atol = 1e-3f, float rtol = 1e-3f) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    const float limit = atol + rtol * std::fabs(expected[i]);
    if (!(diff <= limit)) {
      std::cerr << "Mismatch at " << i << ": got " << actual[i]
                << ", expected " << expected[i]
                << ", diff " << diff << ", limit " << limit << "\n";
      assert(false);
    }
  }
}

std::vector<float> make_input(int size, float offset) {
  std::vector<float> x(size);
  for (int i = 0; i < size; ++i) {
    x[i] = std::sin(0.17f * static_cast<float>(i) + offset) * 0.5f;
  }
  return x;
}

void run_gpu_case(int N, int d_k, int d_v, bool use_shared_memory) {
  const auto Q = make_input(N * d_k, 0.1f);
  const auto K = make_input(N * d_k, 0.7f);
  const auto V = make_input(N * d_v, 1.3f);
  std::vector<float> expected;
  cpu_attention_reference(Q, K, V, &expected, N, d_k, d_v);

  float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
  CUDA_CHECK(cudaMalloc(&d_Q, Q.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, K.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V, V.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O, expected.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));

  if (use_shared_memory) {
    dim3 block(16, 1);
    dim3 grid((d_v + block.x - 1) / block.x, N);
    size_t smem_bytes = SMEM_TILE_K * (d_k + block.x) * sizeof(float);
    naive_attention_smem_kernel<<<grid, block, smem_bytes>>>(d_Q, d_K, d_V, d_O,
                                                             N, d_k, d_v);
  } else {
    dim3 block(16, 16);
    dim3 grid((d_v + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    naive_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d_k, d_v);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> actual(expected.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), d_O, actual.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  expect_close(actual, expected);

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
}

void test_global_kernel_matches_cpu() {
  run_gpu_case(4, 8, 5, false);
  run_gpu_case(7, 16, 9, false);
  std::cout << "  PASS: global kernel matches CPU reference\n";
}

void test_shared_memory_kernel_matches_cpu() {
  run_gpu_case(4, 8, 5, true);
  run_gpu_case(9, 16, 7, true);
  std::cout << "  PASS: shared-memory kernel matches CPU reference\n";
}

}  // namespace

int main() {
  std::cout << "Chapter 02 CUDA Unit Tests\n";
  std::cout << std::string(40, '=') << "\n";

  test_global_kernel_matches_cpu();
  test_shared_memory_kernel_matches_cpu();

  std::cout << std::string(40, '=') << "\n";
  std::cout << "All tests passed!\n";
  return 0;
}
