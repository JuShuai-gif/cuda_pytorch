#define CATCH_CONFIG_MAIN
#include "../../third-party/catch.hpp"
#include "gemm_kernels.cuh"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// 性能测试辅助函数
struct BenchmarkResult {
    double time_ms;           // 平均时间(ms)
    double tflops;           // 性能(TFLOPS)
    double max_error;        // 最大误差
    bool correct;            // 是否正确
};

// 计算 FLOPs
long long compute_flops(int M, int K, int N) {
    // 矩阵乘法 C(M,N) = A(M,K) * B(K,N)
    // 每个输出元素需要 K 次乘法和 (K-1) 次加法
    // 总运算量 ≈ 2 * M * N * K
    return 2LL * M * N * K;
}

// 获取理论峰值性能 (需要根据你的 GPU 调整)
double get_theoretical_peak_tflops(const std::string& precision) {
    // RTX 4090 的理论峰值
    if (precision == "FP32") return 82.6;      // 82.6 TFLOPS
    if (precision == "FP16") return 165.2;     // 165.2 TFLOPS
    if (precision == "TF32") return 82.6;      // Tensor Core TF32 与 FP32 相同
    return 0.0;
}

// 性能测试函数
BenchmarkResult benchmark_gemm(
    const std::string& name,
    std::function<void()> kernel_func,
    int M,int K,int N,
    int warmup = 2,
    int repeats = 25,
    bool verbose = true
){
    BenchmarkResult result;

    // 预热
    if (verbose)
        printf(" Warming up...\n");

    for (int i = 0; i < warmup; ++i)
    {
        kernel_func();
    }

    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repeats; ++i)
    {
        kernel_func();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time_ms = 0;
    cudaEventElapsedTime(&elapsed_time_ms,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 计算性能
    result.time_ms = elapsed_time_ms / repeats;
    long long total_flops = compute_flops(M,K,N);
    double total_time_sec = elapsed_time_ms / 1000.0;
    result.tflops = (repeats * total_flops / total_time_sec) / 1e12;

    if (verbose)
    {
        printf("  Average time: %.4f ms\n", result.time_ms);
        printf("  Performance:  %.2f TFLOPS\n", result.tflops);
        
        double peak = get_theoretical_peak_tflops("FP32");
        if (peak > 0){
            printf("  Utilization:  %.1f%% of theoretical peak\n", 
                (result.tflops / peak) * 100);
        }
    }
    
    return result;
}


TEST_CASE("GEMM Performance Comparison", "[gemm][performance]") {
    
    // 测试不同规模的矩阵
    struct TestConfig {
        int M, K, N;
        std::string description;
    };

    std::vector<TestConfig> tests = {
        {32, 32, 32, "Small (32x32x32)"},
        {128, 128, 128, "Medium (128x128x128)"},
        {512, 512, 512, "Large (512x512x512)"},
        {1024, 1024, 1024, "Very Large (1024x1024x1024)"},
        {2048, 2048, 2048, "Huge (2048x2048x2048)"},
        {4096, 4096, 4096, "Massive (4096x4096x4096)"},
        // 非方形矩阵
        {1024, 512, 2048, "Rectangular (1024x512x2048)"},
        {2048, 1024, 512, "Rectangular (2048x1024x512)"},
        {512, 2048, 1024, "Rectangular (512x2048x1024)"},
    };

    SECTION("FP32 Performance Comparison") {
        printf("\n========== FP32 GEMM Performance Comparison ==========\n");
        printf("%-20s %-15s %-15s %-15s %-15s\n", 
               "Matrix Size", "PyTorch(ms)", "Naive(ms)", "PyTorch(TFLOPS)", "Naive(TFLOPS)");
        printf("--------------------------------------------------------\n");
        
        for (const auto& test : tests) {
            int M = test.M, K = test.K, N = test.N;
            long long flops = compute_flops(M, K, N);
            double theoretical_tflops = flops / 1e12 * 1000; // 理论值参考
            
            printf("\n%s (M=%d,K=%d,N=%d):\n", test.description.c_str(), M, K, N);
            printf("  Theoretical FLOPs: %.2e\n", (double)flops);
            
            // 创建随机矩阵
            auto options = torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(torch::kCUDA);

            auto A = torch::randn({M, K}, options);
            auto B = torch::randn({K, N}, options);
            auto C_torch = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
            auto C_naive = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
            
            // 1. PyTorch matmul (cuBLAS)
            auto pytorch_kernel = [&]() {
                C_torch = torch::matmul(A, B);
            };
            
            BenchmarkResult torch_result = benchmark_gemm(
                "PyTorch", pytorch_kernel, M, K, N, 10, 100, false
            );
            
            // 2. 你的 Naive GEMM
            auto naive_kernel = [&]() {
                sgemm_naive(A, B, C_naive, 1.0f, 0.0f);
            };
            
            BenchmarkResult naive_result = benchmark_gemm(
                "Naive", naive_kernel, M, K, N, 10, 100, false
            );
            
            // 3. 验证正确性
            auto expected = torch::matmul(A, B);
            auto diff = torch::abs(C_naive - expected);
            auto max_diff = torch::max(diff).item<float>();
            naive_result.max_error = max_diff;

            naive_result.correct = (max_diff < 1e-2f);
            
            // 打印对比结果
            printf("  PyTorch (cuBLAS): %10.5f ms  %10.5f TFLOPS\n", 
                   torch_result.time_ms, torch_result.tflops);
            printf("  Naive (Your GEMM): %10.5f ms  %10.5f TFLOPS", 
                   naive_result.time_ms, naive_result.tflops);
            
            if (!naive_result.correct) {
                printf("  [FAILED] Max error: %.2e\n", naive_result.max_error);
            } else {
                printf("  [PASSED] Max error: %.2e\n", naive_result.max_error);
            }
            
            // 计算加速比
            double speedup = naive_result.time_ms / torch_result.time_ms;
            printf("  Speedup (cuBLAS/Naive): %.2fx\n", speedup);
            printf("\n");
            
            // 断言：Naive 实现应该正确
            REQUIRE(naive_result.correct);
        }
    }
    
    SECTION("Detailed Profile with Different Sizes") {
        printf("\n========== Detailed Performance Profile ==========\n");
        
        // 测试不同形状对性能的影响
        struct ProfileCase {
            int M, K, N;
            std::string desc;
        };
        
        std::vector<ProfileCase> profiles = {
            {1024, 1024, 1024, "Square"},
            {1024, 1024, 2048, "Tall"},
            {1024, 2048, 1024, "Wide"},
            {2048, 1024, 1024, "Long"},
            {1024, 512, 2048, "Mixed 1"},
            {512, 1024, 2048, "Mixed 2"},
        };
        
        for (const auto& profile : profiles) {
            int M = profile.M, K = profile.K, N = profile.N;
            long long flops = compute_flops(M, K, N);
            
            printf("\n%s: %dx%dx%d (FLOPs: %.2e)\n", 
                   profile.desc.c_str(), M, K, N, (double)flops);
            
            auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
            auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
            auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
            
            // PyTorch
            auto torch_kernel = [&]() { C = torch::matmul(A, B); };
            BenchmarkResult torch_result = benchmark_gemm("PyTorch", torch_kernel, M, K, N, 5, 50, true);
            
            // Naive
            auto naive_kernel = [&]() { sgemm_naive(A, B, C, 1.0f, 0.0f); };
            BenchmarkResult naive_result = benchmark_gemm("Naive", naive_kernel, M, K, N, 5, 50, true);
            
            printf("  Performance Ratio (cuBLAS/Naive): %.2fx\n", 
                   naive_result.time_ms / torch_result.time_ms);
        }
    }
}




TEST_CASE("SGEMM Naive - Basic functionality", "[sgemm_naive]") {
    
    warmup_gpu();
    // Check if CUDA is available
    REQUIRE(torch::cuda::is_available());

    // Set seed for deterministic tests
    torch::manual_seed(42);

    SECTION("Small square matrices") {
        const int M = 32, K = 32, N = 32;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        sgemm_naive(A, B, C, 1.0f, 0.0f);
        printf("时间\n");
        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);
        REQUIRE(C.device().is_cuda());

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-4f);
    }

    SECTION("Rectangular matrices") {
        const int M = 64, K = 48, N = 32;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        sgemm_naive(A, B, C, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-4f);
    }

    SECTION("Alpha and beta scaling") {
        const int M = 32, K = 32, N = 32;
        float alpha = 2.0f;
        float beta = 0.5f;

        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C_init = torch::rand({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C = C_init.clone();

        sgemm_naive(A, B, C, alpha, beta);

        // Expected: C = alpha * (A @ B) + beta * C_init
        auto expected = alpha * torch::matmul(A, B) + beta * C_init;
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-3f);
    }

    SECTION("Large matrices") {
        const int M = 512, K = 512, N = 512;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        sgemm_naive(A, B, C, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-3f);
    }

    SECTION("Edge case - non-multiple of block size") {
        const int M = 33, K = 47, N = 29;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        sgemm_naive(A, B, C, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-4f);
    }
}