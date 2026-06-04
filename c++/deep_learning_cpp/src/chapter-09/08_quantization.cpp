/*
 * 08_quantization.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * Quantization reduces model size by mapping 32-bit floating-point weights
 * to lower bit-width representations (typically int8). This yields ~4x
 * compression (e.g., 28 GB -> 7 GB for a 7B parameter model).
 *
 * Techniques demonstrated:
 *
 * 1. Asymmetric Quantization
 *    Maps [β, α] to [0, 2^n - 1]:
 *      S = (α - β) / (2^n - 1)                  // scale
 *      Z = clamp(round(-β / S), 0, 2^n - 1)     // zero point
 *      q = clamp(round(x / S) + Z, 0, 2^n - 1)  // quantize
 *      x̂ = S * (q - Z)                           // dequantize
 *
 * 2. Symmetric Quantization
 *    Maps [-α, α] to [-(2^{n-1}-1), 2^{n-1}-1]:
 *      S = α / (2^{n-1} - 1)
 *      q = clamp(round(x / S), -(2^{n-1}-1), 2^{n-1}-1)
 *      x̂ = S * q
 *    No zero-point needed, simpler but may introduce larger errors
 *    for asymmetric distributions.
 *
 * 3. Range Calibration Strategies
 *    - Min-Max: simple, sensitive to outliers
 *    - Percentile: clip extreme values (e.g. p0.1-p99.9)
 *    - MSE: minimize reconstruction error
 *    - Cross-Entropy: preserve output distribution
 *
 * 4. Per-Channel Quantization
 *    Compute separate (S, Z) per output channel instead of
 *    per-tensor. Handles varying magnitude across channels.
 *
 * 5. QAT (Quantization-Aware Training)
 *    Simulates quantization noise during training via fake_quantize.
 *    This pushes the model to converge to "wide" minima that are
 *    robust to quantization error.
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

// ----------------------------------------------------------------
// 1. Asymmetric Quantization
//
//   q = clamp(round(x / S) + Z, 0, 255)     (uint8)
//   x̂ = S * (q - Z)
// ----------------------------------------------------------------
struct AsymmetricQuantParams {
    float scale;
    int zero_point;
};

AsymmetricQuantParams computeAsymmetricParams(const torch::Tensor &x) {
    AsymmetricQuantParams params;
    auto beta = x.min().item<float>();  // minimum value
    auto alpha = x.max().item<float>(); // maximum value

    float range = alpha - beta;
    if (range < 1e-8) range = 1e-8; // prevent division by zero

    params.scale = range / 255.0f;
    params.zero_point = static_cast<int>(
        std::round(-beta / params.scale));
    params.zero_point = std::max(0, std::min(255, params.zero_point));

    return params;
}

// Quantize: float -> uint8 representation (stored as float for demo)
torch::Tensor asymmetricQuantize(const torch::Tensor &x,
                                 const AsymmetricQuantParams &params) {
    auto q = torch::round(x / params.scale) + params.zero_point;
    return torch::clamp(q, 0.0f, 255.0f);
}

// Dequantize: uint8 representation -> float
torch::Tensor asymmetricDequantize(const torch::Tensor &q,
                                   const AsymmetricQuantParams &params) {
    return params.scale * (q - params.zero_point);
}

// ----------------------------------------------------------------
// 2. Symmetric Quantization
//
//   q = clamp(round(x / S), -127, 127)     (int8)
//   x̂ = S * q
// ----------------------------------------------------------------
struct SymmetricQuantParams {
    float scale;
};

SymmetricQuantParams computeSymmetricParams(const torch::Tensor &x) {
    SymmetricQuantParams params;
    float alpha = std::max(std::abs(x.min().item<float>()),
                           std::abs(x.max().item<float>()));
    if (alpha < 1e-8) alpha = 1e-8;
    params.scale = alpha / 127.0f;
    return params;
}

torch::Tensor symmetricQuantize(const torch::Tensor &x,
                                const SymmetricQuantParams &params) {
    auto q = torch::round(x / params.scale);
    return torch::clamp(q, -127.0f, 127.0f);
}

torch::Tensor symmetricDequantize(const torch::Tensor &q,
                                  const SymmetricQuantParams &params) {
    return params.scale * q;
}

// ----------------------------------------------------------------
// 3. Percentile-Based Range Calibration
//
// Uses p-th and (100-p)-th percentiles instead of min/max
// to reduce outlier influence. Typically p = 0.1 or 1.0.
// ----------------------------------------------------------------
AsymmetricQuantParams computePercentileParams(const torch::Tensor &x,
                                              float percentile = 1.0f) {
    AsymmetricQuantParams params;
    auto x_flat = x.view({-1});
    int64_t N = x_flat.size(0);

    auto [sorted, _] = x_flat.sort();
    int64_t lo_idx = static_cast<int64_t>(N * percentile / 100.0f);
    int64_t hi_idx = static_cast<int64_t>(N * (100.0f - percentile) / 100.0f);
    lo_idx = std::max(int64_t(0), std::min(N - 1, lo_idx));
    hi_idx = std::max(int64_t(0), std::min(N - 1, hi_idx));

    float beta = sorted[static_cast<int>(lo_idx)].item<float>();
    float alpha = sorted[static_cast<int>(hi_idx)].item<float>();
    float range = alpha - beta;
    if (range < 1e-8) range = 1e-8;

    params.scale = range / 255.0f;
    params.zero_point = static_cast<int>(std::round(-beta / params.scale));
    params.zero_point = std::max(0, std::min(255, params.zero_point));

    return params;
}

// ----------------------------------------------------------------
// 4. Per-Channel Quantization
//
// Computes quant params for each output channel independently.
// Weight shape: (out_channels, in_channels)
// Returns one (scale, zero_point) per output channel.
// ----------------------------------------------------------------
struct PerChannelQuantParams {
    std::vector<float> scales;
    std::vector<int> zero_points;
    int num_channels;
};

PerChannelQuantParams computePerChannelParams(const torch::Tensor &weight) {
    PerChannelQuantParams pcq;
    pcq.num_channels = static_cast<int>(weight.size(0));

    for (int c = 0; c < pcq.num_channels; c++) {
        auto channel = weight[c]; // (in_channels,)
        auto beta = channel.min().item<float>();
        auto alpha = channel.max().item<float>();
        float range = alpha - beta;
        if (range < 1e-8) range = 1e-8;

        float scale = range / 255.0f;
        int zp = static_cast<int>(std::round(-beta / scale));
        zp = std::max(0, std::min(255, zp));

        pcq.scales.push_back(scale);
        pcq.zero_points.push_back(zp);
    }

    return pcq;
}

torch::Tensor perChannelQuantize(const torch::Tensor &weight,
                                 const PerChannelQuantParams &params) {
    auto q = weight.clone();
    for (int c = 0; c < params.num_channels; c++) {
        q[c] = torch::round(weight[c] / params.scales[c]) + params.zero_points[c];
        q[c] = torch::clamp(q[c], 0.0f, 255.0f);
    }
    return q;
}

torch::Tensor perChannelDequantize(const torch::Tensor &q,
                                   const PerChannelQuantParams &params) {
    auto x = q.clone();
    for (int c = 0; c < params.num_channels; c++) {
        x[c] = params.scales[c] * (q[c] - params.zero_points[c]);
    }
    return x;
}

// ----------------------------------------------------------------
// 5. Fake Quantization (for QAT)
//
// Simulates quantization noise without actually reducing precision:
//   x_fake = S * (clamp(round(x / S) + Z, 0, 255) - Z)
//
// The round() operation is non-differentiable, so during backward pass
// the gradient flows through unchanged (Straight-Through Estimator).
// ----------------------------------------------------------------
torch::Tensor fakeQuantize(const torch::Tensor &x,
                           const AsymmetricQuantParams &params) {
    // Forward: quantize then dequantize to simulate precision loss
    auto q = torch::round(x / params.scale) + params.zero_point;
    q = torch::clamp(q, 0.0f, 255.0f);
    return params.scale * (q - params.zero_point);
}

// ----------------------------------------------------------------
// Simple MLP for quantization demo
// ----------------------------------------------------------------
struct QuantMLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    QuantMLP(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);
    }
};

// ----------------------------------------------------------------
// Demo: Compare quantization methods and measure error
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Quantization Demo ===\n\n";

    // Synthetic weight tensor with outliers
    auto weight = torch::tensor({
                                    0.05, -0.12, 0.33, -0.07, 0.01, 0.42, -0.29, 0.18,
                                    -0.03, 0.25, -0.44, 0.09, -0.17, 0.88, -0.05, -0.33,
                                    0.11, -0.02, 0.50, -0.21, 0.04, -0.19, 0.67, -0.14,
                                    5.20, -0.08, 0.03, -0.37, 0.29, -0.55, 0.12, -0.02 // outlier at 5.20
                                },
                                torch::kFloat32)
                      .view({4, 8});

    std::cout << "Original weight tensor shape: " << weight.sizes() << "\n";
    std::cout << "  Range: [" << weight.min().item<float>()
              << ", " << weight.max().item<float>() << "]\n";
    std::cout << "  (note the outlier at 5.20)\n\n";

    // --- Asymmetric (Min-Max) ---
    std::cout << "--- 1. Asymmetric (Min-Max) Quantization ---\n";
    auto asym_params = computeAsymmetricParams(weight);
    auto q_asym = asymmetricQuantize(weight, asym_params);
    auto dq_asym = asymmetricDequantize(q_asym, asym_params);

    float mse_asym = torch::mse_loss(dq_asym, weight).item<float>();
    std::cout << "  Scale: " << asym_params.scale
              << ", Zero-point: " << asym_params.zero_point << "\n";
    std::cout << "  Reconstruction MSE: " << mse_asym << "\n";
    std::cout << "  Outlier 5.20 -> " << dq_asym[3][0].item<float>() << "\n\n";

    // --- Asymmetric (Percentile p=10%) ---
    std::cout << "--- 2. Asymmetric (Percentile p=10%) Quantization ---\n";
    auto pct_params = computePercentileParams(weight, 10.0f);
    auto q_pct = asymmetricQuantize(weight, pct_params);
    auto dq_pct = asymmetricDequantize(q_pct, pct_params);

    float mse_pct = torch::mse_loss(dq_pct, weight).item<float>();
    std::cout << "  Scale: " << pct_params.scale
              << ", Zero-point: " << pct_params.zero_point << "\n";
    std::cout << "  Reconstruction MSE: " << mse_pct << "\n";
    std::cout << "  Percentile clipping reduces outlier impact at cost of clipping rare values.\n\n";

    // --- Symmetric ---
    std::cout << "--- 3. Symmetric Quantization ---\n";
    auto sym_params = computeSymmetricParams(weight);
    auto q_sym = symmetricQuantize(weight, sym_params);
    auto dq_sym = symmetricDequantize(q_sym, sym_params);

    float mse_sym = torch::mse_loss(dq_sym, weight).item<float>();
    std::cout << "  Scale: " << sym_params.scale << "\n";
    std::cout << "  Reconstruction MSE: " << mse_sym << "\n\n";

    // --- Per-Channel ---
    std::cout << "--- 4. Per-Channel Quantization ---\n";
    auto pc_params = computePerChannelParams(weight);
    auto q_pc = perChannelQuantize(weight, pc_params);
    auto dq_pc = perChannelDequantize(q_pc, pc_params);

    float mse_pc = torch::mse_loss(dq_pc, weight).item<float>();
    std::cout << "  Per-channel scales: [";
    for (size_t i = 0; i < pc_params.scales.size(); i++) {
        std::cout << pc_params.scales[i];
        if (i + 1 < pc_params.scales.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "  Reconstruction MSE: " << mse_pc << "\n";
    std::cout << "  Per-channel handles varying magnitude across channels.\n\n";

    // --- Fake Quantization (QAT) demonstration ---
    std::cout << "--- 5. Fake Quantization (QAT) ---\n";
    int input_dim = 8, hidden_dim = 16, output_dim = 3;
    int n_train = 100;

    auto x_train = torch::randn({n_train, input_dim});
    auto y_train = torch::randint(0, output_dim, {n_train}).to(torch::kLong);

    auto model = std::make_shared<QuantMLP>(input_dim, hidden_dim, output_dim);

    // Train with fake quantization
    auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

    for (int epoch = 0; epoch < 30; epoch++) {
        model->train();

        // Calibrate quant params each epoch (in practice: every few epochs)
        auto w1_params = computeAsymmetricParams(model->fc1->weight);
        auto w2_params = computeAsymmetricParams(model->fc2->weight);

        optimizer.zero_grad();

        auto x = x_train;
        // Fake-quantize weights before forward pass
        auto orig_w1 = model->fc1->weight.data().clone();
        auto orig_w2 = model->fc2->weight.data().clone();

        model->fc1->weight.data().copy_(
            asymmetricDequantize(
                asymmetricQuantize(model->fc1->weight, w1_params), w1_params));
        model->fc2->weight.data().copy_(
            asymmetricDequantize(
                asymmetricQuantize(model->fc2->weight, w2_params), w2_params));

        auto logits = model->forward(x);
        auto loss = torch::nn::functional::cross_entropy(logits, y_train);
        loss.backward();

        // Restore original weights (gradients flow through STE)
        model->fc1->weight.data().copy_(orig_w1);
        model->fc2->weight.data().copy_(orig_w2);

        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "  Epoch " << (epoch + 1) << "/30 loss: "
                      << loss.item<float>() << "\n";
        }
    }

    // Final evaluation
    {
        torch::NoGradGuard no_grad;
        model->eval();
        auto pred = model->forward(x_train).argmax(1);
        auto acc = pred.eq(y_train).to(torch::kFloat32).mean();
        std::cout << "\n  QAT training accuracy: " << acc.item<float>() << "\n";
    }

    // Memory comparison
    float original_mb = weight.numel() * 4.0f / (1024 * 1024);
    float quantized_mb = weight.numel() * 1.0f / (1024 * 1024);
    std::cout << "\n--- Memory Comparison ---\n";
    std::cout << "  FP32: " << original_mb << " MB\n";
    std::cout << "  INT8: " << quantized_mb << " MB\n";
    std::cout << "  Compression ratio: 4x\n\n";

    // Summary
    std::cout << "--- Summary ---\n";
    std::cout << "Method         | MSE     | Notes\n";
    std::cout << "---------------|---------|----------------------------\n";
    std::cout << "Asym (Min-Max) | " << mse_asym << " | Outlier-sensitive\n";
    std::cout << "Asym (p=10%)   | " << mse_pct << " | Outlier-robust\n";
    std::cout << "Symmetric      | " << mse_sym << " | No zero-point\n";
    std::cout << "Per-Channel    | " << mse_pc << " | Finest granularity\n";

    return 0;
}
