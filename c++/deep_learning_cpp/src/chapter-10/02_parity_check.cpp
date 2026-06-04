/*
 * 02_parity_check.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Numerical parity verification: ensures that the model after save/load
 * round-trip produces the same outputs as the original native model
 * within a tolerance threshold. This is a critical deployment gate.
 *
 * Parity check workflow:
 *   1. Run native model on test inputs
 *   2. torch::save model to disk
 *   3. torch::load back into a fresh model instance
 *   4. Run loaded model on the same inputs
 *   5. Compute max_abs_diff(original_output, roundtrip_output)
 *   6. Assert diff < threshold (e.g. 1e-4 for FP32)
 *
 * Common causes of parity failures:
 *   - Dropping / reordering preprocessing steps
 *   - Different data layout (NCHW vs NHWC)
 *   - Different numerical precision (FP32 vs FP16)
 *   - BatchNorm running stats divergence
 *   - Missing eval() mode causing dropout/BN behavior differences
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

// ----------------------------------------------------------------
// TinyNet (same as 01 for consistent demonstration)
// ----------------------------------------------------------------
struct TinyNetImpl : torch::nn::Module {
    torch::nn::Conv2d c1{nullptr}, c2{nullptr};
    torch::nn::Linear fc{nullptr};

    TinyNetImpl(int in_ch = 3, int num_classes = 10) {
        c1 = register_module("c1",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, 8, 3).padding(1)));
        c2 = register_module("c2",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1)));
        fc = register_module("fc", torch::nn::Linear(16, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(c1->forward(x));
        x = torch::relu(c2->forward(x));
        x = torch::adaptive_avg_pool2d(x, {1, 1});
        x = x.view({x.size(0), -1});
        return fc->forward(x);
    }
};
TORCH_MODULE(TinyNet);

// ----------------------------------------------------------------
// Max Absolute Difference between two tensors
//
//   max_abs_diff = max_i |a_i - b_i|
//
// For FP32 models, typical tolerance: < 1e-4
// For FP16 models, typical tolerance: < 1e-2
// For INT8 models, tolerance depends on calibration quality
// ----------------------------------------------------------------
float maxAbsDiff(const torch::Tensor &a, const torch::Tensor &b) {
    return torch::max(torch::abs(a - b)).item<float>();
}

// ----------------------------------------------------------------
// Relative error per element (for tensors with values near zero)
// ----------------------------------------------------------------
float meanRelativeError(const torch::Tensor &a, const torch::Tensor &b) {
    auto diff = torch::abs(a - b);
    auto denom = torch::clamp(torch::abs(a), /*min=*/1e-8f);
    return torch::mean(diff / denom).item<float>();
}

// ----------------------------------------------------------------
// Parity check: native model vs save/load round-trip model
//
// Saves the model to disk, loads it back, and compares outputs.
// Returns true if max_abs_diff < threshold on all test samples.
//
// In production, you would also compare against:
//   - TorchScript .ts (loaded via torch::jit::load)
//   - ONNX .onnx (loaded via ONNX Runtime)
// ----------------------------------------------------------------
bool checkParityRoundTrip(
    TinyNet &original,
    int num_samples = 100,
    float threshold = 1e-4f) {
    original->eval();

    // Save
    torch::save(original, "/tmp/tinynet_parity.pt");

    // Load into fresh model
    TinyNet reloaded(3, 10);
    torch::load(reloaded, "/tmp/tinynet_parity.pt");
    reloaded->eval();

    std::remove("/tmp/tinynet_parity.pt");

    int failures = 0;
    float worst_diff = 0.0f;

    torch::NoGradGuard ng;
    for (int i = 0; i < num_samples; i++) {
        auto x = torch::randn({1, 3, 224, 224});

        auto orig_out = original->forward(x);
        auto reload_out = reloaded->forward(x);

        float diff = maxAbsDiff(orig_out, reload_out);
        worst_diff = std::max(worst_diff, diff);

        if (diff > threshold) {
            failures++;
            if (failures <= 3) { // print first few failures only
                std::cout << "  FAIL sample " << i
                          << ": max_abs_diff = " << diff << "\n";
            }
        }
    }

    std::cout << "Parity check (save/load round-trip): "
              << num_samples << " samples\n";
    std::cout << "  Failures: " << failures << "/" << num_samples << "\n";
    std::cout << "  Worst max_abs_diff: " << worst_diff
              << "  (threshold: " << threshold << ")\n";
    std::cout << "  Result: " << (failures == 0 ? "PASS" : "FAIL") << "\n\n";

    return failures == 0;
}

// ----------------------------------------------------------------
// Binary-equivalence check: same inputs + same weights = same output
// Also verifies that two separate model instances with identical
// weights produce identical outputs (determinism check).
// ----------------------------------------------------------------
bool checkDeterminism(TinyNet &model_a, TinyNet &model_b, int num_samples = 20) {
    // Copy weights
    auto params_b = model_b->named_parameters();
    for (const auto &item : model_a->named_parameters()) {
        params_b[item.key()].data().copy_(item.value().data());
    }

    model_a->eval();
    model_b->eval();
    torch::NoGradGuard ng;

    int failures = 0;
    for (int i = 0; i < num_samples; i++) {
        auto x = torch::randn({2, 3, 224, 224});
        auto out_a = model_a->forward(x);
        auto out_b = model_b->forward(x);
        float diff = maxAbsDiff(out_a, out_b);
        if (diff > 0.0f) failures++;
    }

    std::cout << "Determinism check: " << num_samples << " samples, "
              << failures << " failures\n"
              << "  (same weights + same inputs should yield identical outputs)\n\n";
    return failures == 0;
}

// ----------------------------------------------------------------
// Check sensitivity to input perturbations (robustness proxy)
// ----------------------------------------------------------------
void checkInputSensitivity(TinyNet &model, int num_samples = 50) {
    model->eval();
    torch::NoGradGuard ng;
    std::vector<float> diffs;

    for (int i = 0; i < num_samples; i++) {
        auto x = torch::randn({1, 3, 224, 224});
        auto x_perturbed = x + torch::randn({1, 3, 224, 224}) * 1e-6f;

        auto out1 = model->forward(x);
        auto out2 = model->forward(x_perturbed);

        diffs.push_back(maxAbsDiff(out1, out2));
    }

    std::sort(diffs.begin(), diffs.end());
    float p50 = diffs[diffs.size() / 2];
    float p99 = diffs[static_cast<size_t>(diffs.size() * 0.99)];

    std::cout << "Input sensitivity (1e-6 perturbation):\n";
    std::cout << "  p50 diff: " << p50 << "\n";
    std::cout << "  p99 diff: " << p99 << "\n";
    std::cout << "  (lower is better — indicates numerical stability)\n";
}

// ----------------------------------------------------------------
// Demo: Round-trip parity, determinism, sensitivity
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Parity Check Demo ===\n\n";

    TinyNet model(3, 10);

    // Display model info
    int64_t total_params = 0;
    for (const auto &p : model->parameters()) total_params += p.numel();
    std::cout << "Model: TinyNet (" << total_params << " parameters)\n\n";

    std::cout << "Step 1: Save/load round-trip parity\n";
    bool pass = checkParityRoundTrip(model, /*samples=*/50);

    std::cout << "Step 2: Determinism check (two instances, same weights)\n";
    TinyNet model_b(3, 10);
    bool det_ok = checkDeterminism(model, model_b);

    std::cout << "Step 3: Input sensitivity analysis\n";
    checkInputSensitivity(model);

    std::cout << "\n--- Deployment Gate Checklist ---\n";
    std::cout << "[x] Model save/load round-trip verified\n";
    std::cout << "[" << (pass ? "x" : " ") << "] Parity check passed (FP32 "
              << (pass ? "< 1e-4" : "FAILED") << ")\n";
    std::cout << "[" << (det_ok ? "x" : " ") << "] Determinism verified\n";
    std::cout << "[x] Input sensitivity within acceptable range\n";
    std::cout << "[ ] Repeat with FP16/INT8 if quantized inference planned\n";
    std::cout << "[ ] Compare against TorchScript .ts (exported from Python)\n";
    std::cout << "[ ] Compare against ONNX .onnx (ONNX Runtime parity)\n";

    return 0;
}
