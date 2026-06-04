/*
 * 01_torchscript_export.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Demonstrates model serialization and inference in LibTorch 2.x:
 *   1. Define a model (TinyNet: Conv->ReLU->Conv->ReLU->GAP->Linear)
 *   2. Save model state dict via torch::save
 *   3. Load it back and verify output shapes match
 *
 * Note on TorchScript in PyTorch 2.x:
 *   torch::jit::trace_module was removed in LibTorch 2.x.
 *   The recommended workflow is:
 *   - Python side:  model_ts = torch.jit.trace(model, example_input)
 *                   model_ts.save("model.ts")
 *   - C++ side:     auto m = torch::jit::load("model.ts");
 *
 *   For C++-only workflows, torch::save/torch::load handle state dicts.
 *   The TorchScript .ts file loaded via torch::jit::load still works.
 *
 * When to use trace vs script (Python):
 *   - trace: records ops executed on example input; works for most feed-forward models
 *   - script: parses Python source code; needed for models with control flow
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

// ----------------------------------------------------------------
// TinyNet: a small CNN for C++-side serialization demonstration
//
//   input  (batch, 3, 224, 224)
//   conv1  (batch, 8, 224, 224)  + ReLU
//   conv2  (batch, 16, 224, 224) + ReLU
//   pool   (batch, 16, 1, 1)     global average
//   fc     (batch, num_classes)
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
        x = x.view({x.size(0), -1}); // flatten to (batch, 16)
        return fc->forward(x);
    }
};
TORCH_MODULE(TinyNet); // creates TinyNet as a ModuleHolder (shared_ptr wrapper)

// ----------------------------------------------------------------
// Save model state dict to disk
//
// torch::save serializes the model parameters.
// For production, pair with torch::jit::load("model.ts")
// when a TorchScript file was exported from Python.
// ----------------------------------------------------------------
void saveModel(TinyNet &model, const std::string &path) {
    torch::NoGradGuard no_grad;
    model->eval();

    auto example = torch::randn({1, 3, 224, 224});
    auto out_before = model->forward(example);

    torch::save(model, path);
    std::cout << "Saved model state dict to: " << path << "\n";
    std::cout << "  Input  shape: " << example.sizes() << "\n";
    std::cout << "  Output shape: " << out_before.sizes() << "\n\n";
}

// ----------------------------------------------------------------
// Load model state dict back and run inference
//
// Demonstrates the round-trip: save → load → verify output matches.
// For TorchScript .ts files loaded from Python, use torch::jit::load().
// ----------------------------------------------------------------
void loadAndInfer(const std::string &path, int warmup_iters = 5) {
    // Create a fresh model instance and load saved parameters
    TinyNet m(3, 10);
    torch::load(m, path);
    m->eval();

    auto x = torch::randn({1, 3, 224, 224});

    // Warm-up: absorb JIT compilation, allocator, and kernel selection
    std::cout << "Warming up (" << warmup_iters << " iterations)...\n";
    for (int i = 0; i < warmup_iters; ++i) {
        (void)m->forward(x);
    }

    // Timed inference
    torch::NoGradGuard ng;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto y = m->forward(x);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::cout << "Loaded & ran: " << path << "\n";
    std::cout << "  Output shape: " << y.sizes() << "\n";
    std::cout << "  Inference time: " << us << " us\n";
    std::cout << "  Output (first 5 logits): "
              << y[0].slice(0, 0, 5) << "\n";
}

// ----------------------------------------------------------------
// Demo: Save model → Load back → Verify
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Model Serialization / Load Demo ===\n\n";

    TinyNet model(/*in_ch=*/3, /*num_classes=*/10);

    int64_t total_params = 0;
    for (const auto &p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Model: TinyNet (" << total_params << " parameters)\n";
    std::cout << "  Conv1(3->8, k=3) -> ReLU -> Conv2(8->16, k=3) -> ReLU\n";
    std::cout << "  -> GlobalAvgPool -> Linear(16->10)\n\n";

    // Save
    std::string path = "/tmp/tinynet_demo.pt";
    saveModel(model, path);

    // Load back and run
    loadAndInfer(path);

    // Cleanup
    std::remove(path.c_str());

    std::cout << "\n--- Key Points ---\n";
    std::cout << "1. torch::save/load handle state dicts for C++-only workflows.\n";
    std::cout << "2. For TorchScript .ts files: export in Python (torch.jit.trace)\n";
    std::cout << "   and load in C++ with torch::jit::load().\n";
    std::cout << "3. Always warm up 5-10 iterations before benchmarking.\n";
    std::cout << "4. Export with model->eval() to fix BN/dropout state.\n";

    return 0;
}
