/*
 * 07_pruning.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * Pruning removes redundant weights or structural components to reduce
 * model size, FLOPs, and memory footprint while preserving accuracy.
 *
 * Two categories:
 *
 * 1. Unstructured Pruning — zero out individual weights based on importance.
 *    Creates sparse matrices; requires sparse hardware/software support.
 *    Importance criteria: magnitude, gradient, Wanda (|w| * |activation|).
 *
 * 2. Structured Pruning — remove entire architectural units:
 *    - Layer-level: remove whole Transformer blocks (coarsest)
 *    - Head-level: remove attention heads (medium)
 *    - Neuron-level: remove FFN neurons (finest)
 *
 * Iterative Pruning Strategy:
 *   Gradually increase sparsity (30% -> 50% -> 70% -> 90%),
 *   fine-tuning between pruning steps to recover accuracy.
 *   Critical: reapply masks after each training step to prevent
 *   pruned weights from being revived by the optimizer.
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

// ----------------------------------------------------------------
// Simple model for pruning demonstration
// ----------------------------------------------------------------
struct PrunableMLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    PrunableMLP(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

// ----------------------------------------------------------------
// 1. Magnitude-based Unstructured Pruning
//
// Zeros out weights with smallest absolute values.
// Returns a binary mask (1 for kept weights, 0 for pruned).
// ----------------------------------------------------------------
struct PruningMask {
    torch::Tensor mask; // same shape as weight, binary: 1 = keep, 0 = prune
};

std::unordered_map<std::string, PruningMask> magnitudePrune(
    std::shared_ptr<PrunableMLP> model,
    float sparsity) {
    std::unordered_map<std::string, PruningMask> masks;

    for (auto &p : model->named_parameters()) {
        if (p.key().find("weight") == std::string::npos) continue;

        auto weight = p.value();
        auto w_flat = weight.abs().view({-1});
        int64_t k = static_cast<int64_t>(w_flat.size(0) * sparsity);

        // Find threshold: k-th smallest absolute value
        auto [sorted, indices] = w_flat.sort();
        float threshold = (k > 0) ? sorted[static_cast<int>(k - 1)].item<float>() : 0.0f;

        // Create mask
        PruningMask pmask;
        pmask.mask = (weight.abs() > threshold).to(torch::kFloat32);
        masks[p.key()] = pmask;
    }

    return masks;
}

// ----------------------------------------------------------------
// Apply pruning masks to model weights
//
// mask[key].mask is applied to model.named_parameters()[key]
// ----------------------------------------------------------------
void applyPruningMasks(
    std::shared_ptr<PrunableMLP> model,
    const std::unordered_map<std::string, PruningMask> &masks) {
    for (auto &p : model->named_parameters()) {
        auto it = masks.find(p.key());
        if (it != masks.end()) {
            p.value().data().mul_(it->second.mask); // element-wise multiply
        }
    }
}

// ----------------------------------------------------------------
// Measure actual sparsity
// ----------------------------------------------------------------
float measureSparsity(std::shared_ptr<PrunableMLP> model) {
    int64_t total = 0, zeros = 0;
    for (auto &p : model->named_parameters()) {
        if (p.key().find("weight") == std::string::npos) continue;
        auto w = p.value();
        total += w.numel();
        zeros += (w == 0.0).sum().item<int64_t>();
    }
    return static_cast<float>(zeros) / static_cast<float>(total);
}

// ----------------------------------------------------------------
// 2. Structured Pruning: Head-level for Multi-Head Attention
//
// For a weight matrix W of shape (d_model, d_model) viewed as
// num_heads x d_k per head. Remove entire heads (rows) based on
// their L2 norm.
// ----------------------------------------------------------------
torch::Tensor pruneAttentionHeads(
    const torch::Tensor &weight, // (d_model, d_model)
    int num_heads,
    float head_sparsity) {
    int d_model = weight.size(0);
    int d_k = d_model / num_heads;

    // Compute L2 norm per head
    auto w_reshaped = weight.view({num_heads, d_k, d_model});
    auto head_norms = w_reshaped.pow(2).sum({1, 2}).sqrt(); // (num_heads,)

    // Determine which heads to keep
    int heads_to_prune = static_cast<int>(num_heads * head_sparsity);
    auto [sorted, indices] = head_norms.sort();
    auto prune_indices = indices.slice(0, 0, heads_to_prune); // smallest norms

    // Zero out pruned heads
    auto pruned = weight.clone();
    for (int i = 0; i < heads_to_prune; i++) {
        int head_idx = prune_indices[i].item<int>();
        pruned.index_put_(
            {torch::indexing::Slice(head_idx * d_k, (head_idx + 1) * d_k),
             torch::indexing::Slice()},
            0.0);
    }

    return pruned;
}

// ----------------------------------------------------------------
// 3. Iterative Pruning
//
// Train -> Prune -> Reapply mask & Fine-tune -> Increase sparsity -> Repeat
// ----------------------------------------------------------------
void iterativePruning(
    std::shared_ptr<PrunableMLP> model,
    const torch::Tensor &x_train,
    const torch::Tensor &y_train,
    const std::vector<float> &sparsity_schedule, // e.g., {0.3, 0.5, 0.7, 0.9}
    int fine_tune_epochs = 20) {
    auto optimizer = torch::optim::Adam(model->parameters(), 0.001);

    for (size_t stage = 0; stage < sparsity_schedule.size(); stage++) {
        float target_sparsity = sparsity_schedule[stage];
        std::cout << "\n  Stage " << (stage + 1) << "/" << sparsity_schedule.size()
                  << ": target sparsity = " << (target_sparsity * 100) << "%\n";

        // Prune
        auto masks = magnitudePrune(model, target_sparsity);
        applyPruningMasks(model, masks);
        float actual = measureSparsity(model);
        std::cout << "    After pruning: actual sparsity = "
                  << (actual * 100) << "%\n";

        // Fine-tune with mask enforcement
        for (int epoch = 0; epoch < fine_tune_epochs; epoch++) {
            model->train();
            optimizer.zero_grad();

            auto logits = model->forward(x_train);
            auto loss = torch::nn::functional::cross_entropy(logits, y_train);
            loss.backward();

            // Critical: zero out gradients of pruned weights before optimizer.step()
            // to prevent them from being revived by momentum
            for (auto &p : model->named_parameters()) {
                auto it = masks.find(p.key());
                if (it != masks.end()) {
                    if (p.value().grad().defined()) {
                        p.value().grad().data().mul_(it->second.mask);
                    }
                }
            }

            optimizer.step();

            // Reapply masks (in case optimizer + L2 reg revived some weights)
            applyPruningMasks(model, masks);

            if ((epoch + 1) % 10 == 0) {
                std::cout << "    Fine-tune epoch " << (epoch + 1)
                          << "/" << fine_tune_epochs
                          << " loss: " << loss.item<float>() << "\n";
            }
        }
    }
}

// ----------------------------------------------------------------
// Demo: Unstructured + Structured + Iterative pruning
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Pruning Demo ===\n\n";

    int input_dim = 10;
    int hidden_dim = 32;
    int output_dim = 4;
    int n_train = 100;

    // Synthetic data
    auto x_train = torch::randn({n_train, input_dim});
    auto y_train = torch::randint(0, output_dim, {n_train}).to(torch::kLong);

    // 1. Unstructured magnitude pruning
    std::cout << "--- 1. Unstructured Magnitude Pruning ---\n";
    auto model_unstruct = std::make_shared<PrunableMLP>(input_dim, hidden_dim, output_dim);

    // Train briefly
    {
        auto opt = torch::optim::Adam(model_unstruct->parameters(), 0.01);
        for (int e = 0; e < 30; e++) {
            opt.zero_grad();
            auto loss = torch::nn::functional::cross_entropy(
                model_unstruct->forward(x_train), y_train);
            loss.backward();
            opt.step();
        }
    }

    float before = measureSparsity(model_unstruct);
    auto masks = magnitudePrune(model_unstruct, 0.5); // prune 50%
    applyPruningMasks(model_unstruct, masks);
    float after = measureSparsity(model_unstruct);

    std::cout << "  Sparsity before: " << (before * 100) << "%\n";
    std::cout << "  Sparsity after 50% pruning: " << (after * 100) << "%\n\n";

    // 2. Structured head pruning demonstration
    std::cout << "--- 2. Structured Attention Head Pruning ---\n";
    int d_model = 64;
    int num_heads = 8;

    auto attn_weight = torch::randn({d_model, d_model});
    auto pruned_weight = pruneAttentionHeads(attn_weight, num_heads, 0.25); // remove 25% heads

    int heads_removed = 0;
    for (int h = 0; h < num_heads; h++) {
        int d_k = d_model / num_heads;
        auto head_slice = pruned_weight.slice(0, h * d_k, (h + 1) * d_k);
        if (head_slice.eq(0.0).all().item<bool>()) heads_removed++;
    }
    std::cout << "  Heads removed: " << heads_removed << "/" << num_heads << "\n";
    std::cout << "  Remaining FLOPs: ~" << ((num_heads - heads_removed) * 100.0 / num_heads)
              << "%\n\n";

    // 3. Iterative pruning
    std::cout << "--- 3. Iterative Pruning ---\n";
    auto model_iter = std::make_shared<PrunableMLP>(input_dim, hidden_dim, output_dim);

    // Copy weights from trained model
    {
        auto params_iter = model_iter->named_parameters();
        auto params_base = model_unstruct->named_parameters();
        // Note: we use fresh training here since model_unstruct is already pruned
    }

    // Train briefly then apply iterative pruning
    {
        auto opt = torch::optim::Adam(model_iter->parameters(), 0.01);
        for (int e = 0; e < 20; e++) {
            opt.zero_grad();
            auto loss = torch::nn::functional::cross_entropy(
                model_iter->forward(x_train), y_train);
            loss.backward();
            opt.step();
        }
    }

    std::vector<float> schedule = {0.3f, 0.5f, 0.7f};
    iterativePruning(model_iter, x_train, y_train, schedule, /*fine_tune_epochs=*/15);

    float final_sparsity = measureSparsity(model_iter);
    std::cout << "\n  Final sparsity: " << (final_sparsity * 100) << "%\n";

    // Evaluate
    {
        torch::NoGradGuard no_grad;
        model_iter->eval();
        auto pred = model_iter->forward(x_train).argmax(1);
        auto acc = pred.eq(y_train).to(torch::kFloat32).mean();
        std::cout << "  Training accuracy after iterative pruning: "
                  << acc.item<float>() << "\n";
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "Magnitude pruning: simple & effective for unstructured sparsity.\n";
    std::cout << "Structured pruning: removes entire heads/layers for hardware speedup.\n";
    std::cout << "Iterative pruning: gradual sparsity + fine-tuning preserves accuracy.\n";
    std::cout << "Key: always reapply masks to prevent weight revival.\n";

    return 0;
}
