/*
 * 01_self_attention.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * Scaled Dot-Product Attention is the core building block of the Transformer.
 * Given Query (Q), Key (K), and Value (V) tensors, it computes:
 *
 *   Attention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V
 *
 * Q: "What am I looking for?"
 * K: "What can I offer?"
 * V: "What is my actual content?"
 *
 * The sqrt(d_k) scaling prevents the dot products from growing too large,
 * which would push softmax into regions of extremely small gradients.
 *
 * Variants demonstrated:
 *   - Standard (bidirectional) attention
 *   - Causal (masked) attention for autoregressive decoding
 *   - Self-attention where Q=K=V from the same input
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// ----------------------------------------------------------------
// Scaled Dot-Product Attention (equation 1 in "Attention Is All You Need")
//
//   scores = Q @ K^T          (batch_size, num_heads, seq_len, seq_len)
//   scores = scores / sqrt(d_k)
//   if mask is provided: scores += mask
//   attn_weights = softmax(scores, dim=-1)
//   output = attn_weights @ V
// ----------------------------------------------------------------
torch::Tensor scaledDotProductAttention(
    const torch::Tensor &Q,
    const torch::Tensor &K,
    const torch::Tensor &V,
    const torch::Tensor &mask = {}) {
    auto d_k = static_cast<double>(K.size(-1)); // key dimension
    auto scale = 1.0 / std::sqrt(d_k);

    // attention scores: (batch, heads, seq_q, seq_k)
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;

    // apply mask if provided (e.g. causal mask for decoder)
    if (mask.defined()) {
        scores = scores + mask;
    }

    auto attention_weights = torch::softmax(scores, /*dim=*/-1);
    auto output = torch::matmul(attention_weights, V);

    return output;
}

// ----------------------------------------------------------------
// Self-Attention: Q, K, V are all projections of the same input X.
//
//   Q = X @ W_q,  K = X @ W_k,  V = X @ W_v
//
// Self-attention allows each position to attend to all other positions
// in the same sequence, resolving polysemy and long-range dependencies.
// ----------------------------------------------------------------
struct SelfAttention : torch::nn::Module {
    torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr};
    int d_model;

    SelfAttention(int d_model) : d_model(d_model) {
        auto opts = torch::nn::LinearOptions(d_model, d_model).bias(false);
        W_q = register_module("W_q", torch::nn::Linear(opts));
        W_k = register_module("W_k", torch::nn::Linear(opts));
        W_v = register_module("W_v", torch::nn::Linear(opts));
    }

    torch::Tensor forward(torch::Tensor x, const torch::Tensor &mask = {}) {
        auto Q = W_q->forward(x);
        auto K = W_k->forward(x);
        auto V = W_v->forward(x);

        // reshape for multi-head: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        // For single-head demo, we use 1 head.
        return scaledDotProductAttention(Q, K, V, mask);
    }
};

// ----------------------------------------------------------------
// Causal (Lower-Triangular) Mask
//
// Ensures position i can only attend to positions j <= i.
// Critical for autoregressive generation (GPT-style decoding).
//
//   mask[i, j] = 0    if j <= i   (allowed)
//   mask[i, j] = -inf if j > i    (masked)
// ----------------------------------------------------------------
torch::Tensor createCausalMask(int seq_len) {
    auto mask = torch::triu(
        torch::ones({seq_len, seq_len}) * (-1e9),
        /*diagonal=*/1);
    return mask;
}

// ----------------------------------------------------------------
// Demo: Compare bidirectional vs causal self-attention
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Self-Attention Demo ===\n\n";

    int batch_size = 2;
    int seq_len = 6;
    int d_model = 8;

    // synthetic input
    auto x = torch::randn({batch_size, seq_len, d_model});

    SelfAttention attn(d_model);

    // --- Bidirectional (encoder-style) ---
    std::cout << "--- Bidirectional Self-Attention (Encoder style) ---\n";
    {
        torch::NoGradGuard no_grad;
        auto out = attn.forward(x);
        std::cout << "Input  shape: " << x.sizes() << "\n";
        std::cout << "Output shape: " << out.sizes() << "\n";
        std::cout << "Sample output (batch 0, position 0): "
                  << out[0][0].slice(0, 0, 4) << "\n";
        std::cout << "Position 0 can attend to all " << seq_len
                  << " positions.\n\n";
    }

    // --- Causal (decoder-style) ---
    std::cout << "--- Causal Self-Attention (Decoder style) ---\n";
    {
        torch::NoGradGuard no_grad;
        auto mask = createCausalMask(seq_len);
        auto out = attn.forward(x, mask);

        std::cout << "Causal mask (0=allowed, -inf=masked):\n";
        std::cout << (mask > -1.0).to(torch::kInt32) << "\n";
        std::cout << "Position 0 can only attend to itself.\n";
        std::cout << "Position 5 can attend to positions 0-5.\n\n";
    }

    // --- Attention weight inspection ---
    std::cout << "--- Attention Weight Inspection ---\n";
    {
        torch::NoGradGuard no_grad;
        auto Q = attn.W_q->forward(x);
        auto K = attn.W_k->forward(x);
        auto V = attn.W_v->forward(x);

        auto scores = torch::matmul(Q, K.transpose(-2, -1))
                      / std::sqrt(static_cast<double>(d_model));

        std::cout << "Raw attention scores (batch 0):\n"
                  << scores[0] << "\n\n";

        auto weights = torch::softmax(scores, /*dim=*/-1);
        std::cout << "Attention weights (batch 0, row 0): "
                  << weights[0][0] << "\n";
        std::cout << "Sum of weights (should be 1.0): "
                  << weights[0][0].sum().item<float>() << "\n";
    }

    return 0;
}
