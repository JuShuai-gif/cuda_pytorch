/*
 * 02_multi_head_attention.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * Multi-Head Attention (MHA) runs multiple attention operations in parallel.
 * Analogy: like multiple feature maps in CNNs, each head learns different
 * linguistic patterns (syntax, semantics, long-range dependencies, local
 * interactions).
 *
 * Process:
 *   1. Project Q, K, V into d_model-dimensional space
 *   2. Split into num_heads groups (each of dimension d_k = d_model/num_heads)
 *   3. Compute scaled dot-product attention per head independently
 *   4. Concatenate head outputs
 *   5. Final linear projection to d_model
 *
 * With num_heads = h, each head operates on d_k dimensions, so total
 * computation is equivalent to single-head attention on d_model dimensions.
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// ----------------------------------------------------------------
// Scaled Dot-Product Attention (repeated from 01 for self-contained build)
// ----------------------------------------------------------------
torch::Tensor scaledDotProductAttention(
    const torch::Tensor &Q,
    const torch::Tensor &K,
    const torch::Tensor &V,
    const torch::Tensor &mask = {}) {
    auto d_k = static_cast<double>(K.size(-1));
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k);

    if (mask.defined()) {
        scores = scores + mask;
    }

    auto attn_weights = torch::softmax(scores, /*dim=*/-1);
    return torch::matmul(attn_weights, V);
}

// ----------------------------------------------------------------
// Multi-Head Attention Module
//
// Shapes (batch-first convention):
//   Input:  (batch_size, seq_len, d_model)
//   Output: (batch_size, seq_len, d_model)
//
// Internal projection:
//   d_model -> (Q, K, V) each of d_model, then split into heads
// ----------------------------------------------------------------
struct MultiHeadAttention : torch::nn::Module {
    torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr};
    torch::nn::Linear W_o{nullptr}; // output projection
    int d_model, num_heads, d_k;

    MultiHeadAttention(int d_model, int num_heads) : d_model(d_model),
                                                     num_heads(num_heads),
                                                     d_k(d_model / num_heads) {
        auto opts = torch::nn::LinearOptions(d_model, d_model).bias(false);
        W_q = register_module("W_q", torch::nn::Linear(opts));
        W_k = register_module("W_k", torch::nn::Linear(opts));
        W_v = register_module("W_v", torch::nn::Linear(opts));
        W_o = register_module("W_o", torch::nn::Linear(opts));
    }

    torch::Tensor forward(
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
        const torch::Tensor &mask = {}) {
        int batch_size = query.size(0);
        int seq_len_q = query.size(1);
        int seq_len_k = key.size(1);

        // Linear projection and split into heads
        // (batch, seq, d_model) -> (batch, seq, num_heads, d_k)
        // -> (batch, num_heads, seq, d_k)
        auto Q = W_q->forward(query)
                     .view({batch_size, seq_len_q, num_heads, d_k})
                     .transpose(1, 2);
        auto K = W_k->forward(key)
                     .view({batch_size, seq_len_k, num_heads, d_k})
                     .transpose(1, 2);
        auto V = W_v->forward(value)
                     .view({batch_size, seq_len_k, num_heads, d_k})
                     .transpose(1, 2);

        // Scaled dot-product attention per head (batched)
        // If mask is (seq_len, seq_len), broadcast to (batch, heads, seq, seq)
        auto mask_expanded = mask;
        if (mask.defined() && mask.dim() == 2) {
            mask_expanded = mask.unsqueeze(0).unsqueeze(0); // (1, 1, seq, seq)
        }

        auto attn_output = scaledDotProductAttention(Q, K, V, mask_expanded);

        // Concatenate heads and project
        // (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2)
                          .contiguous()
                          .view({batch_size, seq_len_q, d_model});

        return W_o->forward(attn_output);
    }
};

// ----------------------------------------------------------------
// Demo: Multi-Head Attention with bidirectional and causal modes
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Multi-Head Attention Demo ===\n\n";

    int batch_size = 2;
    int seq_len = 5;
    int d_model = 128;
    int num_heads = 8; // d_k = 128 / 8 = 16

    std::cout << "Configuration:\n";
    std::cout << "  d_model = " << d_model << "\n";
    std::cout << "  num_heads = " << num_heads << "\n";
    std::cout << "  d_k per head = " << (d_model / num_heads) << "\n\n";

    auto x = torch::randn({batch_size, seq_len, d_model});

    MultiHeadAttention mha(d_model, num_heads);

    // --- Bidirectional multi-head attention ---
    std::cout << "--- Bidirectional MHA ---\n";
    {
        torch::NoGradGuard no_grad;
        auto out = mha.forward(x, x, x);
        std::cout << "Input shape:  " << x.sizes() << "\n";
        std::cout << "Output shape: " << out.sizes() << "\n\n";
    }

    // --- Causal multi-head attention ---
    std::cout << "--- Causal MHA (autoregressive) ---\n";
    {
        torch::NoGradGuard no_grad;
        auto mask = torch::triu(
            torch::ones({seq_len, seq_len}) * (-1e9), /*diagonal=*/1);
        auto out = mha.forward(x, x, x, mask);

        std::cout << "Causal mask shape: " << mask.sizes() << "\n";
        std::cout << "Output shape: " << out.sizes() << "\n";
        std::cout << "Position 0 output differs from bidirectional ("
                  << torch::mse_loss(out, mha.forward(x, x, x)).item<float>()
                  << " MSE)\n\n";
    }

    // --- Cross-attention demo (Q != K,V) ---
    std::cout << "--- Cross-Attention Demo ---\n";
    {
        torch::NoGradGuard no_grad;
        int tgt_len = 5;
        int src_len = 7;

        auto decoder_hidden = torch::randn({batch_size, tgt_len, d_model});
        auto encoder_output = torch::randn({batch_size, src_len, d_model});

        auto cross_out = mha.forward(decoder_hidden, encoder_output, encoder_output);

        std::cout << "Query (decoder) shape: " << decoder_hidden.sizes() << "\n";
        std::cout << "Key/Value (encoder) shape: " << encoder_output.sizes() << "\n";
        std::cout << "Cross-attn output shape: " << cross_out.sizes() << "\n";
        std::cout << "Decoder position i attends to ALL encoder positions (no mask).\n";
    }

    return 0;
}
