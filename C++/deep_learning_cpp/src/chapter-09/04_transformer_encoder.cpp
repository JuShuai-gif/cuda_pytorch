/*
 * 04_transformer_encoder.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * The Transformer Encoder processes input sequences with full bidirectional
 * context. Each encoder layer consists of:
 *
 *   x = x + MultiHeadAttention(LayerNorm(x))      // Self-Attention + residual
 *   x = x + FeedForward(LayerNorm(x))             // FFN + residual
 *
 * The encoder stack is the foundation of BERT and other understanding models.
 * It is also used as the "conditioning" network in encoder-decoder models.
 *
 * Key architectural decisions:
 *   - Pre-LayerNorm vs Post-LayerNorm (pre-LN is more stable for deep stacks)
 *   - GeLU activation (BERT) vs ReLU (original Transformer)
 *   - Dropout after each sub-layer for regularization
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// ----------------------------------------------------------------
// Scaled Dot-Product Attention
// ----------------------------------------------------------------
torch::Tensor scaledDotProductAttention(
    const torch::Tensor &Q,
    const torch::Tensor &K,
    const torch::Tensor &V,
    const torch::Tensor &mask = {}) {
    auto d_k = static_cast<double>(K.size(-1));
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k);
    if (mask.defined()) scores = scores + mask;
    auto attn_weights = torch::softmax(scores, /*dim=*/-1);
    return torch::matmul(attn_weights, V);
}

// ----------------------------------------------------------------
// Transformer Encoder Layer (all sub-modules inlined)
//
// Uses Pre-LayerNorm (norm then sub-layer, then residual):
//
//   x = x + SelfAttention(LayerNorm(x))
//   x = x + FeedForward(LayerNorm(x))
// ----------------------------------------------------------------
struct TransformerEncoderLayer : torch::nn::Module {
    // Multi-Head Self-Attention components
    torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr}, W_o{nullptr};
    // Feed-Forward components
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    // Normalization and regularization
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout attn_dropout{nullptr}, ffn_dropout{nullptr};

    int d_model, num_heads, d_k, d_ff;

    TransformerEncoderLayer(
        int d_model,
        int num_heads,
        int d_ff,
        float dropout_rate = 0.1) : d_model(d_model),
                                    num_heads(num_heads),
                                    d_k(d_model / num_heads),
                                    d_ff(d_ff) {
        auto attn_opts = torch::nn::LinearOptions(d_model, d_model).bias(false);
        W_q = register_module("W_q", torch::nn::Linear(attn_opts));
        W_k = register_module("W_k", torch::nn::Linear(attn_opts));
        W_v = register_module("W_v", torch::nn::Linear(attn_opts));
        W_o = register_module("W_o", torch::nn::Linear(attn_opts));

        fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
        fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));

        norm1 = register_module("norm1",
                                torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm2 = register_module("norm2",
                                torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        attn_dropout = register_module("attn_dropout",
                                       torch::nn::Dropout(dropout_rate));
        ffn_dropout = register_module("ffn_dropout",
                                      torch::nn::Dropout(dropout_rate));
    }

    // Multi-head self-attention sub-layer
    torch::Tensor selfAttention(torch::Tensor x, const torch::Tensor &mask = {}) {
        int B = x.size(0), S = x.size(1);

        auto Q = W_q->forward(x).view({B, S, num_heads, d_k}).transpose(1, 2);
        auto K = W_k->forward(x).view({B, S, num_heads, d_k}).transpose(1, 2);
        auto V = W_v->forward(x).view({B, S, num_heads, d_k}).transpose(1, 2);

        auto mask_exp = mask;
        if (mask.defined() && mask.dim() == 2) {
            mask_exp = mask.unsqueeze(0).unsqueeze(0);
        }

        auto attn_out = scaledDotProductAttention(Q, K, V, mask_exp);
        attn_out = attn_out.transpose(1, 2).contiguous().view({B, S, d_model});
        return W_o->forward(attn_dropout->forward(attn_out));
    }

    // Feed-forward sub-layer with GeLU activation (BERT-style)
    torch::Tensor feedForward(torch::Tensor x) {
        x = fc1->forward(x);
        x = torch::gelu(x);
        x = ffn_dropout->forward(x);
        return fc2->forward(x);
    }

    torch::Tensor forward(torch::Tensor x, const torch::Tensor &mask = {}) {
        // Self-Attention sub-layer (pre-LN)
        x = x + selfAttention(norm1->forward(x), mask);
        // Feed-Forward sub-layer (pre-LN)
        x = x + feedForward(norm2->forward(x));
        return x;
    }
};

// ----------------------------------------------------------------
// Full Transformer Encoder (stack of N identical layers)
//
// Input processing:
//   1. Token embedding + positional encoding
//   2. N encoder layers
//   3. Final LayerNorm
// ----------------------------------------------------------------
struct TransformerEncoder : torch::nn::Module {
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers;
    torch::nn::Embedding token_embedding{nullptr};
    torch::nn::LayerNorm final_norm{nullptr};
    torch::nn::Dropout dropout{nullptr};
    int d_model, num_layers;

    TransformerEncoder(
        int vocab_size,
        int d_model,
        int num_heads,
        int d_ff,
        int num_layers,
        float dropout_rate = 0.1) : d_model(d_model), num_layers(num_layers) {
        token_embedding = register_module(
            "token_embedding", torch::nn::Embedding(vocab_size, d_model));

        for (int i = 0; i < num_layers; i++) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                d_model, num_heads, d_ff, dropout_rate);
            register_module("encoder_layer_" + std::to_string(i), layer);
            layers.push_back(layer);
        }

        final_norm = register_module("final_norm", torch::nn::LayerNorm(
                                                       torch::nn::LayerNormOptions({d_model})));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
    }

    torch::Tensor forward(
        torch::Tensor input_ids,
        const torch::Tensor &mask = {}) {
        auto x = token_embedding->forward(input_ids);
        x = x * std::sqrt(static_cast<double>(d_model)); // scale embeddings
        x = dropout->forward(x);

        for (auto &layer : layers) {
            x = layer->forward(x, mask);
        }

        return final_norm->forward(x);
    }
};

// ----------------------------------------------------------------
// Demo: Encoder forward pass with synthetic data
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Transformer Encoder Demo ===\n\n";

    int vocab_size = 1000;
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 512;
    int num_layers = 4;
    int batch_size = 2;
    int seq_len = 10;

    TransformerEncoder encoder(vocab_size, d_model, num_heads, d_ff, num_layers);

    // Synthetic input: random token IDs
    auto input_ids = torch::randint(0, vocab_size, {batch_size, seq_len});

    // Optional padding mask (all ones = no padding for demo)
    auto mask = torch::ones({batch_size, 1, 1, seq_len}); // broadcastable

    // Count parameters
    int64_t total_params = 0;
    for (const auto &p : encoder.parameters()) {
        total_params += p.numel();
    }

    std::cout << "Model configuration:\n";
    std::cout << "  vocab_size = " << vocab_size << "\n";
    std::cout << "  d_model    = " << d_model << "\n";
    std::cout << "  num_heads  = " << num_heads << "\n";
    std::cout << "  d_ff       = " << d_ff << "\n";
    std::cout << "  num_layers = " << num_layers << "\n";
    std::cout << "  Total params: " << total_params << "\n\n";

    // Forward pass
    {
        torch::NoGradGuard no_grad;
        auto output = encoder.forward(input_ids, mask);

        std::cout << "Input IDs  shape: " << input_ids.sizes() << "\n";
        std::cout << "Encoder output shape: " << output.sizes() << "\n";
        std::cout << "Sample output (batch 0, position 0, first 4 dims): "
                  << output[0][0].slice(0, 0, 4) << "\n";
        std::cout << "Output mean: " << output.mean().item<float>()
                  << ", std: " << output.std().item<float>() << "\n\n";

        // [CLS] token (position 0) encodes full sequence context
        auto cls_embedding = output.index({torch::indexing::Slice(),
                                           0,
                                           torch::indexing::Slice()});
        std::cout << "[CLS] embedding (position 0) encodes bidirectional context.\n";
        std::cout << "CLS shape: " << cls_embedding.sizes() << "\n";
    }

    return 0;
}
