/*
 * 05_transformer_decoder.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * The Transformer Decoder generates output sequences autoregressively.
 * Each decoder layer consists of three sub-layers:
 *
 *  1. Masked Self-Attention  — attends to past outputs only (causal)
 *  2. Cross-Attention        — attends to encoder output
 *  3. Feed-Forward Network   — position-wise nonlinear transform
 *
 * Each sub-layer uses Pre-LayerNorm + residual connection.
 *
 * Training vs Inference:
 *   - Training: teacher forcing — feed full target sequence,
 *     use causal mask, compute loss over all positions in parallel
 *   - Inference: autoregressive generation — produce one token at a time,
 *     append to input, repeat until <eos>
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
// Causal Mask: lower-triangular, prevents attending to future tokens
// ----------------------------------------------------------------
torch::Tensor causalMask(int seq_len) {
    return torch::triu(
        torch::ones({seq_len, seq_len}) * (-1e9),
        /*diagonal=*/1);
}

// ----------------------------------------------------------------
// Transformer Decoder Layer
//
// Structure (all sub-modules inlined):
//   x = x + MaskedSelfAttention(LayerNorm(x))
//   x = x + CrossAttention(LayerNorm(x), encoder_output)
//   x = x + FeedForward(LayerNorm(x))
// ----------------------------------------------------------------
struct TransformerDecoderLayer : torch::nn::Module {
    // Masked Self-Attention components
    torch::nn::Linear self_Wq{nullptr}, self_Wk{nullptr}, self_Wv{nullptr}, self_Wo{nullptr};
    // Cross-Attention components
    torch::nn::Linear cross_Wq{nullptr}, cross_Wk{nullptr}, cross_Wv{nullptr}, cross_Wo{nullptr};
    // Feed-Forward components
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    // Normalization and regularization
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
    torch::nn::Dropout self_dropout{nullptr}, cross_dropout{nullptr}, ffn_dropout{nullptr};

    int d_model, num_heads, d_k, d_ff;

    TransformerDecoderLayer(
        int d_model,
        int num_heads,
        int d_ff,
        float dropout_rate = 0.1) : d_model(d_model),
                                    num_heads(num_heads),
                                    d_k(d_model / num_heads),
                                    d_ff(d_ff) {
        auto attn_opts = torch::nn::LinearOptions(d_model, d_model).bias(false);

        self_Wq = register_module("self_Wq", torch::nn::Linear(attn_opts));
        self_Wk = register_module("self_Wk", torch::nn::Linear(attn_opts));
        self_Wv = register_module("self_Wv", torch::nn::Linear(attn_opts));
        self_Wo = register_module("self_Wo", torch::nn::Linear(attn_opts));

        cross_Wq = register_module("cross_Wq", torch::nn::Linear(attn_opts));
        cross_Wk = register_module("cross_Wk", torch::nn::Linear(attn_opts));
        cross_Wv = register_module("cross_Wv", torch::nn::Linear(attn_opts));
        cross_Wo = register_module("cross_Wo", torch::nn::Linear(attn_opts));

        fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
        fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));

        norm1 = register_module("norm1",
                                torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm2 = register_module("norm2",
                                torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm3 = register_module("norm3",
                                torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        self_dropout = register_module("self_dropout",
                                       torch::nn::Dropout(dropout_rate));
        cross_dropout = register_module("cross_dropout",
                                        torch::nn::Dropout(dropout_rate));
        ffn_dropout = register_module("ffn_dropout",
                                      torch::nn::Dropout(dropout_rate));
    }

    torch::Tensor multiHeadAttn(
        torch::Tensor q,
        torch::Tensor kv,
        torch::nn::Linear &wq, torch::nn::Linear &wk,
        torch::nn::Linear &wv, torch::nn::Linear &wo,
        torch::nn::Dropout &do_layer,
        const torch::Tensor &mask = {}) {
        int B = q.size(0), Sq = q.size(1), Sk = kv.size(1);

        auto Q = wq->forward(q).view({B, Sq, num_heads, d_k}).transpose(1, 2);
        auto K = wk->forward(kv).view({B, Sk, num_heads, d_k}).transpose(1, 2);
        auto V = wv->forward(kv).view({B, Sk, num_heads, d_k}).transpose(1, 2);

        auto mask_exp = mask;
        if (mask.defined() && mask.dim() == 2) {
            mask_exp = mask.unsqueeze(0).unsqueeze(0);
        }

        auto attn_out = scaledDotProductAttention(Q, K, V, mask_exp);
        attn_out = attn_out.transpose(1, 2).contiguous().view({B, Sq, d_model});
        return wo->forward(do_layer->forward(attn_out));
    }

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor enc_output,
        const torch::Tensor &causal_mask = {},
        const torch::Tensor &cross_mask = {}) {
        // 1. Masked Self-Attention (pre-LN)
        x = x + multiHeadAttn(norm1->forward(x), x, self_Wq, self_Wk, self_Wv, self_Wo, self_dropout, causal_mask);

        // 2. Cross-Attention: Q from decoder, K,V from encoder (pre-LN)
        x = x + multiHeadAttn(norm2->forward(x), enc_output, cross_Wq, cross_Wk, cross_Wv, cross_Wo, cross_dropout, cross_mask);

        // 3. Feed-Forward (pre-LN)
        auto ffn_input = norm3->forward(x);
        ffn_input = torch::gelu(fc1->forward(ffn_input));
        ffn_input = ffn_dropout->forward(ffn_input);
        x = x + fc2->forward(ffn_input);

        return x;
    }
};

// ----------------------------------------------------------------
// Full Transformer Decoder
// ----------------------------------------------------------------
struct TransformerDecoder : torch::nn::Module {
    std::vector<std::shared_ptr<TransformerDecoderLayer>> layers;
    torch::nn::Embedding token_embedding{nullptr};
    torch::nn::LayerNorm final_norm{nullptr};
    torch::nn::Linear output_proj{nullptr}; // d_model -> vocab_size
    int d_model, num_layers, vocab_size;

    TransformerDecoder(
        int vocab_size,
        int d_model,
        int num_heads,
        int d_ff,
        int num_layers,
        float dropout_rate = 0.1) : d_model(d_model), num_layers(num_layers), vocab_size(vocab_size) {
        token_embedding = register_module(
            "token_embedding", torch::nn::Embedding(vocab_size, d_model));

        for (int i = 0; i < num_layers; i++) {
            auto layer = std::make_shared<TransformerDecoderLayer>(
                d_model, num_heads, d_ff, dropout_rate);
            register_module("decoder_layer_" + std::to_string(i), layer);
            layers.push_back(layer);
        }

        final_norm = register_module("final_norm",
                                     torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        output_proj = register_module("output_proj",
                                      torch::nn::Linear(d_model, vocab_size));
    }

    // Training mode: teacher forcing — process all positions in parallel
    torch::Tensor forward(
        torch::Tensor tgt_ids,
        torch::Tensor enc_output,
        const torch::Tensor &causal_mask = {},
        const torch::Tensor &cross_mask = {}) {
        auto x = token_embedding->forward(tgt_ids);
        x = x * std::sqrt(static_cast<double>(d_model));

        for (auto &layer : layers) {
            x = layer->forward(x, enc_output, causal_mask, cross_mask);
        }

        x = final_norm->forward(x);
        return output_proj->forward(x); // (batch, tgt_len, vocab_size)
    }

    // Autoregressive generation (inference): generate one token at a time
    torch::Tensor generate(
        torch::Tensor enc_output,
        int max_len,
        int start_token,
        int end_token,
        float temperature = 1.0) {
        int batch_size = enc_output.size(0);
        auto generated = torch::full({batch_size, 1}, start_token,
                                     torch::TensorOptions().dtype(torch::kLong));

        torch::NoGradGuard no_grad;

        for (int step = 0; step < max_len - 1; step++) {
            int cur_len = generated.size(1);
            auto mask = causalMask(cur_len);

            auto logits = forward(generated, enc_output, mask);
            auto next_logits = logits.index(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(cur_len - 1, cur_len),
                 torch::indexing::Slice()}); // (batch, 1, vocab_size)

            next_logits = next_logits / temperature;
            auto next_token = next_logits.argmax(/*dim=*/-1); // (batch, 1)

            generated = torch::cat({generated, next_token}, /*dim=*/1);

            auto ended = (next_token == end_token).all();
            if (ended.item<bool>()) break;
        }

        return generated;
    }
};

// ----------------------------------------------------------------
// Demo: Training forward pass + autoregressive generation
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Transformer Decoder Demo ===\n\n";

    int vocab_size = 500;
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 512;
    int num_layers = 2;
    int batch_size = 1;

    TransformerDecoder decoder(vocab_size, d_model, num_heads, d_ff, num_layers);

    int64_t total_params = 0;
    for (const auto &p : decoder.parameters()) {
        total_params += p.numel();
    }
    std::cout << "Decoder params: " << total_params << "\n\n";

    // Synthetic encoder output (simulating a pre-computed encoder)
    int src_len = 6;
    auto enc_output = torch::randn({batch_size, src_len, d_model});

    // ================================================================
    // Training mode: teacher forcing
    // ================================================================
    std::cout << "--- Training Mode (Teacher Forcing) ---\n";
    int tgt_len = 5;
    auto tgt_ids = torch::randint(0, vocab_size, {batch_size, tgt_len}).to(torch::kLong);
    auto mask = causalMask(tgt_len);

    {
        torch::NoGradGuard no_grad;
        auto logits = decoder.forward(tgt_ids, enc_output, mask);
        std::cout << "Target IDs shape: " << tgt_ids.sizes() << "\n";
        std::cout << "Logits shape: " << logits.sizes()
                  << " (batch, tgt_len, vocab_size)\n";
        std::cout << "All " << tgt_len << " positions predicted in parallel.\n\n";
    }

    // ================================================================
    // Inference mode: autoregressive generation
    // ================================================================
    std::cout << "--- Inference Mode (Autoregressive Generation) ---\n";
    int start_token = 0;
    int end_token = 1;
    int max_len = 8;

    auto generated = decoder.generate(enc_output, max_len,
                                      start_token, end_token);

    std::cout << "Generated sequence: ";
    auto gen_flat = generated.flatten();
    for (int i = 0; i < gen_flat.size(0); i++) {
        std::cout << gen_flat[i].item<int>() << " ";
    }
    std::cout << "\n";
    std::cout << "Length: " << generated.size(1) << " tokens\n\n";

    std::cout << "Note: Each step re-encodes the entire growing sequence.\n";
    std::cout << "Production systems use KV-caching to avoid redundant computation.\n";

    return 0;
}
