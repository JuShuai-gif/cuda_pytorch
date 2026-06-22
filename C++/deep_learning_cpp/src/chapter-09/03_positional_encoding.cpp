/*
 * 03_positional_encoding.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * Transformer has no notion of position order (unlike RNNs).
 * Positional encodings inject position information into the input.
 *
 * Evolution of positional encoding methods:
 *
 * 1. Learned Position Embeddings (Vaswani 2017)
 *    - A learnable embedding table indexed by position
 *    - Limitation: cannot generalize beyond max training length
 *
 * 2. Sinusoidal Position Encoding (Vaswani 2017)
 *    - Fixed sine/cosine functions, no learnable parameters
 *    - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *    - Property: PE(pos+k) can be expressed as a linear function of PE(pos)
 *
 * 3. RoPE (Rotary Position Embeddings) — Su et al. 2023
 *    - Used by LLaMA series, latest GPT architectures
 *    - Rotates Q and K vectors by position-dependent angles
 *    - Inherently captures relative position through dot-product invariance
 *    - Key insight: after RoPE, Q_m · K_n depends only on (m-n)
 *
 * 4. ALiBi (Attention with Linear Biases) — Press et al. 2022
 *    - Adds a static, non-learned bias to attention scores
 *    - bias = -m * |i - j|  where m is head-specific slope
 *    - Excellent length extrapolation without any parameters
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// ----------------------------------------------------------------
// 1. Sinusoidal Position Encoding
//
// Returns a (seq_len, d_model) tensor of fixed positional encodings.
// Even indices use sin, odd indices use cos.
// ----------------------------------------------------------------
torch::Tensor sinusoidalPositionEncoding(int seq_len, int d_model) {
    auto pe = torch::zeros({seq_len, d_model});
    auto position = torch::arange(0, seq_len).unsqueeze(1).to(torch::kFloat32); // (seq_len, 1)

    // div_term = 10000^(2i/d_model) for i = 0,1,...,d_model/2-1
    auto div_term = torch::exp(
        torch::arange(0, d_model, 2).to(torch::kFloat32)
        * (-std::log(10000.0) / d_model));

    // even indices: sin
    pe.slice(1, 0, d_model, 2) = torch::sin(position * div_term);
    // odd indices: cos
    pe.slice(1, 1, d_model, 2) = torch::cos(position * div_term);

    return pe;
}

// ----------------------------------------------------------------
// 2. Learned Position Embeddings
//
// A trainable embedding layer indexed by absolute position.
// Used in GPT-2 and many early transformers.
// ----------------------------------------------------------------
struct LearnedPositionEmbedding : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    int max_len, d_model;

    LearnedPositionEmbedding(int max_len, int d_model) : max_len(max_len), d_model(d_model) {
        embedding = register_module("embedding",
                                    torch::nn::Embedding(max_len, d_model));
    }

    torch::Tensor forward(int seq_len) {
        auto positions = torch::arange(0, seq_len).to(torch::kLong);
        return embedding->forward(positions); // (seq_len, d_model)
    }
};

// ----------------------------------------------------------------
// 3. Rotary Position Embeddings (RoPE)
//
// Applies rotation to Q and K vectors based on position.
//   - Pair dimensions (0,1), (2,3), ..., (d-2, d-1) as 2D subspaces
//   - Rotate each pair by angle: pos / (10000^(2i/d))
//
// After rotation: Q_m · K_n depends only on relative position (m-n).
// ----------------------------------------------------------------
torch::Tensor computeRoPEFrequencies(int d_model, int max_seq_len) {
    // freq_i = 1 / (10000^(2i/d_model))
    auto freqs = 1.0 / torch::pow(10000.0, torch::arange(0, d_model, 2).to(torch::kFloat32) / d_model);
    // (max_seq_len, d_model/2)
    auto t = torch::arange(0, max_seq_len).to(torch::kFloat32).unsqueeze(1);
    return t * freqs; // positions * frequencies
}

std::pair<torch::Tensor, torch::Tensor> applyRoPE(
    const torch::Tensor &Q,
    const torch::Tensor &K,
    int start_pos = 0) {
    int seq_len = Q.size(1);
    int d_model = Q.size(2);
    int half_d = d_model / 2;

    auto freqs = computeRoPEFrequencies(d_model, start_pos + seq_len);
    freqs = freqs.slice(0, start_pos, start_pos + seq_len); // (seq_len, half_d)

    auto cos = torch::cos(freqs).unsqueeze(0).unsqueeze(2); // (1, seq_len, 1, half_d)
    auto sin = torch::sin(freqs).unsqueeze(0).unsqueeze(2);

    // Reshape to pair last two dims as (d/2, 2) pairs
    auto reshapeForRoPE = [](const torch::Tensor &x) -> torch::Tensor {
        // (batch, seq, d_model) -> (batch, seq, d_model/2, 2)
        return x.view({x.size(0), x.size(1), -1, 2});
    };

    auto unshapeForRoPE = [](const torch::Tensor &x) -> torch::Tensor {
        return x.view({x.size(0), x.size(1), -1});
    };

    // Apply rotation: (x1, x2) rotated by theta
    auto rotate = [&cos, &sin](const torch::Tensor &x) -> torch::Tensor {
        auto x_reshaped = x.view({x.size(0), x.size(1), -1, 2});
        auto x1 = x_reshaped.slice(/*dim=*/-1, 0, 1);
        auto x2 = x_reshaped.slice(/*dim=*/-1, 1, 2);

        // x1*cos - x2*sin, x1*sin + x2*cos
        auto rot_x1 = x1 * cos - x2 * sin;
        auto rot_x2 = x1 * sin + x2 * cos;

        return torch::cat({rot_x1, rot_x2}, /*dim=*/-1)
            .view({x.size(0), x.size(1), -1});
    };

    return {rotate(Q), rotate(K)};
}

// ----------------------------------------------------------------
// 4. ALiBi (Attention with Linear Biases)
//
// Adds a static bias to attention scores: score -= m * |i - j|
// where m is a head-specific slope. No learnable parameters.
// Slope calculation: m = 2^(-8 * h / num_heads) for head h.
// ----------------------------------------------------------------
torch::Tensor createALiBiMask(int seq_len, int num_heads) {
    auto positions = torch::arange(seq_len).to(torch::kFloat32);
    auto distances = positions.unsqueeze(1) - positions.unsqueeze(0); // (seq_len, seq_len)
    distances = distances.abs();                                      // |i - j|

    // Head-specific slopes: m_h = 2^(-8 * h / num_heads) for h in 1..num_heads
    auto slopes = torch::pow(
        2.0,
        torch::arange(1, num_heads + 1).to(torch::kFloat32) * (-8.0 / num_heads)); // (num_heads)

    // (num_heads, seq_len, seq_len) -> negative bias
    auto alibi = -slopes.unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0);
    return alibi; // add to attention scores before softmax
}

// ----------------------------------------------------------------
// Demo: Compare all four positional encoding methods
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Positional Encoding Demo ===\n\n";

    int seq_len = 8;
    int d_model = 32;
    int num_heads = 4;

    // --- Sinusoidal Encoding ---
    std::cout << "1. Sinusoidal Position Encoding\n";
    auto pe_sin = sinusoidalPositionEncoding(seq_len, d_model);
    std::cout << "   Shape: " << pe_sin.sizes() << "\n";
    std::cout << "   PE(pos=0, dims 0..7): "
              << pe_sin[0].slice(0, 0, 8) << "\n";
    std::cout << "   PE(pos=3, dims 0..7): "
              << pe_sin[3].slice(0, 0, 8) << "\n";

    // Cosine similarity between positions: nearby positions should be similar
    auto sim = torch::cosine_similarity(
        pe_sin.slice(0, 0, 1),
        pe_sin.slice(0, 1, 2),
        /*dim=*/-1);
    std::cout << "   Cos-sim(PE[0], PE[1]): " << sim.item<float>() << "\n\n";

    // --- Learned Position Embeddings ---
    std::cout << "2. Learned Position Embeddings\n";
    LearnedPositionEmbedding lpe(seq_len * 2, d_model);
    {
        torch::NoGradGuard no_grad;
        auto pe_learned = lpe.forward(seq_len);
        std::cout << "   Shape: " << pe_learned.sizes() << "\n";
        std::cout << "   Each position has an independent learnable vector.\n";
        std::cout << "   Cannot generalize beyond max_len="
                  << (seq_len * 2) << "\n\n";
    }

    // --- RoPE ---
    std::cout << "3. RoPE (Rotary Position Embeddings)\n";
    {
        int batch_size = 1;
        auto Q = torch::randn({batch_size, seq_len, d_model});
        auto K = torch::randn({batch_size, seq_len, d_model});

        auto [Q_rope, K_rope] = applyRoPE(Q, K);

        // Verify relative-position property:
        // Q[0] rotated · K[3] rotated should equal
        // Q[1] rotated · K[4] rotated (same relative distance = 3)
        auto dot_0_3 = torch::dot(Q_rope[0][0], K_rope[0][3]);
        auto dot_1_4 = torch::dot(Q_rope[0][1], K_rope[0][4]);

        std::cout << "   Q_rope shape: " << Q_rope.sizes() << "\n";
        std::cout << "   Relative position property test:\n";
        std::cout << "     Q[0]·K[3] = " << dot_0_3.item<float>() << "\n";
        std::cout << "     Q[1]·K[4] = " << dot_1_4.item<float>() << "\n";
        std::cout << "     (should be equal for same relative distance=3)\n\n";
    }

    // --- ALiBi ---
    std::cout << "4. ALiBi (Attention with Linear Biases)\n";
    {
        auto alibi = createALiBiMask(seq_len, num_heads);
        std::cout << "   Shape: " << alibi.sizes()
                  << " (num_heads, seq_len, seq_len)\n";
        std::cout << "   Head 0 slopes: -\n";
        std::cout << "   ALiBi[head=0, pos=0, :]: "
                  << alibi[0][0] << "\n";
        std::cout << "   ALiBi[head=3, pos=0, :]: "
                  << alibi[3][0] << "\n";
        std::cout << "   Each head has a different slope; higher heads discount "
                     "longer distances more aggressively.\n\n";
    }

    // --- Summary ---
    std::cout << "--- Summary ---\n";
    std::cout << "| Method      | Type   | Params | Relative | Extrapolation |\n";
    std::cout << "|-------------|--------|--------|----------|---------------|\n";
    std::cout << "| Sinusoidal  | Fixed  | 0      | Indirect | Moderate      |\n";
    std::cout << "| Learned     | Train  | ML*d   | No       | Poor          |\n";
    std::cout << "| RoPE        | Fixed  | 0      | Yes      | Good          |\n";
    std::cout << "| ALiBi       | Fixed  | 0      | Yes      | Excellent     |\n";

    return 0;
}
