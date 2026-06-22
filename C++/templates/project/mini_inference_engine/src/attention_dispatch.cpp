#include <iostream>
#include <cstdint>
#include <cmath>

#include "include/attention_config.hpp"
#include "include/engine_config.hpp"

// ============================================================================
// Attention Dispatch 实现
// ============================================================================
//
// FlashAttention 风格的多配置 dispatch。
// 核心思想:
//   1. 将 Attention 分解为多个 GEMM-like 操作
//   2. 使用 tiling 避免 O(N^2) 的 attention matrix 写入 HBM
//   3. Online softmax 避免多次 pass
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  FlashAttention-2 算法概述 (for KV-Cache inference)              │
// │                                                                  │
// │  Input:  Q [batch, heads, 1, head_dim]       (single token)    │
// │          K_cache [batch, heads, seq_len, head_dim]             │
// │          V_cache [batch, heads, seq_len, head_dim]             │
// │                                                                  │
// │  Algorithm (单 token 推理，有 KV cache):                         │
// │                                                                  │
// │  1. Initialize: O = zeros, m = -inf, l = zeros                 │
// │                                                                  │
// │  2. For each KV tile (size Bc):                                 │
// │     a. Load K_tile, V_tile from cache to SRAM                   │
// │     b. Compute S = Q @ K_tile^T  (Br × Bc in SRAM)             │
// │        - If causal: mask upper triangle                         │
// │        - If ALiBi: add position bias                            │
// │     c. Compute m_new = max(m, rowmax(S))                        │
// │        P = exp(S - m_new)                                       │
// │        l_new = l * exp(m - m_new) + rowsum(P)                   │
// │     d. O = diag(l/l_new) * diag(exp(m - m_new)) * O            │
// │        O = O + P @ V_tile                                       │
// │     e. Update m = m_new, l = l_new                             │
// │                                                                  │
// │  3. Output: softmax(QK^T/√d) × V                                │
// └──────────────────────────────────────────────────────────────────┘

using namespace mini_inference;

// ============================================================================
// Attention Dispatch 模板
// ============================================================================

template <typename AttnConfig>
struct AttentionDispatch {
  using Config = AttnConfig;

  // ── 编译期常量 ──
  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kBr = Config::kBr;     // Q 的 tile 行数
  static constexpr int kBc = Config::kBc;     // KV 的 tile 行数
  static constexpr int kNumWarps = Config::kNumWarps;
  static constexpr bool kCausal = Config::kCausal;

  // =========================================================================
  // 分派: 根据序列长度选择策略
  // =========================================================================
  //
  // WHY 运行时序列长度影响策略:
  //   - seq_len < 128:  小序列 → 不需要 tiling，直接全在 SRAM 中算
  //   - 128 ≤ seq_len < 512: 中等序列 → 标准 tiling
  //   - seq_len ≥ 512:  长序列 → 需要大 tile 或 multi-stage

  static void dispatch(
      int batch_size,
      int num_heads,
      int seq_len,     // KV cache 的当前长度
      void* Q,         // [batch, heads, 1, head_dim]  (prefill: seq_q 维度)
      void* K_cache,   // [batch, heads, seq_len, head_dim]
      void* V_cache,   // [batch, heads, seq_len, head_dim]
      void* O          // [batch, heads, 1, head_dim]
  ) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Attention Dispatch (FlashAttention-style)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Config:" << std::endl;
    std::cout << "    head_dim:  " << kHeadDim << std::endl;
    std::cout << "    Br (Q tile): " << kBr << std::endl;
    std::cout << "    Bc (KV tile): " << kBc << std::endl;
    std::cout << "    Warps:    " << kNumWarps << std::endl;
    std::cout << "    Causal:   " << (kCausal ? "YES" : "NO") << std::endl;
    std::cout << "    SRAM:     " << Config::kSmemTotal / 1024 << " KB" << std::endl;
    std::cout << "    SM80 fit: " << (Config::kFitsSm80 ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << "    SM90 fit: " << (Config::kFitsSm90 ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << std::endl;

    std::cout << "  Runtime:" << std::endl;
    std::cout << "    batch:    " << batch_size << std::endl;
    std::cout << "    heads:    " << num_heads << std::endl;
    std::cout << "    seq_len:  " << seq_len << std::endl;
    std::cout << std::endl;

    // ── 策略选择 ──
    if (seq_len <= 128) {
      std::cout << "  Strategy: Small sequence (≤128)" << std::endl;
      std::cout << "    → No tiling needed, entire KV in SRAM" << std::endl;
      std::cout << "    → Single-pass attention" << std::endl;
      dispatch_small_seq(batch_size, num_heads, seq_len, Q, K_cache, V_cache, O);
    } else if (seq_len <= 512) {
      std::cout << "  Strategy: Medium sequence (129-512)" << std::endl;
      std::cout << "    → Standard tiling with online softmax" << std::endl;
      std::cout << "    → " << (seq_len + kBc - 1) / kBc
                << " KV tiles of size " << kBc << std::endl;
      dispatch_medium_seq(batch_size, num_heads, seq_len, Q, K_cache, V_cache, O);
    } else {
      std::cout << "  Strategy: Long sequence (>512)" << std::endl;
      std::cout << "    → Multi-stage tiling for long contexts" << std::endl;
      std::cout << "    → " << (seq_len + kBc - 1) / kBc
                << " KV tiles of size " << kBc << std::endl;
      dispatch_long_seq(batch_size, num_heads, seq_len, Q, K_cache, V_cache, O);
    }

    std::cout << "  → Attention dispatch complete\n" << std::endl;
  }

 private:
  // ── 小序列: 单次 pass (无 tiling) ──
  static void dispatch_small_seq(
      int batch, int heads, int seq_len,
      void* Q, void* K, void* V, void* O) {

    // 序列足够短 → 全部数据可以一次放入 SRAM
    // 算法: 标准 attention (无 tiling overhead)
    //
    // 伪代码:
    // for b in 0..batch:
    //   for h in 0..heads:
    //     Q_bh = Q[b][h]   // [1, d]
    //     K_bh = K[b][h]   // [seq, d]
    //     V_bh = V[b][h]   // [seq, d]
    //
    //     S = Q_bh @ K_bh^T           // [1, seq]
    //     S = S / sqrt(d_head)
    //     if causal: S = mask(S)
    //     P = softmax(S)              // [1, seq]
    //     O_bh = P @ V_bh            // [1, d]
    //
    // 此模式不需要 online softmax rescaling

    std::cout << "    [SmallSeq] Computing QK^T in SRAM..." << std::endl;
    std::cout << "    [SmallSeq] Softmax in SRAM..." << std::endl;
    std::cout << "    [SmallSeq] PV in SRAM..." << std::endl;
    std::cout << "    [SmallSeq] Writing output to HBM..." << std::endl;

    // 启动 1 个 kernel per (batch, head)
    // dim3 grid(batch, heads);
    // dim3 block(kNumWarps * 32);
    // attention_small_seq<Config><<<grid, block, kSmemTotal>>>(
    //     Q, K, V, O, seq_len);
  }

  // ── 中等序列: 标准 tiling ──
  static void dispatch_medium_seq(
      int batch, int heads, int seq_len,
      void* Q, void* K, void* V, void* O) {

    int num_kv_tiles = (seq_len + kBc - 1) / kBc;

    std::cout << "    [MediumSeq] Tiling KV into " << num_kv_tiles << " tiles" << std::endl;
    std::cout << "    [MediumSeq] Q tile size: " << kBr << " × " << kHeadDim << std::endl;
    std::cout << "    [MediumSeq] KV tile size: " << kBc << " × " << kHeadDim << std::endl;

    // 伪代码:
    // for b in 0..batch, h in 0..heads:
    //   // Initialize running stats for online softmax
    //   float O_reg[SBr * d];  // accumulated output
    //   float m_reg[SBr];       // running max
    //   float l_reg[SBr];       // running sum
    //
    //   // Initialize m = -inf, l = 0
    //   for i in 0..SBr: m_reg[i] = -inf, l_reg[i] = 0
    //
    //   // Load Q tile to SRAM (stays there for all KV tiles)
    //   Q_smem[SBr][d] = Q[b][h][:SBr]
    //
    //   // Loop over KV tiles
    //   for tile in 0..num_kv_tiles:
    //     // Load K_tile, V_tile to SRAM
    //     K_smem[SBc][d] = K[b][h][tile*Bc : (tile+1)*Bc]
    //     V_smem[SBc][d] = V[b][h][tile*Bc : (tile+1)*Bc]
    //
    //     // Compute S = Q_smem @ K_smem^T  [SBr × SBc]
    //     S_smem = Q_smem @ K_smem^T / sqrt(d)
    //
    //     // Apply causal mask if needed
    //     if causal: mask upper triangle of S
    //
    //     // Online softmax rescaling
    //     m_new = max(m_old, rowmax(S_smem))
    //     P = exp(S_smem - m_new)
    //     l_new = l_old * exp(m_old - m_new) + rowsum(P)
    //
    //     // Rescale old O
    //     O_reg = O_reg * (l_old / l_new) * exp(m_old - m_new)
    //
    //     // Update O with this tile's contribution
    //     O_reg += P_smem @ V_smem
    //
    //     // Update running stats
    //     m_old = m_new, l_old = l_new
    //
    //   // Write O_reg to HBM
    //   O[b][h][:SBr] = O_reg

    std::cout << "    [MediumSeq] Online softmax with rescaling..." << std::endl;
    std::cout << "    [MediumSeq] Computing tile O += P_tile @ V_tile..." << std::endl;
    std::cout << "    [MediumSeq] Writing accumulated output..." << std::endl;

    // Block/grid 配置
    int blocks_per_head = (1 + kBr - 1) / kBr;  // Q 维度上的 block 数
    std::cout << "    [MediumSeq] Launch: grid=(" << batch * heads * blocks_per_head
              << "), block=" << kNumWarps * 32
              << ", smem=" << Config::kSmemTotal / 1024 << "KB" << std::endl;
  }

  // ── 长序列: 多阶段 tiling (更复杂的流水线) ──
  static void dispatch_long_seq(
      int batch, int heads, int seq_len,
      void* Q, void* K, void* V, void* O) {

    // 长序列需要更激进的 tiling 策略:
    //   1. Q 也可能需要 tiling (如果 batch_size × num_heads 大)
    //   2. 使用 multi-stage 流水线: 加载下一个 KV tile 的同时计算当前 tile
    //   3. 考虑使用 persistent kernel 减少 launch overhead

    std::cout << "    [LongSeq] Multi-stage pipeline for long sequences" << std::endl;
    std::cout << "    [LongSeq] Double-buffered KV tile loading" << std::endl;
    std::cout << "    [LongSeq] Software pipelining: load tile N+1 while computing tile N"
              << std::endl;

    int num_kv_tiles = (seq_len + kBc - 1) / kBc;

    // 使用 persistent kernel (一个 block 处理多个 tile)
    // grid = min(SM_count, batch * heads)
    // 每个 block 用 while 循环处理分配给它的 Q 行

    std::cout << "    [LongSeq] Persistent kernel: grid="
              << batch * heads << " blocks" << std::endl;
    std::cout << "    [LongSeq] Each block processes all " << num_kv_tiles
              << " KV tiles sequentially" << std::endl;

    // 预取优化: SM80+ 使用 cp.async 预取下一个 tile 的 K, V
    if constexpr (Config::kFitsSm80) {
      std::cout << "    [LongSeq] Using cp.async for prefetching KV tiles" << std::endl;
    }
  }
};

// ============================================================================
// 公开 API: attention_dispatch
// ============================================================================
//
// 使用方式:
//   using AttnConfig = AttnConfigSm80D128;
//   AttentionDispatch<AttnConfig>::dispatch(batch, heads, seq_len, Q, K, V, O);

// ============================================================================
// Attention 分派辅助函数: 自动根据 head_dim 和架构选择配置
// ============================================================================

template <int HeadDim, int Arch>
void attention_dispatch_auto(
    int batch_size,
    int num_heads,
    int seq_len,
    void* Q, void* K_cache, void* V_cache, void* O) {

  using AttnConfig = default_attention_config_t<HeadDim, Arch>;

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Auto Attention Dispatch" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "  Auto-selected config: head_dim=" << HeadDim << std::endl;

  AttentionDispatch<AttnConfig>::dispatch(
      batch_size, num_heads, seq_len, Q, K_cache, V_cache, O);
}

// ============================================================================
// 预编译的 Attention dispatch 实例化 (可选: 显式实例化减少编译时间)
// ============================================================================

// 常见配置的显式实例化声明
// template struct AttentionDispatch<AttnConfigSm80D64>;
// template struct AttentionDispatch<AttnConfigSm80D128>;
// template struct AttentionDispatch<AttnConfigCausalSm80D128>;
