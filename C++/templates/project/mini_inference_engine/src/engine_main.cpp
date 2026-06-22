#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

#include "include/engine_config.hpp"
#include "include/kernel_registry.hpp"
#include "include/attention_config.hpp"
#include "include/layer_norm_config.hpp"

// ============================================================================
// Mini 推理引擎主入口
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  推理引擎架构总览                                                │
// │                                                                  │
// │  ┌─────────────────────────────────────────────────────────────┐ │
// │  │                   InferenceEngine                           │ │
// │  │                                                             │ │
// │  │  Config (编译期)          Registry (编译期)                   │ │
// │  │  ┌──────────────────┐    ┌─────────────────────────────┐    │ │
// │  │  │ Arch, Dtype,     │    │ KernelRegistry<             │    │ │
// │  │  │ Layout, TileList │    │   GEMM_Kernel_FP16_Sm80,    │    │ │
// │  │  └──────────────────┘    │   GEMM_Kernel_BF16_Sm80,    │    │ │
// │  │                         │   Attn_Kernel_Sm80_D128,    │    │ │
// │  │                         │   LN_Kernel_4096,            │    │ │
// │  │                         │   ...                        │    │ │
// │  │                         └─────────────────────────────┘    │ │
// │  │                                                             │ │
// │  │  Runtime (运行时参数)                                       │ │
// │  │  ┌──────────────────┐                                      │ │
// │  │  │ batch_size,       │                                      │ │
// │  │  │ seq_len,          │                                      │ │
// │  │  │ num_heads,        │                                      │ │
// │  │  │ hidden_dim        │                                      │ │
// │  │  └──────────────────┘                                      │ │
// │  │                                                             │ │
// │  │  执行流程:                                                  │ │
// │  │  Config + Runtime → registry.find_kernel<Config>()        │ │
// │  │                  → Kernel::launch(runtime_params, data)    │ │
// │  │                  → GPU execution → result                  │ │
// │  └─────────────────────────────────────────────────────────────┘ │
// └──────────────────────────────────────────────────────────────────┘

using namespace mini_inference;

// ============================================================================
// 模型层描述 (用于展示推理 Pipeline)
// ============================================================================

struct ModelLayer {
  std::string name;
  std::string op_type;   // "gemm", "attention", "layernorm", "gelu", "residual"
  int M, N, K;           // 矩阵维度 (GEMM 使用; attention/layernorm 忽略)
};

// ============================================================================
// 推理引擎主类 (编译期配置 + 运行时执行)
// ============================================================================

template <typename Config>
class InferenceEngine {
 public:
  using RuntimeConfig = typename Config::RuntimeConfig;

  explicit InferenceEngine(const RuntimeConfig& rcfg)
      : rcfg_(rcfg) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Mini Inference Engine Initialized                  ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Arch:     " << Config::arch_name();
    for (int i = static_cast<int>(std::string(Config::arch_name()).length()); i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "║  Dtype:    " << Config::dtype_name();
    for (int i = static_cast<int>(std::string(Config::dtype_name()).length()); i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    const char* tensor_core_status = Config::kUseTensorCore ? "YES (mma.sync)" : "NO (SIMT)";
    std::cout << "║  TensorCore: " << tensor_core_status;
    for (int i = static_cast<int>(std::string(tensor_core_status).length()); i < 34; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    const char* layout_str = (Config::Layout == LayoutType::kRowMajor) ? "RowMajor (C-contiguous)" :
                             (Config::Layout == LayoutType::kColumnMajor) ? "ColumnMajor (F-contiguous)" :
                             "ChannelsLast (NHWC)";
    std::cout << "║  Layout:   " << layout_str;
    for (int i = static_cast<int>(std::string(layout_str).length()); i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    std::cout << "║  Smem:     " << Config::kSmemBudget / 1024 << " KB";
    for (int i = 20; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    std::cout << "║  Register: " << Config::kRegisterBudget << " per thread";
    for (int i = 27; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    const char* tma_status = Config::kHasTMA ? "YES" : "NO";
    std::cout << "║  TMA:      " << tma_status;
    for (int i = 11; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    std::cout << "╚══════════════════════════════════════════════════════╝" << std::endl;
  }

  // =========================================================================
  // GEMM 操作 (推理中最频繁的操作)
  // =========================================================================
  void gemm(void* A, void* B, void* C, int M, int N, int K,
            bool trans_a = false, bool trans_b = false) {
    std::cout << "\n[Engine::GEMM] M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 选择 tile 策略 (根据矩阵形状)
    bool is_tall_skinny = (M > 8 * N);
    bool is_short_fat = (N > 8 * M);
    bool is_small_k = (K <= 256);
    int tile_m = is_tall_skinny ? 256 : (is_short_fat ? 64 : 128);
    int tile_n = is_short_fat ? 256 : (is_tall_skinny ? 64 : 128);
    int tile_k = is_small_k ? 16 : 32;

    const char* strategy = is_tall_skinny ? "Tall-Skinny" :
                           (is_short_fat ? "Short-Fat" :
                            (is_small_k ? "Small-K" : "Default"));

    std::cout << "  Strategy: " << strategy << std::endl;
    std::cout << "  Tile: " << tile_m << "x" << tile_n << "x" << tile_k << std::endl;

    int grid_m = (M + tile_m - 1) / tile_m;
    int grid_n = (N + tile_n - 1) / tile_n;

    std::cout << "  Grid: (" << grid_m << ", " << grid_n << ") = "
              << grid_m * grid_n << " blocks" << std::endl;

    // 伪代码: 实际 kernel launch
    // using Kernel = Registry::template find_kernel<Config>;
    // Kernel::launch(A, B, C, M, N, K);
  }

  // =========================================================================
  // Attention 操作 (Transformer 核心)
  // =========================================================================
  void attention(void* Q, void* K_cache, void* V_cache, void* O,
                 int seq_len, int head_dim) {
    std::cout << "\n[Engine::Attention] seq_len=" << seq_len
              << ", head_dim=" << head_dim << std::endl;

    // 根据 head_dim 选择最优 attention 配置
    if (head_dim == 64) {
      using AttnConfig = AttnConfigSm80D64;
      std::cout << "  Config: head_dim=64, Br=" << AttnConfig::kBr
                << ", Bc=" << AttnConfig::kBc << std::endl;
      std::cout << "  SRAM: " << AttnConfig::kSmemTotal / 1024 << " KB" << std::endl;

      if constexpr (Config::Arch >= 90) {
        using AttnConfig90 = AttnConfigSm90D64;
        std::cout << "  → SM90 optimized: Br=" << AttnConfig90::kBr
                  << " (larger tiles for H100)" << std::endl;
      }
    } else {
      using AttnConfig = AttnConfigSm80D128;
      std::cout << "  Config: head_dim=128, Br=" << AttnConfig::kBr
                << ", Bc=" << AttnConfig::kBc << std::endl;
    }

    std::cout << "  Causal: " << (rcfg_.seq_len > 1 ? "YES (decoder)" : "NO (encoder)")
              << std::endl;

    int num_kv_tiles = (seq_len + 64 - 1) / 64; // Bc=64
    std::cout << "  KV tiles: " << num_kv_tiles << std::endl;
  }

  // =========================================================================
  // LayerNorm 操作
  // =========================================================================
  void layer_norm(void* input, void* output, void* gamma, void* beta,
                  int rows, int cols, bool use_rms_norm = false) {
    std::cout << "\n[Engine::LayerNorm] rows=" << rows << ", cols=" << cols
              << ", type=" << (use_rms_norm ? "RMSNorm" : "LayerNorm") << std::endl;

    // 选择最优 LayerNorm 配置
    if (cols == 4096) {
      if (use_rms_norm) {
        std::cout << "  Config: RMSNorm, hidden=4096, warps=8" << std::endl;
        std::cout << "  Algorithm: Welford's online variance" << std::endl;
      } else {
        std::cout << "  Config: LayerNorm, hidden=4096, warps=8" << std::endl;
        std::cout << "  Algorithm: Two-pass (mean + variance)" << std::endl;
      }
    } else if (cols == 8192) {
      std::cout << "  Config: hidden=8192, warps=16" << std::endl;
    } else if (cols == 7168) {
      std::cout << "  Config: DeepSeek-V2 style, hidden=7168" << std::endl;
    }

    // Warp-level reduction 性能预估
    int num_warps = (cols <= 4096) ? 8 : 16;
    std::cout << "  Warps: " << num_warps << " (" << num_warps * 32
              << " threads)" << std::endl;
    std::cout << "  Elements per thread: " << cols / (num_warps * 32) << std::endl;
  }

  // =========================================================================
  // Activation (GELU/SiLU/ReLU) - 通常融合在 GEMM epilogue 中
  // =========================================================================
  void activation(void* input, void* output, int size,
                  const std::string& act_type = "gelu") {
    std::cout << "\n[Engine::Activation] " << act_type << ", size=" << size << std::endl;

    if (act_type == "gelu") {
      std::cout << "  GELU: tanh approximation (recommended in epilogue)" << std::endl;
      std::cout << "  → Should be fused with preceding GEMM for best performance" << std::endl;
    } else if (act_type == "silu" || act_type == "swiglu") {
      std::cout << "  SiLU/SwiGLU: element-wise + sigmoid (LLaMA-style)" << std::endl;
    } else if (act_type == "relu") {
      std::cout << "  ReLU: max(0, x)" << std::endl;
    }
  }

  // =========================================================================
  // Residual Add (残差连接)
  // =========================================================================
  void residual_add(void* x, void* residual, void* output, int size) {
    std::cout << "\n[Engine::ResidualAdd] size=" << size << std::endl;
    std::cout << "  → Should be fused with preceding operation's epilogue" << std::endl;
  }

  // =========================================================================
  // 完整 Transformer Layer 推理
  // =========================================================================
  //
  // WHY 一个完整的 layer: 展示推理引擎中各个 operator 如何组合
  //   以及 epilogue fusion 带来的全局优化机会

  void transformer_layer(
      void* hidden_states,       // [batch, seq, hidden_dim]
      void* attention_output,    // [batch, seq, hidden_dim]
      void* ffn_output,          // [batch, seq, hidden_dim]
      int batch_size,
      int seq_len,
      int hidden_dim,
      int intermediate_dim,
      int num_heads,
      int head_dim,
      int num_kv_heads = 0) {    // GQA (Grouped Query Attention)

    if (num_kv_heads == 0) num_kv_heads = num_heads;

    std::cout << "\n╔══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Transformer Layer Inference                        ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Batch:    " << batch_size;
    for (int i = 11; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "║  Seq:      " << seq_len;
    for (int i = 11; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "║  Hidden:   " << hidden_dim;
    for (int i = 11; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "║  Heads:    " << num_heads << " (Q), " << num_kv_heads << " (KV)";
    for (int i = 18; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "║  Head dim: " << head_dim;
    for (int i = 11; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════╝" << std::endl;

    // ── Pre-LayerNorm (RMSNorm in LLaMA) ──
    std::cout << "\n── Step 1: Pre-Attention RMSNorm ──" << std::endl;
    layer_norm(hidden_states, nullptr, nullptr, nullptr,
               batch_size * seq_len, hidden_dim, true);

    // ── QKV Projections ──
    std::cout << "\n── Step 2: QKV Projections ──" << std::endl;

    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;

    std::cout << "  Q projection: [" << batch_size * seq_len
              << ", " << hidden_dim << "] × [" << hidden_dim
              << ", " << q_size << "] (fused bias in epilogue)" << std::endl;

    std::cout << "  K projection: [" << batch_size * seq_len
              << ", " << hidden_dim << "] × [" << hidden_dim
              << ", " << kv_size << "] (fused bias in epilogue)" << std::endl;

    std::cout << "  V projection: [" << batch_size * seq_len
              << ", " << hidden_dim << "] × [" << hidden_dim
              << ", " << kv_size << "] (fused bias in epilogue)" << std::endl;

    // 如果使用 GQA: K,V 维度更小 (num_kv_heads < num_heads)
    if (num_kv_heads < num_heads) {
      std::cout << "  → GQA: KV heads (" << num_kv_heads
                << ") < Q heads (" << num_heads << ")" << std::endl;
      std::cout << "  → KV cache size reduced by "
                << (1.0 - double(num_kv_heads)/num_heads) * 100 << "%"
                << std::endl;
    }

    // ── Attention ──
    std::cout << "\n── Step 3: Multi-Head Attention ──" << std::endl;
    attention(nullptr, nullptr, nullptr, nullptr, seq_len, head_dim);

    // ── Output Projection ──
    std::cout << "\n── Step 4: Output Projection ──" << std::endl;
    std::cout << "  O projection: [" << batch_size * seq_len
              << ", " << q_size << "] × [" << q_size
              << ", " << hidden_dim << "] (fused bias in epilogue)" << std::endl;

    // ── Residual Add + Pre-FFN LayerNorm ──
    std::cout << "\n── Step 5: Residual Add + Pre-FFN RMSNorm ──" << std::endl;
    std::cout << "  → Residual add should be fused in O projection epilogue" << std::endl;
    layer_norm(nullptr, nullptr, nullptr, nullptr,
               batch_size * seq_len, hidden_dim, true);

    // ── FFN (Feed-Forward Network) ──
    std::cout << "\n── Step 6: FFN ──" << std::endl;
    std::cout << "  Gate projection: [" << batch_size * seq_len
              << ", " << hidden_dim << "] × [" << hidden_dim
              << ", " << intermediate_dim << "] (fused bias)" << std::endl;
    std::cout << "  Up projection:   [" << batch_size * seq_len
              << ", " << hidden_dim << "] × [" << hidden_dim
              << ", " << intermediate_dim << "] (fused bias)" << std::endl;
    std::cout << "  SiLU(Gate) * Up  → activation fusion in epilogue" << std::endl;
    std::cout << "  Down projection: [" << batch_size * seq_len
              << ", " << intermediate_dim << "] × [" << intermediate_dim
              << ", " << hidden_dim << "] (fused bias + residual in epilogue)" << std::endl;

    // ── Residual Add ──
    std::cout << "\n── Step 7: Residual Add ──" << std::endl;
    std::cout << "  → Fused in Down projection epilogue (zero extra memory ops)" << std::endl;

    std::cout << "\n── Layer complete ──\n" << std::endl;
  }

  // =========================================================================
  // 性能统计
  // =========================================================================
  void print_performance_summary() const {
    std::cout << "\n╔══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Performance Summary                                 ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════╣" << std::endl;

    // 理论峰值性能 (基于架构)
    double peak_tflops = 0;
    if constexpr (Config::Arch >= 90) peak_tflops = 989;     // H100 FP16
    else if constexpr (Config::Arch >= 80) peak_tflops = 312; // A100 FP16
    else if constexpr (Config::Arch >= 75) peak_tflops = 65;  // T4 FP16
    else peak_tflops = 125;                                   // V100 FP16

    std::cout << "║  Peak TFLOPS: " << peak_tflops;
    for (int i = 14; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    double mem_bw = 0;
    if constexpr (Config::Arch >= 90) mem_bw = 3350;   // H100 HBM3
    else if constexpr (Config::Arch >= 80) mem_bw = 2039; // A100 HBM2e
    else if constexpr (Config::Arch >= 75) mem_bw = 320;  // T4 GDDR6
    else mem_bw = 900;                                    // V100 HBM2

    std::cout << "║  Memory BW:  " << mem_bw << " GB/s";
    for (int i = 14; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    std::cout << "║  TensorCore: " << (Config::kUseTensorCore ? "YES" : "NO");
    for (int i = 14; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    std::cout << "║  Epilogue Fusion: ALL ops fused when possible";
    for (int i = 31; i < 38; i++) std::cout << " ";
    std::cout << "║" << std::endl;

    std::cout << "╚══════════════════════════════════════════════════════╝" << std::endl;
  }

 private:
  RuntimeConfig rcfg_;
};

// ============================================================================
// main: 推理引擎演示
// ============================================================================

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║                                                              ║" << std::endl;
  std::cout << "║     Mini Inference Engine                                    ║" << std::endl;
  std::cout << "║     Compile-time Architecture Language for GPU Computing     ║" << std::endl;
  std::cout << "║                                                              ║" << std::endl;
  std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

  // =========================================================================
  // 配置选择 (编译期)
  // =========================================================================
  //
  // 用户选择编译期配置:
  //   1. 精度: FP16 (常用) 或 BF16 (训练) 或 INT8 (量化推理)
  //   2. 架构: 根据部署硬件选择 (A100/SM80 最常见)
  //   3. Layout: RowMajor (PyTorch 默认)

  // 示例: A100 + FP16 推理
  using MyConfig = ConfigA100Fp16;

  // =========================================================================
  // 运行时配置
  // =========================================================================
  typename MyConfig::RuntimeConfig rcfg;
  rcfg.batch_size = 1;
  rcfg.seq_len = 2048;       // 上下文长度
  rcfg.num_heads = 32;       // LLaMA-7B 配置
  rcfg.head_dim = 128;
  rcfg.hidden_dim = 4096;
  rcfg.intermediate_dim = 11008;

  std::cout << "\n  Runtime Configuration:" << std::endl;
  std::cout << "    " << rcfg.to_string() << std::endl;
  std::cout << "    hidden_dim=" << rcfg.hidden_dim
            << ", intermediate_dim=" << rcfg.intermediate_dim << std::endl;

  // =========================================================================
  // 创建推理引擎实例
  // =========================================================================
  InferenceEngine<MyConfig> engine(rcfg);

  // =========================================================================
  // 完整 Transformer Layer 推理
  // =========================================================================

  engine.transformer_layer(
      nullptr,           // hidden_states (占位)
      nullptr,           // attention_output (占位)
      nullptr,           // ffn_output (占位)
      rcfg.batch_size,
      rcfg.seq_len,
      rcfg.hidden_dim,
      rcfg.intermediate_dim,
      rcfg.num_heads,
      rcfg.head_dim,
      8                  // num_kv_heads (GQA: LLaMA-70B 使用 8)
  );

  // =========================================================================
  // 性能总结
  // =========================================================================
  engine.print_performance_summary();

  // =========================================================================
  // 多架构对比
  // =========================================================================
  std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  Cross-Architecture Comparison (same model, different GPU)   ║" << std::endl;
  std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
  std::cout << std::endl;

  std::cout << "  ┌──────────────────┬──────────┬──────────┬──────────┬──────────┐" << std::endl;
  std::cout << "  │ Metric           │ V100(SM70)│ T4(SM75) │A100(SM80)│H100(SM90)│" << std::endl;
  std::cout << "  ├──────────────────┼──────────┼──────────┼──────────┼──────────┤" << std::endl;
  std::cout << "  │ Peak FP16 TFLOPS │   125    │    65    │   312    │   989    │" << std::endl;
  std::cout << "  │ Memory BW (GB/s) │   900    │   320    │   2039   │   3350   │" << std::endl;
  std::cout << "  │ Shared Mem (KB)  │    96    │    64    │   164    │   228    │" << std::endl;
  std::cout << "  │ Max Tile (M×N×K) │ 128×64×8 │128×128×16│256×128×32│256×256×64│" << std::endl;
  std::cout << "  │ MMA Instruction  │ m16n16k4 │m16n16k8  │m16n8k16  │  wgmma   │" << std::endl;
  std::cout << "  │ cp.async         │    No    │    No    │   Yes    │   Yes    │" << std::endl;
  std::cout << "  │ TMA              │    No    │    No    │    No    │   Yes    │" << std::endl;
  std::cout << "  │ FP8 Support      │    No    │    No    │    No    │   Yes    │" << std::endl;
  std::cout << "  └──────────────────┴──────────┴──────────┴──────────┴──────────┘" << std::endl;
  std::cout << std::endl;

  std::cout << "  Key insight: Each architecture requires DIFFERENT tile sizes," << std::endl;
  std::cout << "  MMA instructions, and shared memory layouts to reach peak perf." << std::endl;
  std::cout << "  Compile-time architecture dispatch handles this automatically." << std::endl;

  std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  Demo Complete.                                              ║" << std::endl;
  std::cout << "║  All dispatch decisions made at compile-time (zero overhead).║" << std::endl;
  std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

  return 0;
}
