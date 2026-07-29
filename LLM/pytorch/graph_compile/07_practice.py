"""
07_practice.py — torch.compile 真实模型实战

演示：
  1. 训练 MiniTransformer 并对比 compile 前后的吞吐
  2. 查找并修复 graph break
  3. 动态 batch size 处理
  4. 推理优化选型 (default vs reduce-overhead vs max-autotune)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


# ═══════════════════════════════════════════════════════════════
# Part 1: 定义一个真实结构的 Transformer（含常见 graph break 陷阱）
# ═══════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_model * 4)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    """6 层，256 维，8 头，约 5M 参数 — 足够真实，又够小便于快速实验"""

    def __init__(self, vocab_size=10000, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos_enc[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════
# Part 2: Benchmark 工具
# ═══════════════════════════════════════════════════════════════


def benchmark(fn, args=(), warmup=5, repeat=30, label=""):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / repeat * 1000
    if label:
        print(f"  {label:<40} {elapsed_ms:8.3f} ms")
    return elapsed_ms


# ═══════════════════════════════════════════════════════════════
# Part 3: 训练实战 — 对比 compile vs eager
# ═══════════════════════════════════════════════════════════════


def demo_training():
    print("=" * 65)
    print("训练实战: compile 带来的吞吐提升")
    print("=" * 65)

    vocab_size = 10000
    batch_size = 32
    seq_len = 128

    model = MiniTransformer(vocab_size=vocab_size).cuda()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    def train_step(m, inp, tgt):
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        opt.zero_grad()
        logits = m(inp)
        loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        opt.step()
        return loss

    # Eager baseline
    m_eager = MiniTransformer(vocab_size=vocab_size).cuda()
    print("\n  ── Eager 模式 ──")
    eager_ms = benchmark(
        lambda: train_step(m_eager, x, y), label="  forward+backward+update"
    )

    print(f"\n  ── compile 对比不同策略 ──")

    # 策略 1: 只编译 forward
    m1 = MiniTransformer(vocab_size=vocab_size).cuda()
    m1.forward = torch.compile(m1.forward, mode="default")
    compile_ms = benchmark(
        lambda: train_step(m1, x, y), label="  只编译 forward (default)"
    )

    # 策略 2: compile 整个 model
    m2 = MiniTransformer(vocab_size=vocab_size).cuda()
    m2 = torch.compile(m2, mode="default")
    compile_full_ms = benchmark(
        lambda: train_step(m2, x, y), label="  compile 整个 model (default)"
    )

    print(f"""
  ── 结果对比 ──
    没有 compile:         {eager_ms:.1f} ms/step
    只编译 forward:        {compile_ms:.1f} ms/step  (speedup: {eager_ms / compile_ms:.1f}x)
    整模型 compile:        {compile_full_ms:.1f} ms/step  (speedup: {eager_ms / compile_full_ms:.1f}x)

  注意:
    - compile 整个 model 时会处理 forward+backward+optimizer
    - 但是在优化器 update 部分可能 graph break（非 tensor 操作）
    - 一般推荐: compile model.forward + 手动写在 step() 外面
    - 这个和 FSDP/DDP 的交互: torch.compile 包裹 forward,
      分布式通信层在外部处理
  """)


# ═══════════════════════════════════════════════════════════════
# Part 4: Graph Break 检测与修复
# ═══════════════════════════════════════════════════════════════


def demo_graph_break_fix():
    print("\n" + "=" * 65)
    print("Graph Break 检测与修复实战")
    print("=" * 65)

    class BadModel(nn.Module):
        """故意写了几个常见 graph break 的模型"""

        def __init__(self, d_model=64):
            super().__init__()
            self.fc = nn.Linear(d_model, d_model * 2)

        def forward(self, x):
            y = self.fc(x)

            # break 1: print（调试遗留下来的）
            # print(f"  y mean: {y.mean().item():.4f}")  # ← 取消注释就会 break

            # break 2: 数据依赖条件
            # if y.sum() > 0:         # ← Dynamo 不知道该走哪条
            #     y = y * 2
            # else:
            #     y = y * 0.5

            # break 3: .item()
            # scale = y.max().item()   # ← Tensor -> Python float
            # y = y / scale

            return F.relu(y) + 1.0

    model = BadModel().cuda()

    # 检测: 看有多少 graph break
    print("\n  检测: 用 torch._dynamo.explain 分析")
    x = torch.randn(4, 64, device="cuda")
    exp = torch._dynamo.explain(model)(x)

    print(f"    graph_break_count: {exp.graph_break_count}")
    print(f"    break_reasons: {exp.break_reasons[:2] if exp.break_reasons else '无'}")

    # 修复: 直接替换 view/scalar_outputs 等方法
    print(f"""
  常见修复手法速查:

  ┌─────────────────────────────────────────────────────┐
  │  错误代码                      修复方式              │
  ├─────────────────────────────────────────────────────┤
  │  print(f"loss={{loss.item():.4f}}")  删掉或移出 forward    │
  │  if y.sum() > 0:               torch.where(y.sum()>0, a, b)│
  │  val = y.max().item()          capture_scalar_outputs=True │
  │  for i in range(y.size(0)):    向量化或 torch.while_loop   │
  │  x = y.cpu().numpy()           全部用 torch 算子代替       │
  │  shape 每次变                   dynamic=True              │
  └─────────────────────────────────────────────────────┘
  """)


# ═══════════════════════════════════════════════════════════════
# Part 5: 动态 batch size
# ═══════════════════════════════════════════════════════════════


def demo_dynamic_shape():
    print("\n" + "=" * 65)
    print("动态 batch size 处理")
    print("=" * 65)

    vocab_size = 10000

    class SmallTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 64)
            self.fc = nn.Linear(64, vocab_size)

        def forward(self, x):
            return self.fc(F.relu(self.embed(x)))

    x_small = torch.randint(0, vocab_size, (4, 128), device="cuda")
    x_large = torch.randint(0, vocab_size, (16, 128), device="cuda")

    # 场景 A: 不用 dynamic → 每个 shape 编译一次
    print(f"\n  ── 场景 A: 不用 dynamic（每个新 shape 一次 recompile）──")
    torch._dynamo.reset()
    m_a = SmallTransformer().cuda()
    m_a_c = torch.compile(m_a)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m_a_c(x_small)
    torch.cuda.synchronize()
    t_first = time.perf_counter()

    m_a_c(x_large)  # 触发 recompile
    torch.cuda.synchronize()
    t_second = time.perf_counter()

    print(f"    第一次前向 (B=4):  {(t_first - t0) * 1000:.1f} ms (含编译)")
    print(f"    第二次前向 (B=16): {(t_second - t_first) * 1000:.1f} ms (recompile!)")

    # 场景 B: 用 dynamic=True
    print(f"\n  ── 场景 B: dynamic=True（一次编译，适配多个 shape）──")
    torch._dynamo.reset()
    m_b = SmallTransformer().cuda()
    m_b_c = torch.compile(m_b, dynamic=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m_b_c(x_small)
    torch.cuda.synchronize()
    t_first = time.perf_counter()

    m_b_c(x_large)  # 不触发 recompile
    torch.cuda.synchronize()
    t_second = time.perf_counter()

    print(f"    第一次前向 (B=4):  {(t_first - t0) * 1000:.1f} ms (含编译)")
    print(f"    第二次前向 (B=16): {(t_second - t_first) * 1000:.1f} ms (不复编译!)")

    print(f"""
  选型建议:
    固定 shape 推理           -> 不用 dynamic（编译更优）
    batch 不定的线上服务        -> dynamic=True
    仅 batch 维度变            -> torch._dynamo.mark_dynamic(x, 0)
    其他维度也可能变            -> dynamic=True
    dynamic=True 代价: 编译产物略慢，因为无法做 hard-coded shape 优化
  """)


# ═══════════════════════════════════════════════════════════════
# Part 6: 推理优化 — mode 对比
# ═══════════════════════════════════════════════════════════════


def demo_inference_modes():
    print("\n" + "=" * 65)
    print("推理优化: mode 对比 (default / reduce-overhead / max-autotune)")
    print("=" * 65)

    vocab_size = 10000

    class MiniTransformerSmall(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 128)
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.head = nn.Linear(128, vocab_size)

        def forward(self, x):
            x = F.relu(self.fc1(self.embed(x)))
            x = F.sigmoid(self.fc2(x))
            return self.head(x)

    model = MiniTransformerSmall().cuda().eval()

    # 小 batch（模拟 LLM decode: batch=1, seq=1）
    x_tiny = torch.randint(0, vocab_size, (1, 1), device="cuda")
    # 中等 batch（正常推理）
    x_med = torch.randint(0, vocab_size, (8, 128), device="cuda")

    print("\n  ── 小 batch (B=1, T=1) — 模拟 LLM decode ──")
    benchmark(lambda: model(x_tiny), label="不 compile (eager)")

    for mode in ["default", "reduce-overhead", "max-autotune"]:
        torch._dynamo.reset()
        m = MiniTransformerSmall().cuda().eval()
        mc = torch.compile(m, mode=mode)
        mc(x_tiny)  # 触发编译
        torch.cuda.synchronize()
        benchmark(lambda: mc(x_tiny), label=f"compile mode={mode}")

    print("\n  ── 中等 batch (B=8, T=128) — 正常推理 ──")
    benchmark(lambda: model(x_med), label="不 compile (eager)")

    for mode in ["default", "reduce-overhead", "max-autotune"]:
        torch._dynamo.reset()
        m = MiniTransformerSmall().cuda().eval()
        mc = torch.compile(m, mode=mode)
        mc(x_med)  # 触发编译
        torch.cuda.synchronize()
        benchmark(lambda: mc(x_med), label=f"compile mode={mode}")

    print("""
  ── 结论 ──
    小 batch (LLM decode): reduce-overhead 优势明显
      → CUDA Graph 合并多次 kernel launch 为一次

    中大 batch: default 就很好
      → GPU 忙于计算，launch 开销被覆盖

    max-autotune 在所有场景下都可能更快
      → 但编译时间长（本次示例模型小所以不明显）
      → 线上部署的首选
  """)


# ═══════════════════════════════════════════════════════════════
# Part 7: 训练专用 — AOTAutograd 图检查
# ═══════════════════════════════════════════════════════════════


def demo_training_debug():
    print("\n" + "=" * 65)
    print("训练专用：查看 AOTAutograd 生成的完整训练图")
    print("=" * 65)

    vocab_size = 10000

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, vocab_size)

        def forward(self, x):
            return F.relu(self.fc(x))

    model = Tiny().cuda()
    x = torch.randint(0, vocab_size, (4, 64), device="cuda")
    y = torch.randint(0, vocab_size, (4,), device="cuda")

    # 查看 AOTAutograd 输出（含 forward + backward）
    torch._dynamo.reset()
    node_count = [0]

    def inspect_aot(gm, inputs):
        node_count[0] = len(list(gm.graph.nodes))
        print(f"\n  AOTAutograd 处理后的图:")
        print(f"    总共 {node_count[0]} 个 node（含 forward+backward）")
        for n in gm.graph.nodes:
            t = n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
            print(f"    [{n.op:14}] {n.name:<12} -> {t}")
        return gm.forward

    # 必须做一次完整的 forward+backward 才能看到训练图
    def train_step(m):
        logits = m(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return loss

    torch.compile(model, backend=inspect_aot)
    train_step(model)

    print(f"""
  ── 解读 ──
    推理时 AOTAutograd 图: 只有 forward node, 约 5-10 个
    训练时 AOTAutograd 图: forward + backward node, 数量翻倍

    这个图就是 Inductor 吃的输入:
      ① lowering: 每个 aten node 映射到 IR (Pointwise/Reduction/ExternKernel)
      ② fusion: scheduler 把相邻 pointwise 合并
      ③ codegen: 生成 Triton kernel 源码

    训练加速的核心:
      - 连续 pointwise ops 融合: relu+gelu+add → 一个 kernel
      - backward 梯度计算也被融合: grad_input 计算链合并
      - min-cut rematerialization: 减少反向所需 saved tensor 显存
  """)


# ═══════════════════════════════════════════════════════════════
# Part 8: 环境变量诊断速查
# ═══════════════════════════════════════════════════════════════


def demo_env_vars():
    print("\n" + "=" * 65)
    print("环境变量诊断速查")
    print("=" * 65)
    print("""
  日常开发最常用的三条命令:

    # 1. 看 graph break
    TORCH_LOGS="graph_breaks" python train.py

    # 2. 看生成的 kernel + graph
    TORCH_LOGS="output_code" python train.py

    # 3. 看全部阶段（排查复杂问题）
    TORCH_LOGS="+dynamo,inductor,output_code,graph_breaks,recompiles" python train.py

  生产环境:
    # dump 全部中间产物
    TORCH_COMPILE_DEBUG=1 python train.py

    # 指定编译缓存目录（避免 home 目录占用）
    TORCHINDUCTOR_CACHE_DIR=/mnt/data/cache python train.py

  高级诊断:
    # 看每次 recompile 原因（shape 变化、dtype 变化等）
    TORCH_LOGS="recompiles" python train.py

    # 看 guard 检查细节
    TORCH_LOGS="guards" python train.py
  """)


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 GPU 运行本脚本")
        exit(1)

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")

    demo_training()
    demo_graph_break_fix()
    demo_dynamic_shape()
    demo_inference_modes()
    demo_training_debug()
    demo_env_vars()

    print("\n" + "=" * 65)
    print("实战总结: 你的 torch.compile 上线 checklist")
    print("=" * 65)
    print("""
  ☐ 1. 跑一遍 TORCH_LOGS="graph_breaks" → 确保热点路径 0 break
  ☐ 2. forward 干净: 去掉 print/item/numpy/data-dep if
  ☐ 3. 选对 mode: 训练 default, LLM decode reduce-overhead, 上线 max-autotune
  ☐ 4. dynamic=True 如果 batch size 可变
  ☐ 5. benchmark 对比: eager vs compile, 确认有无加速
  ☐ 6. 精度对比: 同一输入 eager vs compile 输出误差 < 1e-3
  ☐ 7. TORCHINDUCTOR_CACHE_DIR 指定缓存目录
  ☐ 8. 分布式训练: compile 只包 forward, 通信层在外面
""")
