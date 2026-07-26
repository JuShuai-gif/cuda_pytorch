"""
共享工具函数 —— 模型定义、计时、图可视化
"""

import torch
import torch.nn as nn
import time


# ── 模型定义 ──────────────────────────────────────────


class SimpleMLP(nn.Module):
    """最简单的两层层 MLP，用于演示 compile"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 128)

    def forward(self, x):
        return torch.relu(self.linear(x))


class TransformerBlock(nn.Module):
    """小 Transformer Block —— 用于演示 graph break"""

    def __init__(self, d=512):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(x + self.fc2(torch.relu(self.fc1(x))))


class MultiBlockNet(nn.Module):
    """多个 Block 的模型 —— 用于大规模 compile 测试"""

    def __init__(self, d=512, n=4):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(d) for _ in range(n)])

    def forward(self, x):
        return self.blocks(x)


# ── 计时工具 ──────────────────────────────────────────


def benchmark(fn, *args, warmup=10, repeat=100, label=""):
    """
    简单计时：预热 + 重复测量，返回平均耗时(ms)。
    fn: callable
    args: 传参
    """
    # warmup
    for _ in range(warmup):
        fn(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    ms = (t1 - t0) / repeat * 1000

    if label:
        print(f"  {label:<30} {ms:8.3f} ms")

    return ms


# ── 图结构可视化 ──────────────────────────────────────


def print_graph_structure(gm, title="FX Graph"):
    """
    打印 FX Graph 的 opcode / name / target / 数据流。
    """
    print(f"\n=== {title} ===")
    print(f"{'opcode':<16} {'name':<12} {'target':<30} {'inputs':<30}")
    print("-" * 90)

    for node in gm.graph.nodes:
        inputs = ", ".join(a.name for a in node.all_input_nodes)
        target = str(node.target)
        if len(target) > 28:
            target = target[:25] + "..."
        print(f"{node.op:<16} {node.name:<12} {target:<30} {inputs:<30}")

    print()

    # 打印边（数据流）
    print("数据流 (edges):")
    for node in gm.graph.nodes:
        if node.users:
            users = ", ".join(u.name for u in node.users)
            print(f"  {node.name} ──→ [{users}]")


# ── compile 相关环境变量提示 ──────────────────────────


def show_env_hints():
    """打印常用的 compile 调试环境变量"""
    print("""
常用环境变量:
  TORCH_LOGS="graph_breaks,recompiles"    看到哪里发生了 graph break / 重编译
  TORCH_LOGS="+dynamo,inductor,output_code"  看 Dynamo+Inductor 全过程 + 生成的 kernel 代码
  TORCH_COMPILE_DEBUG=1                     dump 中间产物（含 FX graph、Inductor IR）
  TORCHINDUCTOR_CACHE_DIR=./cache           缓存编译结果
""")
