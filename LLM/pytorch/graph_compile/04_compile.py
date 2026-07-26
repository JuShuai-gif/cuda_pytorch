"""
04_compile.py — backend/mode 对比 + 实际编译产物生成

包含:
  1. 四种 backend 对比 (inductor, eager, aot_eager, cudagraphs)
  2. inductor 三种 mode 对比 (default, reduce-overhead, max-autotune)
  3. 生成实际编译中间产物到 _compile_artifacts/
"""

import torch
import torch.nn as nn
import time
import os
import sys
import subprocess
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Part 1: 四种 backend 对比
# ═══════════════════════════════════════════════════════════════


def bench(fn, args=(), warmup=10, repeat=50, label=""):
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / repeat * 1000
    if label:
        print(f"  {label:<35} {ms:8.3f} ms")
    return ms


def demo_backends():
    print("=" * 60)
    print("四种 backend 对比")
    print("=" * 60)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)

        def forward(self, x):
            return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

    model = Model().cuda().eval()
    x = torch.randn(32, 256, device="cuda")

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    bench(lambda: model(x), label="不 compile (eager)")

    # 四种 backend
    for bk, desc in [
        ("eager", "仅验证图等不等价"),
        ("aot_eager", "AOTAutograd 但不融合"),
        ("inductor", "算子融合+代码生成(默认)"),
        ("cudagraphs", "仅 CUDA Graph 包裹"),
    ]:
        c = torch.compile(model, backend=bk)
        c(x)
        torch.cuda.synchronize()
        bench(lambda: c(x), label=f"backend={bk}")

    print("""
  backend 选型:
    正常用           → inductor（默认），99% 场景
    怀疑编译导致问题  → eager，看是不是图的问题
    怀疑融合有问题    → aot_eager，跳过 fusion
    kernel 大且少    → cudagraphs，不需要融合
""")


# ═══════════════════════════════════════════════════════════════
# Part 2: inductor 三种 mode
# ═══════════════════════════════════════════════════════════════


def demo_modes():
    print("\n" + "=" * 60)
    print("inductor 三种 mode 对比")
    print("=" * 60)

    class ManyOps(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 128)

        def forward(self, x):
            for _ in range(5):
                x = torch.relu(self.fc(x))
                x = x + 0.1
                x = x * 1.01
            return x

    model = ManyOps().cuda().eval()
    x = torch.randn(8, 128, device="cuda")  # 小 batch，launch overhead 明显

    bench(lambda: model(x), label="不 compile (eager)")

    for mode in ["default", "reduce-overhead", "max-autotune"]:
        c = torch.compile(model, backend="inductor", mode=mode)
        c(x)
        torch.cuda.synchronize()
        bench(lambda: c(x), label=f"mode={mode}")

    print("""
  mode 选型:
    日常开发       → default，编译快，效果已经很好
    LLM decode     → reduce-overhead，CUDA Graph 消除 launch 开销
    模型上线       → max-autotune，编译最慢但运行最快
""")


# ═══════════════════════════════════════════════════════════════
# Part 3: 生成实际编译产物
# ═══════════════════════════════════════════════════════════════


def demo_artifacts():
    print("\n" + "=" * 60)
    print("生成编译中间产物")
    print("=" * 60)

    OUT = Path(__file__).parent / "_compile_artifacts"
    OUT.mkdir(exist_ok=True)
    (OUT / ".gitignore").write_text("# 中间产物\n*.py\n*.txt\n*.log\n4_full_debug/\n")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)

        def forward(self, x):
            return torch.sigmoid(torch.relu(self.fc(x))) + 1.0

    model = Model()
    x = torch.randn(4, 8)

    # ── ① Dynamo FX Graph ──
    torch._dynamo.reset()

    def save_fx(gm, inputs):
        (OUT / "1_dynamo_fx_graph.py").write_text(gm.code)
        lines = [f"{n.op} {n.name} → {n.target}" for n in gm.graph.nodes]
        (OUT / "1_dynamo_fx_graph.txt").write_text("\n".join(lines))
        return gm.forward

    torch.compile(model, backend=save_fx)(x)
    print("  ✓ 1_dynamo_fx_graph.py  — Dynamo 输出的可执行 Python 代码")
    print("  ✓ 1_dynamo_fx_graph.txt — Dynamo 输出的图结构")

    # ── ② AOTAutograd ──
    torch._dynamo.reset()

    def save_aot(gm, inputs):
        (OUT / "2_aotautograd.py").write_text(
            "# AOTAutograd 处理后的图 (Inductor 前)\n" + gm.code
        )
        return gm.forward

    torch.compile(model, backend=save_aot)(x)
    print("  ✓ 2_aotautograd.py       — AOTAutograd 处理后的图")

    # ── ③ TORCH_COMPILE_DEBUG dump ──
    script = """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(8, 16)
    def forward(self, x): return torch.sigmoid(torch.relu(self.fc(x))) + 1.0
m = M(); c = torch.compile(m); c(torch.randn(4, 8))
"""
    debug_dir = OUT / "4_full_debug"
    debug_dir.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TORCH_COMPILE_DEBUG": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(debug_dir / "cache"),
        },
        timeout=30,
    )

    # Move dump
    import glob, shutil

    dumps = sorted(Path.cwd().glob("torch_compile_debug/run_*"), key=os.path.getmtime)
    if dumps:
        target = debug_dir / "latest"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(dumps[-1]), str(target))
        for d in dumps[:-1]:
            try:
                shutil.rmtree(str(d))
            except:
                pass
        print(f"  ✓ 4_full_debug/latest/   — TORCH_COMPILE_DEBUG=1 完整 dump")
        kernel = target / "torchinductor" / "model__0_forward_1.0" / "output_code.py"
        if kernel.exists():
            print(f"    → {kernel.relative_to(debug_dir)}  ← Inductor 生成的 kernel")

    print(f"\n  产物在: {OUT.resolve()}")
    print("  用 cat 查看每个阶段生成的真实文件")


if __name__ == "__main__":
    demo_backends()
    demo_modes()
    demo_artifacts()
