"""torch.compile 原理的最小可复现实验, 配套 notes/pytorch_torch_compile_source_analysis.md。

四个实验各演示一条核心机制, 配合 TORCH_LOGS 观察真实行为。
环境: torch 2.x + CUDA, 单 GPU(无 GPU 会自动回退 CPU, 但 outputcode 看不到 Triton)。

运行方式(每个实验配不同的 TORCH_LOGS):

    # A. recompile: shape 一变就重编译, 看 guard 失败原因
    #    (彩蛋: 第三个 shape 因 automatic-dynamic 不再 recompile)
    TORCH_LOGS=recompiles   python torch_compile_demo.py recompile

    # B. dynamic=True: shape 升级成符号, 一份 kernel 复用, recompile 清零
    TORCH_LOGS=recompiles   python torch_compile_demo.py dynamic

    # C. graph break: 数据依赖的 if 无法编译, 看 break 位置与原因
    TORCH_LOGS=graph_breaks python torch_compile_demo.py graphbreak

    # D. 融合: mul/add/relu 融成一个 Triton kernel, 中间结果只活在寄存器
    TORCH_LOGS=output_code  python torch_compile_demo.py outputcode

    # 一次性总览(图数量 / break 数 / 原因)
    #   torch._dynamo.explain(fn)(*args)
"""

import sys

import torch


# ============ A. recompile: shape 变化触发重编译 ============
def exp_recompile(dev):
    # dynamic=False(默认): 每个新 shape 编一份特化 kernel
    @torch.compile
    def f(x):
        return (x * 2 + 1).relu().sum()

    for n in [8, 16, 32]:
        print(f"--- call shape ({n},{n}) ---", flush=True)
        f(torch.randn(n, n, device=dev))


# ============ B. dynamic=True: 符号 shape 消除 recompile ============
def exp_dynamic(dev):
    # shape 变成符号变量 s0, 同一份 kernel 适配多种尺寸
    @torch.compile(dynamic=True)
    def f(x):
        return (x * 2 + 1).relu().sum()

    for n in [8, 16, 32]:
        print(f"--- call shape ({n},{n}) ---", flush=True)
        f(torch.randn(n, n, device=dev))


# ============ C. graph break: 数据依赖控制流 ============
def exp_graphbreak(dev):
    @torch.compile
    def f(x):
        y = x * 2
        if y.sum() > 0:  # data-dependent control flow -> graph break
            y = y + 1
        return y.relu()

    f(torch.randn(4, device=dev))


# ============ D. 算子融合: 三个 pointwise -> 一个 Triton kernel ============
def exp_outputcode(dev):
    @torch.compile
    def f(x):
        return (x * 2 + 1).relu()  # mul, add, relu

    f(torch.randn(1024, device=dev))


EXPERIMENTS = {
    "recompile": exp_recompile,
    "dynamic": exp_dynamic,
    "graphbreak": exp_graphbreak,
    "outputcode": exp_outputcode,
}


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "recompile"
    if exp not in EXPERIMENTS:
        print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
        sys.exit(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch {torch.__version__}] device={dev} exp={exp}", flush=True)
    EXPERIMENTS[exp](dev)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
