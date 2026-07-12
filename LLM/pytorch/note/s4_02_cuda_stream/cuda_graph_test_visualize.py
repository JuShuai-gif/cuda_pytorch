"""可视化 CUDAGraph: 打印内部 kernel 节点 + 导出 dot 图 + FX graph 对比"""

import os
import tempfile

import torch


def print_cuda_graph_kernels():
    """用 enable_debug_mode + debug_dump 导出 CUDA Graph 内部的 kernel 列表"""
    print("=== 1. CUDA Graph 内部 kernel 节点 ===")

    x = torch.randn(128, 128, device="cuda")

    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()  # 必须先开 debug mode
    with torch.cuda.graph(graph):
        y = x * 2 + 1
        z = torch.relu(y)
        w = z.sum()

    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False, mode="w") as f:
        path = f.name
    graph.debug_dump(path)

    with open(path) as f:
        dot = f.read()
    os.unlink(path)

    for line in dot.strip().split("\n"):
        if "label=" in line and ("->" not in line or line.count("label=") > 1):
            start = line.find("label=") + 7
            end = line.find('"', start)
            label = line[start:end] if end > start else line[start:start+80]
            print(f"  {label}")
    print()

    graph.reset()


def print_cuda_graph_kernel_detail():
    """用 profiler 抓 replay 时实际 launch 的 CUDA kernel"""
    print("=== 2. CUDA Graph replay 时 launch 的 kernel ===")

    x = torch.randn(128, 128, device="cuda")
    w = torch.randn(128, 128, device="cuda")

    _ = x @ w
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x @ w
        z = y.relu()
        w2 = z.sum()

    x.copy_(torch.randn(128, 128, device="cuda"))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()

    print(f"  {'Kernel':<35s} {'calls':<6s} {'time(us)':<10s}")
    print(f"  {'-'*55}")
    for evt in prof.key_averages():
        t_us = evt.cuda_time if evt.cuda_time else 0
        print(f"  {evt.key:<35s} {evt.count:<6d} {t_us:<10.1f}")

    graph.reset()
    print()


def show_fx_graph():
    """用 torch.fx 展示计算图（Python op IR）"""
    print("=== 3. FX Graph（Python 计算图 IR）===")

    class MyModule(torch.nn.Module):
        def forward(self, x, w):
            y = x @ w
            z = torch.relu(y)
            return z.sum()

    traced = torch.fx.symbolic_trace(MyModule())
    print(traced.graph)
    print()


def show_torch_compile_graph():
    """用 torch.dynamo.export 展示 trace 的 graph"""
    print("=== 4. torch.compile/torch.export 计算图 ===")

    def fn(x, w):
        y = x @ w
        z = torch.relu(y)
        return z.sum()

    x = torch.randn(128, 128, device="cuda")
    w = torch.randn(128, 128, device="cuda")

    # torch.export.export 输出 ExportedProgram，包含完整的计算图
    try:
        ep = torch.export.export(fn, (x.cpu(), w.cpu()))
        print(ep.graph)
    except Exception as e:
        print(f"  torch.export 导出失败: {e}")

    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[SKIP] CUDA 不可用")
        exit(0)

    print_cuda_graph_kernels()
    print_cuda_graph_kernel_detail()
    show_fx_graph()
    show_torch_compile_graph()
    print("[所有可视化 demo 完成]")