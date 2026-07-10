"""CUDA Graph 演示: 基础捕获/重放 + 内存池共享 + 性能对比"""

import gc
import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def demo_capture_replay():
    """基础 capture → replay 流程"""
    print("=== 基础 capture & replay ===")

    # 输入 tensor，地址必须在 replay 间固定（用 copy_ 换内容不换地址）
    x = torch.randn(4096, 4096, device=DEVICE)

    # 1. 创建 CUDAGraph 对象
    graph = torch.cuda.CUDAGraph()

    # 2. 在 torch.cuda.graph() 上下文中 capture
    with torch.cuda.graph(graph):
        y = x * 2 + 1       # 图里记录这个 op
        z = y.relu()        # 和这个 op

    # 3. 换输入（in-place copy_ 保持地址不变）
    x.copy_(torch.randn(4096, 4096, device=DEVICE))

    # 4. 重放（一次 cudaGraphLaunch 驱动整个图）
    graph.replay()
    result = z.mean().item()
    assert isinstance(result, float)
    print(f"  replay 成功, z.mean() = {result:.4f}")

    # 5. 不再需要时释放资源
    graph.reset()
    print("  PASS\n")


def demo_pool_sharing():
    """共享内存池，减少碎片"""
    print("=== 内存池共享 ===")

    x1 = torch.randn(1024, 1024, device=DEVICE)
    x2 = torch.randn(1024, 1024, device=DEVICE)

    # 第一个 graph 创建自己的私有池
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        y1 = x1 * 2

    # 第二个 graph 复用 g1 的池
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=g1.pool()):
        y2 = x2 * 3

    x1.copy_(torch.randn(1024, 1024, device=DEVICE))
    x2.copy_(torch.randn(1024, 1024, device=DEVICE))
    g1.replay()
    g2.replay()

    g1.reset()
    g2.reset()
    print("  PASS\n")


def demo_speed_compare():
    """graph replay 与 eager 模式的速度对比（小 kernel 效果明显）"""
    print("=== 速度对比: graph vs eager ===")

    N, M = 128, 128
    n_iter = 500
    x = torch.randn(N, M, device=DEVICE)

    # 捕获
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2 + 1

    # --- Graph replay 耗时 ---
    x.copy_(torch.randn(N, M, device=DEVICE))
    for _ in range(10):                     # 预热
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        graph.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # --- Eager 模式耗时 ---
    for _ in range(10):
        x * 2 + 1
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(n_iter):
        x * 2 + 1
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    graph_ms = (t1 - t0) / n_iter * 1000
    eager_ms = (t3 - t2) / n_iter * 1000

    print(f"  Graph replay:  {graph_ms:.4e} ms/iter")
    print(f"  Eager launch:  {eager_ms:.4e} ms/iter")
    if eager_ms > 0:
        print(f"  加速比:        {eager_ms / graph_ms:.2f}x")
    print()


def demo_multi_stream():
    """在非默认 stream 上 capture/replay，避免阻塞主 stream"""
    print("=== 多 stream 上的 CUDAGraph ===")

    x = torch.randn(1024, 1024, device=DEVICE)
    s = torch.cuda.Stream()         # 自定义 stream
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, stream=s):
        y = x * 2

    x.copy_(torch.randn(1024, 1024, device=DEVICE))
    # replay 必须在同一 stream 上
    with torch.cuda.stream(s):
        graph.replay()
    s.synchronize()

    print(f"  replay 成功, 结果 mean = {y.mean().item():.4f}")
    graph.reset()
    print("  PASS\n")


def demo_graph_with_grad():
    """带梯度的图捕获（需要 torch.no_grad 内 capture）"""
    print("=== 带梯度的图（推理场景） ===")

    x = torch.randn(256, 256, device=DEVICE, requires_grad=True)
    graph = torch.cuda.CUDAGraph()

    # 推理场景通常关闭梯度
    with torch.no_grad():
        with torch.cuda.graph(graph):
            y = x * 2 + 1

    x.data.copy_(torch.randn(256, 256, device=DEVICE))
    graph.replay()
    print(f"  replay 成功, y.shape = {y.shape}")
    graph.reset()
    print("  PASS\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[SKIP] CUDA 不可用")
        exit(0)

    print(f"设备: {torch.cuda.get_device_name(0)}")
    demo_capture_replay()
    demo_pool_sharing()
    demo_speed_compare()
    demo_multi_stream()
    demo_graph_with_grad()
    print("[所有 demo 完成]")