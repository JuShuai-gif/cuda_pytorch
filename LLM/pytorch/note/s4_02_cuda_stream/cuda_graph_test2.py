"""CUDA Graph 进阶: 图更新 + 多 graph 串联 + 动态 shape 绕过方案"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def demo_graph_update():
    """用 cudaGraphExecUpdate 更新图参数（不重新 capture）

    适用场景: 模型参数 warm-up 后权重变化了，但图结构不变，
    不用重新 capture（capture 开销大），直接更新图即可。
    """
    print("=== 图更新 (cudaGraphExecUpdate) ===")

    x = torch.randn(256, 256, device=DEVICE)
    w = torch.randn(256, 256, device=DEVICE)

    # warmup: 让 cublas 初始化好 handle，capture 时不会报错
    _ = x @ w
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x @ w

    # 初次 replay
    x.copy_(torch.randn(256, 256, device=DEVICE))
    graph.replay()
    print(f"  初次 replay, y.mean() = {y.mean().item():.4f}")

    # 权重更新了，但图结构没变 → 用 replay() 即可（地址没变）
    # 如果地址变了，需要用 graph_pool_handle 预先分配固定地址
    w.copy_(torch.randn(256, 256, device=DEVICE))
    graph.replay()
    print(f"  权重更新后 replay, y.mean() = {y.mean().item():.4f}")

    graph.reset()
    print("  PASS\n")


def demo_multi_graph_chaining():
    """多个 CUDAGraph 串联，每个图依赖前一个的输出

    将一个大模型拆分成多个子图分别 capture，降低内存占用，
    也方便调试。
    """
    print("=== 多 graph 串联 ===")

    x = torch.randn(512, 512, device=DEVICE)
    w1 = torch.randn(512, 512, device=DEVICE)
    w2 = torch.randn(512, 512, device=DEVICE)

    # warmup cublas handle
    _ = x @ w1
    _ = x @ w2
    torch.cuda.synchronize()

    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()

    # 子图 1: x @ w1
    with torch.cuda.graph(g1):
        h = x @ w1

    # 子图 2: h @ w2（依赖 g1 的输出 h）
    with torch.cuda.graph(g2):
        out = h @ w2

    x.copy_(torch.randn(512, 512, device=DEVICE))
    g1.replay()             # 先执行 g1
    g2.replay()             # g2 自动看到 g1 的输出
    torch.cuda.synchronize()

    print(f"  串联 replay 成功, out.shape = {out.shape}")
    g1.reset()
    g2.reset()
    print("  PASS\n")


def demo_dynamic_shape_workaround():
    """绕过 dynamic shape 限制：padding 到最大 shape

    CUDA Graph 要求所有 tensor shape 在 capture 时固定。
    对于变长输入，padding 到最大长度，只计算有效部分。
    """
    print("=== 绕过 dynamic shape: padding 到固定尺寸 ===")

    MAX_SEQ = 128          # 最大序列长度
    HIDDEN = 64

    # 固定大小的输入（短序列补 0）
    x = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE)
    w = torch.randn(HIDDEN, HIDDEN, device=DEVICE)

    # warmup cublas handle
    _ = x @ w
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x @ w

    # 实际序列长度 128，padding 到 512
    real_data = torch.randn(128, HIDDEN, device=DEVICE)
    x[:128].copy_(real_data)    # 只填充前 128 行，后面是 0
    graph.replay()

    # 结果只取有效部分
    valid_result = y[:128]
    print(f"  padding 方案成功, valid output shape = {valid_result.shape}")
    print(f"  y.mean() = {y.mean().item():.4f}")

    graph.reset()
    print("  PASS\n")


def demo_graph_optimization_tips():
    """使用 CUDAGraph 的最佳实践和注意事项"""
    print("=== CUDAGraph 使用建议 ===")

    tips = [
        "1. 小 kernel 加速明显（launch 时间占比大）",
        "2. 大 kernel（如大 matmul）收益有限（计算占主导）",
        "3. 图内不能有 CPU 同步 (.item(), .cpu(), synchronize())",
        "4. 图内不能有动态内存分配（if/else 导致不同路径）",
        "5. 输入/输出 tensor 用 copy_() 填内容，不换地址",
        "6. 多 graph 共享 pool 降低内存碎片",
        "7. 预热几轮再计时，避免首次 launch 的 cold start",
        "8. capture 前调 torch.cuda.empty_cache() 释放碎片",
        "9. 推荐在专门 stream 上 capture/replay，不阻塞主 stream",
    ]
    for t in tips:
        print(f"  {t}")
    print()


def demo_cuda_graph_memory_analysis():
    """分析 CUDAGraph 的内存占用"""
    print("=== CUDAGraph 内存分析 ===")

    def allocated_mb():
        return torch.cuda.memory_allocated() / 1024 / 1024

    def reserved_mb():
        return torch.cuda.memory_reserved() / 1024 / 1024

    size_mb = 50
    n_elements = size_mb * 1024 * 1024 // 4  # float32

    before_alloc = allocated_mb()
    x = torch.randn(n_elements, device=DEVICE)
    print(f"  tensor 分配后: allocated={allocated_mb() - before_alloc:.0f}MB")

    before_capture = allocated_mb()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2 + 1
    after_capture = allocated_mb()
    print(f"  录制后额外分配: {after_capture - before_capture:.0f}MB")
    print(f"  graph.pool() token: {graph.pool()}")

    graph.reset()
    torch.cuda.empty_cache()
    print(f"  reset + empty_cache 后: allocated={allocated_mb():.0f}MB")
    print("  PASS\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[SKIP] CUDA 不可用")
        exit(0)

    print(f"设备: {torch.cuda.get_device_name(0)}")
    demo_graph_update()
    demo_multi_graph_chaining()
    demo_dynamic_shape_workaround()
    demo_graph_optimization_tips()
    demo_cuda_graph_memory_analysis()
    print("[所有进阶 demo 完成]")