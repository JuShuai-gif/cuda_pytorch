#include <torch/extension.h>

#include <vector>

// 前向声明：CUDA kernel 函数

// 多 stream 并发执行：在 N 个独立 stream 上同时执行 vector_add
std::vector<torch::Tensor> multi_stream_concurrent_exec(
    const std::vector<torch::Tensor>& inputs_a,
    const std::vector<torch::Tensor>& inputs_b,
    const std::vector<torch::Tensor>& outputs);

// pinned memory + async copy + kernel overlap 完整 pipeline
std::vector<torch::Tensor> pinned_async_pipeline(
    const std::vector<torch::Tensor>& host_chunks,
    int num_streams);

// 带 CUDA event 精确计时的 kernel 性能测量
torch::Tensor kernel_timing_with_events(
    const torch::Tensor& a,
    const torch::Tensor& b,
    int num_launches);

// cudaStreamWaitEvent 跨 stream 同步示例
std::vector<torch::Tensor> stream_wait_event_demo(
    const torch::Tensor& a,
    const torch::Tensor& b);

// WAR 同步正反例对比
std::vector<torch::Tensor> war_sync_correct_vs_wrong(
    const torch::Tensor& a,
    const torch::Tensor& b);

// ============================================================================
// Python 封装函数：将 vector<torch::Tensor> 拆分为更友好的 Python API
// ============================================================================

torch::Tensor multi_stream_concurrent(const torch::Tensor& a, const torch::Tensor& b, int num_streams) {
    // 将 a 和 b 复制 N 份，在 N 个 stream 上并发执行
    std::vector<torch::Tensor> a_vec, b_vec, c_vec;
    auto shape = a.sizes().vec();
    for (int i = 0; i < num_streams; ++i) {
        a_vec.push_back(a.clone());
        b_vec.push_back(b.clone());
        c_vec.push_back(torch::empty_like(a));
    }
    multi_stream_concurrent_exec(a_vec, b_vec, c_vec);
    return c_vec[0];
}

// ============================================================================
// pybind11 模块定义
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA stream 编程实战：多 stream 并发、异步拷贝、event 同步、pipeline overlap";

    m.def("multi_stream_concurrent_exec", &multi_stream_concurrent_exec,
          "在多个独立 CUDA stream 上并发执行 vector_add kernel。\n"
          "参数:\n"
          "    inputs_a (list[Tensor]): 输入张量 a 列表\n"
          "    inputs_b (list[Tensor]): 输入张量 b 列表\n"
          "    outputs (list[Tensor]): 输出张量列表（预分配）\n"
          "返回:\n"
          "    list[Tensor]: 每个 stream 的 kernel 执行时间（ms）",
          pybind11::arg("inputs_a"), pybind11::arg("inputs_b"), pybind11::arg("outputs"));

    m.def("multi_stream_concurrent", &multi_stream_concurrent,
          "将单个张量对在多个 stream 上并发执行（便捷函数）。\n"
          "参数:\n"
          "    a (Tensor): 输入张量 a\n"
          "    b (Tensor): 输入张量 b\n"
          "    num_streams (int): stream 数量\n"
          "返回:\n"
          "    Tensor: 计算结果",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("num_streams"));

    m.def("pinned_async_pipeline", &pinned_async_pipeline,
          "pinned memory + async copy + kernel overlap 完整 pipeline。\n"
          "将数据分成多个 chunk，使用多个 stream 同时执行 H2D→compute→D2H。\n"
          "参数:\n"
          "    host_chunks (list[Tensor]): pinned memory 中的 chunk 列表\n"
          "    num_streams (int): 使用的 CUDA stream 数量（轮转分配）\n"
          "返回:\n"
          "    list[Tensor]: 每个 chunk 的输出 + 最后一个元素是总 pipeline 时间（ms）",
          pybind11::arg("host_chunks"), pybind11::arg("num_streams"));

    m.def("kernel_timing_with_events", &kernel_timing_with_events,
          "使用 CUDA event 精确测量 kernel 执行时间。\n"
          "参数:\n"
          "    a (Tensor), b (Tensor): 输入张量\n"
          "    num_launches (int): kernel launch 次数\n"
          "返回:\n"
          "    Tensor: [总GPU时间(ms), 平均GPU时间(ms)]",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("num_launches"));

    m.def("stream_wait_event_demo", &stream_wait_event_demo,
          "cudaStreamWaitEvent 跨 stream 同步示例。\n"
          "stream A 计算 intermediate = a + b，stream B 等待 stream A 完成后计算 final。\n"
          "参数:\n"
          "    a, b (Tensor): 输入张量\n"
          "返回:\n"
          "    list[Tensor]: [intermediate, final_result]",
          pybind11::arg("a"), pybind11::arg("b"));

    m.def("war_sync_correct_vs_wrong", &war_sync_correct_vs_wrong,
          "WAR 同步正反例对比。\n"
          "对比 cudaDeviceSynchronize（错误）和 cudaStreamSynchronize（正确）。\n"
          "参数:\n"
          "    a, b (Tensor): 输入张量\n"
          "返回:\n"
          "    list[Tensor]: [out_wrong_a, out_wrong_b, out_correct_c, out_correct_d, timing]",
          pybind11::arg("a"), pybind11::arg("b"));
}
