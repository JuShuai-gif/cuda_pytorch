#include <torch/extension.h>

// 前向声明：CUDA 带宽测试 kernel

torch::Tensor bench_copy_float(const torch::Tensor& src);
torch::Tensor bench_copy_float2(const torch::Tensor& src);
torch::Tensor bench_copy_float4(const torch::Tensor& src);
torch::Tensor bench_strided_copy(const torch::Tensor& src, int64_t stride);
torch::Tensor bench_elem_mul_float4(const torch::Tensor& a, const torch::Tensor& b);

// ============================================================================
// pybind11 模块定义
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA memory bandwidth benchmark kernels：float/float2/float4 向量化、coalesced vs strided 访问对比";

    m.def("bench_copy_float", &bench_copy_float,
          "float 标量 copy kernel 基准测试。\n"
          "参数:\n"
          "    src (Tensor): 源张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 平均 kernel 时间（微秒）",
          pybind11::arg("src"));

    m.def("bench_copy_float2", &bench_copy_float2,
          "float2 向量化 copy kernel 基准测试。每个线程一次读取 2 个 float。\n"
          "参数:\n"
          "    src (Tensor): 源张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 平均 kernel 时间（微秒）",
          pybind11::arg("src"));

    m.def("bench_copy_float4", &bench_copy_float4,
          "float4 向量化 copy kernel 基准测试。每个线程一次读取 4 个 float。\n"
          "参数:\n"
          "    src (Tensor): 源张量 (float32, CUDA, contiguous)\n"
          "返回:\n"
          "    Tensor: 平均 kernel 时间（微秒）",
          pybind11::arg("src"));

    m.def("bench_strided_copy", &bench_strided_copy,
          "strided copy kernel 基准测试。模拟非 contiguous 内存访问。\n"
          "参数:\n"
          "    src (Tensor): 源张量 (float32, CUDA)\n"
          "    stride (int): 读取间隔（1=coalesced, >1=strided）\n"
          "返回:\n"
          "    Tensor: 平均 kernel 时间（微秒）",
          pybind11::arg("src"), pybind11::arg("stride"));

    m.def("bench_elem_mul_float4", &bench_elem_mul_float4,
          "float4 向量化 elementwise multiply kernel。\n"
          "参数:\n"
          "    a, b (Tensor): 输入张量 (float32, CUDA, 相同形状)\n"
          "返回:\n"
          "    Tensor: [平均 kernel 时间（微秒）, 计算结果]",
          pybind11::arg("a"), pybind11::arg("b"));
}
