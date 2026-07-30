// ===================== 显式管理 vs Unified Memory =====================
//
// 显式管理（cudaMalloc + cudaMemcpy）
//   - 物理内存分开：malloc 在 host 侧，cudaMalloc 在 device 显存。
//   - 数据移动由你显式控制，什么时候拷到 GPU、什么时候拷回来完全可预测。
//   - 适合：
//     1) 性能敏感场景，需要精确控制数据搬移时机；
//     2) 显存和 host 内存都够用；
//     3) 数据复用多次、kernel 链式调用（一次拷入，多次计算）。
//
// Unified Memory（cudaMallocManaged）
//   - 一份虚拟地址，CPU 和 GPU 都能直接访问。
//   - 物理页按需迁移：谁用谁拿，不用的页留在原地。
//   - 适合：
//     1) 快速原型，不用管 cudaMemcpy 的方向和时机；
//     2) 数据太大显存放不下（oversubscription），UM 自动换页；
//     3) 数据结构复杂（链表、树等），无法用朴素的 cudaMemcpy 拷贝。
//
// 简单原则：先写 UM 把逻辑跑通，确认后再切显式管理做性能优化。
// 已经确定数据流的推理/训练管线直接用显式管理。
// =====================================================================

// 问题 2.3：把显式内存管理改成 Unified Memory（ MODIFY ）。
// 下面是一份完整可运行的显式管理版本。任务：
//   0. 先按原样跑一次，记下耗时——这一版会被你的改动覆盖掉，
//      第 4 步的对比要拿它做基准；
//   1. 用 cudaMallocManaged 替换 cudaMalloc + malloc；
//   2. 删掉所有 cudaMemcpy，kernel 直接读写同一组指针，CPU 也直接读；
//   3. 想清楚哪里需要 cudaDeviceSynchronize；
//   4. 对比两版的耗时。两版的计时窗口要保持一致：分配和填数据都在窗口
//      外，窗口从"数据已经在内存里备好"开始，到 CPU 把结果全部读完为止
//      （下面用一个累加校验和的循环代表"CPU 读完全部结果"，别把它删了）。
// 改完仍要 PASS。
#include <chrono>
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 24;  // 16M 元素
    size_t bytes = (size_t)n * sizeof(float);

    // 先把 CUDA context 建起来。首次调用 CUDA API 要花几百毫秒初始化，
    // 放进计时窗口会把要观察的差距完全淹掉。
    CUDA_CHECK(cudaFree(0));

    // Unified Memory：一份内存，CPU 和 GPU 都能直接访问。
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    // 分配和填数据都在计时窗口外。
    fill_random(a, n, 1);
    fill_random(b, n, 2);

    // 期望的校验和，host 上先算好，同样不计入计时。
    double want = 0;
    for (int i = 0; i < n; i++) want += (double)(a[i] + b[i]);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ================= 计时窗口开始 =================
    auto t0 = std::chrono::steady_clock::now();

    // kernel 访问 a、b 时，Unified Memory 会按需把页搬到 GPU。
    vectorAdd<<<blocks, threads>>>(a, b, c, n);
    // 等 kernel 跑完，否则 CPU 读 c 时数据还没算出来。
    CUDA_CHECK_KERNEL();

    // CPU 读完全部结果。unified memory 版里，这一步才会把结果页搬回 host。
    double got = 0;
    for (int i = 0; i < n; i++) got += (double)c[i];

    auto t1 = std::chrono::steady_clock::now();
    // ================= 计时窗口结束 =================

    printf("搬运 + kernel + 读回: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    REPORT(fabs(got - want) <= 1e-3 * (1.0 + fabs(want)));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return 0;
}
