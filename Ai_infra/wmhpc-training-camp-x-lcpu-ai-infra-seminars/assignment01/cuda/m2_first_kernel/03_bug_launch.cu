// 问题 2.5：找 bug。
// 这个程序不报错，直接 FAIL（kernel 好像压根没跑。。。）
// 任务：先定位到具体error（提示在文件末尾），再解释原因，并修好它。
//
// ===================== Bug 分析与修复 =====================
//
// Bug：int threads = 2048 超出 maxThreadsPerBlock 上限（1024，见 02_device_query）。
//
// 为什么 kernel 没跑却不报错？
//   - kernel 启动是异步的，nvcc 不会在启动时检查执行配置是否合法。
//   - 只有显式调用 cudaGetLastError() / cudaDeviceSynchronize()
//     （即 CUDA_CHECK_KERNEL() 做的事）才能抓到 launch error。
//   - 不加的话，kernel 静默失败，d_c 保持 cudaMemset 写入的零值，
//     cudaMemcpy 拷回 host 后和 h_ref 对比自然 FAIL。
//
// 修复：threads 改回合法值（≤1024），并加上 CUDA_CHECK_KERNEL()。
// ==========================================================
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    int threads = 256;  // 上限 maxThreadsPerBlock = 1024，2048 超了
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();  // 不主动查错的话 kernel 启动失败只会一声不吭

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}

// 提示：kernel 启动语句后面补一行 CUDA_CHECK_KERNEL() 再跑一次，
// 报错信息会告诉你该往哪个方向查。查完记得回答：为什么不加这一行时
// 程序一声不吭？（问题 0.2 打印过的哪个上限和这里有关？）
