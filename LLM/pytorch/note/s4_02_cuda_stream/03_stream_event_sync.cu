// 03_stream_event_sync.cu
//
// 一个更贴近实战的模式: 用 event 做跨 stream 的 fork/join(分叉/汇合)。
// PyTorch 就是用它来重叠 计算 与 H2D/D2H 拷贝，DDP 也用它来重叠 反向传播
// 与 梯度 all-reduce。
//
//   主 stream --生产 x--> record(ev_x)
//        |                     |
//        +--> 旁路 stream: wait(ev_x), 在副本上做独立分支
//        |
//   主 stream: 继续做自己的工作
//        |
//   主 stream: wait(ev_side) 汇合，再使用分支的结果
//
// event 是两个 stream 之间唯一的顺序保证; 去掉 wait 就会得到数据竞争
// (参见 03_cuda_stream/cuda_stream.md 坑点 #3)。我们通过校验数值结果来
// 证明顺序是正确的。
//
// 编译:  make 03_stream_event_sync && ./03_stream_event_sync

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t _err = (expr);                                             \
    if (_err != cudaSuccess) {                                            \
      printf("CUDA error %s at %s:%d -> %s\n", cudaGetErrorName(_err),    \
             __FILE__, __LINE__, cudaGetErrorString(_err));               \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

__global__ void fill(float* x, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}
__global__ void add_scalar(float* x, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += v;
}
// out = a + b, 逐元素相加
__global__ void add_vec(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  const int block = 256;
  const int grid = (n + block - 1) / block;

  float *x, *branch, *out;
  CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&branch, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));

  cudaStream_t main_s, side_s;
  CUDA_CHECK(cudaStreamCreate(&main_s));
  CUDA_CHECK(cudaStreamCreate(&side_s));

  cudaEvent_t ev_x, ev_side;
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_x, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_side, cudaEventDisableTiming));

  // ---- 主 stream: 生产 x = 3 ----------------------------------------------
  fill<<<grid, block, 0, main_s>>>(x, n, 3.0f);
  CUDA_CHECK(cudaEventRecord(ev_x, main_s));  // "x 已就绪" 的标记

  // ---- 分叉(FORK): 旁路 stream 等 x 就绪，再计算 branch = x + 100 ---------
  CUDA_CHECK(cudaStreamWaitEvent(side_s, ev_x, 0));  // 依赖主 stream 的 x
  CUDA_CHECK(cudaMemcpyAsync(branch, x, n * sizeof(float),
                             cudaMemcpyDeviceToDevice, side_s));
  add_scalar<<<grid, block, 0, side_s>>>(branch, n, 100.0f);  // branch = 103
  CUDA_CHECK(cudaEventRecord(ev_side, side_s));  // "branch 已就绪" 的标记

  // ---- 主 stream 继续做自己独立的工作: x += 1 -> x = 4 --------------------
  add_scalar<<<grid, block, 0, main_s>>>(x, n, 1.0f);

  // ---- 汇合(JOIN): 主 stream 必须等旁路完成后才能合并结果 -----------------
  CUDA_CHECK(cudaStreamWaitEvent(main_s, ev_side, 0));
  add_vec<<<grid, block, 0, main_s>>>(x, branch, out, n);  // out = 4 + 103 = 107

  CUDA_CHECK(cudaStreamSynchronize(main_s));

  float host = 0;
  CUDA_CHECK(cudaMemcpy(&host, out, sizeof(float), cudaMemcpyDeviceToHost));
  printf("[fork/join] out[0] = %.1f (期望 107.0)\n", host);
  if (host != 107.0f) { printf("FAIL: 跨 stream 顺序被打乱了\n"); return 1; }

  CUDA_CHECK(cudaEventDestroy(ev_x));
  CUDA_CHECK(cudaEventDestroy(ev_side));
  CUDA_CHECK(cudaStreamDestroy(main_s));
  CUDA_CHECK(cudaStreamDestroy(side_s));
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(branch));
  CUDA_CHECK(cudaFree(out));

  printf("\nOK. 核心要点:\n");
  printf("  * 分叉 = 旁路 stream wait_event(主 stream 的标记) 后运行自己的分支\n");
  printf("  * 汇合 = 主 stream wait_event(旁路的标记) 后再使用结果\n");
  printf("  * event 是跨 stream 唯一的顺序保证\n");
  return 0;
}
