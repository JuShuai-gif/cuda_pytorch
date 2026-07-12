// 01_stream_basics.cu
//
// CUDA Stream = 提交 GPU 工作的"命令队列"。同一个 stream 内的工作严格按序执行；
// 不同 stream 之间的工作可以并发执行。
//
// 本例复现 PyTorch 的 c10::cuda::CUDAStream (c10/cuda/CUDAStream.h) 所封装的东西:
// 一个裸 cudaStream_t 外加 priority()、query()、synchronize()、is_capturing() 等辅助方法。
// PyTorch 每个设备维护一个 32 流的"池"(kStreamsPerPool = 1 << 5)，轮询复用而不是每次
// 重新创建/销毁 stream。这里我们把这个思想复刻出来。
//
// 编译:  make 01_stream_basics && ./01_stream_basics

#include <cstdio>
#include <vector>
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

// 一个故意让 GPU 忙一阵子的 kernel，好让我们"看得见"两个 stream 确实在并发。
// 它只是做一串有依赖的浮点运算(避免被编译器优化掉)。
__global__ void busy_kernel(float* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = out[i];
  for (int k = 0; k < iters; ++k) {
    acc = acc * 1.0000001f + 1.0f;  // 有依赖的运算链，不会被优化掉
  }
  out[i] = acc;
}

// -----------------------------------------------------------------------------
// PyTorch 的 stream 池的极简版。c10 每个设备预分配 kStreamsPerPool = 32 个 stream，
// 通过 getStreamFromPool() 轮询返回。复用 stream 让"创建"一个 stream 几乎零开销。
// ponytail: 这里只用 8 个槽(不是 32)且只支持单设备，足够展示这个思想。
// -----------------------------------------------------------------------------
struct StreamPool {
  static constexpr int kStreamsPerPool = 8;  // c10 用的是 1 << 5 = 32
  cudaStream_t streams[kStreamsPerPool];
  int next = 0;

  StreamPool() {
    for (int i = 0; i < kStreamsPerPool; ++i) {
      // 低优先级池用默认标志; c10 还额外有一个高优先级池。
      CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, 0));
    }
  }
  ~StreamPool() {
    for (int i = 0; i < kStreamsPerPool; ++i) cudaStreamDestroy(streams[i]);
  }
  // getStreamFromPool(): 轮询取用，到达 kStreamsPerPool 后回绕到 0。
  cudaStream_t get() {
    cudaStream_t s = streams[next];
    next = (next + 1) % kStreamsPerPool;
    return s;
  }
};

int main() {
  const int n = 1 << 20;      // 100 万个元素
  const int iters = 20000;    // 足够重，才能看出并发
  const int block = 256;
  const int grid = (n + block - 1) / block;

  // --- (1) Stream 优先级范围，对应 CUDAStream::priority_range() -------------
  int least = 0, greatest = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  printf("[priority] least=%d greatest=%d  (数字越小 = 优先级越高)\n",
         least, greatest);

  // --- (2) 池: 复用 stream 而不是每次都新建 --------------------------------
  StreamPool pool;
  cudaStream_t s0 = pool.get();
  cudaStream_t s1 = pool.get();
  printf("[pool] 从 8 槽轮询池中取出了 2 个 stream\n");

  float *a, *b;
  CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(a, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(b, 0, n * sizeof(float)));

  // --- (3) 启动前 query(): 队列里没有待办 -> true --------------------------
  printf("[query] 启动前 s0 空闲吗? %s\n",
         cudaStreamQuery(s0) == cudaSuccess ? "是" : "否");

  // --- (4) 在两个 stream 上启动独立的工作: 它们会并发 ----------------------
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  CUDA_CHECK(cudaEventRecord(t0, 0));  // 在默认 stream 上记录
  busy_kernel<<<grid, block, 0, s0>>>(a, n, iters);
  busy_kernel<<<grid, block, 0, s1>>>(b, n, iters);  // 与 s0 并发
  CUDA_CHECK(cudaEventRecord(t1, 0));

  // --- (5) 启动后立刻 query(): 工作还在执行 -> 大概率还没完成 --------------
  cudaError_t q = cudaStreamQuery(s0);
  printf("[query] 启动后立刻查询 s0 完成了吗? %s\n",
         q == cudaSuccess ? "是" : "否(cudaErrorNotReady, 符合预期)");
  if (q != cudaSuccess && q != cudaErrorNotReady) CUDA_CHECK(q);

  // --- (6) synchronize() 单个 stream (而不是整个设备) ----------------------
  // 这就是 PyTorch 里的 stream.synchronize(); 热路径上应优先用它，
  // 而不是全局的 torch.cuda.synchronize() (即 cudaDeviceSynchronize)。
  CUDA_CHECK(cudaStreamSynchronize(s0));
  CUDA_CHECK(cudaStreamSynchronize(s1));
  printf("[sync] 两个 stream 都执行完了\n");

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  printf("[time] 双 stream 墙钟时间(launch->launch 窗口): %.3f ms\n", ms);

  // --- (7) is_capturing(): 对应 CUDAStream::is_capturing() -----------------
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(s0, &status));
  printf("[capture] s0 正在被 graph 捕获吗? %s\n",
         status != cudaStreamCaptureStatusNone ? "是" : "否");

  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));

  printf("\nOK. 核心要点:\n");
  printf("  * 单个 stream = 有序队列; 不同 stream 之间可并发\n");
  printf("  * 池 + 轮询 = '创建' stream 几乎零开销 (c10 的模式)\n");
  printf("  * query() = 非阻塞轮询; synchronize() = 阻塞直到完成\n");
  return 0;
}
