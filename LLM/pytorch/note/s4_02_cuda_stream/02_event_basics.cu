// 02_event_basics.cu
//
// CUDA Event = 插入到 stream 里的一个"标记点"。它有两个独立用途:
//   1. 计时: cudaEventElapsedTime 测量两个 event 之间的 GPU 时间
//      (对应 torch.cuda.Event(enable_timing=True) + elapsed_time())。
//   2. 同步: cudaStreamWaitEvent 让 stream B 等待 stream A 中记录的 event
//      完成后才继续(跨 stream 的顺序保证)。
//
// 对应 c10::cuda::CUDAEvent (c10/cuda/CUDAEvent.h):
//   - record(stream)  -> cudaEventRecordWithFlags
//   - block(stream)   -> cudaStreamWaitEvent  (PyTorch 叫 "block", API 叫 "wait")
//   - query()         -> cudaEventQuery
//   - synchronize()   -> cudaEventSynchronize
//   - elapsed_time()  -> cudaEventElapsedTime  (必须开启计时)
//
// 注意 flags: PyTorch 默认是 cudaEventDisableTiming (CUDAEvent.h:246);
// 必须传 enable_timing=True 才能得到可计时的 event。两种我们都演示。
//
// 编译:  make 02_event_basics && ./02_event_basics

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

__global__ void scale_add(float* x, int n, float mul, float add) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * mul + add;
}

__global__ void busy_kernel(float* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = out[i];
  for (int k = 0; k < iters; ++k) acc = acc * 1.0000001f + 1.0f;
  out[i] = acc;
}

int main() {
  const int n = 1 << 20;
  const int block = 256;
  const int grid = (n + block - 1) / block;

  float* x;
  CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(x, 0, n * sizeof(float)));

  cudaStream_t sA, sB;
  CUDA_CHECK(cudaStreamCreate(&sA));
  CUDA_CHECK(cudaStreamCreate(&sB));

  // ------------------------------------------------------------------------
  // 第一部分: 计时用的 event。默认的 cudaEventCreate 是开启计时的; 而 PyTorch
  // 反过来默认 DisableTiming，需要通过 enable_timing=True 显式开启。
  // ------------------------------------------------------------------------
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault));  // 开启计时
  CUDA_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));

  CUDA_CHECK(cudaEventRecord(start, sA));
  busy_kernel<<<grid, block, 0, sA>>>(x, n, 30000);
  CUDA_CHECK(cudaEventRecord(stop, sA));

  // 完成前 query(): 非阻塞轮询。
  printf("[query] 刚 record 完 stop event 就完成了吗? %s\n",
         cudaEventQuery(stop) == cudaSuccess ? "是" : "否(还没就绪)");

  // 对 event 做 synchronize(): 阻塞 CPU 直到它完成。
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("[time] busy_kernel 在 sA 上耗时 %.3f ms (纯 GPU 时间, 不含 CPU 开销)\n", ms);

  // 计时被禁用的 event 不能用于 elapsed_time (会报错)。
  // 这正是 CUDAEvent::elapsed_time 里的 TORCH_CHECK_VALUE 所做的检查。

  // ------------------------------------------------------------------------
  // 第二部分: 跨 stream 同步。sB 必须等 sA 的工作做完才能运行。
  //   sA:  scale_add (x = x*2 + 1)  --record(dep)-->
  //   sB:  wait(dep) 之后再 scale_add (x = x*10 + 0)
  // 如果没有这个 wait，sB 可能与 sA 在同一个 buffer 上产生数据竞争。
  // ------------------------------------------------------------------------
  CUDA_CHECK(cudaMemsetAsync(x, 0, n * sizeof(float), sA));
  scale_add<<<grid, block, 0, sA>>>(x, n, 2.0f, 1.0f);  // x = 0*2+1 = 1

  cudaEvent_t dep;
  CUDA_CHECK(cudaEventCreateWithFlags(&dep, cudaEventDisableTiming));  // 仅用于同步
  CUDA_CHECK(cudaEventRecord(dep, sA));      // 标记 sA 的工作之后的这一点

  CUDA_CHECK(cudaStreamWaitEvent(sB, dep, 0));  // sB 阻塞直到 dep 完成
  scale_add<<<grid, block, 0, sB>>>(x, n, 10.0f, 0.0f);  // x = 1*10 = 10

  CUDA_CHECK(cudaStreamSynchronize(sB));

  float host = 0;
  CUDA_CHECK(cudaMemcpy(&host, x, sizeof(float), cudaMemcpyDeviceToHost));
  printf("[sync] x[0] = %.1f (期望 10.0: 证明 sB 在 sA 之后才运行)\n", host);
  if (host != 10.0f) { printf("FAIL: 顺序被打乱了\n"); return 1; }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(dep));
  CUDA_CHECK(cudaStreamDestroy(sA));
  CUDA_CHECK(cudaStreamDestroy(sB));
  CUDA_CHECK(cudaFree(x));

  printf("\nOK. 核心要点:\n");
  printf("  * event = stream 中的标记点; 两大用途: 计时 和 同步\n");
  printf("  * elapsed_time 需要开启计时 (PyTorch: enable_timing=True)\n");
  printf("  * 在 stream A 里 record() + 在 stream B 里 wait_event() = 跨 stream 顺序保证\n");
  return 0;
}
