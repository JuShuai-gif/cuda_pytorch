// 05_graph_mempool.cu
//
// CUDA graph 的头号正确性规则(参见 cuda_stream.md 坑点 #2):
// 被捕获的 graph 固化了"固定的设备地址"。重放时它读写的正是那些地址。所以只有
// 当你通过"稳定的输入/输出 buffer"喂数据、并原地更新内容时，graph 才有用。
//
// 本例展示 PyTorch 在 make_graphed_callables / CUDA-graph 训练步 中用到的
// 经典 "静态 IO" 模式:
//
//   static_in  = cudaMalloc(...)   // 作为 graph 的输入地址被捕获
//   static_out = cudaMalloc(...)   // 作为 graph 的输出地址被捕获
//   捕获:  static_out = f(static_in)
//   每一步:
//       把新数据拷入 static_in      (原地，地址不变)
//       graph.replay()             (读 static_in, 写 static_out)
//       把 static_out 拷出到结果    (地址不变)
//
// 我们用 5 个不同的输入跑 5 步，逐一校验输出，证明即使数据在变，graph 也能
// 被正确复用。
//
// (PyTorch 还额外把捕获期间的分配路由到一个"私有 mempool"，从而保证地址在
// 多次 replay 之间稳定 -- CUDAGraph.cpp:143 MemPool::graph_pool_handle()。
// 这里我们通过"从不释放 static_in/static_out"来天然获得稳定性，本质是同一
// 种保证。)
//
// 编译:  make 05_graph_mempool && ./05_graph_mempool

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

// out = in * 2 + 1, 代表一个固定的计算步骤。
__global__ void affine(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

int main() {
  const int n = 1 << 16;
  const int block = 256, grid = (n + block - 1) / block;

  // 稳定 buffer: 只分配一次，直到最后才释放。它们的地址就是 graph 将要捕获的地址。
  float *static_in, *static_out;
  CUDA_CHECK(cudaMalloc(&static_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&static_out, n * sizeof(float)));

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // 先预热，再针对稳定 buffer 捕获这个单一计算步骤。
  affine<<<grid, block, 0, s>>>(static_in, static_out, n);
  CUDA_CHECK(cudaStreamSynchronize(s));

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  affine<<<grid, block, 0, s>>>(static_in, static_out, n);  // 地址被固化进 graph
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph,
                                           cudaGraphInstantiateFlagAutoFreeOnLaunch));

  printf("static_in  @ %p\n", (void*)static_in);
  printf("static_out @ %p  (在 graph 整个生命周期内都固定不变)\n\n",
         (void*)static_out);

  bool all_ok = true;
  for (int step = 0; step < 5; ++step) {
    float val = float(step + 1);  // 第 k 步喂入全 (k+1)

    // 原地更新输入 buffer(地址不变)，写入新数据。
    // 设备端的 memset 无法写入任意浮点值，所以借一个临时 host buffer 拷入
    // static_in。static_in 的地址从头到尾不变。
    float host_in = val;
    // 用一个填好的 host 数组把同一个值广播到整个 buffer，简单廉价。
    // ponytail: 在 host 上用循环填充，buffer 很小; 用 fill kernel 也行但对
    // 本课没有额外收益。
    {
      float* tmp = (float*)malloc(n * sizeof(float));
      for (int i = 0; i < n; ++i) tmp[i] = host_in;
      CUDA_CHECK(cudaMemcpyAsync(static_in, tmp, n * sizeof(float),
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaStreamSynchronize(s));  // 确保拷贝完成后再 free(tmp)
      free(tmp);
    }

    // 重放: 读 static_in, 写 static_out。从 CPU 视角看没有重新启动 kernel
    // —— 只有一次 graph launch。
    CUDA_CHECK(cudaGraphLaunch(graph_exec, s));

    // 从稳定的输出地址读回结果。
    float host_out = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_out, static_out, sizeof(float),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    float expected = val * 2.0f + 1.0f;
    bool ok = (host_out == expected);
    all_ok &= ok;
    printf("step %d: in=%.1f -> out=%.1f (期望 %.1f) %s\n",
           step, val, host_out, expected, ok ? "OK" : "不匹配");
  }

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(static_in));
  CUDA_CHECK(cudaFree(static_out));

  printf("\n%s\n", all_ok ? "所有步骤都正确。" : "FAIL: 有步骤不匹配。");
  printf("核心要点:\n");
  printf("  * graph 捕获的是固定的「地址」，不是「值」\n");
  printf("  * 复用方式: 原地把新数据拷入同一个输入 buffer\n");
  printf("  * 绝不要在两次 replay 之间释放+重新分配被捕获的 buffer(非法访问)\n");
  return all_ok ? 0 : 1;
}
