// 04_cuda_graph_basics.cu
//
// CUDA Graph = 把一系列 GPU 操作录制一次，之后用一次 launch 重放整个序列。
// 这消除了逐 kernel 的 CPU launch 开销(每个约 5-10us)。当你在循环里启动大量
// 小 kernel 时，这个开销会成为主导。
//
// 这正是 ATen 里 CUDAGraph.cpp 的生命周期:
//   capture_begin()  -> cudaStreamBeginCapture   (必须是非默认 stream)
//   ... 入队各种操作 ...
//   capture_end()    -> cudaStreamEndCapture (-> cudaGraph_t)
//                    -> instantiate() -> cudaGraphInstantiateWithFlags (-> cudaGraphExec_t)
//   replay()         -> cudaGraphLaunch
//   reset()/析构      -> cudaGraphExecDestroy + cudaGraphDestroy
//
// 直接来自源码的两条硬性规则:
//   1. CUDAGraph.cpp:115  "graph 必须在非默认 stream 上捕获。"
//   2. 地址在捕获时就被固化进了 graph。两次 replay 之间你可以原地修改 tensor
//      的"内容"，但绝不能改"指针"。我们用 拷入/replay/拷出 来演示这个
//      地址稳定契约。
//
// 我们对比: 普通逐次启动 N 轮 vs. 从 graph 重放同样的 N 轮，并打印加速比。
//
// 编译:  make 04_cuda_graph_basics && ./04_cuda_graph_basics

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

// 故意写得很小的 kernel: 每次 launch 的实际计算量小，所以 CPU launch 开销
// 占总时间的比重大 -> graph 重放的优势就明显。
__global__ void step(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 1.001f + 0.5f;
}

// 一"轮"迭代 = 一串很多个小 kernel，就像 RNN/transformer 解码一步由几十个
// 算子组成那样。
static void one_iteration(float* x, int n, cudaStream_t s, int chain) {
  int block = 256, grid = (n + block - 1) / block;
  for (int k = 0; k < chain; ++k) step<<<grid, block, 0, s>>>(x, n);
}

int main() {
  const int n = 4096;      // buffer 小: kernel 便宜，launch 开销才是重点
  const int chain = 50;    // 每轮 50 个 kernel
  const int iters = 200;   // 重放 200 轮

  float* x;
  CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));

  // 捕获必须在非默认 stream 上进行 (CUDAGraph.cpp:115)。
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  // ---- 基线: 每一轮都真的把每个 kernel 启动一遍 --------------------------
  CUDA_CHECK(cudaMemset(x, 0, n * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(t0, s));
  for (int it = 0; it < iters; ++it) one_iteration(x, n, s, chain);
  CUDA_CHECK(cudaEventRecord(t1, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  float ms_eager = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_eager, t0, t1));
  printf("[eager]  %d 轮 x %d 个 kernel = %d 次 launch: %.3f ms\n",
         iters, chain, iters * chain, ms_eager);

  // ---- 把"一轮"捕获进 graph -----------------------------------------------
  // 先预热: allocator / 惰性初始化 绝不能发生在捕获期间，这正是 PyTorch 在
  // torch.cuda.graph(...) 之前要做预热运行的原因。
  CUDA_CHECK(cudaMemset(x, 0, n * sizeof(float)));
  one_iteration(x, n, s, chain);
  CUDA_CHECK(cudaStreamSynchronize(s));

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  // capture_begin: cudaStreamCaptureModeGlobal 就是 CUDAGraph.cpp:158 用的模式。
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  one_iteration(x, n, s, chain);            // 只是被录制，并未执行
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));  // capture_end -> cudaGraph_t

  // 检查: graph 里应该有 chain 个节点 (对应 CUDAGraph.cpp:214 的空图警告)。
  size_t num_nodes = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  printf("[graph]  捕获到 %zu 个节点\n", num_nodes);

  // instantiate -> 可执行图。flags 对应 CUDAGraph.cpp:257。
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph,
                                           cudaGraphInstantiateFlagAutoFreeOnLaunch));

  // ---- 重放 graph iters 次: 每轮只有一次 launch ---------------------------
  CUDA_CHECK(cudaMemset(x, 0, n * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(t0, s));
  for (int it = 0; it < iters; ++it) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, s));  // replay()
  }
  CUDA_CHECK(cudaEventRecord(t1, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  float ms_graph = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_graph, t0, t1));
  printf("[graph]  %d 轮用 %d 次 launch 重放: %.3f ms\n",
         iters, iters, ms_graph);

  printf("[speedup] %.2fx (graph 重放 vs 逐次启动)\n", ms_eager / ms_graph);

  // ---- 清理，对应 CUDAGraph::reset() --------------------------------------
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(x));

  printf("\nOK. 核心要点:\n");
  printf("  * 在非默认 stream 上捕获，捕获前要预热\n");
  printf("  * 实例化一次 (cudaGraph_t -> cudaGraphExec_t)，重放很多次\n");
  printf("  * 重放把 N*chain 次 launch 减少为 N 次 -> 降低 CPU 开销\n");
  return 0;
}
