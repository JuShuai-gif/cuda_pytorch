/*
 * 编译命令:
 *   nvcc -arch=sm_70 -O2 -o bench_launch_event_graph_stream \
 *        bench_launch_event_graph_stream.cu
 *
 * 测试内容:
 *   1. kernel launch 开销
 *   2. CUDA Event 的 host-timer vs device-timer 差异
 *   3. CUDA Stream 的串行 vs 并发差异
 *   4. CUDA Graph 捕获回放 vs 手动 launch 效率对比
 *   5. CUDA Event / Stream / Graph 三者间交互关系
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 主机端微秒级计时器
static double host_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// 空核函数 —— 什么都不做，仅用来测量 launch / prologue / epilogue 耗时
__global__ void empty_kernel() {}

__global__ void small_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// CUDA 错误检查宏
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ===========================================================================
// 测试 1: Kernel Launch 开销 —— 空核函数反复发射，测量平均每次耗时
// ===========================================================================
void test_launch_overhead() {
    printf("=== 测试 1: Kernel Launch 开销 ===\n");

    const int ITER = 100000;
    const int WARMUP = 1000;

    // 预热，让 GPU 进入稳态
    for (int i = 0; i < WARMUP; i++) {
        empty_kernel<<<1, 1>>>();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 大量空核函数发射，取平均耗时
    double t0 = host_us();
    for (int i = 0; i < ITER; i++) {
        empty_kernel<<<1, 1>>>();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = host_us();

    double avg_us = (t1 - t0) / ITER;
    printf("  empty_kernel<<<1,1>>> : %.2f us/launch  (%d 次平均)\n", avg_us, ITER);

    // 更大的 grid 维度，看看 launch 开销是否增加
    for (int i = 0; i < WARMUP; i++) {
        empty_kernel<<<1024, 256>>>();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    t0 = host_us();
    for (int i = 0; i < ITER / 10; i++) {
        empty_kernel<<<1024, 256>>>();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = host_us();

    avg_us = (t1 - t0) / (ITER / 10);
    printf("  empty_kernel<<<1024,256>>> : %.2f us/launch\n\n", avg_us);
}

// ===========================================================================
// 测试 2: CUDA Event —— host 计时 vs device 计时，差值即为 launch 开销
// ===========================================================================
void test_cuda_events() {
    printf("=== 测试 2: CUDA Event —— Host 计时 vs Device 计时 ===\n");

    const int N = 4 * 1024 * 1024;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 主机端计时 —— 包含 launch 开销 + 同步等待
    double t_host0 = host_us();
    small_add<<<GRID, BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_host1 = host_us();
    printf("  Host 计时 (含 launch):  %.2f us\n", t_host1 - t_host0);

    // 设备端计时 —— 纯 GPU 执行时间，通过 cudaEvent 测量
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    small_add<<<GRID, BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    printf("  Device 计时 (纯 GPU):  %.2f us\n", gpu_ms * 1000.0f);
    printf("  Launch 开销 ~ %.2f us\n\n",
           (t_host1 - t_host0) - gpu_ms * 1000.0f);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// ===========================================================================
// 测试 3: CUDA Stream 并发 —— default stream 串行 vs 多 stream 并发
// ===========================================================================
void test_cuda_streams() {
    printf("=== 测试 3: CUDA Stream 并发 ===\n");

    const int N = 4 * 1024 * 1024;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const int N_STREAMS = 4;

    cudaStream_t streams[N_STREAMS];
    float *d_a[N_STREAMS], *d_b[N_STREAMS], *d_c[N_STREAMS];

    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_a[s], N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[s], N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c[s], N * sizeof(float)));
    }

    // 串行执行 —— 全部发射到 default stream，严格按顺序执行
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = host_us();
    for (int s = 0; s < N_STREAMS; s++) {
        small_add<<<GRID, BLOCK>>>(d_a[s], d_b[s], d_c[s], N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = host_us();
    printf("  串行 (%d 个 kernel, default stream): %.2f us\n", N_STREAMS, t1 - t0);

    // 并发执行 —— 每个 kernel 发射到不同的 stream，GPU 尽力并行
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = host_us();
    for (int s = 0; s < N_STREAMS; s++) {
        small_add<<<GRID, BLOCK, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s], N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = host_us();
    printf("  并发 (%d 个 kernel, %d 个 stream): %.2f us\n\n",
           N_STREAMS, N_STREAMS, t1 - t0);

    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(d_a[s]));
        CUDA_CHECK(cudaFree(d_b[s]));
        CUDA_CHECK(cudaFree(d_c[s]));
    }
}

// ===========================================================================
// 测试 4: CUDA Graph —— 手动发射 N 个 kernel vs 捕获成 Graph 后回放
// ===========================================================================
void test_cuda_graph() {
    printf("=== 测试 4: CUDA Graph —— 手动 Launch vs 图回放 ===\n");

    const int N = 4 * 1024 * 1024;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const int N_LAUNCHES = 8;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 每次迭代发射 N_LAUNCHES 个 kernel，重复 REPEAT 次
    const int REPEAT = 1000;
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 预热
    for (int r = 0; r < 100; r++) {
        for (int i = 0; i < N_LAUNCHES; i++) {
            small_add<<<GRID, BLOCK>>>(d_a, d_b, d_c, N);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 手动发射计时
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int r = 0; r < REPEAT; r++) {
        for (int i = 0; i < N_LAUNCHES; i++) {
            small_add<<<GRID, BLOCK>>>(d_a, d_b, d_c, N);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float manual_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&manual_ms, ev_start, ev_stop));
    printf("  手动 (%d launch x %d):      %.2f us/iter\n",
           N_LAUNCHES, REPEAT, manual_ms * 1000.0f / REPEAT);

    // 捕获为 CUDA Graph —— 将 N_LAUNCHES 个 kernel 录制成一个图
    cudaStream_t cap_stream;
    CUDA_CHECK(cudaStreamCreate(&cap_stream));

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < N_LAUNCHES; i++) {
        small_add<<<GRID, BLOCK, 0, cap_stream>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaStreamEndCapture(cap_stream, &graph));

    // 实例化 —— 将图编译为可执行形式
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    // 预热回放
    for (int r = 0; r < 100; r++) {
        CUDA_CHECK(cudaGraphLaunch(instance, cap_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(cap_stream));

    // 图回发计时
    CUDA_CHECK(cudaEventRecord(ev_start, cap_stream));
    for (int r = 0; r < REPEAT; r++) {
        CUDA_CHECK(cudaGraphLaunch(instance, cap_stream));
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, cap_stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float graph_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graph_ms, ev_start, ev_stop));
    printf("  图回放(%d nodes x %d):      %.2f us/iter\n",
           N_LAUNCHES, REPEAT, graph_ms * 1000.0f / REPEAT);
    printf("  Graph 加速比: %.2fx\n\n", manual_ms / graph_ms);

    CUDA_CHECK(cudaGraphExecDestroy(instance));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(cap_stream));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// ===========================================================================
// 测试 5: Event / Stream / Graph 交互关系
// ===========================================================================
void test_event_stream_interaction() {
    printf("=== 测试 5: Event / Stream / Graph 交互 ===\n");

    cudaStream_t s1, s2, s3;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));
    CUDA_CHECK(cudaStreamCreate(&s3));

    cudaEvent_t e1, e2;
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));

    const int N = 1 << 24;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    float *d_a, *d_b, *d_c, *d_d, *d_e, *d_f, *d_g, *d_h, *d_i;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_e, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_i, N * sizeof(float)));

    // Event 链式依赖: s1 做完 -> 通知 event e1 -> s2 等 e1 后执操作 -> 通知 e2 -> s3 等 e2 后执操作
    // s1: A -> event e1 -> s2 等待 e1 后执行
    // s2: B -> event e2 -> s3 等待 e2 后执行
    // s3: C

    // 预热
    small_add<<<GRID, BLOCK, 0, s1>>>(d_a, d_b, d_c, N);
    small_add<<<GRID, BLOCK, 0, s2>>>(d_d, d_e, d_f, N);
    small_add<<<GRID, BLOCK, 0, s3>>>(d_g, d_h, d_i, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_all_start, ev_all_stop;
    CUDA_CHECK(cudaEventCreate(&ev_all_start));
    CUDA_CHECK(cudaEventCreate(&ev_all_stop));

    // 链: s1 -> e1 -> s2 -> e2 -> s3
    CUDA_CHECK(cudaEventRecord(ev_all_start, 0));

    small_add<<<GRID, BLOCK, 0, s1>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(e1, s1));          // 在 s1 工作完成后记录 e1
    CUDA_CHECK(cudaStreamWaitEvent(s2, e1, 0));   // s2 等待 e1 完成后再执行
    small_add<<<GRID, BLOCK, 0, s2>>>(d_d, d_e, d_f, N);
    CUDA_CHECK(cudaEventRecord(e2, s2));          // 在 s2 工作完成后记录 e2
    CUDA_CHECK(cudaStreamWaitEvent(s3, e2, 0));   // s3 等待 e2 完成后再执行
    small_add<<<GRID, BLOCK, 0, s3>>>(d_g, d_h, d_i, N);

    CUDA_CHECK(cudaEventRecord(ev_all_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_all_stop));

    float chain_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&chain_ms, ev_all_start, ev_all_stop));
    printf("  Event 链式依赖 (s1->s2->s3): %.2f us\n", chain_ms * 1000.0f);

    // 测试: 在 s1 上捕获的 Graph，在 s2 上回放
    printf("  Graph 跨 stream 执行:\n");

    cudaGraph_t graph_a;
    CUDA_CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    small_add<<<GRID, BLOCK, 0, s1>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaStreamEndCapture(s1, &graph_a));

    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph_a, NULL, NULL, 0));

    // s1 上捕获的图，在 s2 上 launch
    CUDA_CHECK(cudaGraphLaunch(graph_exec, s2));
    CUDA_CHECK(cudaStreamSynchronize(s2));
    printf("    在 s1 捕获的 Graph 在 s2 上回放: OK\n");

    // 测试: Graph 依赖外部 event —— Graph 在 s1 上执行，但 s1 需先等待 s2 上的 event
    printf("  Graph 等待外部 Event:\n");

    CUDA_CHECK(cudaEventRecord(e1, s2));          // 在 s2 上记录 e1
    CUDA_CHECK(cudaStreamWaitEvent(s1, e1, 0));   // s1 等待 e1

    CUDA_CHECK(cudaGraphLaunch(graph_exec, s1));
    CUDA_CHECK(cudaStreamSynchronize(s1));
    printf("    Graph 在 stream-wait-event 后执行: OK\n\n");

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph_a));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaEventDestroy(ev_all_start));
    CUDA_CHECK(cudaEventDestroy(ev_all_stop));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUDA_CHECK(cudaStreamDestroy(s3));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d)); CUDA_CHECK(cudaFree(d_e)); CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_h)); CUDA_CHECK(cudaFree(d_i));
}

// ===========================================================================
int main() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    test_launch_overhead();
    test_cuda_events();
    test_cuda_streams();
    test_cuda_graph();
    test_event_stream_interaction();

    printf("全部测试通过。\n");
    return 0;
}
