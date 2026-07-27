/*
 * cuda_stream_demo.cu — CUDA Stream 完整示例
 *
 * 编译:
 *   nvcc -O3 -std=c++17 cuda_stream_demo.cu -o cuda_stream_demo
 *
 * 运行:
 *   ./cuda_stream_demo
 *
 * nsys 性能分析 (在时间线中观察不同 stream 的重叠情况):
 *   nsys profile --stats=true -o /tmp/stream_report ./cuda_stream_demo
 *   nsys-ui /tmp/stream_report.nsys-rep
 *
 * 仅抓 GPU trace:
 *   nsys profile --trace=cuda -o /tmp/stream_gpu ./cuda_stream_demo
 *
 * 导出 kernel 统计:
 *   nsys stats --report cuda_gpu_kern_sum /tmp/stream_report.nsys-rep
 *
 * 查看编译资源 (寄存器数等):
 *   nvcc -O3 -std=c++17 -Xptxas=-v cuda_stream_demo.cu -o cuda_stream_demo
 *
 * nsys 时间线中要关注:
 *   - 场景1 (小 kernel):  默认流串行 vs 多流重叠 (2x+ 加速)
 *   - 场景2 (大 kernel):  默认流串行 vs 多流仍串行 (GPU 饱和, 无加速)
 *   - 场景3 (拷贝+计算):  H2D|kernel|D2H 在不同 stream 上流水线重叠
 *   - 场景4 (event 同步): 流间部分依赖, 无需全局同步
 *
 * 核心结论:
 *   1. Stream 不增加 GPU 硬件, 只是让已有的 copy engine 和 SM 并行干活
 *   2. 大 kernel 占满 SM 时, 多流无加速 — 没有多余硬件可供重叠
 *   3. 最大收益场景: 数据传输与计算重叠 (H2D/D2H 走 copy engine, kernel 走 SM)
 *   4. PyTorch DataLoader(pin_memory=True) + non_blocking=True 自动实现拷贝-计算流水线
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            throw std::runtime_error(                                      \
                std::string("CUDA error at ") + __FILE__ + ":" +           \
                std::to_string(__LINE__) + ": " +                          \
                cudaGetErrorString(error));                                \
        }                                                                  \
    } while (0)

// ============================================================================
// 两种 kernel: "light" (短, 不饱和 GPU, 多流可重叠) 和 "heavy"
// (长, 饱和 GPU, 多流无法重叠 — 教学要点)
// ============================================================================
constexpr int LIGHT_N = 1 << 17;     // 128K floats ≈ 0.5 MB, 小 kernel
constexpr int HEAVY_N = 1 << 24;     // 16M floats  ≈ 64 MB, 大 kernel, 占满 GPU
constexpr int THREADS = 256;
constexpr int NUM_STREAMS = 4;
constexpr int LIGHT_REPEAT = 200;
constexpr int HEAVY_REPEAT = 50;

// 轻量 kernel: 一次乘加, 极短 — 多流可实现重叠
__global__ void light_kernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * 1.0001f + 0.0001f;
    }
}

// 重量 kernel: 每元素做多次 sin/cos/sqrt, 计算密集 — 占满所有 SM
__global__ void heavy_kernel(float* a, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < iters; ++k) {
            x = sinf(x) * cosf(x) + sqrtf(fabsf(x) + 1.0f);
        }
        a[i] = x;
    }
}

// 初始化 CPU 端数组
void init_host(float* h, int n) {
    for (int i = 0; i < n; ++i)
        h[i] = static_cast<float>(i % 1000) * 0.001f;
}

// ============================================================================
// 场景 1a: 小 kernel — 默认流 (所有 chunk 串行, 无重叠)
// ============================================================================
double run_light_default(float* d) {
    const int blocks = (LIGHT_N / NUM_STREAMS + THREADS - 1) / THREADS;
    const int chunk_n = LIGHT_N / NUM_STREAMS;

    cudaEvent_t st, sp;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // 默认流: 第4个参数 = 0 (NULL stream), 所有操作按提交顺序串行
    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < LIGHT_REPEAT; ++r) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            light_kernel<<<blocks, THREADS, 0, 0>>>(d + s * chunk_n, chunk_n);
        }
    }
    CUDA_CHECK(cudaEventRecord(sp));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    return ms;
}

// ============================================================================
// 场景 1b: 小 kernel — 多流 (不同 chunk 在不同 stream 上并发, 可重叠)
//   cudaStreamNonBlocking: 关键! 避免默认流隐式同步所有 blocking stream
// ============================================================================
double run_light_multi(float* d) {
    const int blocks = (LIGHT_N / NUM_STREAMS + THREADS - 1) / THREADS;
    const int chunk_n = LIGHT_N / NUM_STREAMS;

    cudaStream_t ss[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamCreateWithFlags(&ss[s], cudaStreamNonBlocking));

    cudaEvent_t st, sp;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // 启动 event 记录在默认流, 确保多流工作在此之前已提交
    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < LIGHT_REPEAT; ++r) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            // 第4个参数 = ss[s]: 每个 chunk 提交到独立 stream
            light_kernel<<<blocks, THREADS, 0, ss[s]>>>(d + s * chunk_n, chunk_n);
        }
    }
    // 在每个 stream 上记录结束 event, 最后一个完成的 stream 触发 cudaEventSynchronize
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaEventRecord(sp, ss[s]));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamDestroy(ss[s]));
    return ms;
}

// ============================================================================
// 场景 2a: 大 kernel — 默认流
//   kernel 太大, 单个就占满所有 SM, 即使用默认流也无所谓
// ============================================================================
double run_heavy_default(float* d) {
    const int blocks = (HEAVY_N + THREADS - 1) / THREADS;

    cudaEvent_t st, sp;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // 每个 kernel 处理全部 64MB, 饱和所有 76 个 SM
    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < HEAVY_REPEAT; ++r) {
        heavy_kernel<<<blocks, THREADS, 0, 0>>>(d, HEAVY_N, 128);
    }
    CUDA_CHECK(cudaEventRecord(sp));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    return ms;
}

// ============================================================================
// 场景 2b: 大 kernel — 多流
//   切成 4 份分别提交到 4 个 stream, 但每份仍有 16MB, 依然饱和 GPU
//   所以与默认流耗时几乎相同 — 这是正常的, 也是重要的教学点
// ============================================================================
double run_heavy_multi(float* d) {
    const int chunk_n = HEAVY_N / NUM_STREAMS;
    const int blocks = (chunk_n + THREADS - 1) / THREADS;

    cudaStream_t ss[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamCreateWithFlags(&ss[s], cudaStreamNonBlocking));

    cudaEvent_t st, sp;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < HEAVY_REPEAT; ++r) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            heavy_kernel<<<blocks, THREADS, 0, ss[s]>>>(d + s * chunk_n, chunk_n, 128);
        }
    }
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaEventRecord(sp, ss[s]));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamDestroy(ss[s]));
    return ms;
}

// ============================================================================
// 场景 3: 异步拷贝 + 计算重叠 — Stream 最核心的使用场景
//   原理: H2D/D2H 走 copy engine (独立于 SM), kernel 走 SM
//   不同 stream 上的操作可以同时进行:
//     stream 0: H2D → kernel → D2H
//     stream 1:          H2D → kernel → D2H
//     stream 2:                   H2D → kernel → D2H
//   形成流水线, copy engine 和 SM 同时干活
//
//   前提: 必须使用 pinned (page-locked) 内存, pageable 内存无法异步
// ============================================================================
double run_copy_compute_overlap(int n, int repeat) {
    const int chunk_n = n / NUM_STREAMS;
    const int blocks = (chunk_n + THREADS - 1) / THREADS;

    // pinned memory: 物理内存锁定不换页, GPU DMA 引擎可直接访问
    float* h_src[NUM_STREAMS];
    float* h_dst[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaMallocHost(&h_src[s], chunk_n * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_dst[s], chunk_n * sizeof(float)));
        init_host(h_src[s], chunk_n);
    }

    float* d[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaMalloc(&d[s], chunk_n * sizeof(float)));

    cudaStream_t ss[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamCreateWithFlags(&ss[s], cudaStreamNonBlocking));

    cudaEvent_t st, sp;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < repeat; ++r) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            // 每个 stream 内操作严格串行 (H2D → kernel → D2H)
            // 但不同 stream 间 copy engine 和 SM 可并行
            CUDA_CHECK(cudaMemcpyAsync(
                d[s], h_src[s], chunk_n * sizeof(float),
                cudaMemcpyHostToDevice, ss[s]));
            heavy_kernel<<<blocks, THREADS, 0, ss[s]>>>(d[s], chunk_n, 64);
            CUDA_CHECK(cudaMemcpyAsync(
                h_dst[s], d[s], chunk_n * sizeof(float),
                cudaMemcpyDeviceToHost, ss[s]));
        }
    }
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaEventRecord(sp, ss[s]));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));

    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamDestroy(ss[s]));
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaFree(d[s]));
        CUDA_CHECK(cudaFreeHost(h_src[s]));
        CUDA_CHECK(cudaFreeHost(h_dst[s]));
    }
    return ms;
}

// ============================================================================
// 场景 4: 基于 Event 的流间依赖 (细粒度同步)
//   stream 0: compute → cudaEventRecord(ev, s0)
//   stream 1: cudaStreamWaitEvent(s1, ev) → compute
//   只同步两个流的特定点, 比 cudaDeviceSynchronize (全 GPU 阻塞) 精细得多
// ============================================================================
double run_event_dependency(int n, int repeat) {
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, n * sizeof(float)));

    const int blocks = (n + THREADS - 1) / THREADS;

    cudaStream_t s0, s1;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreate(&ev));

    cudaEvent_t st, sp;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    CUDA_CHECK(cudaEventRecord(st));
    for (int r = 0; r < repeat; ++r) {
        // stream 0: 独立计算
        heavy_kernel<<<blocks, THREADS, 0, s0>>>(d_a, n, 64);
        // 在 stream 0 上记录事件: "s0 的 kernel 已完成"
        CUDA_CHECK(cudaEventRecord(ev, s0));

        // stream 1: 等待 stream 0 的事件, 然后使用其结果
        CUDA_CHECK(cudaStreamWaitEvent(s1, ev));
        heavy_kernel<<<blocks, THREADS, 0, s1>>>(d_b, n, 64);
    }
    CUDA_CHECK(cudaEventRecord(sp, s1));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));

    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaEventDestroy(ev));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return ms;
}

// ============================================================================
// PyTorch 中 Stream 的使用参考
// ============================================================================
void print_pytorch_stream_info() {
    std::cout << R"(
========================================
 PyTorch CUDA Stream 用法参考
========================================

--- 基础 API ---
  s = torch.cuda.Stream(device=0)           # 创建流
  default = torch.cuda.default_stream()     # 默认流 (legacy NULL stream)
  current = torch.cuda.current_stream()     # 当前线程绑定的流
  s.query()                                 # 检查流是否完成 (不阻塞)
  s.synchronize()                           # CPU 阻塞等待流完成
  e = s.record_event()                      # 在流上记录事件
  s.wait_event(e)                           # 流等待事件
  s.wait_stream(other)                      # 跨流依赖插入

--- 模式 1: 数据传输与计算重叠 (最常见) ---
  s = torch.cuda.Stream()
  with torch.cuda.stream(s):
      # 在流 s 上做异步 H2D (需要 pin_memory)
      data_gpu = data_cpu.to('cuda', non_blocking=True)
  # 默认流并行做前一批的计算
  output = model(prev_batch)
  # 使用前插入依赖
  torch.cuda.current_stream().wait_stream(s)
  output = model(data_gpu)

  等价内部实现: torch 的 DataLoader 使用 pin_memory=True +
  non_blocking=True 自动完成上述流水线。

--- 模式 2: DDP 通信与反向传播重叠 ---
  PyTorch DDP 内部自动:
  - 反向传播在 default stream
  - all-reduce 在专用 NCCL stream
  - 通过 event 同步, 使通信和计算重叠

--- 模式 3: 多请求并发推理 ---
  s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
  with torch.cuda.stream(s1): out1 = model(inp1)
  with torch.cuda.stream(s2): out2 = model(inp2)
  torch.cuda.synchronize()

--- 注意事项 ---
  1. 默认流 (legacy NULL stream) 会隐式同步所有 blocking 流
  2. 使用 cudaStreamNonBlocking / non_blocking=True 避免隐式同步
  3. 不同流间的 kernel 依赖必须显式通过 event 管理
  4. 多流不能加速已被单个 kernel 饱和的 GPU (见 heavy kernel 测试)
)";
}

// ============================================================================
// main
// ============================================================================
int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

        std::cout << "GPU:               " << props.name << "\n";
        std::cout << "计算能力:          " << props.major << "."
                  << props.minor << "\n";
        std::cout << "并发 Kernel 支持:  "
                  << (props.concurrentKernels ? "是" : "否") << "\n";
        std::cout << "异步引擎数:        " << props.asyncEngineCount << "\n";
        std::cout << "SM 数量:           "
                  << (props.multiProcessorCount) << "\n\n";

        // ------------------------------------------------------------------
        // 分配显存 + 预热
        // ------------------------------------------------------------------
        float* d_light = nullptr;
        float* d_heavy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_light, LIGHT_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_heavy, HEAVY_N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_light, 0, LIGHT_N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_heavy, 0, HEAVY_N * sizeof(float)));

        // 预热: 避免首次 CUDA 调用影响计时
        light_kernel<<<1, 1>>>(d_light, 1);
        heavy_kernel<<<1, 1>>>(d_heavy, 1, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ==================================================================
        // 1. 小 kernel: 多流可重叠 → 显著加速
        // ==================================================================
        std::cout << "========================================\n";
        std::cout << " 1. 小 KERNEL ("
                  << (LIGHT_N * sizeof(float) / 1024.0f) << " KB, 重复 "
                  << LIGHT_REPEAT << " 次)\n";
        std::cout << "    单个 kernel 极短, GPU 有空闲 SM\n";
        std::cout << "    多流让调度器交错执行 → 可实现重叠\n";
        std::cout << "========================================\n";

        double t = run_light_default(d_light);
        std::cout << "  默认流:           " << t << " ms\n";

        t = run_light_multi(d_light);
        std::cout << "  多流 (" << NUM_STREAMS << " 个):     " << t << " ms\n";
        std::cout << "  加速比:           "
                  << run_light_default(d_light) / t << "x\n";
        std::cout << "  >>> nsys 时间线中可见 4 个 stream 的 kernel 条重叠\n\n";

        // ==================================================================
        // 2. 大 kernel: GPU 饱和 → 多流无加速 (教学要点)
        // ==================================================================
        std::cout << "========================================\n";
        std::cout << " 2. 大 KERNEL ("
                  << (HEAVY_N * sizeof(float) / (1024.0f * 1024.0f)) << " MB, 重复 "
                  << HEAVY_REPEAT << " 次)\n";
        std::cout << "    单个 kernel 占满所有 SM\n";
        std::cout << "    多流无法重叠 — GPU 没有空闲硬件资源\n";
        std::cout << "========================================\n";

        t = run_heavy_default(d_heavy);
        std::cout << "  默认流:           " << t << " ms\n";

        t = run_heavy_multi(d_heavy);
        std::cout << "  多流 (" << NUM_STREAMS << " 个):     " << t << " ms\n";
        std::cout << "  加速比:           ~1x (无加速, 这是正常的)\n";
        std::cout << "  >>> Stream 不会凭空增加 SM, 大 kernel 无重叠空间\n\n";

        // ==================================================================
        // 3. 拷贝+计算重叠: Stream 最主要的使用场景
        // ==================================================================
        std::cout << "========================================\n";
        std::cout << " 3. 拷贝 + 计算重叠\n";
        std::cout << "    H2D 走 copy engine, kernel 走 SM\n";
        std::cout << "    不同硬件单元可在不同 stream 上同时工作\n";
        std::cout << "    要求: pinned (page-locked) 内存\n";
        std::cout << "========================================\n";

        constexpr int COPY_N = 1 << 22;  // 4M floats = 16 MB
        t = run_copy_compute_overlap(COPY_N, 20);
        std::cout << "  耗时:             " << t << " ms\n";
        std::cout << "  >>> nsys 时间线中可见 H2D/kernel/D2H 在不同 stream 上交错重叠\n\n";

        // ==================================================================
        // 4. Event 依赖: 流间细粒度同步
        // ==================================================================
        std::cout << "========================================\n";
        std::cout << " 4. 基于 EVENT 的流间依赖\n";
        std::cout << "    stream 0 计算 → record event → stream 1 wait → 计算\n";
        std::cout << "    仅同步两个流的特定点, 不阻塞整个 GPU\n";
        std::cout << "========================================\n";

        t = run_event_dependency(1 << 20, 10);
        std::cout << "  耗时:             " << t << " ms\n";
        std::cout << "  >>> nsys 中 stream 0 的 kernel 结束后, stream 1 的 kernel 才开始\n\n";

        // ==================================================================
        // 5. PyTorch stream 用法
        // ==================================================================
        print_pytorch_stream_info();

        // ==================================================================
        // 总结
        // ==================================================================
        std::cout << "\n========================================\n";
        std::cout << " 总结: 什么时候用 CUDA Stream?\n";
        std::cout << "========================================\n\n";
        std::cout << "1. 数据传输与计算重叠 (最核心场景)\n";
        std::cout << "   H2D/D2H 走 copy engine, kernel 走 SM — 天然可并行\n";
        std::cout << "   需要: pinned memory + async memcpy + 非默认流\n\n";
        std::cout << "2. 大量小 kernel 的并发提交\n";
        std::cout << "   小 kernel 不占满 GPU, 多流让调度器填充空闲 SM\n\n";
        std::cout << "3. 多 GPU 通信重叠 (NCCL/DDP)\n";
        std::cout << "   计算流与通信流并行\n\n";
        std::cout << "4. 多请求并发推理\n";
        std::cout << "   每个请求用独立 stream\n\n";
        std::cout << "5. 跨流精细依赖 (event)\n";
        std::cout << "   cudaEventRecord + cudaStreamWaitEvent\n";
        std::cout << "   只同步必要的流, 不阻塞整个 GPU\n\n";
        std::cout << "!! 大 kernel 已占满 GPU → 多流无加速 (没有更多 SM 可用)\n";

        CUDA_CHECK(cudaFree(d_light));
        CUDA_CHECK(cudaFree(d_heavy));

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
