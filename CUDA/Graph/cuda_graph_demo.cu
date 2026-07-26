#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>

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

constexpr int N = 500000;
constexpr int NUM_KERNELS = 20;
constexpr int NUM_STEPS = 1000;
constexpr int THREADS = 256;

/*
 * cudaStreamSynchronize(stream): CPU 阻塞等待，直到指定流中之前提交的
 * 所有 GPU 操作全部执行完成。本文件中它的作用：
 *
 *   - 暴露异步运行时错误（非法内存访问等）
 *   - 保证数据在计时/拷贝/清零之前已经就绪
 *   - 让每步的边界清晰，便于公平计时
 *   - 不能出现在 cudaStreamBeginCapture/EndCapture 之间
 *
 * 按步同步会牺牲一定吞吐，换来确定性的时序和数据安全。
 * 如果各步之间相互独立，可以改成「全部提交后统一等一次」来最大化吞吐。
 */

// 一个计算量很小的 kernel。CUDA Graph 对这类短小、重复执行的 kernel 更容易产生明显收益。
__global__ void shortKernel(float* data, int n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        data[index] = data[index] * 1.000001f + 0.000001f;
    }
}

struct GraphTiming {
    double buildMs;
    double executionMs;
};

// 普通执行方式：
// 每个 timestep 提交 NUM_KERNELS 个 kernel，然后同步一次。
double runNormally(
    float* deviceData,
    cudaStream_t stream,
    int blocks)
{
    const auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < NUM_STEPS; ++step) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            shortKernel<<<blocks, THREADS, 0, stream>>>(
                deviceData,
                N);
        }

        // 每步同步一次即可：同一流内操作天然有序，无需每个 kernel 单独同步。
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const auto end = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaGetLastError());

    return std::chrono::duration<double, std::milli>(
               end - start)
        .count();
}

// CUDA Graph 执行方式。
GraphTiming runWithGraph(
    float* deviceData,
    cudaStream_t stream,
    int blocks)
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    /*
     * 第一阶段：Capture
     *
     * cudaStreamBeginCapture 启动 Graph 捕获阶段。
     * 进入捕获后，提交到该流的所有 GPU 操作（kernel、memcpy 等）
     * 不会立即执行，而是被记录为 cudaGraph_t 中的节点，
     * 最后由 cudaStreamEndCapture 打包并生成 Graph。
     *
     * cudaStreamCaptureModeGlobal:
     *   全局捕获模式（默认、最安全）。
     *   同一时刻整个 CUDA 上下文中只允许一个流处于捕获状态，
     *   其他流再调用 BeginCapture 会返回错误。
     *
     *   其他可用模式：
     *   - ThreadLocal: 已废弃（CUDA 12.3+），仅当前线程范围隔离
     *   - Relaxed:    允许捕获期间跨流的 cudaEventWait/cudaEventRecord
     */
    const auto buildStart = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaStreamBeginCapture(
        stream,
        cudaStreamCaptureModeGlobal));

    for (int k = 0; k < NUM_KERNELS; ++k) {
        shortKernel<<<blocks, THREADS, 0, stream>>>(
            deviceData,
            N);
    }

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    /*
     * 第二阶段：Instantiate
     *
     * 把 cudaGraph_t 模板转换成可重复执行的
     * cudaGraphExec_t。
     */
    CUDA_CHECK(cudaGraphInstantiate(
        &graphExec,
        graph,
        nullptr,
        nullptr,
        0));

    const auto buildEnd = std::chrono::steady_clock::now();

    const double buildMs =
        std::chrono::duration<double, std::milli>(
            buildEnd - buildStart)
            .count();

    /*
     * 第一次 Graph launch 通常包含额外初始化成本，
     * 因此先进行一次预热。
     *
     * cudaGraphLaunch 是异步的：CPU 提交 Graph 后立刻返回，
     * 不会等待 GPU 真正执行完。这里必须同步的原因是：
     *   下一步要 cudaMemsetAsync 清零数组，如果不等待预热完成，
     *   memset 可能覆盖还在被 Graph 读取的数据，导致数据竞争。
     */
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 预热修改了数组，重置为零。
    CUDA_CHECK(cudaMemsetAsync(
        deviceData,
        0,
        N * sizeof(float),
        stream));

    // 同步：确保清零完成后再开始计时的 Graph 执行。
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     * 第三阶段：重复执行。
     *
     * 每个 timestep 不再提交 20 个 kernel，
     * 而只提交一次 cudaGraphLaunch。
     *
     * cudaGraphLaunch 是异步的，每次 launch 后必须同步：
     *   - 如果不 sync，for 循环会瞬间将所有 Graph 推入流中，
     *     CPU 计时器几乎立刻结束，测出的是提交速度而非 GPU 实际耗时；
     *   - 每步 sync 让 CPU 阻塞到该步 GPU 工作完成，
     *     保证计时精确反映 GPU 执行时间。
     */
    const auto executionStart =
        std::chrono::steady_clock::now();

    for (int step = 0; step < NUM_STEPS; ++step) {
        // 一次 GraphLaunch 替代 20 次独立 kernel 启动。
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const auto executionEnd =
        std::chrono::steady_clock::now();

    const double executionMs =
        std::chrono::duration<double, std::milli>(
            executionEnd - executionStart)
            .count();

    /*
     * graphExec 是实例化后的独立可执行对象。
     * 两个对象都需要销毁。
     */
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    return {buildMs, executionMs};
}

int main()
{
    try {
        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp deviceProperties{};
        CUDA_CHECK(cudaGetDeviceProperties(
            &deviceProperties,
            0));

        std::cout << "GPU: "
                  << deviceProperties.name
                  << '\n';

        cudaStream_t stream = nullptr;

        /*
         * Stream Capture 不能使用 legacy NULL stream，
         * 因此显式创建一个非阻塞 stream。
         */
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking));

        float* deviceData = nullptr;

        CUDA_CHECK(cudaMalloc(
            &deviceData,
            N * sizeof(float)));

        const int blocks =
            (N + THREADS - 1) / THREADS;

        // 整体预热，避免首次 CUDA 调用影响测试结果。
        CUDA_CHECK(cudaMemsetAsync(
            deviceData,
            0,
            N * sizeof(float),
            stream));

        shortKernel<<<blocks, THREADS, 0, stream>>>(
            deviceData,
            N);

        // 同步：等待预热完成并捕获可能的运行时错误。
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        // ---------------- 普通方式 ----------------

        CUDA_CHECK(cudaMemsetAsync(
            deviceData,
            0,
            N * sizeof(float),
            stream));

        // 同步：确保数组清零后再开始计时的普通执行。
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const double normalMs =
            runNormally(deviceData, stream, blocks);

        float normalResult = 0.0f;

        CUDA_CHECK(cudaMemcpy(
            &normalResult,
            deviceData,
            sizeof(float),
            cudaMemcpyDeviceToHost));

        // ---------------- Graph 方式 ----------------

        CUDA_CHECK(cudaMemsetAsync(
            deviceData,
            0,
            N * sizeof(float),
            stream));

        // 同步：确保数组清零后再开始计时的 Graph 执行。
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const GraphTiming graphTiming =
            runWithGraph(deviceData, stream, blocks);

        float graphResult = 0.0f;

        CUDA_CHECK(cudaMemcpy(
            &graphResult,
            deviceData,
            sizeof(float),
            cudaMemcpyDeviceToHost));

        // ---------------- 输出结果 ----------------

        const int totalKernelExecutions =
            NUM_STEPS * NUM_KERNELS;

        std::cout << "\nConfiguration\n";
        std::cout << "  Elements:        " << N << '\n';
        std::cout << "  Steps:           " << NUM_STEPS << '\n';
        std::cout << "  Kernels/step:    " << NUM_KERNELS << '\n';
        std::cout << "  Total kernels:   "
                  << totalKernelExecutions << '\n';

        std::cout << "\nTiming\n";
        std::cout << "  Normal execution: "
                  << normalMs << " ms\n";

        std::cout << "  Graph build:      "
                  << graphTiming.buildMs << " ms\n";

        std::cout << "  Graph execution:  "
                  << graphTiming.executionMs << " ms\n";

        std::cout << "  Normal/kernel:    "
                  << normalMs * 1000.0 /
                         totalKernelExecutions
                  << " us\n";

        std::cout << "  Graph/kernel:     "
                  << graphTiming.executionMs * 1000.0 /
                         totalKernelExecutions
                  << " us\n";

        std::cout << "  Speedup:          "
                  << normalMs /
                         graphTiming.executionMs
                  << "x\n";

        std::cout << "\nValidation\n";
        std::cout << "  Normal result:    "
                  << normalResult << '\n';

        std::cout << "  Graph result:     "
                  << graphResult << '\n';

        std::cout << "  Difference:       "
                  << std::fabs(
                         normalResult - graphResult)
                  << '\n';

        CUDA_CHECK(cudaFree(deviceData));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
