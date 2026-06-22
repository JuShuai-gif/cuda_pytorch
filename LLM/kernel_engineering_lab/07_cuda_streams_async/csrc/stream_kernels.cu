#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <vector>

// ============================================================================
// 多 stream 并发 kernel 执行示例
// 工业背景：Triton Inference Server / vLLM 底层使用 CUDA stream 实现 pipeline overlap
// H2D 拷贝 batch_{N+1} 的同时执行 batch_N 的 kernel
// ============================================================================

// ---------------------------------------------------------------------------
// 1. 向量加法 kernel（每个 thread 处理一个元素）
// ---------------------------------------------------------------------------

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------------------------------------------------------------------
// 2. 向量乘法 kernel（模拟计算密集型操作，每个元素做 pow 运算）
// ---------------------------------------------------------------------------

__global__ void vector_mul_pow_kernel(const float* a, const float* b, float* c, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx] * b[idx];
        // 模拟一定计算量，避免 kernel 过于简单导致 launch overhead 主导
        c[idx] = x * x * x * x;
    }
}

// ---------------------------------------------------------------------------
// 3. 向量乘加融合 kernel（模拟 MLP 中常见的 multiply-add 模式）
// ---------------------------------------------------------------------------

__global__ void vector_fma_kernel(const float* a, const float* b, float* c, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FMA: c = a * b + a  (每线程 2 FLOP)
        c[idx] = fmaf(a[idx], b[idx], a[idx]);
    }
}

// ---------------------------------------------------------------------------
// 4. 多 stream 并发 kernel 执行
//    在 N 个独立 stream 上并发执行 vector_add_kernel
//    返回每个 stream 的 wall-clock 时间（通过 event 精确计时）
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> multi_stream_concurrent_exec(
    const std::vector<torch::Tensor>& inputs_a,
    const std::vector<torch::Tensor>& inputs_b,
    const std::vector<torch::Tensor>& outputs) {

    int num_streams = static_cast<int>(inputs_a.size());
    TORCH_CHECK(num_streams > 0, "至少需要一个输入对");
    TORCH_CHECK(static_cast<int>(inputs_b.size()) == num_streams, "a 和 b 数量必须相等");
    TORCH_CHECK(static_cast<int>(outputs.size()) == num_streams, "输出数量必须匹配");

    // 创建独立的 CUDA stream
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<cudaEvent_t> start_events(num_streams);
    std::vector<cudaEvent_t> end_events(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
    }

    // 在每个 stream 上记录起始 event 并 launch kernel
    const int threads = 256;
    for (int i = 0; i < num_streams; ++i) {
        auto& a = inputs_a[i];
        auto& b = inputs_b[i];
        auto& c = outputs[i];
        TORCH_CHECK(a.sizes() == b.sizes() && b.sizes() == c.sizes(), "张量形状必须一致");
        int64_t n = a.numel();
        int blocks = (static_cast<int>(n) + threads - 1) / threads;

        cudaEventRecord(start_events[i], streams[i]);
        vector_add_kernel<<<blocks, threads, 0, streams[i]>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
        cudaEventRecord(end_events[i], streams[i]);
    }

    // 同步所有 stream 并收集计时
    auto timing_ms = torch::empty({num_streams}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    for (int i = 0; i < num_streams; ++i) {
        cudaEventSynchronize(end_events[i]);
        float ms = 0.0f;
        // 检查 kernel launch 后是否有错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Stream %d kernel error: %s\n", i, cudaGetErrorString(err));
        }
        cudaEventElapsedTime(&ms, start_events[i], end_events[i]);
        timing_ms[i] = ms;
    }

    // 清理资源
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }

    return std::vector<torch::Tensor>{timing_ms};
}

// ---------------------------------------------------------------------------
// 5. pinned memory + async copy + kernel overlap 完整 pipeline
//    这是生产级 inference server 的核心优化模式：
//    将数据分成多个 chunk，在 stream A 上执行 H2D→compute→D2H，
//    同时 stream B 正在处理另一个 chunk
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> pinned_async_pipeline(
    const std::vector<torch::Tensor>& host_chunks,
    int num_streams) {

    TORCH_CHECK(num_streams >= 1, "至少需要一个 stream");
    int num_chunks = static_cast<int>(host_chunks.size());
    TORCH_CHECK(num_chunks > 0, "至少需要一个 chunk");

    // 验证所有 chunk 形状一致且是 pinned memory
    auto chunk_size = host_chunks[0].numel();
    for (int i = 0; i < num_chunks; ++i) {
        TORCH_CHECK(host_chunks[i].numel() == chunk_size, "所有 chunk 大小必须相同");
        TORCH_CHECK(host_chunks[i].is_pinned(), "输入必须是 pinned memory");
    }

    // 为每个 stream 预分配 device buffer（双缓冲）
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<torch::Tensor> dev_buffers(num_streams);
    std::vector<torch::Tensor> dev_outputs(num_streams);
    std::vector<torch::Tensor> host_outputs;

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
        dev_buffers[i] = torch::empty({chunk_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        dev_outputs[i] = torch::empty({chunk_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }
    // CPU 端输出（pinned memory）
    for (int i = 0; i < num_chunks; ++i) {
        host_outputs.push_back(
            torch::empty({chunk_size}, torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true)));
    }

    // 全局计时 event
    cudaEvent_t pipeline_start, pipeline_end;
    cudaEventCreate(&pipeline_start);
    cudaEventCreate(&pipeline_end);
    cudaEventRecord(pipeline_start, 0);

    const int threads = 256;
    // 循环处理所有 chunk，使用轮转方式分配到不同 stream
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int stream_idx = chunk_idx % num_streams;
        cudaStream_t stream = streams[stream_idx];
        auto& dev_buf = dev_buffers[stream_idx];
        auto& dev_out = dev_outputs[stream_idx];
        auto& host_chunk = host_chunks[chunk_idx];
        auto& host_out = host_outputs[chunk_idx];

        // H2D 异步拷贝
        cudaMemcpyAsync(
            dev_buf.data_ptr<float>(),
            host_chunk.data_ptr<float>(),
            chunk_size * sizeof(float),
            cudaMemcpyHostToDevice,
            stream);

        // 计算 kernel（在 device buffer 上执行乘方运算）
        int blocks = (static_cast<int>(chunk_size) + threads - 1) / threads;
        vector_mul_pow_kernel<<<blocks, threads, 0, stream>>>(
            dev_buf.data_ptr<float>(),
            dev_buf.data_ptr<float>(),   // in-place: dev_buf 既是输入也是输出
            dev_out.data_ptr<float>(),
            chunk_size);

        // D2H 异步拷贝
        cudaMemcpyAsync(
            host_out.data_ptr<float>(),
            dev_out.data_ptr<float>(),
            chunk_size * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream);
    }

    // 同步所有 stream
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Pipeline stream %d error: %s\n", i, cudaGetErrorString(err));
        }
    }

    cudaEventRecord(pipeline_end, 0);
    cudaEventSynchronize(pipeline_end);
    float pipeline_ms = 0.0f;
    cudaEventElapsedTime(&pipeline_ms, pipeline_start, pipeline_end);

    // 清理
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(pipeline_start);
    cudaEventDestroy(pipeline_end);

    auto timing = torch::empty({1}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    timing[0] = pipeline_ms;
    host_outputs.push_back(timing);
    return host_outputs;
}

// ---------------------------------------------------------------------------
// 6. 带 event 精确计时的 kernel 性能测量
//    使用 cudaEventRecord 包裹 kernel launch，通过 cudaEventElapsedTime 获取
//    纯 GPU 执行时间（不含 host 端开销）
// ---------------------------------------------------------------------------

torch::Tensor kernel_timing_with_events(
    const torch::Tensor& a,
    const torch::Tensor& b,
    int num_launches) {

    TORCH_CHECK(a.device().is_cuda(), "a 必须在 CUDA 上");
    TORCH_CHECK(b.device().is_cuda(), "b 必须在 CUDA 上");
    TORCH_CHECK(a.sizes() == b.sizes(), "a 和 b 形状必须相同");
    TORCH_CHECK(num_launches > 0, "launch 次数必须大于 0");

    auto c = torch::empty_like(a);
    int64_t n = a.numel();
    const int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;

    // 记录多次 kernel launch 的总 GPU 时间
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // 预热
    for (int i = 0; i < 3; ++i) {
        vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    for (int i = 0; i < num_launches; ++i) {
        vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel timing error: %s\n", cudaGetErrorString(err));
    }

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    auto result = torch::empty({2}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    result[0] = total_ms;                        // 总 GPU 时间
    result[1] = total_ms / num_launches;         // 平均每次 GPU 时间
    return result;
}

// ---------------------------------------------------------------------------
// 7. cudaStreamWaitEvent 跨 stream 同步示例
//    演示如何用 event 实现精确的跨 stream 依赖，避免全局同步
//
//    场景：stream A 计算中间结果，stream B 需要等待 stream A 的中间结果完成
//    错误做法：cudaDeviceSynchronize() 阻塞所有 stream
//    正确做法：cudaStreamWaitEvent(streamB, eventA) 只阻塞 stream B
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> stream_wait_event_demo(
    const torch::Tensor& a,
    const torch::Tensor& b) {

    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(a.sizes() == b.sizes(), "形状必须一致");

    int64_t n = a.numel();
    const int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;

    auto intermediate = torch::empty_like(a);
    auto final_result = torch::empty_like(a);

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    // stream A 上的 event，标记中间结果计算完成
    cudaEvent_t step1_done;
    cudaEventCreate(&step1_done);

    // 第一步：stream A 计算 intermediate = a + b
    vector_add_kernel<<<blocks, threads, 0, stream_a>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), intermediate.data_ptr<float>(), n);
    cudaEventRecord(step1_done, stream_a);

    // 第二步：stream B 等待 stream A 的 intermediate 完成后，计算 final = intermediate * 2
    // 这是正确的跨 stream 依赖方式：只阻塞 stream B，不影响其他 stream
    cudaStreamWaitEvent(stream_b, step1_done, 0);
    // final_result[i] = intermediate[i] * 2.0f（使用简化的 multiply-by-2 kernel）
    vector_mul_pow_kernel<<<blocks, threads, 0, stream_b>>>(
        intermediate.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        final_result.data_ptr<float>(),
        n);

    cudaStreamSynchronize(stream_b);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "StreamWaitEvent error: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    cudaEventDestroy(step1_done);

    return std::vector<torch::Tensor>{intermediate, final_result};
}

// ---------------------------------------------------------------------------
// 8. WAR (write-after-read) 同步正反例对比
//    错误方式：使用 cudaDeviceSynchronize() 阻塞所有 stream
//    正确方式：使用 cudaStreamSynchronize(stream) 只等特定 stream
//
//    演示场景：stream A 写入 out_a，stream B 读取 out_a 做进一步计算
//    通过对比两种同步方式的 wall-clock 时间来展示局部同步的优势
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> war_sync_correct_vs_wrong(
    const torch::Tensor& a,
    const torch::Tensor& b) {

    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(a.sizes() == b.sizes(), "形状必须一致");

    int64_t n = a.numel();
    const int threads = 256;
    int blocks = (static_cast<int>(n) + threads - 1) / threads;

    auto out_a = torch::empty_like(a);
    auto out_b = torch::empty_like(a);

    // ---- 错误方式：cudaDeviceSynchronize ----
    cudaEvent_t wrong_start, wrong_end;
    cudaEventCreate(&wrong_start);
    cudaEventCreate(&wrong_end);

    cudaStream_t stream_w1, stream_w2;
    cudaStreamCreate(&stream_w1);
    cudaStreamCreate(&stream_w2);

    cudaEventRecord(wrong_start, 0);

    vector_add_kernel<<<blocks, threads, 0, stream_w1>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out_a.data_ptr<float>(), n);

    // 错误：阻塞整个 GPU，包括其他所有 stream
    cudaDeviceSynchronize();

    vector_mul_pow_kernel<<<blocks, threads, 0, stream_w2>>>(
        out_a.data_ptr<float>(),
        b.data_ptr<float>(),
        out_b.data_ptr<float>(),
        n);

    cudaDeviceSynchronize();
    cudaEventRecord(wrong_end, 0);
    cudaEventSynchronize(wrong_end);

    float wrong_ms = 0.0f;
    cudaEventElapsedTime(&wrong_ms, wrong_start, wrong_end);

    // ---- 正确方式：cudaStreamSynchronize ----
    cudaEvent_t correct_start, correct_end;
    cudaEventCreate(&correct_start);
    cudaEventCreate(&correct_end);

    cudaStream_t stream_c1, stream_c2;
    cudaStreamCreate(&stream_c1);
    cudaStreamCreate(&stream_c2);

    auto out_c = torch::empty_like(a);
    auto out_d = torch::empty_like(a);

    cudaEventRecord(correct_start, 0);

    vector_add_kernel<<<blocks, threads, 0, stream_c1>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out_c.data_ptr<float>(), n);

    // 正确：只等待 stream_c1，stream_c2 和其他 stream 不受影响
    cudaStreamSynchronize(stream_c1);

    vector_mul_pow_kernel<<<blocks, threads, 0, stream_c2>>>(
        out_c.data_ptr<float>(),
        b.data_ptr<float>(),
        out_d.data_ptr<float>(),
        n);

    cudaStreamSynchronize(stream_c2);
    cudaEventRecord(correct_end, 0);
    cudaEventSynchronize(correct_end);

    float correct_ms = 0.0f;
    cudaEventElapsedTime(&correct_ms, correct_start, correct_end);

    // 清理
    cudaStreamDestroy(stream_w1);
    cudaStreamDestroy(stream_w2);
    cudaStreamDestroy(stream_c1);
    cudaStreamDestroy(stream_c2);
    cudaEventDestroy(wrong_start);
    cudaEventDestroy(wrong_end);
    cudaEventDestroy(correct_start);
    cudaEventDestroy(correct_end);

    auto timing = torch::empty({2}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    timing[0] = wrong_ms;
    timing[1] = correct_ms;
    return std::vector<torch::Tensor>{out_a, out_b, out_c, out_d, timing};
}
