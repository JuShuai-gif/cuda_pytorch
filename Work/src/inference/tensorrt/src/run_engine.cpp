// Run a serialized TensorRT engine (C++): inference + correctness + benchmark.
//
// Deserializes the engine, sets the dynamic input shape, runs inference, and
// compares the output against the PyTorch reference (output_ref.bin).  Then
// benchmarks single-request latency (CUDA events) and throughput (burst wall
// time).  This is the runtime path a production C++ server would use.
#include <NvInfer.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "trt_common.h"

using namespace trt_lab;

int main(int argc, char** argv) {
    std::string engine_path = "/tmp/trt_model/model.engine";
    std::string input_path = "/tmp/trt_model/input.bin";
    std::string ref_path = "/tmp/trt_model/output_ref.bin";
    int batch = 1, seq = 16, hidden = 1024;
    int warmup = 20, iterations = 200;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--engine") engine_path = next();
        else if (a == "--input") input_path = next();
        else if (a == "--output-ref") ref_path = next();
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--seq") seq = std::stoi(next());
        else if (a == "--hidden") hidden = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--iterations") iterations = std::stoi(next());
    }

    Logger logger;
    auto engine_data = read_file(engine_path);
    auto runtime = nvinfer1::createInferRuntime(logger);
    TRT_CHECK(runtime != nullptr);
    auto engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    TRT_CHECK(engine != nullptr);

    std::string input_name, output_name;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        auto name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
        else output_name = name;
    }
    TRT_CHECK(!input_name.empty() && !output_name.empty());

    size_t n_elem = static_cast<size_t>(batch) * seq * hidden;
    size_t bytes = n_elem * sizeof(float);

    float *d_input = nullptr, *d_output = nullptr;
    cuda_alloc(&d_input, bytes);
    cuda_alloc(&d_output, bytes);
    std::vector<float> h_input(n_elem), h_output(n_elem);

    // If the recorded input covers the requested shape, use it and compare the
    // output against the reference.  Otherwise (e.g. a larger batch sweep)
    // fill with random data and skip the correctness check.
    auto in_data = read_file(input_path);
    bool do_check = (in_data.size() >= bytes);
    if (do_check) {
        std::memcpy(h_input.data(), in_data.data(), bytes);
    } else {
        for (size_t i = 0; i < n_elem; i++) {
            h_input[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    auto context = engine->createExecutionContext();
    TRT_CHECK(context != nullptr);
    TRT_CHECK(context->setInputShape(input_name.c_str(), nvinfer1::Dims3{batch, seq, hidden}));
    context->setTensorAddress(input_name.c_str(), d_input);
    context->setTensorAddress(output_name.c_str(), d_output);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm up.
    for (int i = 0; i < 10; i++) {
        TRT_CHECK(context->enqueueV3(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Correctness: run once, copy output, compare against PyTorch reference.
    TRT_CHECK(context->enqueueV3(stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    double max_diff = -1.0;  // -1 means "correctness check skipped"
    if (do_check) {
        auto ref_data = read_file(ref_path);
        if (ref_data.size() >= bytes) {
            const float* ref = reinterpret_cast<const float*>(ref_data.data());
            max_diff = 0.0;
            for (size_t i = 0; i < n_elem; i++) {
                double d = std::fabs(static_cast<double>(h_output[i]) - static_cast<double>(ref[i]));
                if (d > max_diff) max_diff = d;
            }
        }
    }

    // Latency benchmark: CUDA-event device time per inference.
    EventTimer timer;
    std::vector<double> lat_ms;
    for (int i = 0; i < iterations; i++) {
        timer.start(stream);
        TRT_CHECK(context->enqueueV3(stream));
        timer.stop(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        lat_ms.push_back(timer.ms());
    }

    // Throughput benchmark: burst of inferences, synchronize once at the end.
    int burst = iterations;
    WallTimer w;
    w.start();
    for (int i = 0; i < burst; i++) {
        TRT_CHECK(context->enqueueV3(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double burst_ms = w.ms();
    double samples_per_sec = (burst * batch) / (burst_ms * 1e-3);

    JsonReport r;
    r.begin();
    r.put("engine", engine_path);
    r.put("batch", static_cast<long long>(batch));
    r.put("seq", static_cast<long long>(seq));
    r.put("hidden", static_cast<long long>(hidden));
    r.put("correct_max_abs_diff", max_diff);
    r.put("latency_mean_ms", mean(lat_ms));
    r.put("latency_median_ms", median(lat_ms));
    r.put("throughput_samples_per_sec", samples_per_sec);
    r.put("throughput_burst_ms", burst_ms);
    std::printf("%s", r.end().c_str());

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
