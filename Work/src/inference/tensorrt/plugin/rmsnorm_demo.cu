// RMSNorm plugin demo: build -> serialize -> deserialize -> infer -> verify.
//
// Builds a network containing the custom RMSNorm plugin (created via the
// plugin registry), builds a serialized engine, deserializes it (which
// requires the creator), runs inference, and verifies against a CPU reference.
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "trt_common.h"

using namespace trt_lab;

int main() {
    const int batch = 2, seq = 8, hidden = 1024;
    const float eps = 1e-5f;
    const std::string engine_path = "/tmp/trt_model/rmsnorm.engine";

    Logger logger;
    auto builder = nvinfer1::createInferBuilder(logger);
    auto network = builder->createNetworkV2(0);

    // Build network: input -> RMSNorm plugin -> output (dynamic batch/seq).
    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT,
                                   nvinfer1::Dims3{-1, -1, hidden});

    auto creator = getPluginRegistry()->getPluginCreator("RMSNormPlugin", "1");
    TRT_CHECK(creator != nullptr);
    nvinfer1::PluginField field{"eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1};
    nvinfer1::PluginFieldCollection fc{1, &field};
    auto plugin = creator->createPlugin("RMSNormPlugin", &fc);
    TRT_CHECK(plugin != nullptr);

    nvinfer1::ITensor* plugin_inputs[1] = {input};
    auto layer = network->addPluginV2(plugin_inputs, 1, *plugin);
    layer->getOutput(0)->setName("output");
    network->markOutput(*layer->getOutput(0));

    auto config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1u << 30);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims3{1, 1, hidden});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims3{batch, seq, hidden});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims3{32, 64, hidden});
    config->addOptimizationProfile(profile);

    auto plan = builder->buildSerializedNetwork(*network, *config);
    TRT_CHECK(plan != nullptr);
    write_file(engine_path, plan->data(), plan->size());

    // Deserialize (exercises the plugin creator path).
    auto engine_data = read_file(engine_path);
    auto runtime = nvinfer1::createInferRuntime(logger);
    auto engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    TRT_CHECK(engine != nullptr);

    size_t n_elem = static_cast<size_t>(batch) * seq * hidden;
    size_t bytes = n_elem * sizeof(float);

    std::vector<float> h_x(n_elem), h_y(n_elem);
    for (size_t i = 0; i < n_elem; i++) h_x[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;

    float *d_x, *d_y;
    cuda_alloc(&d_x, bytes);
    cuda_alloc(&d_y, bytes);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    auto context = engine->createExecutionContext();
    TRT_CHECK(context->setInputShape("input", nvinfer1::Dims3{batch, seq, hidden}));
    context->setTensorAddress("input", d_x);
    context->setTensorAddress("output", d_y);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    TRT_CHECK(context->enqueueV3(stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    // CPU reference: y = x * rsqrt(mean(x^2) + eps) per row.
    double max_diff = 0.0;
    for (int r = 0; r < batch * seq; r++) {
        double sum = 0.0;
        for (int c = 0; c < hidden; c++) sum += h_x[r * hidden + c] * h_x[r * hidden + c];
        float rstd = 1.0f / std::sqrt(static_cast<float>(sum / hidden + eps));
        for (int c = 0; c < hidden; c++) {
            float ref = h_x[r * hidden + c] * rstd;
            double d = std::fabs(static_cast<double>(h_y[r * hidden + c]) - ref);
            if (d > max_diff) max_diff = d;
        }
    }

    JsonReport rpt;
    rpt.begin();
    rpt.put("plugin", "RMSNormPlugin");
    rpt.put("batch", static_cast<long long>(batch));
    rpt.put("seq", static_cast<long long>(seq));
    rpt.put("hidden", static_cast<long long>(hidden));
    rpt.put("engine_size_bytes", static_cast<long long>(plan->size()));
    rpt.put("correct_max_abs_diff", max_diff);
    std::printf("%s", rpt.end().c_str());

    return 0;
}
