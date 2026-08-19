// Build a TensorRT engine from an ONNX model (C++).
//
// Covers the full builder path: ONNX parse -> network -> builder config ->
// optimization profile (dynamic batch/seq) -> serialized engine.  FP16 is a
// flag; INT8 needs calibration and is deferred to the quantization stage.
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdio>
#include <string>

#include "trt_common.h"

using namespace trt_lab;

int main(int argc, char** argv) {
    std::string onnx_path = "/tmp/trt_model/model.onnx";
    std::string engine_path = "/tmp/trt_model/model.engine";
    bool fp16 = false;
    int min_batch = 1, opt_batch = 8, max_batch = 32;
    int min_seq = 1, opt_seq = 16, max_seq = 64;
    size_t workspace_gb = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--onnx") onnx_path = next();
        else if (a == "--engine") engine_path = next();
        else if (a == "--fp16") fp16 = true;
        else if (a == "--min-batch") min_batch = std::stoi(next());
        else if (a == "--opt-batch") opt_batch = std::stoi(next());
        else if (a == "--max-batch") max_batch = std::stoi(next());
        else if (a == "--min-seq") min_seq = std::stoi(next());
        else if (a == "--opt-seq") opt_seq = std::stoi(next());
        else if (a == "--max-seq") max_seq = std::stoi(next());
        else if (a == "--workspace-gb") workspace_gb = std::stoul(next());
        else if (a == "--help") { std::printf("usage: build_engine [--onnx P] [--engine P] [--fp16] ...\n"); return 0; }
    }

    Logger logger;
    auto builder = nvinfer1::createInferBuilder(logger);
    TRT_CHECK(builder != nullptr);

    auto network = builder->createNetworkV2(0);
    TRT_CHECK(network != nullptr);

    auto parser = nvonnxparser::createParser(*network, logger);
    TRT_CHECK(parser != nullptr);
    TRT_CHECK(parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)));

    auto config = builder->createBuilderConfig();
    TRT_CHECK(config != nullptr);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_gb << 30);
    if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Optimization profile: dynamic batch and seq dims on the single input.
    auto input = network->getInput(0);
    std::string input_name = input->getName();
    auto profile = builder->createOptimizationProfile();
    int hidden = input->getDimensions().d[2];
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims3{min_batch, min_seq, hidden});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims3{opt_batch, opt_seq, hidden});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims3{max_batch, max_seq, hidden});
    config->addOptimizationProfile(profile);

    WallTimer timer;
    timer.start();
    auto plan = builder->buildSerializedNetwork(*network, *config);
    TRT_CHECK(plan != nullptr);
    double build_ms = timer.ms();

    write_file(engine_path, plan->data(), plan->size());

    JsonReport r;
    r.begin();
    r.put("onnx", onnx_path);
    r.put("engine", engine_path);
    r.put("fp16", fp16 ? "true" : "false");
    r.put("input", input_name);
    r.put("min_shape", std::to_string(min_batch) + "x" + std::to_string(min_seq) + "x" + std::to_string(hidden));
    r.put("opt_shape", std::to_string(opt_batch) + "x" + std::to_string(opt_seq) + "x" + std::to_string(hidden));
    r.put("max_shape", std::to_string(max_batch) + "x" + std::to_string(max_seq) + "x" + std::to_string(hidden));
    r.put("build_ms", build_ms);
    r.put("engine_size_bytes", static_cast<long long>(plan->size()));
    std::printf("%s", r.end().c_str());

    return 0;
}
