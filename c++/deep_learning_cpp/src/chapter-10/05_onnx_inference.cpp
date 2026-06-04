/*
 * 05_onnx_inference.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * ONNX Runtime Inference in C++.
 *
 * ONNX (Open Neural Network Exchange) is an interoperable format
 * that allows models trained in PyTorch, TensorFlow, etc. to be
 * deployed with a single C++ runtime. ONNX Runtime provides:
 *   - Hardware-specific optimizations (oneDNN on CPU, TensorRT on GPU)
 *   - Graph-level optimizations (constant folding, node fusion)
 *   - INT8 quantization via Q/DQ (Quantize/DeQuantize) operators
 *
 * This demo shows:
 *   1. Setting up Ort::Session with graph optimizations
 *   2. Creating input/output tensors with Ort::MemoryInfo
 *   3. Running inference and extracting results
 *
 * Note: ONNX model export is typically done in Python:
 *   torch.onnx.export(model, example_input, "model.onnx")
 *
 * The C++ side only consumes the .onnx file via Ort::Session.
 */

// Wrap in conditional compilation — ONNX Runtime is optional
#ifdef HAS_ONNX

#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

// ----------------------------------------------------------------
// Helper: create an ONNX Runtime tensor from raw float data
// ----------------------------------------------------------------
Ort::Value createTensor(
    Ort::MemoryInfo &mem_info,
    const std::vector<float> &data,
    const std::vector<int64_t> &shape) {
    return Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float *>(data.data()),
        data.size(),
        shape.data(),
        shape.size());
}

// ----------------------------------------------------------------
// ONNX Model Loader and Inference Runner
// ----------------------------------------------------------------
class ONNXInferenceRunner {
public:
    ONNXInferenceRunner(const std::string &model_path,
                        int intra_threads = 4,
                        OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING) : env_(log_level, "onnx_demo"),
                                                                                 session_(nullptr) {
        session_options_.SetIntraOpNumThreads(intra_threads);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        // session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        session_ = std::make_unique<Ort::Session>(
            env_, model_path.c_str(), session_options_);

        // Print model I/O info
        Ort::AllocatorWithDefaultOptions alloc;
        std::cout << "ONNX model loaded: " << model_path << "\n";
        std::cout << "  Inputs:\n";
        for (size_t i = 0; i < session_->GetInputCount(); i++) {
            auto name = session_->GetInputNameAllocated(i, alloc);
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            std::cout << "    [" << i << "] " << name.get()
                      << "  dtype=" << tensor_info.GetElementType()
                      << "  shape=[";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << (j > 0 ? ", " : "") << shape[j];
            }
            std::cout << "]\n";
        }
        std::cout << "  Outputs:\n";
        for (size_t i = 0; i < session_->GetOutputCount(); i++) {
            auto name = session_->GetOutputNameAllocated(i, alloc);
            std::cout << "    [" << i << "] " << name.get() << "\n";
        }
        std::cout << "\n";
    }

    // Run inference on a single preprocessed tensor
    // input_tensor: shape [1, C, H, W] — must match model's expected shape
    std::vector<float> run(const std::vector<float> &data,
                           const std::vector<int64_t> &shape) {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor
        auto input_name = session_->GetInputNameAllocated(0,
                                                          Ort::AllocatorWithDefaultOptions());
        auto output_name = session_->GetOutputNameAllocated(0,
                                                            Ort::AllocatorWithDefaultOptions());

        const char *input_names[] = {input_name.get()};
        const char *output_names[] = {output_name.get()};

        auto input_tensor = createTensor(mem_info, data, shape);
        std::vector<Ort::Value> input_values;
        input_values.push_back(std::move(input_tensor));

        // Run
        auto output_values = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_values.data(),
            input_values.size(),
            output_names,
            1);

        // Extract data
        auto *output_data = output_values[0].GetTensorMutableData<float>();
        auto output_shape = output_values[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_count = 1;
        for (auto d : output_shape) output_count *= d;

        return std::vector<float>(output_data, output_data + output_count);
    }

    // Benchmark throughput
    double benchmark(const std::vector<float> &data,
                     const std::vector<int64_t> &shape,
                     int iters = 100) {
        // Warm-up
        for (int i = 0; i < 10; i++) run(data, shape);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) run(data, shape);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 1000.0 * iters / ms; // inferences/sec
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
};

// ----------------------------------------------------------------
// Demo: Create a simple ONNX model via Python export, then load it
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== ONNX Runtime Inference Demo ===\n\n";

    // This demo requires a pre-exported .onnx file.
    // For demonstration, we show the setup code.
    //
    // To generate a model:
    //   python -c "
    //   import torch
    //   class TinyNet(torch.nn.Module):
    //       def __init__(self):
    //           super().__init__()
    //           self.c1 = torch.nn.Conv2d(3,8,3,padding=1)
    //           self.c2 = torch.nn.Conv2d(8,16,3,padding=1)
    //           self.fc = torch.nn.Linear(16,10)
    //       def forward(self, x):
    //           x = torch.relu(self.c1(x))
    //           x = torch.relu(self.c2(x))
    //           x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
    //           x = x.view(x.size(0), -1)
    //           return self.fc(x)
    //   model = TinyNet()
    //   model.eval()
    //   torch.onnx.export(model,
    //       torch.randn(1,3,224,224),
    //       '/tmp/tinynet.onnx',
    //       input_names=['input'],
    //       output_names=['output'])
    //   "

    std::string model_path = "/tmp/tinynet.onnx";

    // Check if model file exists
    std::ifstream f(model_path);
    if (!f.good()) {
        std::cout << "ONNX model not found: " << model_path << "\n";
        std::cout << "\nTo generate the model, run the Python code in the comment above.\n";
        std::cout << "Then re-run this demo.\n";
        return 0;
    }
    f.close();

    ONNXInferenceRunner runner(model_path, /*intra_threads=*/4);

    // Create dummy input [1, 3, 224, 224]
    std::vector<int64_t> shape = {1, 3, 224, 224};
    std::vector<float> data(1 * 3 * 224 * 224);
    std::generate(data.begin(), data.end(),
                  []() { return (float)rand() / RAND_MAX; });

    // Single inference
    auto result = runner.run(data, shape);
    std::cout << "Single inference: output size = " << result.size()
              << " (expecting 10 logits)\n";
    std::cout << "  Logits: [";
    for (int i = 0; i < 10 && i < (int)result.size(); i++) {
        std::cout << result[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // Benchmark
    double ips = runner.benchmark(data, shape, /*iters=*/200);
    std::cout << "Throughput: " << ips << " inferences/sec\n";

    return 0;
}

#else // !HAS_ONNX

#include <iostream>
int main() {
    std::cout << "ONNX Runtime not available.\n";
    std::cout << "Install ONNX Runtime and rebuild with -DHAS_ONNX=ON\n";
    return 0;
}

#endif
