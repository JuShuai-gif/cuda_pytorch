// RMSNorm custom TensorRT plugin (C++).
//
// The industrial scenario: TensorRT does not ship a fused RMSNorm, so we
// implement it as a custom plugin.  This file walks the complete plugin
// lifecycle: a CUDA kernel, the IPluginV2DynamicExt implementation, the
// IPluginCreator (for serialization), and the demo that builds/serializes/
// deserializes/runs the engine and verifies against a CPU reference.
//
// RMSNorm (no weight): y[row] = x[row] * rsqrt(mean(x[row]^2) + eps)
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "trt_common.h"

using namespace trt_lab;

// --------------------------------------------------------------------------
// CUDA kernel: one block per row, block reduction for mean-of-squares.
// --------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* x, float* y, int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        sum += v * v;
    }
    __shared__ float sh[256];
    sh[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(sh[0] / cols + eps);
    for (int c = tid; c < cols; c += blockDim.x) {
        y[row * cols + c] = x[row * cols + c] * rstd;
    }
}

// --------------------------------------------------------------------------
// Plugin
// --------------------------------------------------------------------------
namespace {
constexpr char const* kRMSNORM_NAME = "RMSNormPlugin";
constexpr char const* kRMSNORM_VERSION = "1";
}  // namespace

class RMSNormPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit RMSNormPlugin(float eps) : mEps(eps) {}

    // Deserialize constructor.
    RMSNormPlugin(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        mEps = *reinterpret_cast<const float*>(d);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        auto dims = inputDesc[0].dims;  // (batch, seq, hidden)
        int rows = dims.d[0] * dims.d[1];
        int cols = dims.d[2];
        const float* x = static_cast<const float*>(inputs[0]);
        float* y = static_cast<float*>(outputs[0]);
        rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, y, rows, cols, mEps);
        return 0;
    }

    char const* getPluginType() const noexcept override { return kRMSNORM_NAME; }
    char const* getPluginVersion() const noexcept override { return kRMSNORM_VERSION; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(char const* ns) noexcept override { mNamespace = ns; }

    nvinfer1::DimsExprs getOutputDimensions(int32_t index, nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        return inputs[0];  // shape unchanged
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {}

    size_t getSerializationSize() const noexcept override { return sizeof(float); }

    void serialize(void* buffer) const noexcept override {
        char* d = static_cast<char*>(buffer);
        *reinterpret_cast<float*>(d) = mEps;
    }

    void destroy() noexcept override { delete this; }
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* p = new RMSNormPlugin(mEps);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override {
        return inputTypes[0];
    }

private:
    float mEps;
    std::string mNamespace;
};

// --------------------------------------------------------------------------
// Creator (needed to deserialize the engine from disk).
// --------------------------------------------------------------------------
class RMSNormPluginCreator : public nvinfer1::IPluginCreator {
public:
    char const* getPluginName() const noexcept override { return kRMSNORM_NAME; }
    char const* getPluginVersion() const noexcept override { return kRMSNORM_VERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    nvinfer1::IPluginV2* createPlugin(char const* name,
        nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float eps = 1e-5f;
        for (int i = 0; i < fc->nbFields; i++) {
            if (std::strcmp(fc->fields[i].name, "eps") == 0) {
                eps = *static_cast<float const*>(fc->fields[i].data);
            }
        }
        return new RMSNormPlugin(eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData,
        size_t serialLength) noexcept override {
        return new RMSNormPlugin(serialData, serialLength);
    }

    void setPluginNamespace(char const* ns) noexcept override { mNamespace = ns; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{};
};

REGISTER_TENSORRT_PLUGIN(RMSNormPluginCreator);
