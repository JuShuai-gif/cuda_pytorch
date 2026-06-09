#include "gpu_inference.cuh"

#include <cmath>
#include <random>

// ============================================================================
// GPU 内核: 3x3 卷积, valid padding
// 输入: [C_in][H][W], 权重: [C_out][C_in][3][3]
// 输出: [C_out][H-2][W-2]
// ============================================================================
__global__ void conv2d_kernel(const float* __restrict__ input,
                               const float* __restrict__ weights,
                               float* __restrict__ output,
                               int H, int W, int C_in, int C_out) {
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;

    int out_H = H - 2;
    int out_W = W - 2;

    if (out_w >= out_W || out_h >= out_H) return;

    for (int co = 0; co < C_out; ++co) {
        float accum = 0.0f;
        for (int ci = 0; ci < C_in; ++ci) {
            for (int ky = 0; ky < 3; ++ky) {
                int in_h = out_h + ky;
                for (int kx = 0; kx < 3; ++kx) {
                    int in_w = out_w + kx;
                    float pixel = input[ci * H * W + in_h * W + in_w];
                    // 权重布局: [C_out][C_in][3][3]
                    float w = weights[((co * C_in + ci) * 3 + ky) * 3 + kx];
                    accum += pixel * w;
                }
            }
        }
        output[co * out_H * out_W + out_h * out_W + out_w] = accum;
    }
}

// ============================================================================
// GPU 内核: ReLU 激活 (逐元素)
// ============================================================================
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = fmaxf(0.0f, data[idx]);
}

// ============================================================================
// GPU 内核: 2x2 最大池化，步长 2
// 输入: [C][H][W], 输出: [C][H/2][W/2]
// ============================================================================
__global__ void maxpool_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int H, int W, int C) {
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_H = H / 2;
    int out_W = W / 2;

    if (out_w >= out_W || out_h >= out_H) return;

    for (int ch = 0; ch < C; ++ch) {
        int base = ch * H * W;
        int in_h0 = out_h * 2;
        int in_w0 = out_w * 2;
        float v00 = input[base + in_h0 * W + in_w0];
        float v01 = input[base + in_h0 * W + in_w0 + 1];
        float v10 = input[base + (in_h0 + 1) * W + in_w0];
        float v11 = input[base + (in_h0 + 1) * W + in_w0 + 1];
        float mx = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
        output[ch * out_H * out_W + out_h * out_W + out_w] = mx;
    }
}

// ============================================================================
// GPU 内核: 检测头 (每个空间位置的线性层)
// 特征: [C_in][H][W], 检测头权重: [N_det][C_in], 偏置: [N_det]
// 输出: [N_det][H][W] (置信度, 中心_x, 中心_y, 宽度, 高度)
// ============================================================================
__global__ void detection_head_kernel(const float* __restrict__ features,
                                       const float* __restrict__ head_w,
                                       const float* __restrict__ head_b,
                                       float* __restrict__ output,
                                       int H, int W, int C_in, int N_det) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= W || py >= H) return;

    for (int d = 0; d < N_det; ++d) {
        float accum = head_b[d];
        for (int c = 0; c < C_in; ++c) {
            accum += features[c * H * W + py * W + px] * head_w[d * C_in + c];
        }
        // 简单非线性: 对于类置信度值，钳位到 [0, 1]
        output[d * H * W + py * W + px] = 1.0f / (1.0f + expf(-accum));
    }
}

// ============================================================================
// 封装: 启动 conv2d 内核
// ============================================================================
void gpu_conv2d(const float* d_input, const float* d_weights,
                float* d_output,
                int H, int W, int C_in, int C_out,
                cudaStream_t stream) {
    int out_W = W - 2;
    int out_H = H - 2;
    dim3 block(16, 16);
    dim3 grid((out_W + 15) / 16, (out_H + 15) / 16);
    conv2d_kernel<<<grid, block, 0, stream>>>(
        d_input, d_weights, d_output, H, W, C_in, C_out);
}

// ============================================================================
// 封装: 启动 ReLU 内核
// ============================================================================
void gpu_relu(float* d_data, int total_elements, cudaStream_t stream) {
    int block = 256;
    int grid = (total_elements + block - 1) / block;
    relu_kernel<<<grid, block, 0, stream>>>(d_data, total_elements);
}

// ============================================================================
// 封装: 启动 maxpool 内核
// ============================================================================
void gpu_maxpool(const float* d_input, float* d_output,
                 int H, int W, int C,
                 cudaStream_t stream) {
    int out_W = W / 2;
    int out_H = H / 2;
    dim3 block(16, 16);
    dim3 grid((out_W + 15) / 16, (out_H + 15) / 16);
    maxpool_kernel<<<grid, block, 0, stream>>>(d_input, d_output, H, W, C);
}

// ============================================================================
// 封装: 启动检测头内核
// ============================================================================
void gpu_detection_head(const float* d_features, const float* d_head_weights,
                        const float* d_head_bias,
                        float* d_output,
                        int H, int W, int C_in, int N_det,
                        cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    detection_head_kernel<<<grid, block, 0, stream>>>(
        d_features, d_head_weights, d_head_bias, d_output, H, W, C_in, N_det);
}

// ============================================================================
// 使用类 Xavier 均匀分布初始化卷积权重
// ============================================================================
float* gpu_alloc_init_conv_weights(int C_in, int C_out, int seed) {
    int num_weights = C_out * C_in * 3 * 3;
    std::vector<float> h_weights(num_weights);
    std::mt19937 rng(seed);
    float limit = std::sqrt(6.0f / static_cast<float>(C_in * 9 + C_out * 9));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < num_weights; ++i) {
        h_weights[i] = dist(rng);
    }

    float* d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weights, num_weights * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                          num_weights * sizeof(float),
                          cudaMemcpyHostToDevice));
    return d_weights;
}

// ============================================================================
// 初始化检测头权重和偏置
// ============================================================================
void gpu_alloc_init_head_weights(float** d_weights, float** d_bias,
                                 int C_in, int N_det, int seed) {
    std::mt19937 rng(seed);
    float limit = std::sqrt(6.0f / static_cast<float>(C_in + N_det));
    std::uniform_real_distribution<float> dist(-limit, limit);

    std::vector<float> h_w(N_det * C_in);
    std::vector<float> h_b(N_det);
    for (int i = 0; i < N_det * C_in; ++i) h_w[i] = dist(rng);
    for (int i = 0; i < N_det; ++i) h_b[i] = dist(rng);

    CUDA_CHECK(cudaMalloc(d_weights, N_det * C_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_bias, N_det * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_weights, h_w.data(),
                          N_det * C_in * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_bias, h_b.data(),
                          N_det * sizeof(float),
                          cudaMemcpyHostToDevice));
}
