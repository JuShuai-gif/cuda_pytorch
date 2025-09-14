#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Conv2D forward kernel
// conv2d 前向推理
__global__ void conv2d_forward_kernel(const float *input, const float *weights, const float *bias, float *output,
                                      int batch_size, int in_channels, int height, int width, int out_channels,
                                      int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                                      int out_h, int out_w) {
    int b = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / out_w;
    int w_out = blockIdx.x % out_w;

    if (b >= batch_size || c_out >= out_channels || h_out >= out_h || w_out >= out_w) {
        return;
    }

    float val = bias[c_out];

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;

                    val += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
    output[output_idx] = val;
}

// Conv2D backward kernel for input gradients
__global__ void conv2d_backward_input_kernel(const float *weights, const float *grad_output, float *grad_input,
                                             int batch_size, int in_channels, int height, int width, int out_channels,
                                             int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                                             int stride_w, int out_h, int out_w) {
    int b = blockIdx.z;
    int c_in = blockIdx.y;
    int h_in = blockIdx.x / width;
    int w_in = blockIdx.x % width;

    if (b >= batch_size || c_in >= in_channels || h_in >= height || w_in >= width) {
        return;
    }

    float val = 0.0f;

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_out = (h_in + pad_h - kh) / stride_h;
                int w_out = (w_in + pad_w - kw) / stride_w;

                // Check if the output position is valid
                if (h_out >= 0 && h_out < out_h && w_out >= 0 && w_out < out_w && (h_in + pad_h - kh) % stride_h == 0 && (w_in + pad_w - kw) % stride_w == 0) {
                    int grad_output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;

                    val += grad_output[grad_output_idx] * weights[weight_idx];
                }
            }
        }
    }

    int grad_input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
    grad_input[grad_input_idx] = val;
}

// Conv2D backward kernel for weight gradients
__global__ void conv2d_backward_weights_kernel(const float *input, const float *grad_output, float *grad_weights,
                                               int batch_size, int in_channels, int height, int width, int out_channels,
                                               int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                                               int stride_w, int out_h, int out_w) {
    int c_out = blockIdx.z;
    int c_in = blockIdx.y;
    int kh = blockIdx.x / kernel_w;
    int kw = blockIdx.x % kernel_w;

    if (c_out >= out_channels || c_in >= in_channels || kh >= kernel_h || kw >= kernel_w) {
        return;
    }

    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        for (int h_out = 0; h_out < out_h; ++h_out) {
            for (int w_out = 0; w_out < out_w; ++w_out) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                    int grad_output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;

                    val += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }

    int grad_weights_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
    grad_weights[grad_weights_idx] = val;
}

// Conv2D backward kernel for bias gradients
__global__ void conv2d_backward_bias_kernel(const float *grad_output, float *grad_bias, int batch_size,
                                            int out_channels, int out_h, int out_w) {
    int c_out = blockIdx.x;

    if (c_out >= out_channels) {
        return;
    }

    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        for (int h_out = 0; h_out < out_h; ++h_out) {
            for (int w_out = 0; w_out < out_w; ++w_out) {
                int grad_output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                val += grad_output[grad_output_idx];
            }
        }
    }

    grad_bias[c_out] = val;
}

// MaxPool2D forward kernel
__global__ void maxpool2d_forward_kernel(const float *input, float *output, int *indices, int batch_size, int channels,
                                         int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w,
                                         int out_h, int out_w) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h_out = blockIdx.x / out_w;
    int w_out = blockIdx.x % out_w;

    if (b >= batch_size || c >= channels || h_out >= out_h || w_out >= out_w) {
        return;
    }

    float max_val = -FLT_MAX;
    int max_idx = -1;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;

            if (h_in < height && w_in < width) {
                int input_idx = ((b * channels + c) * height + h_in) * width + w_in;
                float val = input[input_idx];

                if (val > max_val) {
                    max_val = val;
                    max_idx = kh * kernel_w + kw; // Store relative index within kernel
                }
            }
        }
    }

    int output_idx = ((b * channels + c) * out_h + h_out) * out_w + w_out;
    output[output_idx] = max_val;
    indices[output_idx] = max_idx;
}

// MaxPool2D backward kernel
__global__ void maxpool2d_backward_kernel(const float *grad_output, const int *indices, float *grad_input,
                                          int batch_size, int channels, int height, int width, int kernel_h,
                                          int kernel_w, int stride_h, int stride_w, int out_h, int out_w) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h_out = blockIdx.x / out_w;
    int w_out = blockIdx.x % out_w;

    if (b >= batch_size || c >= channels || h_out >= out_h || w_out >= out_w) {
        return;
    }

    int output_idx = ((b * channels + c) * out_h + h_out) * out_w + w_out;
    float grad_val = grad_output[output_idx];
    int max_idx = indices[output_idx];

    int kh = max_idx / kernel_w;
    int kw = max_idx % kernel_w;

    int h_in = h_out * stride_h + kh;
    int w_in = w_out * stride_w + kw;

    if (h_in < height && w_in < width) {
        int input_idx = ((b * channels + c) * height + h_in) * width + w_in;
        atomicAdd(&grad_input[input_idx], grad_val);
    }
}

// Conv2D Forward Pass
extern "C" int conv2d_forward(float *input, float *weights, float *bias, float *output, int batch_size, int in_channels,
                              int height, int width, int out_channels, int kernel_h, int kernel_w, int pad_h, int pad_w,
                              int stride_h, int stride_w) {
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Allocate device memory
    float *d_input, *d_weights, *d_bias, *d_output;

    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    size_t weights_size = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    size_t output_size = batch_size * out_channels * out_h * out_w * sizeof(float);

    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_weights, weights_size);
    cudaMalloc((void **)&d_bias, bias_size);
    cudaMalloc((void **)&d_output, output_size);

    // Copy input data to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

    // Set up grid and blocks for kernel launch
    dim3 grid(out_h * out_w, out_channels, batch_size);
    conv2d_forward_kernel<<<grid, 1>>>(d_input, d_weights, d_bias, d_output, batch_size, in_channels, height, width,
                                       out_channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, out_h,
                                       out_w);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv2d_forward: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}

// Conv2D Backward Pass
extern "C" int conv2d_backward(float *input, float *weights, float *d_output, float *d_input, float *d_weights,
                               float *d_bias, int batch_size, int in_channels, int height, int width, int out_channels,
                               int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Allocate device memory
    float *d_input_data, *d_weights_data, *d_output_data;
    float *d_input_grad, *d_weights_grad, *d_bias_grad;

    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    size_t weights_size = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    size_t output_size = batch_size * out_channels * out_h * out_w * sizeof(float);

    cudaMalloc((void **)&d_input_data, input_size);
    cudaMalloc((void **)&d_weights_data, weights_size);
    cudaMalloc((void **)&d_output_data, output_size);
    cudaMalloc((void **)&d_input_grad, input_size);
    cudaMalloc((void **)&d_weights_grad, weights_size);
    cudaMalloc((void **)&d_bias_grad, bias_size);

    // Initialize gradients to zero
    cudaMemset(d_input_grad, 0, input_size);
    cudaMemset(d_weights_grad, 0, weights_size);
    cudaMemset(d_bias_grad, 0, bias_size);

    // Copy data to device
    cudaMemcpy(d_input_data, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_data, weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_data, d_output, output_size, cudaMemcpyHostToDevice);

    // Calculate gradients for input
    dim3 grid_input(height * width, in_channels, batch_size);
    conv2d_backward_input_kernel<<<grid_input, 1>>>(d_weights_data, d_output_data, d_input_grad, batch_size,
                                                    in_channels, height, width, out_channels, kernel_h, kernel_w, pad_h,
                                                    pad_w, stride_h, stride_w, out_h, out_w);

    // Calculate gradients for weights
    dim3 grid_weights(kernel_h * kernel_w, in_channels, out_channels);
    conv2d_backward_weights_kernel<<<grid_weights, 1>>>(d_input_data, d_output_data, d_weights_grad, batch_size,
                                                        in_channels, height, width, out_channels, kernel_h, kernel_w,
                                                        pad_h, pad_w, stride_h, stride_w, out_h, out_w);

    // Calculate gradients for bias
    conv2d_backward_bias_kernel<<<out_channels, 1>>>(d_output_data, d_bias_grad, batch_size, out_channels, out_h,
                                                     out_w);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv2d_backward: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(d_input, d_input_grad, input_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_weights, d_weights_grad, weights_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_bias, d_bias_grad, bias_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_data);
    cudaFree(d_weights_data);
    cudaFree(d_output_data);
    cudaFree(d_input_grad);
    cudaFree(d_weights_grad);
    cudaFree(d_bias_grad);

    return 0;
}

// MaxPool2D Forward Pass
extern "C" int maxpool2d_forward(float *input, float *output, int *indices, int batch_size, int channels, int height,
                                 int width, int kernel_h, int kernel_w, int stride_h, int stride_w) {
    // Calculate output dimensions
    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    // Allocate device memory
    float *d_input, *d_output;
    int *d_indices;

    size_t input_size = batch_size * channels * height * width * sizeof(float);
    size_t output_size = batch_size * channels * out_h * out_w * sizeof(float);
    size_t indices_size = batch_size * channels * out_h * out_w * sizeof(int);

    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, output_size);
    cudaMalloc((void **)&d_indices, indices_size);

    // Copy input data to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    // Set up grid and blocks for kernel launch
    dim3 grid(out_h * out_w, channels, batch_size);
    maxpool2d_forward_kernel<<<grid, 1>>>(d_input, d_output, d_indices, batch_size, channels, height, width, kernel_h,
                                          kernel_w, stride_h, stride_w, out_h, out_w);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in maxpool2d_forward: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, d_indices, indices_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);

    return 0;
}

// MaxPool2D Backward Pass
extern "C" int maxpool2d_backward(float *input, float *d_output, int *indices, float *d_input, int batch_size,
                                  int channels, int height, int width, int kernel_h, int kernel_w, int stride_h,
                                  int stride_w) {
    // Calculate output dimensions
    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    // Allocate device memory
    float *d_grad_output, *d_grad_input;
    int *d_indices;

    size_t input_size = batch_size * channels * height * width * sizeof(float);
    size_t output_size = batch_size * channels * out_h * out_w * sizeof(float);
    size_t indices_size = batch_size * channels * out_h * out_w * sizeof(int);

    cudaMalloc((void **)&d_grad_output, output_size);
    cudaMalloc((void **)&d_grad_input, input_size);
    cudaMalloc((void **)&d_indices, indices_size);

    // Initialize gradient input to zero
    cudaMemset(d_grad_input, 0, input_size);

    // Copy data to device
    cudaMemcpy(d_grad_output, d_output, output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, indices_size, cudaMemcpyHostToDevice);

    // Set up grid and blocks for kernel launch
    dim3 grid(out_h * out_w, channels, batch_size);
    maxpool2d_backward_kernel<<<grid, 1>>>(d_grad_output, d_indices, d_grad_input, batch_size, channels, height, width,
                                           kernel_h, kernel_w, stride_h, stride_w, out_h, out_w);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in maxpool2d_backward: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(d_input, d_grad_input, input_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_indices);

    return 0;
}

// Init function (placeholder - not using cuDNN)
extern "C" int init_cudnn() {
    printf("Using CUDA kernel implementation (no cuDNN)\n");
    return 0;
}

// Cleanup function (placeholder - not using cuDNN)
extern "C" int cleanup_cudnn() {
    printf("Cleaning up CUDA kernel implementation (no cuDNN)\n");
    return 0;
}
