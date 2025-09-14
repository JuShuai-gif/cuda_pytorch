// cudnn_wrapper.cu
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking cuDNN calls
#define CHECK_CUDNN(call)                                                                                   \
    do {                                                                                                    \
        cudnnStatus_t status = (call);                                                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                                                               \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
            return -1;                                                                                      \
        }                                                                                                   \
    } while (0)

static cudnnHandle_t cudnnHandle;

extern "C" int init_cudnn() {
    CHECK_CUDNN(cudnnCreate(&cudnnHandle));
    printf("cuDNN initialized successfully\n");
    return 0;
}

extern "C" int cleanup_cudnn() {
    CHECK_CUDNN(cudnnDestroy(cudnnHandle));
    printf("cuDNN cleaned up successfully\n");
    return 0;
}

extern "C" int conv2d_forward(float* input, float* weights, float* bias, float* output, int batch_size, int in_channels,
                              int height, int width, int out_channels, int kernel_h, int kernel_w, int pad_h, int pad_w,
                              int stride_h, int stride_w) {
    // Compute output dims
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Allocate device memory
    float *d_input, *d_filter, *d_output, *d_bias;
    size_t in_bytes = batch_size * in_channels * height * width * sizeof(float);
    size_t fil_bytes = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t out_bytes = batch_size * out_channels * out_h * out_w * sizeof(float);
    size_t bias_bytes = out_channels * sizeof(float);

    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_filter, fil_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMalloc(&d_bias, bias_bytes);

    cudaMemcpy(d_input, input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, weights, fil_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_bytes, cudaMemcpyHostToDevice);

    // Create tensor descriptors
    cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t fil_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&fil_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, in_channels, height, width);
    cudnnSetFilter4dDescriptor(fil_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_h,
                               kernel_w);
    cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, out_h, out_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);

    // Choose algorithm - using modern method with heuristics
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returned_algo_count;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnnHandle, in_desc, fil_desc, conv_desc, out_desc, 1,
                                                     &returned_algo_count, &perf));
    cudnnConvolutionFwdAlgo_t algo = perf.algo;

    // Workspace
    size_t ws_size = 0;
    CHECK_CUDNN(
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, in_desc, fil_desc, conv_desc, out_desc, algo, &ws_size));
    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, ws_size);

    float alpha = 1.0f, beta = 0.0f;
    // Forward
    CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, in_desc, d_input, fil_desc, d_filter, conv_desc, algo,
                                        d_workspace, ws_size, &beta, out_desc, d_output));

    // Add bias
    CHECK_CUDNN(cudnnAddTensor(cudnnHandle, &alpha, bias_desc, d_bias, &alpha, out_desc, d_output));

    // Copy back
    cudaMemcpy(output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(fil_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return 0;
}

extern "C" int conv2d_backward(float* input, float* weights, float* d_output, float* d_input, float* d_weights,
                               float* d_bias, int batch_size, int in_channels, int height, int width, int out_channels,
                               int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    // Compute dims
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Allocate device memory
    size_t in_bytes = batch_size * in_channels * height * width * sizeof(float);
    size_t fil_bytes = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t out_bytes = batch_size * out_channels * out_h * out_w * sizeof(float);
    size_t bias_bytes = out_channels * sizeof(float);

    float *d_in, *d_fil, *d_out, *d_in_grad, *d_fil_grad, *d_bias_grad;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_fil, fil_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMalloc(&d_in_grad, in_bytes);
    cudaMalloc(&d_fil_grad, fil_bytes);
    cudaMalloc(&d_bias_grad, bias_bytes);

    cudaMemcpy(d_in, input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fil, weights, fil_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, d_output, out_bytes, cudaMemcpyHostToDevice);

    // Descriptors (reuse forward ones)
    cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t fil_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&fil_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, in_channels, height, width);
    cudnnSetFilter4dDescriptor(fil_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_h,
                               kernel_w);
    cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, out_h, out_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);

    float alpha = 1.0f, beta = 0.0f;

    // Backward Data - using newer algorithm selection method
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
    int returned_bwd_data_algo_count;
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle, fil_desc, out_desc, conv_desc, in_desc, 1,
                                                          &returned_bwd_data_algo_count, &bwd_data_perf));
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo = bwd_data_perf.algo;

    size_t ws_data_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, fil_desc, out_desc, conv_desc, in_desc,
                                                             bwd_data_algo, &ws_data_size));
    void* d_ws_data = nullptr;
    cudaMalloc(&d_ws_data, ws_data_size);
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, fil_desc, d_fil, out_desc, d_out, conv_desc,
                                             bwd_data_algo, d_ws_data, ws_data_size, &beta, in_desc, d_in_grad));

    // Backward Filter - using newer algorithm selection method
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_fil_perf;
    int returned_bwd_fil_algo_count;
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle, in_desc, out_desc, conv_desc, fil_desc, 1,
                                                            &returned_bwd_fil_algo_count, &bwd_fil_perf));
    cudnnConvolutionBwdFilterAlgo_t bwd_fil_algo = bwd_fil_perf.algo;

    size_t ws_fil_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, in_desc, out_desc, conv_desc, fil_desc,
                                                               bwd_fil_algo, &ws_fil_size));
    void* d_ws_fil = nullptr;
    cudaMalloc(&d_ws_fil, ws_fil_size);
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, in_desc, d_in, out_desc, d_out, conv_desc,
                                               bwd_fil_algo, d_ws_fil, ws_fil_size, &beta, fil_desc, d_fil_grad));

    // Backward Bias
    CHECK_CUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, out_desc, d_out, &beta, bias_desc, d_bias_grad));

    // Copy back to host
    cudaMemcpy(d_input, d_in_grad, in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_weights, d_fil_grad, fil_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_bias, d_bias_grad, bias_bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_fil);
    cudaFree(d_out);
    cudaFree(d_in_grad);
    cudaFree(d_fil_grad);
    cudaFree(d_bias_grad);
    cudaFree(d_ws_data);
    cudaFree(d_ws_fil);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(fil_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return 0;
}

extern "C" int maxpool2d_forward(float* input, float* output, int* indices, int batch_size, int channels, int height,
                                 int width, int kernel_h, int kernel_w, int stride_h, int stride_w) {
    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    size_t in_b = batch_size * channels * height * width * sizeof(float);
    size_t out_b = batch_size * channels * out_h * out_w * sizeof(float);
    size_t idx_b = batch_size * channels * out_h * out_w * sizeof(int);

    float *d_in, *d_out;
    int* d_idx;
    cudaMalloc(&d_in, in_b);
    cudaMalloc(&d_out, out_b);
    cudaMalloc(&d_idx, idx_b);
    cudaMemcpy(d_in, input, in_b, cudaMemcpyHostToDevice);

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, out_h, out_w);
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, kernel_h, kernel_w, 0, 0, stride_h,
                                stride_w);

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnPoolingForward(cudnnHandle, pool_desc, &alpha, in_desc, d_in, &beta, out_desc, d_out));

    // Note: cuDNN doesn't give indices; you may skip indices or implement argmax manually if needed.

    cudaMemcpy(output, d_out, out_b, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_idx);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);

    return 0;
}

extern "C" int maxpool2d_backward(float* input, float* d_output, int* indices, float* d_input, int batch_size,
                                  int channels, int height, int width, int kernel_h, int kernel_w, int stride_h,
                                  int stride_w) {
    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    size_t in_b = batch_size * channels * height * width * sizeof(float);
    size_t out_b = batch_size * channels * out_h * out_w * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_b);
    cudaMalloc(&d_out, out_b);
    cudaMemcpy(d_out, d_output, out_b, cudaMemcpyHostToDevice);

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, out_h, out_w);
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, kernel_h, kernel_w, 0, 0, stride_h,
                                stride_w);

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnPoolingBackward(cudnnHandle, pool_desc, &alpha, out_desc, d_out, out_desc, d_out, in_desc, d_in,
                                     &beta, in_desc, d_in));

    cudaMemcpy(d_input, d_in, in_b, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);

    return 0;
}
