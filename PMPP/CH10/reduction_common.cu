#include "reduction_common.cuh"

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

bool is_close(float a, float b, float rtol, float atol) {
    float diff = fabs(a - b);
    float tolerance = atol + rtol * fabs(b);
    return diff <= tolerance;
}

float benchmark_sum_reduction(float (*func)(float*, int), float* data, unsigned int length, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) {
        func(data, length);
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);

    float totalTime_ms = 0.0f;

    for (int i = 0; i < reps; ++i) {
        clear_l2();
        cudaEventRecord(iterStart);
        func(data, length);
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return totalTime_ms / reps;
}

float sequential_sum_reduction(float* data, int length) {
    double total = 0.0;
    for (unsigned int i = 0; i < length; ++i) {
        total += data[i];
    }
    return (float)total;
}
