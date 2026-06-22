#include <cuda_runtime.h>
#include <stdio.h>

constexpr int N = 1 << 20;  // 1M elements
constexpr int BLOCK_SIZE = 256;

// Simple vector add kernel used for CUDA Graph demonstration
__global__ void vec_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel with two inputs to demonstrate graph replay with different inputs
__global__ void vec_add_scale(float *a, float *b, float *c, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (a[idx] + b[idx]) * scale;
    }
}

void run_cuda_graph_demo() {
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_a, *d_b, *d_c1, *d_c2;
    float *h_a, *h_b, *h_c1_ref, *h_c2_ref;
    float *h_c1_graph, *h_c2_graph;

    size_t bytes = N * sizeof(float);

    // Host allocations
    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c1_ref = (float *)malloc(bytes);
    h_c2_ref = (float *)malloc(bytes);
    h_c1_graph = (float *)malloc(bytes);
    h_c2_graph = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Device allocations
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c1, bytes);
    cudaMalloc(&d_c2, bytes);

    // Copy input data once
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Warm-up launch (required before graph capture)
    printf("Warm-up launch...\n");
    vec_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c1, N);
    cudaDeviceSynchronize();

    // ====== Capture CUDA Graph ======
    printf("Capturing CUDA Graph...\n");

    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // Begin capture on the stream
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    // Record the kernel into the graph
    vec_add<<<grid_size, BLOCK_SIZE, 0, capture_stream>>>(d_a, d_b, d_c1, N);

    // End capture
    cudaStreamEndCapture(capture_stream, &graph);

    printf("Graph captured. Creating executable graph...\n");

    // Instantiate the graph (creates executable)
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);

    // ====== Replay the graph (no CPU launch overhead) ======
    printf("Replaying graph (iteration 1)...\n");
    cudaGraphLaunch(graph_exec, capture_stream);
    cudaStreamSynchronize(capture_stream);

    cudaMemcpy(h_c1_graph, d_c1, bytes, cudaMemcpyDeviceToHost);

    printf("Replaying graph (iteration 2)...\n");
    cudaGraphLaunch(graph_exec, capture_stream);
    cudaStreamSynchronize(capture_stream);

    cudaMemcpy(h_c2_graph, d_c1, bytes, cudaMemcpyDeviceToHost);

    // Compute reference results on CPU
    for (int i = 0; i < N; i++) {
        h_c1_ref[i] = h_a[i] + h_b[i];
    }

    // Verify
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (h_c1_graph[i] != h_c1_ref[i]) {
            match = false;
            printf("Mismatch at %d: graph=%f ref=%f\n", i, h_c1_graph[i], h_c1_ref[i]);
            break;
        }
    }
    printf("Graph replay correctness: %s\n", match ? "PASS" : "FAIL");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaStreamDestroy(capture_stream);

    // ====== Demonstrate whole-graph execution pattern ======
    // This models the "Piecewise CUDA Graph" concept from SGLang:
    // multiple sub-graphs are captured and replayed independently.
    printf("\n=== Piecewise Sub-Graph Demo ===\n");

    // Sub-graph 1: vec_add
    cudaGraph_t sub1_graph;
    cudaGraphExec_t sub1_exec;

    cudaStream_t s1;
    cudaStreamCreate(&s1);

    cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
    vec_add<<<grid_size, BLOCK_SIZE, 0, s1>>>(d_a, d_b, d_c1, N);
    cudaStreamEndCapture(s1, &sub1_graph);
    cudaGraphInstantiate(&sub1_exec, sub1_graph, NULL, NULL, 0);
    cudaGraphDestroy(sub1_graph);

    // Sub-graph 2: vec_add_scale
    cudaGraph_t sub2_graph;
    cudaGraphExec_t sub2_exec;

    cudaStream_t s2;
    cudaStreamCreate(&s2);

    float scale = 2.0f;
    cudaStreamBeginCapture(s2, cudaStreamCaptureModeGlobal);
    vec_add_scale<<<grid_size, BLOCK_SIZE, 0, s2>>>(d_a, d_b, d_c2, scale, N);
    cudaStreamEndCapture(s2, &sub2_graph);
    cudaGraphInstantiate(&sub2_exec, sub2_graph, NULL, NULL, 0);
    cudaGraphDestroy(sub2_graph);

    printf("Sub-graph 1 (vec_add) launch -> sub-graph 2 (vec_add_scale) launch\n");
    cudaGraphLaunch(sub1_exec, s1);
    cudaGraphLaunch(sub2_exec, s2);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    cudaMemcpy(h_c1_graph, d_c1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2_graph, d_c2, bytes, cudaMemcpyDeviceToHost);

    match = true;
    for (int i = 0; i < N; i++) {
        float expected1 = h_a[i] + h_b[i];
        float expected2 = (h_a[i] + h_b[i]) * 2.0f;
        if (h_c1_graph[i] != expected1 || h_c2_graph[i] != expected2) {
            match = false;
            printf("Sub-graph mismatch at %d\n", i);
            break;
        }
    }
    printf("Sub-graph results: %s\n", match ? "PASS" : "FAIL");

    // Cleanup
    cudaGraphExecDestroy(sub1_exec);
    cudaGraphExecDestroy(sub2_exec);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    free(h_a); free(h_b); free(h_c1_ref); free(h_c2_ref);
    free(h_c1_graph); free(h_c2_graph);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c1); cudaFree(d_c2);
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    if (prop.major < 8) {
        printf("This demo works on any GPU but CUDA Graph requires SM80+ for stream capture\n");
        printf("Current GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    }

    printf("Running CUDA Graph demo on: %s\n\n", prop.name);
    run_cuda_graph_demo();
    printf("\nDone.\n");

    return 0;
}
