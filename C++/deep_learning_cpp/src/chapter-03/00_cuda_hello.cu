/*
 * 00_cuda_hello.cu
 * 第 3 章：CUDA GPU 加速深度学习
 *
 * CUDA 环境验证与基本设备查询工具。
 * 在编写任何 GPU 加速代码之前，必须先确认 CUDA 运行时可用、
 * 设备已检测到、且内核启动路径正常工作。
 *
 * 涵盖的技术：
 *   - cudaGetDeviceCount：查询系统中可用的 CUDA 设备数量
 *   - cudaGetDeviceProperties：获取每块 GPU 的详细信息
 *     （名称、计算能力、多处理器数量、全局内存、最大线程数等）
 *   - 简单的 Hello World 内核启动：验证编译工具链和内核执行路径
 *   - cudaDeviceSynchronize + cudaGetLastError：检查内核启动错误
 *
 * 计算能力 (Compute Capability) 的含义：
 *   - major.minor 格式（如 8.6），决定支持的 CUDA 特性集
 *   - major=8, minor=6 → sm_86，支持 BF16 Tensor Core 等
 *   - 详情参见 NVIDIA CUDA C++ Programming Guide 附录
 */

#include <cstdio>
#include <cuda_runtime.h>

// ----------------------------------------------------------------
// Hello World 内核：每个线程打印其线程索引和所在块索引。
// 用于验证内核启动路径是否正常工作。
// ----------------------------------------------------------------
__global__ void helloFromGPU() {
    // 计算全局线程 ID：blockIdx * blockDim + threadIdx
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("  你好，来自 GPU 的线程 %d（块 %d，块内线程 %d）\n",
           tid, blockIdx.x, threadIdx.x);
}

// ----------------------------------------------------------------
// 查询并打印所有 CUDA 设备的信息。
// 对每个设备调用 cudaGetDeviceProperties 获取详细信息。
// ----------------------------------------------------------------
void queryAllDevices() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("  [错误] 无法获取设备数量: %s\n",
               cudaGetErrorString(err));
        return;
    }

    if (deviceCount == 0) {
        printf("  [警告] 未检测到任何 CUDA 设备。\n");
        printf("  请确认：\n");
        printf("    1. NVIDIA GPU 驱动已安装\n");
        printf("    2. CUDA 工具包版本与驱动兼容\n");
        return;
    }

    printf("  检测到 %d 个 CUDA 设备:\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("  ───────────────────────────────────────\n");
        printf("  设备 %d: %s\n", i, prop.name);
        printf("  ───────────────────────────────────────\n");
        printf("    计算能力:            %d.%d\n",
               prop.major, prop.minor);
        printf("    多处理器 (SM) 数量:   %d\n",
               prop.multiProcessorCount);
        printf("    全局内存总量:         %.2f GB (%zu 字节)\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.totalGlobalMem);
        printf("    每个块的最大线程数:   %d\n",
               prop.maxThreadsPerBlock);
        printf("    每个块的共享内存:     %zu KB\n",
               prop.sharedMemPerBlock / 1024);
        printf("    常量内存总量:         %zu 字节\n",
               prop.totalConstMem);
        printf("    Warp 大小:            %d\n",
               prop.warpSize);
        printf("    最大网格维度:         (%d, %d, %d)\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("    最大块维度:           (%d, %d, %d)\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("    时钟频率:             %.2f GHz\n",
               prop.clockRate / 1e6);
        printf("    是否支持统一寻址:     %s\n",
               prop.unifiedAddressing ? "是" : "否");
        printf("    是否支持并发内核:     %s\n",
               prop.concurrentKernels ? "是" : "否");
        printf("\n");
    }
}

// ----------------------------------------------------------------
// 启动一个简单的 Hello World 内核来验证内核执行路径。
// 使用 2 个块，每块 4 个线程，共 8 个线程。
// 每个线程会打印一行问候信息。
// ----------------------------------------------------------------
bool testKernelLaunch() {
    printf("  [测试] 正在启动 Hello World 内核（2 个块 × 4 个线程）...\n");

    helloFromGPU<<<2, 4>>>();

    // 等待 GPU 完成所有工作
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("  [错误] cudaDeviceSynchronize 失败: %s\n",
               cudaGetErrorString(syncErr));
        return false;
    }

    // 检查是否有内核启动错误
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("  [错误] 内核启动失败: %s\n",
               cudaGetErrorString(kernelErr));
        return false;
    }

    printf("  [通过] 内核执行成功。\n");
    return true;
}

// ----------------------------------------------------------------
// 主函数：依次执行设备查询和内核启动测试。
// ----------------------------------------------------------------
int main() {
    printf("=== CUDA 设备查询与内核启动测试 ===\n\n");

    // --- 查询 CUDA 设备 ---
    printf("[步骤 1] 查询 CUDA 设备信息\n");
    printf("--------------------------------------------------\n");
    queryAllDevices();

    // --- 测试内核启动 ---
    printf("[步骤 2] 测试 GPU 内核启动\n");
    printf("--------------------------------------------------\n");
    bool kernelOk = testKernelLaunch();

    // --- 总结 ---
    printf("\n--------------------------------------------------\n");
    if (kernelOk) {
        printf("[总结] CUDA 环境一切正常，可以开始 GPU 编程。\n");
    } else {
        printf("[总结] CUDA 内核启动测试未通过，请检查驱动和工具链。\n");
    }

    return kernelOk ? 0 : 1;
}
