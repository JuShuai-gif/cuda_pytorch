#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc,char** argv){
    printf("%s Starting ...\n",argv[0]);
    int deviceCount = 0;
    
    // 获取设备数量
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n -> %s\n",
        (int)error_id,cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }else{
        printf("Detected %d CUDA Capable device(s)\n",deviceCount);
    }

    int dev = 0,driverVersion = 0,runtimeVersion=0;

    // 设置设备
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    
    printf("Device %d:\"%s\"\n",dev,deviceProp.name);
    // 驱动版本
    cudaDriverGetVersion(&driverVersion);
    // 运行库版本
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n",
        driverVersion/1000,(driverVersion%100)/10,
        runtimeVersion/1000,(runtimeVersion%100)/10);

        // 计算能力
    printf("  CUDA Capability Major/Minor version number:   %d.%d\n",
        deviceProp.major,deviceProp.minor);

// 显存总共多少
    printf("  Total amount of global memory:                %.2f GBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/pow(1024.0,3),deviceProp.totalGlobalMem);
    
    // GPU 时钟周期
            printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
    
    // 内存宽度
            printf("  Memory Bus width:                             %d-bits\n",
            deviceProp.memoryBusWidth);
// L2 缓存大小
    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                            	%d bytes\n",
                deviceProp.l2CacheSize);
    }
    printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1]
            ,deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
            deviceProp.maxTexture1DLayered[0],deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);

            // 常量内存大小
    printf("  Total amount of constant memory               %lu bytes\n",
            deviceProp.totalConstMem);

            // 每个块的共享内存
    printf("  Total amount of shared memory per block:      %lu bytes\n",
            deviceProp.sharedMemPerBlock);

            // 每个块可获取的寄存器数量
    printf("  Total number of registers available per block:%d\n",
            deviceProp.regsPerBlock);

            // warp大小
    printf("  Wrap size:                                    %d\n",deviceProp.warpSize);

// 一个SM中最多有多少线程
    printf("  Maximun number of thread per multiprocesser:  %d\n",
            deviceProp.maxThreadsPerMultiProcessor);

// 每个块最多有多少线程
    printf("  Maximun number of thread per block:           %d\n",
            deviceProp.maxThreadsPerBlock);

// 每个维度最多有多少线程
    printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);


    printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
            deviceProp.maxGridSize[0],
	    deviceProp.maxGridSize[1],
	    deviceProp.maxGridSize[2]);


    printf("  Maximu memory pitch                           %lu bytes\n",deviceProp.memPitch);
    printf("----------------------------------------------------------\n");


    printf("Number of multiprocessors:                      %d\n", deviceProp.multiProcessorCount);
    printf("Total amount of constant memory:                %4.2f KB\n",
	deviceProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block:        %4.2f KB\n",
     deviceProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers available per block:  %d\n",
    deviceProp.regsPerBlock);
    printf("Warp size                                       %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);

    /*
    指一个 SM（Streaming Multiprocessor，多处理器） 上同时能驻留的最大线程数
    比如某个 GPU 的每个 SM 最多只能同时容纳 1536 个线程
    这决定了 occupancy（硬件利用率）

    如果你启动一个 block 里面有 512 个线程，那么在一个 SM 上可以同时驻留 1536 / 512 = 3 个 block
    如果 block 里面有 1024 个线程，那么一个 SM 上只能驻留 1 个 block（因为再放一个就超过 1536 了）

    SM (Streaming Multiprocessor)：GPU 上的计算单元，一个 SM 可以同时驻留多个 block。

        Warp：32 个线程组成的执行单位（同时执行一条指令）

        利用率（occupancy）指的是：一个 SM 实际驻留的 warp 数 / 该 SM 最大能容纳的 warp 数

        每个 block 最多 1024 线程（= 32 个 warp）。每个 SM 最多 1536 线程（= 48 个 warp）。所以一个 SM 最多只能放下 48 个 warp（1536 / 32）

        所以说每个块的线程并不是越多越好

        情况 A：blockDim = 512 (16 warp)

每个 SM 能放下 1536 / 512 = 3 个 block。

所以一个 SM 上能放 3 × 16 = 48 warp。

正好满载 = 100% occupancy。

情况 B：blockDim = 1024 (32 warp)

每个 SM 能放下 1536 / 1024 = 1 个 block（再放就超了）。

所以一个 SM 上能放 1 × 32 = 32 warp。

实际占用率 = 32 / 48 = 66% occupancy。

情况 C：blockDim = 256 (8 warp)

每个 SM 能放下 1536 / 256 = 6 个 block。

一共能放 6 × 8 = 48 warp。

也是 100% occupancy。
    */
    printf("Maximum number of threads per multiprocessor:  %d\n",
	deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor:     %d\n",
	deviceProp.maxThreadsPerMultiProcessor/32);
    return EXIT_SUCCESS;
    
    


}
