# HPC拓扑、NUMA与多节点性能

## 单节点拓扑

```bash
lscpu -e
lstopo
numactl --hardware
numastat -p PID
taskset -c 0-7 ./program
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./program
```

first-touch决定常见页归属。远端访问高、带宽低时，应并行初始化并让计算线程与内存同node。绑定必须A/B验证，避免与IRQ和其他进程争核。

## 计数器与Roofline工具

- PAPI：可移植硬件计数器API。
- LIKWID：topology、pinning、MEM/FLOPS性能组。
- Intel PCM：socket、DRAM、UPI。
- Intel Advisor：vectorization与Roofline。
- AMD uProf：AMD CPU。
- Arm Performance Studio/Streamline：Arm/Jetson CPU。

```bash
papi_avail
likwid-topology
likwid-perfctr -C 0-7 -g MEM_DP ./program
pcm-memory 1
```

## 多节点

HPCToolkit、TAU、Score-P/Vampir用于调用路径和MPI/OpenMP trace；mpiP用于轻量MPI统计；IMB和OSU用于通信基准。

```bash
mpirun -n 4 ./IMB-MPI1 PingPong
mpirun -n 2 osu_latency
mpirun -n 2 osu_bw
```

关注communication/computation ratio、collective、message size、load imbalance、network latency/bandwidth和NUMA。本机无MPI与多NUMA环境，因此只提供方案，不伪造实验结果。

## Strong与Weak Scaling

Strong scaling固定总问题规模增加资源；Weak scaling让每个进程工作量固定并随资源扩大总规模。两者回答不同问题。

```text
Parallel Efficiency = Speedup / Resource Count
```

线程增加后效率下降可能来自串行比例、通信、同步、带宽饱和、NUMA remote access或负载不均衡。

## MPI通信诊断

先测平台基线，再测应用：

- OSU latency/bandwidth给点对点能力；
- IMB测试collective与常见模式；
- mpiP找MPI函数时间占比；
- Score-P/Vampir看跨rank时间线；
- HPCToolkit/TAU连接调用路径和硬件事件。

小消息常受latency限制，大消息偏bandwidth；collective性能依拓扑和算法。

## NUMA First Touch实验设计

在多node机器上：

1. 线程固定node0；
2. 分别在node0/node1初始化内存；
3. 保持计算工作量一致；
4. 比较local/remote bandwidth、latency和numastat；
5. 重新用并行first-touch验证。

不要用numactl命令结果代替程序内部实际页归属。

## OpenMP Affinity

OMP_PLACES定义可放置位置，OMP_PROC_BIND定义线程与位置关系。close适合共享cache的紧密协作，spread可能提高总带宽。最佳策略依工作负载，应测量。

## HPC与AI推理交叉

大模型推理也具有HPC问题：NUMA到GPU拓扑、PCIe/NVLink、CPU feeding、线程池、collective通信、memory bandwidth和负载均衡。多GPU吞吐高不代表单请求P99好。

## 练习

1. 绘制线程数到speedup/efficiency曲线。
2. 比较OMP_PROC_BIND close与spread。
3. 在多NUMA机器复现local/remote；单node机器不伪造结果。
4. 用hwloc标出CPU、NUMA、GPU和PCIe关系。

## 可选libnuma目标

CMake检测到numa.h和libnuma时构建`33_numa_local_remote`。单node机器安全跳过。

```bash
./src/scripts/run_numa_target.sh
# 手动：
./src/build/33_numa_local_remote 0 0 256 8
./src/build/33_numa_local_remote 0 1 256 8
```

参数依次为CPU node、memory node、MiB和iteration。程序绑定CPU、在指定node分配内存、warm-up、输出GB/s与checksum。
