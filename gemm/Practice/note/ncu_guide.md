# NVIDIA Nsight Compute (ncu) 使用指南

## 编译选项

### 基本编译
```bash
nvcc -o test1 test1.cu -O3
```

### 生成PTX中间代码
```bash
nvcc -o test1 test1.cu -O3 -ptx
```
生成 `.ptx` 文件，可查看SASS指令。

### 生成CUDA可执行文件（arch和code指定）
```bash
nvcc -o test1 test1.cu -O3 -arch=sm_80 -code=sm_80
```

### 生成详细编译日志（含register、shared memory信息）
```bash
nvcc -o test1 test1.cu -O3 -Xptxas -v
```

### 生成各阶段编译产物
```bash
# 生成cubin（CUDA二进制）
nvcc -o test1 test1.cu -O3 -cubin

# 生成fatbin
nvcc -o test1 test1.cu -O3 -fatbin

# 生成ptx
nvcc -o test1 test1.cu -O3 -ptx
```

### 查看SASS（机器码）
```bash
# 方法1: cuobjdump
nvcc -o test1 test1.cu -O3
cuobjdump -sass ./test1

# 方法2: nvdisasm
nvcc -o test1 test1.cu -O3 -cubin
nvdisasm ./test1.cubin
```

### 关键编译选项

| 选项 | 说明 |
|------|------|
| `-Xptxas -v` | 显示寄存器、shared memory、constant memory使用情况 |
| `-cubin` | 生成CUDA二进制文件 |
| `-ptx` | 生成PTX中间代码 |
| `-arch=sm_xx` | 指定GPU架构 |
| `-code=sm_xx` | 指定生成的cubin架构 |
| `-lineinfo` | 生成行号信息，用于源码级别profiling |
| `-dlto` | 启用Device Link Time Optimization |

### 推荐编译命令（用于profiling）
```bash
nvcc -o test1 test1.cu -O3 -Xptxas -v -lineinfo
```

### 运行时查看寄存器/SM信息
```bash
# 使用nvidia-smi查看实时SM利用率
nvidia-smi -l 1

# 查看CUDA设备信息
nvidia-smi -L

# 查看详细设备属性
ncu --query-gpu-metrics
```

## 基本使用

sudo /usr/local/cuda/bin/ncu --set full ./sgemm 1

### 1. 运行ncu分析

```bash
ncu --set full ./test1
```

这会运行程序并收集所有可用的性能指标。

### 2. 只运行kernel分析

```bash
ncu --kernel-name gemm_naive_gpu ./test1
```

### 3. 输出到文件

```bash
ncu --set full -o profile_result ./test1
```

生成的文件可以用 ncu-ui 打开查看。

## 常用命令

### 收集特定指标

```bash
# GPU核心时间
ncu --metrics smsp__average_runtime_extension.pct ./test1

# 内存相关
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
           l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
           dram__bytes.sum ./test1

# 计算吞吐
ncu --metrics sm__inst_executed.sum \
           sm__inst_fp32.sum \
           sm__inst_fp64.sum ./test1

# 屋顶线分析
ncu --set roofline ./test1
```

### 常用指标分类

**执行效率**
- `smsp__average_runtime_extension.pct` - SM运行时间占比
- `sm__warps_active.avg.pct_of_peak_sustained` - 活跃warp占比

**内存**
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` - 全局内存加载
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` - 全局内存存储
- `dram__bytes.sum` - DRAM带宽
- `l2__bytes.sum` - L2缓存带宽

**计算**
- `sm__inst_executed.sum` - 总指令数
- `sm__inst_fp32.sum` - FP32指令数
- `sm__inst_fp64.sum` - FP64指令数

**Warp效率**
- `sm__warps_active.avg.pct_of_peak_sustained` - 活跃warp
- `sm__issue_active.avg.pct_of_peak_sustained` - 发射效率

## 分析重点指标

### 1. 内存带宽
```bash
ncu --metrics dram__bytes.sum,gpu__time_duration ./test1
```
计算: `dram__bytes.sum / gpu__time_duration / 1e9` GB/s

### 2. 计算效率
```bash
ncu --metrics sm__inst_fp32.sum,gpu__time_duration ./test1
```

### 3. L1/L2缓存命中率
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
           l1tex__t_sectors_pipe_lsu_mem_global_op_ld.l2_hit_ratio.pct \
           l2__t_sectors_pipe_lsu_mem_global_op_ld.sum \
           l2__t_sectors_pipe_lsu_mem_global_op_ld.l2_hit_ratio.pct ./test1
```

### 4. 屋顶线分析
```bash
ncu --set roofline ./test1
```

## GUI查看

```bash
ncu-ui profile_result.ncu-rep
```

## 建议的分析流程

1. 先运行基本profile: `ncu --set full ./test1`
2. 查看GPU kernel耗时占比
3. 检查内存带宽是否达到峰值
4. 分析L1/L2缓存效率
5. 查看warp效率，找出瓶颈
