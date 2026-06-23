# Profiling 工具速查(torch.profiler / nsys / ncu)

性能分析分三个粒度,先用上层工具定位热点,再用下层工具深挖原因。

## 1. 三个工具的分工

| 工具             | 粒度          | 看什么                              | 速度 |
| ---------------- | ------------- | ----------------------------------- | ---- |
| `torch.profiler` | 算子(op)级    | 哪个算子耗时多、CPU/GPU 时间分布    | 快   |
| `nsys`(Nsight Systems) | 系统/时间线级 | kernel 调度、CPU-GPU 重叠、空隙     | 中   |
| `ncu`(Nsight Compute)  | kernel 级     | 单个 kernel 内部的硬件效率(占用率/带宽/寄存器) | 慢   |

典型流程:`torch.profiler` / `nsys` 找到热点 kernel → `ncu` 深入分析该 kernel 为何慢。

---

## 2. ncu(Nsight Compute):kernel 级分析

### 基本命令

```bash
# 用默认基础指标分析全部 kernel, 结果打印到终端
ncu python train.py

# 采集完整指标集, 保存为报告文件(可在 Nsight Compute GUI 打开)
ncu --set full -o output $(which python) train.py
```

参数说明:

| 部分            | 作用                                                              |
| --------------- | ----------------------------------------------------------------- |
| `--set full`    | 采集完整指标集(寄存器、occupancy、内存、roofline 等),信息全但最慢 |
| `-o output`     | 结果存为 `output.ncu-rep`,可在 Nsight Compute GUI 可视化            |
| `$(which python)` | 用解释器绝对路径作为目标程序,避免 PATH 解析问题                    |

### 实用提示(必看)

ncu 会对每个 kernel **replay(多次重放)** 读硬件计数器,训练脚本里 kernel 极多,
直接 `--set full` 会慢到不可用。务必缩小范围:

```bash
# -k: 只分析名字匹配正则的 kernel; -c: 只采前 N 个 kernel
ncu --set full -k "regex_of_kernel" -c 10 -o output $(which python) train.py
```

也可在代码里圈定区间(配合 `ncu --profile-from-start off`):

```python
import torch
torch.cuda.cudart().cudaProfilerStart()
# ... 只分析这一段 ...
torch.cuda.cudart().cudaProfilerStop()
```

- 权限:遇到 `ERR_NVGPUCTRPERM` 需 root 运行,或让管理员开放 GPU 性能计数器权限。

---

## 3. nsys(Nsight Systems):时间线分析

```bash
# 生成 report.nsys-rep, 用 Nsight Systems GUI 打开看时间线
nsys profile -o report $(which python) train.py
```

适合看 CPU 与 GPU 是否重叠、kernel 之间有没有空隙、数据搬运是否阻塞计算。

---

## 4. torch.profiler:导出 JSON + Chrome 可视化

### 4.1 导出 chrome trace(JSON)

最简用法:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        torch.square(torch.randn(10000, 10000).cuda())

# 打印算子耗时排行
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# 导出 chrome trace 格式的 JSON
prof.export_chrome_trace("trace.json")
```

带 schedule 的训练循环用法(在回调里导出):

```python
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(f"/tmp/test_trace_{prof.step_num}.json")

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=trace_handler,
) as p:
    for _ in range(10):
        torch.square(torch.randn(10000, 10000).cuda())
        p.step()   # 推进 schedule 状态机, 必须放在循环里
```

### 4.2 用 Google Chrome 可视化 JSON

**方式 A:chrome://tracing(经典)**

1. 打开 Google Chrome,地址栏输入 `chrome://tracing`(Edge 用 `edge://tracing`)。
2. 点左上角 **Load**,选中导出的 `trace.json`。
3. 用 `W/S` 缩放、`A/D` 平移,点击算子方块看耗时详情。

**方式 B:Perfetto UI(推荐,更现代)**

1. 打开 <https://ui.perfetto.dev>(纯前端,trace 不会上传到服务器)。
2. 左上角 **Open trace file**,选 `trace.json`。
3. 功能比 chrome://tracing 更强(搜索、SQL 查询、火焰图)。

### 4.3 用 TensorBoard 可视化(可选)

```python
on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
```

```bash
pip install torch_tb_profiler
tensorboard --logdir ./log
# 浏览器打开 http://localhost:6006 , 进入 PyTorch Profiler 标签页
```

---

## 5. 一句话总结

- **找热点**:`torch.profiler` 看算子耗时表 + 导 JSON 用 Chrome/Perfetto 看时间线。
- **看调度**:`nsys` 看 CPU-GPU 时间线与空隙。
- **挖 kernel**:`ncu --set full` 看单个 kernel 的硬件效率(记得用 `-k`/`-c` 限范围)。
