# PyTorch、Transformer与AI推理性能

## 三级诊断

```text
torch.profiler（operator）→ nsys（CPU/CUDA timeline）→ ncu（慢kernel）
```

Demo：`07_operator_hotspot_bad_good.py`，包含Linear、Matmul、Softmax、LayerNorm、Attention-like和SwiGLU-like计算，并用assert_close验证Bad/Good。

```bash
python3 src/python/07_operator_hotspot_bad_good.py
nsys profile -t cuda,nvtx,osrt python3 src/python/07_operator_hotspot_bad_good.py
```

Self CPU/CUDA是算子自身归因时间；CPU/CUDA Total包含子调用。层级指标不能直接相加。

## GPU计时

`time.time(); model(x)`在CUDA异步执行下通常只测enqueue。正确方法：

```python
torch.cuda.synchronize()
start = time.perf_counter()
model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

start_event.record()
model(x)
end_event.record()
end_event.synchronize()
ms = start_event.elapsed_time(end_event)
```

## PyTorch显存

Demo：`08_memory_bad_good.py`比较clone/contiguous/temporary与inference buffer reuse。

关注：

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.memory._snapshot()  # 版本支持时
```

allocated是活跃tensor，reserved包含allocator cache，peak是观测窗口峰值。reserved高不等于leak。in-place/reuse不能破坏Autograd、alias或correctness语义。

## Transformer Kernel地图

| Kernel/算子 | 常见候选瓶颈 | 关键变量 |
|---|---|---|
| GEMM | Compute/Tensor Core | M/N/K、dtype、layout |
| RMSNorm | Bandwidth + Reduction | hidden size、fusion |
| Softmax | Reduction + Memory | sequence、实现 |
| Attention | GEMM/Memory/Capacity | batch、heads、sequence |
| SwiGLU/Elementwise | Memory/Launch | fusion、tensor规模 |
| KV Cache | Bandwidth/Capacity | sequence、dtype、layout |

这些不是固定结论。shape、dtype、hardware、kernel implementation会改变bound，必须Profiler验证。

## TensorRT与ONNX Runtime

```bash
trtexec --loadEngine=model.plan --warmUp=3000 --duration=30 --iterations=1000
trtexec --loadEngine=model.plan --useCudaGraph --noDataTransfers
trtexec --loadEngine=model.plan --dumpProfile --profilingVerbosity=detailed
```

关注Throughput、Latency、Enqueue、H2D、GPU Compute、D2H。`--noDataTransfers`只用于隔离传输，不代表E2E。

ONNX Runtime启用`SessionOptions.enable_profiling=True`，分析operator latency、Execution Provider fallback和数据copy。RKNN/TensorRT/ORT前处理必须保持resize、layout、dtype、量化和归一化一致。

## torch.profiler配置代价

record_shapes帮助识别异常shape，profile_memory定位tensor峰值，with_stack连接源码，但都增加开销。长服务应用schedule只采集少量active step，并用on_trace_ready导出TensorBoard trace。

```python
schedule(wait=2, warmup=2, active=5, repeat=1)
```

prof.step必须与迭代边界一致，否则schedule不会正确推进。

## Operator与Kernel映射

一个aten operator可能启动多个CUDA kernel；多个operator也可能被compile/fusion合成一个kernel。torch.profiler先回答模型语义热点，nsys回答实际时序，ncu只分析选中的慢kernel。

## CPU Overhead来源

PyTorch GPU慢不一定在GPU：

- Python循环和小operator dispatch；
- DataLoader或camera decode；
- shape/dtype转换；
- CPU tensor到pinned memory；
- 每步.item()触发同步；
- 动态shape导致重新compile；
- allocator和线程池初始化。

nsys中GPU gap前的CPU stack和CUDA API最有价值。

## Attention复杂度与显存

标准attention score矩阵随sequence平方增长。Flash/SDPA类实现通过tiling避免物化完整score，减少HBM流量。是否使用高效backend取决于dtype、head dimension、mask、硬件和框架版本。

## KV Cache

KV cache容量近似随batch、layers、heads、head_dim、sequence和bytes/dtype线性增长。Decode阶段每token计算量较小但需要读取大量历史KV，常表现为capacity/bandwidth限制。量化、分页KV和layout会改变行为。

## Precision

FP32、TF32、FP16、BF16、FP8和INT8具有不同吞吐、带宽和数值特性。比较前必须固定精度策略并做任务级correctness，不应仅比较kernel时间。

## torch.compile与CUDA Graph

compile可能融合operator并减少Python/launch overhead，但存在首次编译成本、graph break和动态shape重编译。分别报告cold compile、warm steady state和缓存命中后的性能。

## 显存峰值案例

```text
症状：allocated最终不高，但max_memory_allocated很高
profiler：clone/contiguous产生多个短命tensor
优化：统一layout，删除无意义clone，inference workspace复用
验证：peak下降、P99下降、assert_close通过
```

reserved不随tensor释放立刻下降通常是caching allocator行为，不应直接判leak。

## In-place风险

in-place可能破坏autograd保存的中间值、view alias、并发读或后续复用。仅在明确inference_mode、所有权和生命周期时使用；每次用assert_close和模型级精度验证。

## Transformer练习

1. 改变sequence为128/512/2048，记录attention latency和memory。
2. 比较FP32/BF16/FP16并验证误差。
3. 从torch.profiler Top operator追到nsys kernel。
4. 检查一次item()对timeline同步的影响。

## CUDA版PyTorch目标机实验

`src/python/10_cuda_inference_target_lab.py`提供Transformer-like Block、CUDA Event分布、NVTX、torch.profiler、显存统计、CSV原始样本和assert_close。

```bash
python3 src/python/10_cuda_inference_target_lab.py --batch 1 --sequence 512 --hidden 768 --heads 12 --dtype fp16 --iterations 200 --profile
nsys profile -t cuda,nvtx,osrt -o torch_target python3 src/python/10_cuda_inference_target_lab.py --iterations 100
```

当前CPU版PyTorch会安全跳过；换CUDA版后无需修改代码。
