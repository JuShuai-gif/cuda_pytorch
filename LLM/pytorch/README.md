# PyTorch 源码笔记 — 由浅及深学习路径

> 39 个知识点，按依赖关系编号。01-10 基础 → 11-20 进阶 → 21-30 编译 → 31-39 高阶。

## 01-03 基础设施

| # | 目录 | 要点 |
|---|------|------|
| 01 | [tensor](./01_tensor/) | Tensor 三层结构、view/stride/storage、contiguous、channels_last |
| 02 | [device_copy](./02_device_copy/) | to()/copy_() 语义、DeviceGuard/StreamGuard、pinned memory、P2P |
| 03 | [cuda_stream](./03_cuda_stream/) | CUDA stream/event、同步、CUDA Graph、流优先级 |

## 04-07 模型构建

| # | 目录 | 要点 |
|---|------|------|
| 04 | [module](./04_module/) | nn.Module、Parameter/Buffer、hook、state_dict |
| 05 | [dataloader](./05_dataloader/) | DataLoader 多进程、collate_fn、pin_memory、worker_init_fn |
| 06 | [serialization](./06_serialization/) | torch.save/load、pickle/zip 格式、weights_only |
| 07 | [checkpoint_format](./07_checkpoint_format/) | state_dict、SafeTensors、tied weights、FSDP shard、LoRA |

## 08-11 训练基础

| # | 目录 | 要点 |
|---|------|------|
| 08 | [optimizer](./08_optimizer/) | Optimizer.step()、state_dict、param_groups |
| 09 | [rng](./09_rng/) | CPU/CUDA Generator、Philox、checkpoint 随机数、reproducibility |
| 10 | [amp](./10_amp/) | Automatic Mixed Precision、autocast、Conv-BN 融合 |
| 11 | [grad_scaler](./11_grad_scaler/) | Gradient scaling、overflow 检测、unscale |

## 12-13 自动微分

| # | 目录 | 要点 |
|---|------|------|
| 12 | [autograd](./12_autograd/) | Autograd Engine、Node 拓扑、backward()、高阶导数 |
| 13 | [checkpoint](./13_checkpoint/) | Activation checkpoint、reentrant vs non-reentrant |

## 14-16 算子体系

| # | 目录 | 要点 |
|---|------|------|
| 14 | [dispatcher](./14_dispatcher/) | DispatchKey/DispatchKeySet、OperatorEntry、TLS exclude |
| 15 | [torchgen](./15_torchgen/) | native_functions.yaml→Register*.cpp、Composite*、structured kernel |
| 16 | [custom_ops](./16_custom_ops/) | torch.library、TensorIterator、meta kernel、backend override |

## 17-18 函数变换

| # | 目录 | 要点 |
|---|------|------|
| 17 | [functorch](./17_functorch/) | vmap/BatchedTensor、grad、jacfwd/jacrev、per-sample grad |
| 18 | [functionalization](./18_functionalization/) | mutation→functional 改写、view_copy、alias annotation |

## 19-21 编译基础设施

| # | 目录 | 要点 |
|---|------|------|
| 19 | [meta_fake_tensor](./19_meta_fake_tensor/) | Meta tensor、FakeTensorMode、meta kernel、SymInt/shape guard |
| 20 | [fx_graphs](./20_fx_graphs/) | symbolic_trace、FX Graph IR、subgraph_rewrite |
| 21 | [graph_passes](./21_graph_passes/) | Graph optimization pass、pattern matching、DCE |

## 22-27 编译流水线

| # | 目录 | 要点 |
|---|------|------|
| 22 | [dynamo](./22_dynamo/) | 字节码帧拦截、VariableTracker、guard、graph break |
| 23 | [torch_compile](./23_torch_compile/) | 三层流水线（Dynamo→AOTAutograd→Inductor）、mode 选择 |
| 24 | [aot_autograd](./24_aot_autograd/) | joint graph tracing、partitioner、decomposition |
| 25 | [inductor](./25_inductor/) | Inductor IR lowering、scheduler fusion、buffer reuse |
| 26 | [triton_kernel](./26_triton_kernel/) | 自定义 Triton kernel、torch.compile 集成 |
| 27 | [compile_debug](./27_compile_debug/) | graph break 排查、recompilation 诊断、精度对比 |

## 28-30 性能

| # | 目录 | 要点 |
|---|------|------|
| 28 | [profiler](./28_profiler/) | torch.profiler、trace 导出、CUDA kernel 级分析 |
| 29 | [memory_allocator](./29_memory_allocator/) | CUDA caching allocator、fragmentation、expandable_segments |
| 30 | [sdpa_attention](./30_sdpa_attention/) | FlashAttention v2/v3 dispatch、causal mask |

## 31-33 量化与部署

| # | 目录 | 要点 |
|---|------|------|
| 31 | [quantization](./31_quantization/) | 对称/非对称量化、LLM.int8()、AWQ、SmoothQuant、GPTQ |
| 32 | [ptq_qat](./32_ptq_qat/) | Post-Training Quantization / Quantization-Aware Training |
| 33 | [deploy](./33_deploy/) | 模型部署、TorchScript/torch.export |

## 34-38 分布式

| # | 目录 | 要点 |
|---|------|------|
| 34 | [collective_operations](./34_collective_operations/) | NCCL all-reduce/gather/broadcast、ring vs tree |
| 35 | [torch_dist](./35_torch_dist/) | ProcessGroup、init_process_group、backend 选择 |
| 36 | [ddp_reducer](./36_ddp_reducer/) | DDP Reducer、bucket 分组、通信计算重叠、comm hook |
| 37 | [parallelism_strategies](./37_parallelism_strategies/) | DP/DDP/ZeRO/TP/PP、FSDP、sequence parallelism |
| 38 | [moe](./38_moe/) | Mixture of Experts、load balancing、expert parallelism |

## 附录

| # | 目录 | 要点 |
|---|------|------|
| 39 | [jax_scaling_book](./39_jax_scaling_book/) | JAX/TPU Roofline 分析练习 |

## 40-45 设计模式与底层

| # | 目录 | 要点 |
|---|------|------|
| 40 | [design_patterns](./40_design_patterns/) | RAII、Singleton、Observer、Wrapper — PyTorch C++ 设计模式 |
| 41 | [cpu_arch](./41_cpu_arch/) | CPU cache hierarchy、SIMD 向量化、NUMA、线程池 |
| 42 | [cuda_arch](./42_cuda_arch/) | GPU SM/warp、occupancy、shared memory、Tensor Core |
| 43 | [intrusive_ptr](./43_intrusive_ptr/) | c10::intrusive_ptr 侵入式引用计数、vs shared_ptr |
| 44 | [ivalue_type](./44_ivalue_type/) | IValue tagged union、JIT 类型系统、Type Erasure |
| 45 | [cudnn_backend](./45_cudnn_backend/) | cuDNN/cuBLAS 算法选择、benchmark、backend 调度 |
