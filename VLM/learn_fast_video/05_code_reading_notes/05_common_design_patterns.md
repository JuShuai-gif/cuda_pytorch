# 常见设计模式

> FastVideo 反复使用的设计模式。认识它们能加速读任何模块。
>
> 阅读提示：FastVideo 大量借鉴 **vLLM** 的工程范式（GroupCoordinator、Executor/Worker、参数类、registry）。若你读过 vLLM，会感到熟悉；没读过也没关系，本文逐一拆解。

## 1. Facade（门面）

`VideoGenerator` 隐藏 Executor/Worker/Pipeline 的复杂度，用户只看到 `from_pretrained` / `generate_video`。

**识别**：一个简单类背后是复杂子系统。

## 2. Registry + Lazy Import（注册表 + 延迟导入）

model / pipeline / config / metric 全部通过注册表 + `EntryClass` 约定 + AST 扫描发现，用 `_LazyRegisteredModel` 延迟导入。

**为什么**：避免主进程过早 import 触发 CUDA；解耦（新增模型不改注册中心）。

**源码**：`models/registry.py`, `pipelines/pipeline_registry.py`, `registry.py`, `eval/metrics/__init__.py`。

**识别**：`EntryClass = XxxClass`；`walk_packages`/`os.walk`+`ast.parse`；`importlib.import_module`。

## 3. Composed Pipeline（组合式管线）

pipeline = stage 列表，每个 stage 输入输出同一个 `ForwardBatch`。

**为什么**：支持十几个模型而不写巨型类；stage 可复用、可测试。

**源码**：`pipelines/composed_pipeline_base.py:forward` 的 `for stage in stages`。

**识别**：`add_stage`；stage 的 `forward(batch, args) -> batch`。

## 4. Strategy（策略）—— 后端选择

attention 后端、executor 后端、quant 方法都通过工厂选择具体实现。

**源码**：`attention/selector.py`（后端）、`worker/executor.py:get_class`（executor）、`quant_config.get_quant_method`（量化）。

**识别**：抽象基类 + 多实现 + 运行时选择。

## 5. Data Carrier（数据载体）

`ForwardBatch` 贯穿所有 stage，携带全部中间状态。pipeline 本身 stateless。

**为什么**：状态集中，pipeline 可复用；便于调试（打印 batch 即知全貌）。

**源码**：`pipelines/pipeline_batch_info.py:ForwardBatch`。

## 6. Template Method（模板方法）

`PipelineStage.__call__` 定义 `verify_input → forward → verify_output` 骨架，子类只实现 `forward`。`TrainingMethod` 类似。

**源码**：`pipelines/stages/base.py:__call__`。

## 7. Lazy Proxy（延迟代理）—— __getattr__

`ModelConfig.__getattr__` 代理到 `arch_config`；`WorkerWrapperBase.__getattr__` 代理到 `worker`。

**源码**：`configs/models/base.py`, `worker/worker_base.py`。

## 8. SPMD + RPC（多进程编排）

主进程 executor 通过 `collective_rpc` 向所有 worker 子进程广播命令，SPMD 执行。

**源码**：`worker/multiproc_executor.py:collective_rpc`。

## 9. Autograd-aware 通信

所有分布式通信原语（AllToAll4D/AllReduce/Slice）都是 `autograd.Function`，支持训练反向。

**源码**：`distributed/device_communicators/base_device_communicator.py:DistributedAutograd`。

## 10. Custom Op 包装（torch.compile 兼容）

不可追踪的 kernel（flash-attn）用 `torch.library.custom_op` 包装成可追踪算子。

**源码**：`attention/backends/flash_attn.py:65`。

## 11. Fallback Chain（回退链）

attention 后端不支持 → SDPA；kernel 未编译 → Triton；视频解码 decord → PyAV → torchvision。

**为什么**：跨硬件/环境健壮性。

**源码**：`selector.py`, `ops.py`（`sta_fwd is None`）, `eval/io/video.py`。

## 12. Config-driven Assembly（配置驱动组装）—— 新训练栈

`_target_` 全限定类路径 + `instantiate` 从 YAML 动态构建 Model/Method/Callback（类 Hydra）。

**源码**：`train/utils/instantiate.py`, `train/utils/builder.py`。

## 13. 反模式（AGENTS.md 明确禁止）

- 单文件写完整 pipeline（应组装 stages）。
- 在 stage 里读 `os.getenv`（应从 config 读）。
- module-level dict 跨 stage 传状态（应用 ForwardBatch）。

## 14. 识别模式加速读码

读到陌生模块时先问：
- 是不是 registry？找 `EntryClass`。
- 是不是 stage？找 `forward(batch)`。
- 是不是后端选择？找 selector/factory。
- 状态在哪？多半在 `ForwardBatch` 或 config。
