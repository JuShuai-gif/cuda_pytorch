# VLA Profiler 使用指南

一套面向 **Vision-Language-Action（VLA）策略** 的工业级分析器，对标
NVIDIA Nsight / TensorRT profiler / Torch profiler，专门针对 SmolVLA / pi0.5
这类模型，覆盖：参数/MACs 模块拆分、理论与实测延迟、Roofline、动作 chunk
rollout、KV cache 带宽、多相机、ROS 实时性，以及 kernel 级 profiling。

---

## 1. 它能告诉你什么

- **哪一部分最耗算力**：vision / language / fusion / action 的 params% 与 MACs%
- **哪里是 memory-bound**：Roofline 判定 + arithmetic intensity
- **为什么 robot 会卡顿**：理论延迟 vs 实测延迟、efficiency、P99
- **chunk action 的成本**：50 步 chunk 摊销了多少、能否掩盖推理延迟
- **kernel 级热点**：哪个 CUDA kernel（addmm / GEMM / attention bmm）最慢
- **剪枝/量化收益方向**：compute-bound 还是 memory-bound，对症下药

---

## 2. 环境与依赖

工具位于 `project/vla_profiler/`，运行前确保以下依赖（已在 `lerobot_ghr` 环境装好）：

```bash
# 核心
torch torchvision numpy matplotlib
# MACs 后端（任一可用即可，auto 会自动回退）
fvcore torchprofile ptflops thop
# 如缺失：
uv pip install fvcore torchprofile ptflops thop
```

Nsight 工具（`ncu` / `nsys`）为可选，仅 `--run-ncu / --run-nsys` 时需要。

---

## 3. 快速开始（CLI）

> 所有命令都在 `project/` 目录下，用 `python -m vla_profiler.main` 运行。

```bash
cd project

# 最简：A100 FP16 上分析内置 705M 合成模型
python -m vla_profiler.main --gpu a100 --precision fp16

# 不测延迟（纯理论，无需 GPU）
python -m vla_profiler.main --no-measure --device cpu

# 一次性产出全部：文本报告 + markdown + roofline 图 + kernel 分解 + trace
python -m vla_profiler.main \
  --gpu a100 --precision fp16 --chunk-steps 50 \
  --markdown report.md \
  --plot roofline.png \
  --kernels --trace trace.json \
  --print-ncu
```

---

## 4. CLI 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `synthetic` | 仅内置合成模型；真实模型请用 Python API |
| `--preset` | `705M` | 合成模型规格 |
| `--gpu` | `a100` | `a100` / `h100` / `rtx4090` / `jetson_orin` / `jetson_nano` |
| `--precision` | `fp16` | `fp32`/`tf32`/`fp16`/`bf16`/`fp8`/`int8`（按 GPU 支持） |
| `--device` | `cuda` | `cuda` / `cpu`（无 GPU 自动回退 CPU） |
| `--batch` | `1` | 推理 batch；机器人通常 1 |
| `--backend` | `auto` | MACs 后端 `auto`→`fvcore`→`torchprofile`→`ptflops`→`thop`→`hook` |
| `--chunk-steps` | `50` | 动作 chunk 步数 |
| `--control-hz` | `30` | 机器人控制频率（用于 ROS 实时性判定） |
| `--num-cameras` | `1` | 相机路数（多相机会放大 vision 成本） |
| `--resolution` | `224` | 输入分辨率（影响 vision token 数） |
| `--warmup` / `--repeat` | `20` / `50` | 延迟测量预热/重复次数 |
| `--no-measure` | off | 跳过实测延迟，仅理论 |
| `--markdown PATH` | — | 导出 markdown 报告 |
| `--plot PATH` | — | 导出 roofline + 模块条形 PNG |
| `--kernels` | off | 跑 torch.profiler kernel 分解 |
| `--kernel-top N` | `15` | 显示最热的 N 个 kernel |
| `--kernel-steps N` | `20` | kernel profiling 迭代次数 |
| `--trace PATH` | — | 导出 Chrome/Perfetto trace（隐含 `--kernels`） |
| `--print-ncu` | off | 打印并检测 ncu / nsys 命令 |
| `--run-ncu` / `--run-nsys` | off | 直接拉起 Nsight Compute / Systems |

GPU 预设（峰值算力 TFLOPs / 带宽 GB/s）：

| GPU | 带宽 | fp32 | tf32 | fp16/bf16 | fp8 | int8 |
|-----|------|------|------|-----------|-----|------|
| a100 | 2039 | 19.5 | 156 | 312 | — | — |
| h100 | 3350 | 67 | 495 | 989 | 1979 | — |
| rtx4090 | 1008 | 82.6 | — | 165 | — | — |
| jetson_orin | 204 | — | — | 137 | — | 275 |
| jetson_nano | 25.6 | — | — | 0.47 | — | — |

---

## 5. 报告字段怎么读

```text
[Latency Estimate]
Theoretical Latency : 0.78 ms     # 理想下界（峰值算力跑满）
Measured Latency    : 21.80 ms    # 实测（CUDA Event 计时）
Efficiency          : 3.6%        # 理论/实测，越低越说明被内存/launch 拖住
```

- **Efficiency < 10%** → batch=1 机器人推理典型：kernel launch + 内存延迟主导，
  靠减 FLOPs 无效，应做 kernel fusion / 量化减带宽 / 增大 batch。
- **Roofline regime = memory-bound** → AI 在 ridge 左侧，优先减「字节搬运」而非 FLOPs。
- **Action Chunk Rollout / Amortization** → chunk_once 相比逐步重规划省多少倍。
- **KV Cache / Bandwidth-bound** → attention 解码是否被带宽限制。
- **ROS Latency Coupling** → 端到端（compute+sensor+actuation）能否满足控制频率，
  以及 chunk 是否能掩盖推理延迟（`Chunk hides latency`）。

---

## 6. Python API（接入真实 SmolVLA / pi0 模型）

CLI 只跑内置合成模型；**真实模型用 API**，只要能 `model(*inputs)` 前向即可。

```python
from vla_profiler import VLAProfiler, ProfilerConfig
from vla_profiler.report import render_text, save_markdown
from vla_profiler.plot import save_roofline_plot

# 1. 你的真实模型 + 一组 dummy 输入（与 forward 签名一致的 tuple）
model = load_my_smolvla()                  # 任意 nn.Module
dummy = (images, lang_tokens)              # forward(images, lang_tokens)

# 2. 配置目标硬件 + VLA 参数
cfg = ProfilerConfig(
    gpu_name="A100", precision="fp16",
    gpu_tflops=312.0, bandwidth_gbps=2039.0,
    device="cuda",
    chunk_steps=50, control_hz=30.0, num_cameras=2,
    # 可选：让 KV cache 分析生效（填 fusion transformer 几何）
    kv_layers=18, kv_heads=8, kv_head_dim=128, kv_seq_len=256,
)

# 3. 一行出结果
result = VLAProfiler(model, cfg).run(dummy)

print(render_text(result))
save_markdown(result, "report.md")
save_roofline_plot(result, "roofline.png")

# 4. 直接取数值
print(result.macs.total_macs, result.latency.efficiency)
print(result.macs.category_fraction)       # {'vision':.., 'fusion':.., ...}
```

### 模块拆分关键字

`module_splitter` 按参数名关键字归类，已内置覆盖常见命名
（`siglip` / `dino` / `gemma` / `qwen` / `connector` / `expert` …）。
若你的模型命名特殊，用 `SplitConfig.overrides` 强制指定顶层前缀：

```python
from vla_profiler import SplitConfig, ProfilerConfig

split = SplitConfig(overrides={
    "model.visual": "vision",
    "model.llm":    "language",
    "policy_head":  "action",
})
cfg = ProfilerConfig(split_config=split, ...)
```

---

## 7. Roofline 绘图

```bash
python -m vla_profiler.main --plot roofline.png
```

或 API：`save_roofline_plot(result, "roofline.png")`。输出双图：

- **左**：log-log Roofline，内存屋顶 + 计算屋顶 + ridge point + 模型工作点
  （memory-bound 红点 / compute-bound 绿点）
- **右**：vision/language/fusion/action 的 Params% vs MACs% 对比条形

---

## 8. Kernel 级 Profiling 与 Nsight

### torch.profiler（进程内，直接出表）

```bash
python -m vla_profiler.main --kernels --kernel-top 10 --trace trace.json
```

输出最热 kernel（self CUDA time / 占比 / 调用次数 / GFLOPs），例如
`aten::addmm`（线性层 GEMM）、cutlass kernel、`aten::bmm`（attention）。
`--trace` 导出的 `.json` 可在 `chrome://tracing` 或 `nsys-ui` 打开时间线。

> 提示：trace 文件可能上百 MB，建议输出到 `/tmp` 避免污染仓库。

### Nsight Compute / Systems（进程外，SM 级指标）

```bash
# 仅打印 + 检测可用性
python -m vla_profiler.main --print-ncu

# 直接拉起（较慢，ncu 需要相应权限）
python -m vla_profiler.main --run-ncu     # Nsight Compute（occupancy / 内存吞吐 / warp stall）
python -m vla_profiler.main --run-nsys    # Nsight Systems（CUDA/NVTX 时间线）
```

API 取命令：

```python
from vla_profiler import build_ncu_command, build_nsys_command, has_ncu
print(has_ncu(), " ".join(build_ncu_command(output="vla_ncu")))
```

---

## 9. 模块一览

| 文件 | 职责 |
|------|------|
| `module_splitter.py` | 按关键字把参数归到 vision/language/fusion/action |
| `model_analyzer.py` | 模块级参数统计 + 按真实 dtype 算模型大小 |
| `macs_analyzer.py` | 多后端 MACs：fvcore → torchprofile → thop → hook 回退 |
| `latency_estimator.py` | 理论延迟 + CUDA Event 实测（mean/p50/p99/efficiency） |
| `roofline.py` | arithmetic intensity / ridge / compute-vs-memory 比例 |
| `vla_extensions.py` | chunk rollout / KV cache / 多相机 / ROS 耦合建模 |
| `profiler.py` | `VLAProfiler` 主编排 + 自动瓶颈分析 |
| `report.py` | 文本 + markdown 报告渲染 |
| `plot.py` | roofline + 模块条形 PNG |
| `kernel_profiler.py` | torch.profiler kernel 分解 + ncu/nsys 命令 |
| `main.py` | CLI 入口 |
| `models/synthetic_vla.py` | 内置 ~705M SmolVLA 风格合成模型 |

---

## 10. 注意事项

- **MACs 单位**：所有后端统一返回 **MACs**（1 乘加）。fvcore 内部 1 MAC = 1 flop，
  其 `total()` 已是 MACs，无需再除 2。论文里的 "FLOPs" 多为 2×MACs，对比前先对齐口径。
- **GPU 计时**：必须用 CUDA Event（工具已内置），切勿用 `time.perf_counter()` 测 GPU。
- **Unsupported Ops**：报告会列出 fvcore 未统计的算子（softmax/gelu/add 等 elementwise
  属正常未计；若出现大头 matmul/bmm 说明 MACs 被低估，需注意）。
- **真实模型 trace 限制**：含动态控制流的模型 fvcore/thop 可能 trace 失败，
  会自动回退到 hook 估算；实测延迟与 kernel 分解不受影响。
