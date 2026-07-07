# Scheduler / 采样 / 求解器

> 知识点扩展：DDPM/DDIM/DPM-Solver/Flow Matching 对比、采样步数影响，回扣 FastVideo scheduler。

## 1. 采样器解决什么

扩散反向过程是求解一个（随机）微分方程。采样器（scheduler/solver）决定如何离散化这个过程，即"每步怎么从 x_t 到 x_{t-1}"。核心权衡：**步数 vs 质量**。

## 2. 主流采样器对比

| 采样器 | 类型 | 步数需求 | 特点 |
|--------|------|---------|------|
| **DDPM** | 随机 | 1000 | 原始，慢 |
| **DDIM** | 确定性 | 50-100 | 加速，可跳步 |
| **DPM-Solver** | 高阶 ODE | 10-25 | 高阶，少步高质量 |
| **Flow Matching** | ODE（直线流） | 20-50 | 直线速度场，视频常用 |
| **UniPC** | 高阶多步 | 10-20 | 预测-校正 |

### 2.1 SDE vs ODE（随机 vs 确定性）

扩散反向过程可以写成两种等价形式：
- **反向 SDE**（随机微分方程）：每步注入随机噪声，DDPM 属于此类。多样性好但慢。
- **概率流 ODE**（probability flow ODE）：去掉随机项的确定性版本，同一起点必得同一结果。DDIM/DPM-Solver/Flow Matching 属于此类。可用高阶数值方法加速。

FastVideo 主要用 ODE 采样器（flow matching），因为可少步、可蒸馏、可复现（固定 seed 结果确定）。`FlowMatchEulerDiscreteScheduler` 也支持 `stochastic_sampling`（加随机项）。

### 2.2 数值求解器阶数

采样 = 对 ODE 数值积分。积分方法有阶数：
- **一阶（Euler）**：`x_{t-1} = x_t + dt·v`。简单，需较多步。
- **二阶/高阶（Heun / DPM-Solver-2 / UniPC）**：用多个点估计，单步更准，可减少总步数。
- **多步（multistep）**：复用前几步的预测（类似 Adams 方法），UniPC 属此类。

FastVideo 的 `FlowMatchEulerDiscreteScheduler` 是一阶 Euler；`FlowUniPCMultistepScheduler` 是高阶多步（Wan 默认，少步高质量）。

## 3. FastVideo 用 Flow Matching

主力：`FlowMatchEulerDiscreteScheduler`。Wan 默认 `FlowUniPCMultistepScheduler`（高阶）。

### Flow Matching 核心

学习从噪声到数据的"直线"路径的速度场：
```
x_t = (1-σ)·x_0 + σ·ε        # 直线插值
v = ε - x_0                   # 目标速度（常数）
x_{t-1} = x_t + dt·v          # Euler 积分
```

对比 DDPM 的曲线路径，flow matching 路径更直，需要更少步数。

### 源码

```
models/schedulers/scheduling_flow_match_euler_discrete.py
```
- `scale_noise`（L198）：加噪。
- `set_timesteps`（L285）：时间步生成 + 可选 shift/Karras schedule。
- `step`（L450）：Euler 一步。

## 4. flow_shift（视频关键参数）

```python
scheduler.set_shift(flow_shift)   # SchedulerLoader
```
调整 sigma 分布。高分辨率视频用较大 shift（Wan 常用 8.0），让采样更关注高噪声区，改善大分辨率生成质量。配置在 `PipelineConfig.flow_shift`。

### 4.1 为什么高分辨率要更大 shift

分辨率越高，同样的噪声水平对视觉的破坏越"局部"（信噪比不同）。shift 把时间步分布向高噪声端偏移，让模型在高噪声阶段（决定全局结构）花更多步。`set_timesteps` 里有两种 shift：
- **静态 shift**：`σ' = shift·σ / (1 + (shift-1)·σ)`。
- **动态 shift（dynamic shifting）**：根据序列长度 `mu` 自适应（`use_dynamic_shifting`），token 越多 shift 越大。

### 4.2 sigma schedule 变体

`set_timesteps`（L285）支持多种 sigma 采样：
- 均匀（默认）。
- **Karras sigmas**（`use_karras_sigmas`）：在低噪声端更密，改善细节。
- **exponential / beta sigmas**：其他非均匀分布。
这些影响"步数怎么分配到不同噪声水平"，是画质调参的旋钮。

## 5. 步数对速度/质量的影响

| 步数 | 场景 | 质量 |
|------|------|------|
| 50 | 标准扩散 | 高 |
| 20-25 | UniPC 高阶 | 高 |
| 1-4 | 蒸馏后（DMD） | 接近 |

FastVideo 加速的核心之一：通过蒸馏把步数从 50 降到 1-4（`DmdDenoisingStage` 用 3 步）。

## 6. 各 scheduler（源码）

| Scheduler | 文件 |
|-----------|------|
| `FlowMatchEulerDiscreteScheduler` | `scheduling_flow_match_euler_discrete.py` |
| `FlowUniPCMultistepScheduler` | `scheduling_flow_unipc_multistep.py` |
| `UniPCMultistepScheduler` | `scheduling_unipc_multistep.py` |
| `SelfForcingFlowMatchScheduler` | `scheduling_self_forcing_flow_match.py` |
| `RCMScheduler` | `scheduling_rcm.py` |

## 7. 去噪循环中的调用

```python
# stages/denoising.py:567
latents = scheduler.step(noise_pred, t, latents)[0]
```
scheduler 在去噪循环里每步调用一次，是唯一更新 latent 的地方。

## 8. Cosmos 的 EDM

Cosmos 用 EDM preconditioning（不同噪声参数化），`CosmosDenoisingStage` 单独实现。

## 9. 回扣源码
| 概念 | 源码 |
|------|------|
| flow matching | `scheduling_flow_match_euler_discrete.py` |
| flow_shift | `SchedulerLoader.load` + `PipelineConfig.flow_shift` |
| 步数 | `SamplingParam.num_inference_steps` |
| 去噪调用 | `stages/denoising.py:step` |

## 10. 延伸
- 采样流程：[`../03_core_flows/05_scheduler_and_sampling_flow.md`](../03_core_flows/05_scheduler_and_sampling_flow.md)
- 蒸馏（少步）：[`10_distillation_dmd_sparse_distill.md`](10_distillation_dmd_sparse_distill.md)
