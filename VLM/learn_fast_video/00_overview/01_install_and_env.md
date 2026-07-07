# 安装与环境

> 目标：把 FastVideo 跑起来，并理解它的依赖结构，为后续源码调试做准备。

## 1. 官方推荐安装（uv）

来源：`/home/hpc/ghr_code/FastVideo/README.md`

```bash
# 创建并激活干净的 uv 环境
uv venv --python 3.12 --seed
source .venv/bin/activate

# 在 NVIDIA CUDA 12 上安装
UV_TORCH_BACKEND=cu126 uv pip install fastvideo
```

> 本机 AGENTS 约定优先用 `uv`。若无 `uv` 再退回 `pip`。

## 2. 从源码安装（推荐用于读源码 / 改代码）

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu126 uv pip install -e .
```

`-e`（editable）模式下改动 `fastvideo/` 里的 Python 代码立即生效，方便加断点、打印张量形状。

## 3. 核心依赖（`pyproject.toml`）

| 类别 | 关键依赖 | 说明 |
|------|---------|------|
| 深度学习 | `torch==2.12.0`, `torchvision`, `torchaudio` | 固定 torch 版本，FSDP2 / DTensor 需要新版本 |
| 模型库 | `transformers>=4.57.3`, `diffusers>=0.38.0`, `peft>=0.15.0` | text encoder 从 transformers 加载，VAE/scheduler 从 diffusers 加载 |
| tokenizer | `tokenizers>=0.20.1,<0.23` | 上界锁定，避免 CLIP tokenizer 加载报错（源码注释有详细说明） |
| 加速 | `flashinfer-python`, `accelerate==1.0.1` | flashinfer 用于 FP4 量化路径 |
| 视频 IO | `opencv-python`, `imageio`, `imageio-ffmpeg`, `einops` | 视频读写与张量重排 |
| 实验跟踪 | `wandb`, `loguru` | 训练日志 |

**可选依赖**（`[project.optional-dependencies]`）：
- `flash-attn`：需单独装 `uv pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation`。
- `test`：`av` 等测试依赖。
- `lint`：`pre-commit`。

## 4. CUDA Kernel 扩展（`fastvideo-kernel/`）

这是**独立的包**，包含 VSA、Sliding Tile Attention、INT8 GEMM/Norm 等自定义 kernel。

```bash
cd fastvideo-kernel
bash build.sh      # 自动检测 GPU 架构、更新 submodule、scikit-build-core 编译
```

编译细节（架构选择、CMake、submodule）见 [`02_source_by_directory/11_fastvideo_kernel.md`](../02_source_by_directory/11_fastvideo_kernel.md)。

**关键点**：
- ThunderKittens kernel（STA、block sparse）只在 **sm_90a（Hopper / H100）** 编译。
- FP4 attention（`attn_qat_infer`）只在 **sm_120a（Blackwell）+ CUDA≥12.8** 编译。
- TurboDiffusion（INT8 GEMM / RMSNorm / LayerNorm）**通用架构**都编译。

## 5. 硬件与 OS 支持

来源 README：H100 / A100 / 4090；Linux / Windows / MacOS（MacOS 上 FSDP 被禁用，见 `fsdp_load.py`）。

## 6. 环境变量速查（读源码常用）

这些环境变量会直接影响运行时行为，调试时非常有用：

| 环境变量 | 作用 | 定义位置 |
|---------|------|---------|
| `FASTVIDEO_ATTENTION_BACKEND` | 强制指定 attention 后端（如 `FLASH_ATTN`、`VIDEO_SPARSE_ATTN`） | `attention/selector.py` |
| `FASTVIDEO_FA4` | 启用 FlashAttention 4（CuTe DSL，sm90+） | `attention/backends/flash_attn.py` |
| `FASTVIDEO_DISABLE_ATTENTION_COMPILE` | 关闭 attention 的 torch.compile 禁用逻辑 | `attention/layer.py` |
| `FASTVIDEO_NCCL_SO_PATH` | 指定 NCCL 动态库路径 | `distributed/device_communicators/pynccl_wrapper.py` |
| `FASTVIDEO_FSDP2_AUTOWRAP` / `FASTVIDEO_FSDP2_MIN_PARAMS` | FSDP2 按参数量自动分片 | `models/loader/fsdp_load.py` |
| `LOCAL_RANK` / `WORLD_SIZE` / `RANK` | torchrun 设置的分布式变量 | `distributed/parallel_state.py` |

完整环境变量定义集中在 `fastvideo/envs.py`（**待确认**：可用 `grep` 查看完整列表）。

## 7. 验证安装

```bash
python -c "from fastvideo import VideoGenerator, PipelineConfig, SamplingParam, __version__; print(__version__)"
fastvideo --help    # CLI 是否可用
```

## 8. 常见问题

- **首次加载慢**：模型从 HuggingFace 下载（`maybe_download_model`），会缓存到 `~/.cache/huggingface`。
- **OOM**：打开 CPU offload（`dit_cpu_offload=True`、`vae_cpu_offload=True`），见 [`04_knowledge_expansion/13_memory_optimization.md`](../04_knowledge_expansion/13_memory_optimization.md)。
- **attention 后端不可用**：会自动回退到 SDPA（`selector.py` 的 fallback 逻辑）。
