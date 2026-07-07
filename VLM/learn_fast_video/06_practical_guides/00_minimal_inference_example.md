# 最小推理示例

> 从零跑通一个视频生成，理解每行代码对应的源码。

## 1. 最简三行

```python
from fastvideo import VideoGenerator
generator = VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1)
video = generator.generate_video(prompt="A cat playing piano", output_path="outputs/")
generator.shutdown()
```

对应源码：
- `from_pretrained` → `entrypoints/video_generator.py:144`（加载模型到 worker）。
- `generate_video` → 同上 L359 → `_generate_single_video` L658（生成 + 保存）。
- `shutdown` → 释放 worker 进程。

## 2. 快速调试配置（几秒跑完）

```python
video = generator.generate_video(
    prompt="test",
    num_frames=17, height=256, width=256,
    num_inference_steps=4,   # 步数少
    guidance_scale=1.0,      # 关 CFG，省一半算力
    output_path="outputs/", save_video=True)
```

## 3. 省显存配置（单卡小显存）

```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1,
    dit_cpu_offload=True,
    text_encoder_cpu_offload=True,
    vae_cpu_offload=True,
    pin_cpu_memory=True)
```

## 4. 多 GPU（序列并行）

```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", num_gpus=4)   # sp_size 默认=num_gpus
```

## 5. 复用 generator 生成多个视频

```python
generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
for p in ["prompt1", "prompt2", "prompt3"]:
    generator.generate_video(prompt=p, output_path="outputs/")
generator.shutdown()   # 模型只加载一次
```

## 6. 返回值

```python
result = generator.generate_video(prompt="test", num_frames=17, height=256, width=256)
print(result["video_path"])       # mp4 路径
print(result["samples"].shape)    # [1, 3, 17, 256, 256]
print(result["generation_time"])  # 生成耗时
print(result["peak_memory_mb"])   # 显存峰值
```

## 7. 只要 latent（不 decode）

```python
result = generator.generate_video(prompt="test", output_type="latent")
# result 含 latent 张量，跳过 VAE decode
```

## 8. CLI 方式

```bash
fastvideo generate --config scripts/inference/inference_wan.yaml
```

## 9. 换模型

```python
# LTX2
VideoGenerator.from_pretrained("Davids048/LTX2-Base-Diffusers", num_gpus=1)
# Hunyuan、Cosmos 等同理，registry 自动匹配 pipeline
```

## 10. 官方示例

```bash
python examples/inference/basic/basic.py        # Wan
python examples/inference/basic/basic_ltx2.py   # LTX2
```

## 11. 理解这个流程后

去读 [`../03_core_flows/00_video_generation_flow.md`](../03_core_flows/00_video_generation_flow.md) 看完整调用链。
