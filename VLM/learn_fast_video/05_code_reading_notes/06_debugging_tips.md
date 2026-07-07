# 调试技巧

> FastVideo 是多进程 + 分布式 + 大模型，调试有特殊性。本文给实用入口。

## 1. 用小配置快速跑通

```python
generator.generate_video(prompt="test", num_frames=17, height=256, width=256, num_inference_steps=4)
```
帧数/分辨率/步数都调小，几秒跑完一轮，快速验证代码路径。

## 2. 强制单 GPU + 单进程调试

多进程 worker 难打断点。调试时：
```python
VideoGenerator.from_pretrained(model_path, num_gpus=1)
```
`num_gpus=1` 仍是子进程，但只有一个。若要主进程调试，可在 worker 代码里加 `import pdb; pdb.set_trace()`（注意子进程 stdin）。

## 3. 强制 attention 后端

```bash
FASTVIDEO_ATTENTION_BACKEND=SDPA python script.py   # 最稳，排除 kernel 问题
```
SDPA 通用可用，先用它确认逻辑正确，再换加速后端。

## 4. 观察 stage 数据流

在 `ComposedPipelineBase.forward`（`composed_pipeline_base.py:488`）循环里加：
```python
for stage in self.stages:
    batch = stage(batch, fastvideo_args)
    print(type(stage).__name__,
          "latents:", getattr(batch, "latents", None) and batch.latents.shape,
          "embeds:", getattr(batch, "prompt_embeds", None) and len(batch.prompt_embeds))
```
一眼看出每个 stage 后张量形状变化。

## 5. 观察 DiT 输入张量

在 `WanTransformer3DModel.forward`（`models/dits/wanvideo.py:632`）开头：
```python
print("hidden:", hidden_states.shape, "text:", encoder_hidden_states.shape, "t:", timestep)
```

## 6. 检查后端选择结果

在 `selector.py:_cached_get_attn_backend` 打印 `selected_backend`，确认用了哪个 attention。

## 7. 检查 pipeline 选择

在 `build_pipeline`（`pipelines/__init__.py:27`）打印 `pipeline_cls.__name__`，确认 model_path 匹配到正确 pipeline。

## 8. 分布式调试

```bash
# 单卡训练排除分布式问题
torchrun --nproc_per_node 1 -m fastvideo.train.entrypoint.train --config x.yaml
```
分布式挂起常见原因：某 rank 走了不同分支导致集合通信不匹配。检查是否所有 rank 都执行相同的 `collective_rpc`/`all_reduce`。

## 9. 显存调试

```python
print(torch.cuda.max_memory_allocated() / 1e9, "GB")
```
`_generate_single_video` 返回的 `peak_memory_mb` 也有峰值。OOM 时逐个打开 offload 开关定位。

## 10. logging_info（内置耗时）

`ForwardBatch.logging_info` 记录每个 stage 耗时（`PipelineStage.__call__` 自动记）。检查哪个 stage 慢。

## 11. stage verification

`enable_stage_verification=True` 打开 stage 的 `verify_input`/`verify_output`，在数据异常时早失败，定位问题 stage。

## 12. 常见坑

| 现象 | 可能原因 |
|------|---------|
| 分布式挂起 | rank 间分支不一致 / 集合通信不匹配 |
| OOM | offload 未开 / VAE tiling 未开 / 分辨率太大 |
| 黑屏/噪点输出 | latent 反归一化错 / scheduler 配错 |
| attention 报错 | 后端不支持当前 GPU（换 SDPA） |
| kernel import 失败 | fastvideo-kernel 未编译（回退 Triton） |
| 加载慢 | HF 首次下载 |

## 13. 系统化调试流程

遇到 bug 别急着改：
1. 复现（最小配置）。
2. 定位（哪个 stage/模块，打印张量）。
3. 假设 → 验证（改一处，观察）。
4. 修复 + 回归测试。

（参考 systematic-debugging skill）

## 14. 相关
- 性能分析：[`../06_practical_guides/06_how_to_profile_performance.md`](../06_practical_guides/06_how_to_profile_performance.md)
