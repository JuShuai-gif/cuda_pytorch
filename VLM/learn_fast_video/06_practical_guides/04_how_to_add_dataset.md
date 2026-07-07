# 如何添加数据集

> 添加新的训练数据源。FastVideo 数据流是"原始视频 → 预处理 → Parquet → 训练读取"。

## 1. 两种添加方式

1. **准备符合现有格式的数据**（推荐）：把你的视频整理成 FastVideo 期望的格式，走现有预处理。
2. **实现新 Dataset 类**：数据格式特殊时。

## 2. 方式一：准备现有格式数据

### 组织原始数据
```
my_dataset/
├── video1.mp4
├── video2.mp4
└── ...
```
生成 caption JSON（`scripts/dataset_preparation/prepare_json_file.py`）：
```json
[{"path": "video1.mp4", "cap": ["a caption"],
  "resolution": {"width": 1280, "height": 720}, "fps": 24.0, "duration": 5.0}]
```

索引文件（txt）每行：`<folder_path>,<json_path>`。

### 运行预处理
```bash
torchrun ... fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL --data_merge_path $INDEX_TXT \
    --output_dir ./processed --preprocess_task "t2v" --num_frames 81 --train_fps 16
```
产出 Parquet（VAE latent + text embedding）。

### 训练读取
YAML 里 `training.data.data_path` 指向 Parquet 目录，`LatentsParquetIterStyleDataset` 自动读取。

## 3. 方式二：实现新 Dataset

参考 `dataset/parquet_dataset_iterable_style.py`。若要新 Dataset：
```python
class MyDataset(torch.utils.data.IterableDataset, Stateful):
    def __iter__(self):
        yield (latents, embs, masks, captions)   # 保持输出契约
    def state_dict(self): ...   # 支持 checkpoint resume
    def load_state_dict(self, state): ...
```
输出契约（iterable-style）：`(latents [B,C,T,H,W], embs [B,L,D], masks [B,L], captions)`。

## 4. Parquet schema（若自定义）

参考 `dataset/dataloader/schema.py`。tensor 用 `*_bytes + *_shape + *_dtype` 三元组存。
新增字段（如 I2V 的图像）参考 `pyarrow_schema_i2v`。

## 5. Record creator（预处理写入）

参考 `dataset/dataloader/record_schema.py` 的 `basic_t2v_record_creator` / `i2v_record_creator`，定义预处理输出如何变成 Parquet 行。

## 6. Sharding 注意

iterable-style dataset 要处理 SP group + worker 分片（参考 `parquet_dataset_iterable_style.py` 的贪心分片逻辑），否则多 GPU 训练数据会重复。

## 7. 帧采样/变换（若处理原始视频）

复用 `dataset/preprocessing_datasets.py` 的 stage：`FrameSamplingStage`（fps 重采样）、`VideoTransformStage`（crop/resize/归一化）。

## 8. CFG dropout

训练需要无条件分支，`LatentDataset` 以 `cfg_rate` 概率把 text embedding 置零。新 dataset 应支持类似机制。

## 9. 测试

用 `dataset/benchmarks/benchmark_parquet_dataset_iterable_style.py` 验证吞吐量 + resume 正确性。

## 10. 检查清单

- [ ] 输出契约正确（latents/embs/masks/captions）。
- [ ] 支持 SP + worker 分片（不重复）。
- [ ] 支持 checkpoint resume（Stateful）。
- [ ] Parquet schema 匹配。
- [ ] CFG dropout。

## 11. 参考
- `dataset/parquet_dataset_iterable_style.py`
- `dataset/dataloader/schema.py` + `record_schema.py`
- `dataset/preprocessing_datasets.py`
- 知识：[`../04_knowledge_expansion/14_video_dataset_and_preprocessing.md`](../04_knowledge_expansion/14_video_dataset_and_preprocessing.md)
