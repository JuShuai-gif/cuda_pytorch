# dataset —— 数据层

> 模块作用：视频/图像/文本数据的读取、预处理、Parquet 存储、dataloader。训练和评测的数据来源。

## 1. 模块结构

```
dataset/
├── __init__.py                        # getdataset() / gettextdataset()
├── preprocessing_datasets.py          # 原始视频/文本处理（713 行）
├── latent_datasets.py                 # 预计算 latent（JSON 索引）
├── parquet_dataset_iterable_style.py  # Parquet iterable（推荐，305 行）
├── parquet_dataset_map_style.py       # Parquet map（兼容用）
├── ltx2_precomputed_dataset.py        # LTX-2 预计算
├── validation_dataset.py              # 验证集
├── transform.py                       # 视频变换
├── utils.py                           # collation
├── dataloader/                        # Parquet schema/writer/record
└── benchmarks/                        # dataloader 基准
```

## 2. 两种数据形态

FastVideo 训练**不直接读原始视频**，而是先预处理成 Parquet（存 VAE latent + text embedding），训练时读 Parquet。

```mermaid
graph LR
    RAW["原始视频.mp4 + caption.json"] --> PRE["preprocessing_datasets<br/>帧采样/变换/tokenize"]
    PRE --> VAE["VAE.encode + TextEncoder"]
    VAE --> PQ["ParquetDatasetWriter<br/>.parquet 文件"]
    PQ --> PDS["LatentsParquetIterStyleDataset"]
    PDS --> DL["DataLoader → 训练"]
```

## 3. 原始数据处理（preprocessing_datasets.py）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/dataset/preprocessing_datasets.py
关键类：VideoCaptionMergedDataset (L363), TextDataset (L585)
```

`VideoCaptionMergedDataset`（`IterableDataset` + `Stateful`）用 Stage 管道处理：
```python
DataValidationStage()          # 检查 cap/resolution/fps/duration
FrameSamplingStage(...)        # fps 重采样 + temporal crop
VideoTransformStage(transform) # 读视频→TCHW→crop/resize/归一化
TextEncodingStage(tokenizer)   # tokenize
```

输出（`_get_item`, L545）：
```python
{"pixel_values": (C,T,H,W) float32 [-1,1],   # 视频帧
 "input_ids": (1, text_max_length) int64,     # tokenizer 输出
 "cond_mask": (1, text_max_length) int64,      # attention mask
 "text": str, "path": str}
```

### 帧采样细节（FrameSamplingStage, L151）
- `should_keep`：过滤太长视频，以 `drop_short_ratio` 丢弃太短的。
- `process`：`num_frames = ceil(fps * duration)`，按 `train_fps` 重采样 `frame_indices = arange(0, num_frames, fps/train_fps)`。

### 变换（transform.py）
- `CenterCropResizeVideo`（L77）：先按宽高比裁剪再 resize。
- `Normalize255`（L119）：uint8 → [0,1]。
- Video→[-1,1]：`/ 127.5 - 1.0`。

## 4. Parquet 格式

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/dataset/dataloader/schema.py
```

用 `*_bytes + *_shape + *_dtype` 三元组存任意形状 tensor（绕过 Arrow 嵌套限制）：
```
pyarrow_schema_t2v:
  vae_latent_bytes: binary       # tensor 原始字节
  vae_latent_shape: list(int64)  # [C, T, H, W]
  vae_latent_dtype: string       # 'bfloat16'
  text_embedding_bytes/shape/dtype
  caption, width, height, num_frames, fps, duration_sec
```

`pyarrow_schema_i2v` 额外有 `clip_feature_*`, `first_frame_latent_*`, `pil_image_*`。

`ParquetDatasetWriter`（`dataloader/parquet_io.py` L49）：
- `append_table` 缓冲 → `flush` 按 `samples_per_file` 写文件（临时文件 + 原子 rename）。
- 支持 `ProcessPoolExecutor` 并行写。

## 5. Parquet 数据集（训练读取）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/dataset/parquet_dataset_iterable_style.py
关键类：LatentsParquetIterStyleDataset (L58)
```

- **Sharding**：rank 0 扫描所有 parquet，贪心分配到 `num_sp_groups × num_workers` 个 shard，pickle 缓存 plan，barrier 同步。
- **输出**（`BatchIterator.__iter__`, L33）：
```python
(all_latents,   # (B, C, T, H, W) bfloat16
 all_embs,      # (B, L, D) float32，已 padding
 all_masks,     # (B, L) float32 attention mask
 caption_text)  # list[str]
```

`LatentDataset`（`latent_datasets.py`）：从 JSON 索引加载 `.pt` latent + prompt embedding，支持 CFG dropout（`cfg_rate` 概率替换为零 embedding）。

## 6. 验证集（validation_dataset.py）

`ValidationDataset`（L18）：从 csv/json/parquet/arrow 加载验证 prompt，支持 SP group 分片，自动加载 image/video/action 附件。输出 `{prompt, image, video, keyboard_cond, mouse_cond, ...}`。

## 7. collation（utils.py）

`collate_rows_from_parquet_schema`（L99）：从 parquet 行提取 tensor，处理 text embedding padding + CFG dropout，其他 tensor 直接 `torch.stack`。

## 8. 源码阅读重点
1. `schema.py` 的 Parquet 三元组设计。
2. `parquet_dataset_iterable_style.py` 的 sharding 逻辑（如何跨 SP group / worker 分片）。
3. `preprocessing_datasets.py` 的 stage 管道。

## 9. 调试入口
```python
from fastvideo.dataset import getdataset
# 观察一个 batch 的形状与内容
```
用 `benchmarks/benchmark_parquet_dataset_iterable_style.py` 测吞吐量和 resume 正确性。

## 10. 相关笔记
- 数据知识：[`04_knowledge_expansion/14_video_dataset_and_preprocessing.md`](../04_knowledge_expansion/14_video_dataset_and_preprocessing.md)
- 数据集实践：[`06_practical_guides/04_how_to_add_dataset.md`](../06_practical_guides/04_how_to_add_dataset.md)
