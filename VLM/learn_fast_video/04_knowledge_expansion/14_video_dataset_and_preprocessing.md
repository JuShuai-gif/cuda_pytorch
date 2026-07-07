# 视频数据集与预处理

> 知识点扩展：视频读取、帧采样、fps、resolution、caption/text pair、parquet/webdataset、dataloader 瓶颈，回扣 FastVideo。

## 1. 视频训练数据的挑战

- 视频文件大、解码慢。
- 需要配对 caption。
- 每次训练重新 VAE encode + text encode 太贵。

FastVideo 方案：**预处理一次，存 Parquet**（VAE latent + text embedding），训练直接读 latent。

### 1.1 为什么"预处理成 latent"是关键设计

如果训练时每步都做"读视频 → 解码 → VAE encode → text encode"，GPU 会大量空等 CPU/IO，利用率极低。FastVideo 把这些**一次性离线做完**，存成 latent Parquet：
- 训练时 dataloader 只读小 tensor（latent 比像素小 ~48 倍）。
- 省掉训练时的 VAE/text encoder 显存（不用加载）。
- GPU 利用率高。

代价：预处理耗时 + 存储（但 latent 比原视频小）。这是所有大规模视频训练框架的标配。

### 1.2 数据集选型考量

| 格式 | 优点 | 缺点 | FastVideo |
|------|------|------|-----------|
| 原始视频 + json | 灵活 | 训练时解码慢 | 仅预处理阶段 |
| **Parquet（latent）** | 随机访问、列式、易 shard | 需预处理 | 训练主力 |
| webdataset（tar） | 顺序流式好 | 随机访问弱 | 未用 |
| HDF5 | 结构化 | 并发弱 | 未用 |

## 2. 数据流

```mermaid
graph LR
    RAW["视频.mp4 + caption.json"] --> PRE["preprocessing_datasets"]
    PRE --> ENC["VAE.encode + TextEncoder"]
    ENC --> PQ["Parquet (latent+embedding)"]
    PQ --> DS["LatentsParquetIterStyleDataset"]
    DS --> TRAIN["训练"]
```

## 3. 原始数据格式

索引文件（txt）每行 `<folder_path>,<json_file_path>`。JSON：
```json
[{"path": "video.mp4", "cap": ["caption"],
  "resolution": {"width": 1920, "height": 1080}, "fps": 25.0, "duration": 6.88}]
```

## 4. 帧采样（FrameSamplingStage）

```
源码：dataset/preprocessing_datasets.py:FrameSamplingStage (L151)
```
- **fps 重采样**：原视频 30fps → 训练 16fps，抽帧 `arange(0, num_frames, fps/train_fps)`。
- **temporal crop**：超过 `num_frames` 随机裁剪。
- **过滤**：太长的丢弃，太短的按 `drop_short_ratio` 概率丢。

关键参数：`num_frames`, `train_fps`。

## 5. 分辨率处理

```
dataset/transform.py:CenterCropResizeVideo (L77)
```
先按目标宽高比 center crop，再 resize。归一化 `/127.5 - 1.0` → [-1,1]。

## 6. Parquet 格式

```
dataset/dataloader/schema.py
```
用 `*_bytes + *_shape + *_dtype` 三元组存任意 tensor：
```
vae_latent_bytes/shape/dtype
text_embedding_bytes/shape/dtype
caption, width, height, num_frames, fps
```
`ParquetDatasetWriter`（parquet_io.py）：缓冲 + 按 `samples_per_file` flush + 原子 rename。

为什么 Parquet 而非 webdataset：列式存储、随机访问、易 sharding、生态成熟。

## 7. Dataloader 与 sharding

```
dataset/parquet_dataset_iterable_style.py:LatentsParquetIterStyleDataset (L58)
```
- rank 0 扫描所有 parquet，贪心分配到 `num_sp_groups × num_workers` 个 shard。
- pickle 缓存 plan，barrier 同步。
- 输出 `(latents, embs, masks, captions)`。

## 8. Dataloader 性能瓶颈

| 瓶颈 | 缓解 |
|------|------|
| 解码慢 | 预处理成 Parquet（已解码） |
| I/O | iterable-style 顺序读 + 多 worker |
| 内存 | 流式读，不全加载 |
| shuffle | `DP_SP_BatchSampler` 确定性 shuffle |

预处理成 latent 后，训练 dataloader 基本只是读 tensor，瓶颈从解码转移到磁盘 I/O。

## 9. CFG dropout

`LatentDataset` 以 `cfg_rate` 概率把 text embedding 替换为零（训练无条件分支，支持 CFG）。

## 10. 回扣源码
| 概念 | 源码 |
|------|------|
| 帧采样 | `preprocessing_datasets.py:FrameSamplingStage` |
| 变换 | `dataset/transform.py` |
| Parquet schema | `dataset/dataloader/schema.py` |
| 写入 | `dataset/dataloader/parquet_io.py` |
| 读取 | `parquet_dataset_iterable_style.py` |
| 预处理入口 | `pipelines/preprocess/v1_preprocess.py` |

## 11. 延伸
- 数据目录：[`../02_source_by_directory/06_dataset.md`](../02_source_by_directory/06_dataset.md)
- 添加数据集：[`../06_practical_guides/04_how_to_add_dataset.md`](../06_practical_guides/04_how_to_add_dataset.md)
