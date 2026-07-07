# 评测指标

> 知识点扩展：PSNR/SSIM/LPIPS/FVD/VBench 的原理、适用场景、局限，回扣 FastVideo eval。

## 0. 评测的两难

视频生成评测比图像更难：
- **无唯一答案**：同一 prompt 可生成无数合理视频，不能用像素级对比。
- **多维度**：画质、时序一致性、运动合理性、prompt 遵循度……单一指标覆盖不全。
- **主观性**：最终要人眼评判，自动指标只是近似。

所以实践中组合使用多个指标 + 人工评估。FastVideo `eval/` 提供全套。

## 1. 指标分类

| 类型 | 指标 | 需要参考视频? | 衡量什么 |
|------|------|-------------|---------|
| 像素级重建 | PSNR, SSIM | 是（成对） | 与 GT 的逐像素/结构相似 |
| 感知相似 | LPIPS | 是 | 与 GT 的感知相似（网络特征） |
| 分布距离 | FVD | 否（vs 参考集） | 生成集与真实集的分布差异 |
| 无参考质量 | VBench, VideoScore | 否 | 单视频的多维质量 |

**参考型（reference-based）** 用于有 GT 的任务（超分、重建、V2V）；**无参考型（reference-free）** 用于纯生成（T2V/I2V）。

## 2. PSNR（Peak Signal-to-Noise Ratio）

```
metrics/common/psnr/metric.py
```
`PSNR = 10·log10(max²/MSE)`。逐帧 MSE。
- **适合**：有 ground truth 的重建任务（超分、VAE 重建）。
- **不适合**：生成任务（没有唯一正确答案）；对感知质量不敏感（模糊图可能 PSNR 高）。
- **典型范围**：20-40 dB，越高越好。>30 通常视觉接近。

## 3. SSIM（Structural Similarity）

```
metrics/common/ssim/metric.py
```
比较亮度/对比度/结构：`SSIM = f(μ, σ, 协方差)`，高斯核 depthwise conv。范围 [0,1]，越高越相似。
- **适合**：结构保真度（超分、重建）。
- **不适合**：生成多样性评估。
- **为什么比 PSNR 好**：PSNR 只看逐像素误差，SSIM 考虑局部结构（人眼对结构更敏感）。

## 4. LPIPS（Learned Perceptual Image Patch Similarity）

```
metrics/common/lpips/metric.py，依赖 lpips 库（AlexNet）
```
用预训练网络特征的距离衡量感知相似度。比 PSNR/SSIM 更符合人眼。越低越相似。需 reference。
- **原理**：PSNR/SSIM 是手工设计的低层指标；LPIPS 用深度网络（AlexNet/VGG）的多层特征算距离，捕捉高层语义相似性，与人类感知判断相关性最高。
- **典型范围**：0-1，越低越好。

## 5. FVD（Fréchet Video Distance）

```
metrics/common/fvd/metric.py（set-vs-set）
```
衡量**生成视频集**和**真实视频集**的分布距离：
```
FVD = ||μ_gen - μ_real||² + tr(Σ_gen + Σ_real - 2√(Σ_gen·Σ_real))
```
用视频特征提取器（I3D/CLIP/VideoMAE）提特征，拟合高斯，算 Fréchet 距离。
- **适合**：生成质量+多样性的整体评估（文献标准）。
- **注意**：需 ≥256 视频才有统计意义；对提取器选择敏感。

FastVideo 三个提取器：`i3d`（Kinetics-400, 文献标准）、`clip`、`videomae`。参考特征缓存到 `${FASTVIDEO_EVAL_CACHE}/fvd/`。

## 6. VBench

```
metrics/vbench/（16 个子指标）
```
多维度无参考评测：
- `motion_smoothness`（运动平滑，AMT-S 光流插值）
- `aesthetic_quality`（美学）
- `subject_consistency`（主体一致性）
- `dynamic_degree`（动态程度）
- `color`, `temporal_flickering` 等。
- **适合**：T2V 生成的综合质量评估（无需 ground truth）。
- **注意**：每个维度模型不同，需分别下载 checkpoint。

## 7. 其他指标

- `videoscore2`：Qwen2.5-VL 打分。
- `judge.*`：VLM pairwise 对比。
- `physics_iq.*`：物理合理性。
- `audio.*`：音频质量（LTX-2）。

## 8. 如何选指标

| 任务 | 推荐指标 |
|------|---------|
| 超分/重建（有GT） | PSNR + SSIM + LPIPS |
| T2V 生成质量 | FVD + VBench |
| 主体一致性 | VBench subject_consistency |
| 运动质量 | VBench motion_smoothness + dynamic_degree |
| 与参考对比 | LPIPS + judge |

## 9. 评测调用

```python
from fastvideo.eval import create_evaluator, samples_from
ev = create_evaluator(metrics=["common.ssim", "common.fvd", "vbench.motion_smoothness"], device="cuda:0")
results = ev.evaluate(samples=samples_from(video="gen/", reference="ref/", fps=24))
```
CLI：`fastvideo eval list` / `fastvideo eval run`。

## 10. 回扣源码
| 指标 | 源码 |
|------|------|
| PSNR | `eval/metrics/common/psnr/metric.py` |
| SSIM | `eval/metrics/common/ssim/metric.py` |
| LPIPS | `eval/metrics/common/lpips/metric.py` |
| FVD | `eval/metrics/common/fvd/metric.py` |
| VBench | `eval/metrics/vbench/` |

## 11. 延伸
- eval 目录：[`../02_source_by_directory/09_eval.md`](../02_source_by_directory/09_eval.md)
