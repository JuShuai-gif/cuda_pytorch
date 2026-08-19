# 10｜VLM 推理：Image → Vision Encoder → LLM 的性能模型

## 本模块解决的问题

VLM（视觉语言模型）推理不是"一个模型跑一遍"，而是**多个异构 stage 串联**：图像解码在 CPU、vision encoder 和 LLM 在 GPU、中间还有数据搬运。本章回答：

```text
VLM 推理的完整 pipeline 有哪些 stage？
每个 stage 花多少时间？瓶颈在哪里？
CPU 阶段和 GPU 阶段怎么 overlap？
```

配套代码：`src/inference/vlm/`（`pipeline.py` 各 stage + `benchmark.py` 分解测量）。

---

## 1. VLM 的 pipeline 结构

```text
Image bytes（JPEG）
   ↓ decode（CPU：JPEG 解码 → RGB 像素）
   ↓ preprocess（CPU：resize + normalize）
   ↓ H2D（CPU → GPU）
   ↓ Vision Encoder（GPU：ViT，patch embedding + transformer）
   ↓ Connector（GPU：Linear 投影，把 vision token 映射到 LLM 空间）
   ↓ Language Model（GPU：transformer，生成回答）
   ↓ Output
```

真实 VLM（LLaVA、Qwen-VL、GPT-4V）就是这条链，只是 vision encoder 用预训练 ViT（如 ViT-L 的 300M 参数）、LLM 用 7B+ 的 decoder。本模块用 from-scratch 小 ViT + 小 LLM 演示**结构**，不演示**规模**。

---

## 2. 各 stage 实测（本机 Thor，224×224 图片）

| stage | device | latency | 占比 |
|---|---|---|---|
| decode | CPU | 0.448ms | 6.3% |
| preprocess | CPU | 1.880ms | 26.4% |
| H2D | H2D | 0.030ms | 0.4% |
| vision encoder | GPU | 2.638ms | 37.1% |
| LLM | GPU | 2.118ms | 29.8% |
| **total** | | **7.113ms** | |

### 读法

1. **Vision encoder 是最大单阶段（37%）**：ViT 的 196 个 patch token 过 6 层 transformer，是计算主力。真实 VLM 里 ViT-L 会更重（几 ms 到几十 ms）。

2. **CPU 阶段合计占 33%（decode 6% + preprocess 26%）**：这是最容易被忽略的瓶颈——大家盯着 GPU 的 vision/LLM，但 CPU 上的 JPEG 解码 + resize + normalize 已经占了 1/3。高分辨率图片（1024×1024）时 preprocess 会更慢。

3. **H2D 几乎免费（0.4%）**：一张 224×224 的 fp32 图只有 600KB，搬运开销可忽略。但**批量图片**（video 帧、多图 VLM）时 H2D 会线性增长。

4. **Connector 极轻**：一个 Linear 投影，融合在 vision encoder 里测量，几乎不可见。

---

## 3. 关键洞察：CPU/GPU 的 overlap

上面的测量是**串行**的（一个 stage 接着一个）。真实 serving 里，CPU 的 decode/preprocess 可以和 GPU 的 vision/LLM **重叠**：

```text
串行：decode → preprocess → H2D → vision → LLM       （7.1ms）
流水：              request N+1 的 preprocess
                    └─ request N 的 vision/LLM ─┘     （吞吐↑，接近 GPU 阶段 4.8ms）
```

所以 VLM serving 的关键优化是 **pipeline**：用多线程/多 stream 让 CPU preprocess 和 GPU 推理重叠，端到端吞吐由 GPU 阶段决定，而不是 CPU+GPU 之和。

这正是 Stage 2 的"异步 H2D + 多 stream"思想在 VLM 场景的应用——**preprocess 和上一个请求的 GPU 计算 overlap**。

---

## 4. 优化方向（按瓶颈排序）

| 瓶颈 | 优化手段 |
|---|---|
| Vision encoder（37%） | 量化（int8/fp8）、TensorRT、减少 patch 数（更高 patch size） |
| LLM（30%） | KV cache、量化、speculative decoding（Stage 11/12） |
| CPU preprocess（26%） | 多线程 decode、GPU 上做 resize/normalize（NVDEC + CUDA preprocess）、批量预处理 |
| decode（6%） | 硬件解码（NVDEC）、避免重复解码（cache） |

真实 VLM 推理（尤其机器人 VLA，Stage 14）里，**CPU preprocess 和 image decode 往往是端到端延迟的隐藏杀手**——因为机器人要"相机帧进来 → 尽快出动作"，串行的 CPU 阶段直接加到 sensor-to-action latency 上。

---

## 5. 与机器人 VLA 的衔接

VLA（Vision-Language-Action）推理在 VLM 基础上多了两段：

```text
VLM：   image → vision → LLM → text
VLA：   image → vision → LLM → action decoder → robot control
```

VLA 的额外关注点（Stage 14 详述）：
- **sensor-to-action latency**：从相机帧到控制指令的端到端时间
- **jitter / p99**：实时控制要求稳定，不看平均
- **batch=1**：机器人在线控制几乎总是 batch=1，launch overhead 和 CPU preprocess 占比更高

---

## 6. 本模块闭环小结

```text
问题：VLM 推理慢在哪一段
      ↓
分解：decode → preprocess → H2D → vision encoder → connector → LLM
      ↓
实测：vision 37% + LLM 30% + CPU preprocess 26% + decode 6% + H2D 0.4%
      ↓
结论：CPU 阶段占 1/3 是隐藏瓶颈，流水化后吞吐由 GPU 阶段决定
      ↓
下一步：Stage 14 VLA / Robot Policy 推理（sensor-to-action latency、jitter、batch=1）
```

要继续就说「继续」。
