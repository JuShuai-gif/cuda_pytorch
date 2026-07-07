# Text Encoder 与 Prompt 编码

> 知识点扩展：text encoder 的作用、各家 encoder、prompt 如何变成条件 embedding，回扣 FastVideo。

## 1. Text Encoder 的作用

把文本 prompt 编码成向量序列 `[B, L, D]`，作为扩散模型的条件（cross attention 的 K/V）。质量直接影响 prompt 遵循度。

### 1.1 为什么 text encoder 如此关键

生成模型的"理解能力"上限由 text encoder 决定：
- encoder 抓不到的语义（如复杂空间关系、否定），DiT 也生成不出来。
- 这就是趋势从 CLIP（77 token，弱语义）→ T5（长序列，强语义）→ 大 LLM（Llama/Qwen，最强）的原因。
- FastVideo 新模型多用大 LLM 做 encoder，代价是 encoder 本身也很大（需 offload/TP）。

## 2. FastVideo 支持的 encoder

```
源码位置：models/encoders/
基类：base.py 的 TextEncoder / ImageEncoder
```

| encoder | 文件 | 用于 | 特点 |
|---------|------|------|------|
| T5 / UMT5 | `t5.py` | Wan | encoder-only，相对位置编码，4096 维 |
| CLIP | `clip.py` | 多模型 | 文本+视觉双塔 |
| Llama | `llama.py` | Hunyuan | decoder-only，RoPE，SwiGLU |
| Qwen2.5/Qwen3 | `qwen2_5.py`/`qwen3.py` | - | LLM |
| Gemma | `gemma.py` | LTX-2 | - |
| Mistral3 | `mistral3.py` | Flux2 | - |
| SigLIP | `siglip.py` | 图像 | 视觉编码 |
| Bert | `bert.py` | - | - |
| Reason1 | `reason1.py` | - | 推理增强 |
| Stable Audio conditioner | `stable_audio_conditioner.py` | Stable Audio | T5 + 数值条件 |

### 2.1 encoder-only vs decoder-only 做条件编码

| | encoder-only（T5/Bert） | decoder-only（Llama/Qwen） |
|--|------------------------|---------------------------|
| 注意力 | 双向（每 token 看全序列） | 因果（只看前面） |
| 语义 | 双向理解，适合表征 | 生成式，语义丰富 |
| 用法 | 直接取 hidden state | 取最后层 hidden / 特定 token |
| 位置编码 | 相对位置（T5） | RoPE |

视频生成用它们提取文本表征（不做生成），所以 decoder-only 也常取 `last_hidden_state` 当条件。

## 3. 统一输出接口

```python
# base.py
class TextEncoder(nn.Module, ABC):
    def forward(self, input_ids, attention_mask, ...) -> BaseEncoderOutput:
        ...
# BaseEncoderOutput(last_hidden_state, attention_mask)
```
输出 `last_hidden_state` `[B, L, D]`。

## 4. Prompt 编码流程（TextEncodingStage）

```
源码位置：pipelines/stages/text_encoding.py，encode_text (L117)
```
```python
processed_text = preprocess_func(prompt)       # prompt template（如加特定前缀）
text_inputs = tokenizer(processed_text)         # → input_ids, attention_mask
outputs = text_encoder(input_ids, attention_mask)
prompt_embeds = postprocess_func(outputs)       # 后处理（如取 last_hidden_state）
```

- `preprocess_func`/`postprocess_func` 来自 `PipelineConfig`，各模型不同。
- 多 encoder 模型（Hunyuan）会有两个 encoder（text_encoder + text_encoder_2），拼接输出。

## 5. 为什么 T5 vs LLM

- **T5（Wan）**：encoder-only，专为文本理解，输出稠密语义。
- **Llama/Qwen（Hunyuan/新模型）**：decoder-only LLM，语义更丰富，可支持指令式 prompt。

趋势是用更大的 LLM 做 encoder 提升文本理解。

## 6. Negative Prompt 与 CFG

`SamplingParam.negative_prompt` 也被编码成 `negative_prompt_embeds`，供 CFG 使用：
```python
noise_pred = uncond(negative) + guidance * (cond(prompt) - uncond(negative))
```

## 7. TP 支持

encoder 用 `QKVParallelLinear`/`RowParallelLinear`（`layers/linear.py`）支持张量并行，大 LLM encoder 可切分到多卡。

## 8. Prompt 增强（dreamverse）

`apps/dreamverse/prompt_enhancer.py` 用外部 LLM（Cerebras/Groq）把用户简短 prompt 扩展成 cinematic 详细描述，再喂给 text encoder。这是应用层的 prompt engineering，不在核心库。

## 9. 回扣源码
| 概念 | 源码 |
|------|------|
| 编码流程 | `stages/text_encoding.py:encode_text` |
| T5 实现 | `models/encoders/t5.py` |
| 输出接口 | `models/encoders/base.py:BaseEncoderOutput` |
| cross attention 注入 | `models/dits/wanvideo.py:WanT2VCrossAttention` |

## 10. 调试
在 `encode_text` 后打印 `prompt_embeds.shape`（确认 seq_len 和 hidden_dim）。
