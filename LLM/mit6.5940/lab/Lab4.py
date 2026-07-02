# 使用 AWQ 进行 LLM 量化

"""
LLM 量化提供了代码和框架。可以学习如何量化一个大语言模型，使其能够高效运行。
本文将实现 AWQ (激活感知的仅权重量化)，用于4比特的仅权重量化

在边缘端运行大语言模型(LLM)非常重要，这不仅能提升用户体验，还能解决隐私问题--敏感数据保留在本地，降低了潜在泄漏的风险。

然而，在边缘端部署 LLM 有两个主要挑战：1) LLM 的参数量在不断变大(例如 LLaMA 从最初到 LLaMA-3 已增长超过 10 倍)，对设备的内存容量提出了更高要求；2) 自回归解码过程的 token 生成速度受限于内存带宽。因此，高效地压缩 LLM 模型、加速推理，是边缘端部署的必要条件。

借助 AWQ 4比特仅权重量化算法，再配合高效的4比特kernel，我们可以在RTX 4090上实现下面的加速效果。
在下一节实验中，将使用 TinyChatEngine 来实现实际的性能加速。
"""

# AWQ (激活感知的仅权重量化)

"""
大语言模型（LLM）在各类任务上都展现出出色的性能，但其庞大的模型规模抬高了部署的硬件门槛（内存容量），也拖慢了 token 生成速度（内存带宽）。LLM 的规模和计算量呈指数级增长，而内存带宽的增长却很缓慢。这一差距是 LLM 的主要瓶颈。在本次实验中，我们将探索使用一种新颖的量化算法（AWQ）来减少 LLM 的内存占用，并实现推理加速。

前面的课程中，我们学习了量化的基本方法。量化分为两种类型：
- 同时量化权重和激活
    - 更适合计算受限的场景：上下文阶段、大批量推理
    - 例如 SmoothQuant: W8A8量化
- 仅权重量化
    - 更适合内存受限的场景：解码阶段、单批量推理
    - AWQ：W4A16 量化

以 LLaMA-65B 模型为例，在单批量推理的解码阶段，我们需要执行 GEMV [1,8192] X [8192,8192].
以NVIDIA A100 80G为例，其半精度(FP16)性能为 312TFLOPS，内存带宽约为 2000GB/s.
因此，其计算强度为：

FLOP/Byte = 2 X 8192^2/8192^2  << (3.12 X 10^11) /(2X 10^9)
"""

import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import gc


"""
evaluate: 在 wikitext-2 测试集上计算模型的困惑度（perplexity）。
困惑度越低说明模型语言建模能力越好。取前 40 段，每段 2048 token，计算交叉熵损失的指数均值。
"""


# ── 函数：evaluate ──
# 在 wikitext-2 测试集上计算模型困惑度(perplexity)。困惑度越低=语言建模越好。
# 取前 40 段、每段 2048 token，计算交叉熵损失的指数均值。
# 使用 wikitext-2 数据集进行评估
def evaluate(model, tokenizer):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")

    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))


"""
get_model_size: 计算模型的总比特数（bits）。
data_width: 每个参数的平均比特数（FP32=32，INT3=3）。
如果使用了 group_size，额外加上 scale/zero_point 的存储开销：(16+4)/group_size 比特/参数。
"""


# ── 函数：get_model_size ──
# 计算模型总比特数(bits)。data_width 是每个参数的平均比特数(FP32=32, INT3=3)。
# 如果 group_size>0，额外加上每组存储 scale(FP16) 和 zero_point(INT4) 的开销。
# 下面的代码用于计算模型大小
def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# 评估 FP32 模型的困惑度和模型大小
model_path = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=32, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size / MiB:.2f} MiB")


"""
均匀量化就是把范围 [beta, alpha] 内的实数值映射到 [0, 2^b - 1] 之内。

记号说明：
- 量化后的权重：w_q
- 缩放因子：s_q = (alpha - beta) / (2^b - 1)
- 零点：z = -round(beta / s_q)
- 量化过程：w_q = clamp(round(w / s_q) + z, 0, 2^b - 1)
- 反量化过程：w' = (w_q - z) * s_q
"""

"""
伪量化

下面的代码用于伪量化

伪量化用于模拟量化对模型的影响，而不真正量化模型的权重。
即先舍入到最近的量化值，然后再反量化回浮点数
"""


"""
pseudo_quantize_tensor: 对给定张量执行"伪量化"——先量化到 n_bit 整数，再反量化回浮点数。
这样可以在 FP 精度下模拟量化带来的精度损失，而不需要真正的 INT 推理引擎。
q_group_size: 量化的组大小，-1 表示全张量一起量化，>0 表示每组独立量化（per-group）。
"""


# ── 函数：pseudo_quantize_tensor ──
# 核心伪量化函数：把张量量化到 n_bit 整数再反量化回浮点，模拟量化精度损失。
# q_group_size: 组大小，-1=全张量一起量化，>0=每组独立(per-group)量化。
# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w


"""
pseudo_quantize_model_weight: 对模型中所有 nn.Linear 层执行伪量化（均匀量化，无保护策略）。
这是一个基准函数，困惑度通常非常高，演示了普通量化的问题。
"""


# ── 函数：pseudo_quantize_model_weight ──
# 基准量化：对所有 nn.Linear 层做普通均匀量化（不保护任何权重）。
# 困惑度通常很高(~284)，演示了朴素量化的问题。
@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_group_size,
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )


# 评估量化后的 3 比特模型的困惑度和模型大小
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size / MiB:.2f} MiB")

"""
可以看出，模型大小减小了，但困惑度显著升高了

在LLM的激活中有一个现象：离群值只出现在一小部分通道里。如果某个通道存在离群值，它会在所有 token 中持续出现。
对于给定的 token， 各通道之间的方差很大(某些通道的激活非常大，但大多数都很小)；而对于给定得通道，其幅度在不同token
之间的方差却很小(离群值通道始终保持较大的值)

根据 AWQ 的观察，与激活离群值对应的权重通道更为重要(salient),保留这些重要权重可以带来
显著的性能提升。接下来，我们尝试找出这些重要权重并将其保留为原始值，观察困惑度的变化

下面的代码用于加载校准数据集，从而得到激活离群值，进而识别出重要权重
"""


"""
get_calib_dataset: 加载 Pile 验证集的一个子集作为校准数据。
返回 n_samples 条、每条截断到 block_size token 的输入序列列表。
校准数据用于统计每层激活的分布，从而识别重要权重通道。
"""


# ── 函数：get_calib_dataset ──
# 从 Pile 验证集子集加载校准数据。返回 n_samples 条截断到 block_size token 的序列。
# 校准数据用于统计每层激活的分布，从而识别重要权重通道。
def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


"""
get_calib_feat: 在校准数据集上运行前向传播，通过 forward hook 收集每个 nn.Linear 层的
输入激活的绝对值均值（按通道维度）。返回值 input_dict[name] 是每个 batch 的激活均值列表，
对这些数据做 sum 即可得到每个通道的重要性分数。
"""


# ── 函数：get_calib_feat ──
# 在校准数据上跑前向传播，通过 forward hook 收集每个 nn.Linear 输入激活的绝对值均值(按通道)。
# 返回 input_dict[name] — 对每个 batch 的激活均值做 sum 即得各通道的重要性分数。
@torch.no_grad()
def get_calib_feat(model, tokenizer):
    input_dict = dict()

    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(partial(stat_input_max_hook, name=name))
            )

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return input_dict


del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
input_feat = get_calib_feat(model, tokenizer)

"""
### 问题 1（50 分）

#### 问题 1.1（20 分）

接下来，在量化前后添加代码，以保护 1% 的重要权重通道(重要性最高的 1% 通道)，确保
它们的值在量化后保持不变。(期望的困惑度为 17.15)
"""


# ── 函数：pseudo_quantize_model_salient_weight_fp16 ──
# 混合精度保护：找最重要的1%权重→量化全部→把1%恢复为FP16。期望困惑度≈17.15。
@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(model, w_bit, q_group_size, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 1A】找出 1% 最重要通道 — 你需要用 torch.topk(importance, k, largest=True).indices
            # k = max(1, int(0.01 * importance.shape[0]))，保证至少保留 1 个通道
            outlier_indices = None

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the salient weight channels
            outlier = m.weight.data[:, outlier_indices].clone()

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 1B】恢复重要通道 — 将备份的 outlier 覆盖回量化后权重的 outlier_indices 列
            # 即 m.weight.data[:, outlier_indices] = outlier

            ############### YOUR CODE ENDS HERE #################


del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_salient_weight_fp16(
    model, w_bit=3, q_group_size=128, input_feat=input_feat
)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size / MiB:.2f} MiB")


"""
#### 问题 1.2（15 分）

做一个消融实验：随机保护 1% 的权重通道，确保它们的值在量化后保持不变，然后观察困惑度

期望的困惑度超过 100

"""


# ── 函数：pseudo_quantize_model_random_weight_fp16 ──
# 消融实验：随机选1%通道保护为FP16（而非按重要性）。期望困惑度>100，证明随机无效。
@torch.no_grad()
def pseudo_quantize_model_random_weight_fp16(model, w_bit, q_group_size, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 2A】随机选 1% 通道 — 用 torch.randperm(通道数)[:k] 随机取索引
            # k = max(1, int(0.01 * importance.shape[0]))
            outlier_mask = None

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the selected weight channels
            outlier = m.weight.data[:, outlier_mask].clone()

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 2B】恢复选中通道 — 同 1B: m.weight.data[:, outlier_mask] = outlier

            ############### YOUR CODE ENDS HERE #################


del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_random_weight_fp16(
    model, w_bit=3, q_group_size=128, input_feat=input_feat
)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size / MiB:.2f} MiB")

"""
#### 问题 1.3（15 分）

给出一个可能的解释，说明为什么这些重要权重如此重要

#### 回答 1.3：

############### YOUR ANSWER STARTS HERE #################

# 【回答 1.3】请解释为什么与激活离群值对应的权重通道如此重要。
# 提示：从量化误差公式 Err = Delta * RoundErr(w/Delta) * x 出发，
#       说明 x 大的通道对量化误差贡献更大，因此保护这些通道的权重能显著降低整体误差。

############### YOUR ANSWER ENDS HERE #################
"""

"""
### 问题 2（50 分）

虽然将 0.1% 的权重保留为 FP16 可以提升量化后的性能，
而且不会明显增加模型大小（以总比特数衡量），但这种混合精度的数据类型会让系统实现变得困难。
我们需要想出一种方法，在不真正把重要权重保留为 FP16 的前提下保护它们。

根据 AWQ 的方法，只需将重要权重通道放大就能保护它们。原理如下：

- 考虑一个线性层通道 y = w * x（来自 Wx）。我们关心的是 Q(w)x 带来的量化误差。
- Err(Q(w)x) = Delta * RoundErr(w / Delta) * x
- 误差与权重本身无关，而是与输入 x 的大小和 RoundErr() 有关

关键思想：将输入 x 缩放为 x / s，将 w 缩放为 w * s 使结果不变 (w * s) * (x / s) = w * x。
由于重要通道的 x 更大，误差 Err 也会更大，因此将 x 缩小就等价于将误差缩小。
但另一方面，缩放 w 则不会影响 RoundErr，因为 w 被整体缩放了（相对误差不变）。

因此，通过将重要通道的输入缩小(除以 s)，我们可以减少量化误差，而对应的权重则放大(乘以 s)来保持结果不变。
权重放大后更容易被量化(相对误差更小)，之后在输入处再缩小回来即可。

"""

"""
#### 问题 2.1（20 分）
编写代码，将重要权重通道放大，然后量化，最后再缩小回去，并观察困惑度变化。

期望的困惑度为 18.93
"""


# ── 函数：pseudo_quantize_model_weight_scaleup ──
# AWQ 放大-量化-缩小策略：重要权重×scale_factor→量化→÷scale_factor 恢复。期望困惑度≈18.93。
@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 3A】找出 1% 重要通道 — 与 1A 相同: torch.topk(importance, k, largest=True).indices
            outlier_mask = None

            ############### YOUR CODE ENDS HERE #################

            # To simulate applying the scale factor, we can simply multiply it before quantization, and then divide by the scale factor after quantization.
            # Scale up the values of the salient weight channels
            m.weight.data[:, outlier_mask] *= scale_factor

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 3B】缩小回来 — 将 outlier_mask 列除以 scale_factor: m.weight.data[:, outlier_mask] /= scale_factor

            ############### YOUR CODE ENDS HERE #################


del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_scaleup(
    model, w_bit=3, q_group_size=128, input_feat=input_feat, scale_factor=2
)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size / MiB:.2f} MiB")


"""
#### 问题 2.2（15 分）
在代码中尝试不同的缩放因子(例如 1 2 3 4),并观察困惑度的变化

是否观察到困惑度先下降后上升？请根据上面提到的原理解释为什么会出现这种情况。

#### 回答 2.2：

############### YOUR ANSWER STARTS HERE #################

# 【回答 2.2】缩放因子(1,2,3,4)下困惑度先降后升的原因：
# 提示：缩放太小(≈1)时重要权重未得到充分保护，量化误差仍大；
#       缩放太大时非重要权重被过度放大，它们的量化相对误差增加。
#       最优缩放因子在"保护重要通道"与"不损害普通通道"之间取得平衡。

############### YOUR ANSWER ENDS HERE #################
"""

"""
#### 问题 2.3（15 分）

由于微调过程不稳定，在预先定义的搜索空间内寻找最优的 s 是更好的选择。我们可以在搜索空间中找到
最优的缩放值，在保护重要权重的同时也兼顾其他值。实践中可以观察到，仅考虑激活就足以得到不错的结果。

目标函数：

  L(s) = || Q(W * s) * (s^{-1} * X) - W * X ||,
  s = s_X^{alpha}

  s* = argmin_s L(s)

即搜索最优的 alpha 使量化前后输出误差最小。添加搜索的代码并运行，观察困惑度。(期望的困惑度为 17.92)
"""


# ── 函数：scale_ln_fcs ──
# 层间缩放：LayerNorm 输出除以 scales，线性层权重乘以 scales。
# 效果是把权重的放大转移到前一层的 LayerNorm 缩小上，保持数学等价。
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


# ── 函数：scale_fc_fc ──
# 层间缩放：fc1 输出通道权重除以 scales，fc2 输入通道权重乘以 scales。
# 用于在两个线性层之间传递缩放，保持整体输出不变。
@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


# ── 函数：auto_scale_block ──
# 自动搜索最优缩放因子：在 [0,1] 以 20 个网格点搜索最优 alpha（scales = s_x^alpha）。
# 内部 _search_module_scale 枚举所有 alpha，找量化输出误差最小的那个。期望困惑度≈17.92。
@torch.no_grad()
def auto_scale_block(module, name, w_bit, q_group_size, input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        ############### YOUR CODE STARTS HERE ###############

        # 【填空 4A】初始化搜索变量: best_error = float('inf'), best_ratio = -1, best_scales = None
        best_error = None
        best_ratio = None
        best_scales = None

        ############### YOUR CODE ENDS HERE #################

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 4B】计算 scales = s_x^alpha: scales = s_x ** ratio
            scales = None

            ############### YOUR CODE ENDS HERE #################

            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:
                scales = scales.to(fc.weight.device)

                # Scale up the values of the weight channels
                fc.weight.mul_(scales)

                fc.weight.data = pseudo_quantize_tensor(
                    fc.weight.data, w_bit, q_group_size
                )

                ############### YOUR CODE STARTS HERE ###############

                # 【填空 4C】缩小回来 — fc.weight.div_(scales.view(1, -1))

                ############### YOUR CODE ENDS HERE #################

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # attention input
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # attn out
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(
        module.self_attn.out_proj, [module.self_attn.out_proj], inp
    )
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # fc1
    inp = input_feat[name + ".fc1"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # fc2
    inp = input_feat[name + ".fc2"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)


# ── 函数：pseudo_quantize_model_weight_auto_scale ──
# 对模型每个 DecoderLayer 自动搜索最优缩放因子，然后将缩放应用到 LayerNorm/FC 之间。
# 最后对所有 Linear 层执行伪量化。这是 AWQ 的完整自动缩放流程。
@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(model, w_bit, q_group_size, input_feat):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )


del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_auto_scale(
    model, w_bit=3, q_group_size=128, input_feat=input_feat
)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size / MiB:.2f} MiB")

"""
## 附加分

你有没有想到什么不使用混合精度的优化技术？试着实现它们，进一步降低困惑度吧！如果你能把困惑度进一步降低到 x,
就可以获得max(0,(17.92-x)x10)的附加分！

总之，我们无需使用混合精度就能显著降低困惑度。通过高效的 kernel实现，4比特模型可以在推理时获得不错的加速。
下一节学习 TinyChatEngine 之后，我们就能像简介中的演示那样，在自己的笔记本电脑上运行 LLaMA-7B 模型。
"""
