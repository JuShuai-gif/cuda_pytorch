# 使用 AWQ 进行 LLM 量化

"""
=============================================================================
AWQ (Activation-aware Weight Quantization) — 一句话核心
=============================================================================
LLM 的激活中存在"离群通道"：少数通道的激活值远大于其他通道，且在所有 token 上
持续出现。这些通道对应的权重如果被量化，误差会被大激活放大，严重损害模型质量。

AWQ 的解决方案：找到这些离群通道 → 把它们的权重放大 s 倍 → 量化全部权重 →
再把权重缩小 s 倍。数学上等价于"输入缩小 s 倍后再量化"，但不需要改前向代码。

效果对比（3-bit, OPT-6.7B, WikiText-2 PPL）：
    朴素均匀量化         ≈ 284
    AWQ (固定 s=2)       ≈ 18.93
    AWQ (自动搜 α)       ≈ 17.92
    混合精度 (1% FP16)   ≈ 17.15  （精度最高，但需要 FP16 推理单元）

为什么放大权重后量化误差反而变小？
    量化误差公式：Err = Δ · round_err(w/Δ) · x
    放大后 w' = w·s, 反推输入等效为 x/s：Err' ≈ Err / s
    round_err 是取整损失（始终在 [-0.5, 0.5]），与数值大小无关。
    所以放大 s 倍后，误差直接缩小为原来的 1/s。

核心概念：per-group 量化、校准数据集、激活离群值、重要性通道、grid search
=============================================================================
"""

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
exit(0)

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
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    # 保存原始形状，最后恢复用
    org_w_shape = w.shape
    # 如果开启了 per-group 量化
    if q_group_size > 0:
        # 确保最后一维可以均匀分割成组
        assert org_w_shape[-1] % q_group_size == 0
        # 将张量重排成 (num_groups, group_size) 形状
        w = w.reshape(-1, q_group_size)

    # 确保当前形状是 2 维（num_groups, group_size 或 1, dim）
    assert w.dim() == 2

    # 沿每行（每组）取最大值，即公式中的 alpha
    max_val = w.amax(dim=1, keepdim=True)
    # 断言形状正确：每行有一个最大值
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    # 沿每行取最小值，即公式中的 beta
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # n_bit 量化能表示的最大整数值（如 4bit 时 max_int = 15）
    max_int = 2**n_bit - 1
    # 计算缩放因子 scale = (max - min) / max_int，避免除以 0 而 clamp 到 1e-5
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    # 计算零点 zero_point = round(-min / scale)，截断到 [0, max_int]
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    # 检查缩放因子和权重中都没有 NaN
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # 量化：w_int = clamp(round(w / scale) + zero_point, 0, max_int)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    # 断言量化后形状不变且组大小正确
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # 反量化（伪量化）：w_fp = (w_int - zero_point) * scale，恢复回浮点值
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # 反量化后也不应有 NaN
    assert torch.isnan(w).sum() == 0

    # 恢复原始形状并返回
    w = w.reshape(org_w_shape)
    return w


"""
pseudo_quantize_model_weight: 对模型中所有 nn.Linear 层执行伪量化（均匀量化，无保护策略）。
这是一个基准函数，困惑度通常非常高，演示了普通量化的问题。
"""


# ── 函数：pseudo_quantize_model_weight ──
# 基准量化：遍历模型所有子模块，对 nn.Linear 层的权重做普通的均匀伪量化。
# 不保护任何"重要"权重，困惑度通常非常高（~284），用作对比基线。
@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_group_size,
):
    # 遍历模型的所有模块（含嵌套子模块）
    for n, m in model.named_modules():
        # 只处理线性层，跳过 embedding / norm 等其他层
        if isinstance(m, nn.Linear):
            # 原地替换该层的权重为伪量化后的版本
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )


# ── 评估 3 比特均匀量化的效果 ──
# 先释放之前的模型，清理显存，防止 OOM
del model
gc.collect()
torch.cuda.empty_cache()
# 重新加载原始模型（FP16）
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# 应用 3 比特均匀量化，组大小为 128
pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)

# 计算量化后模型的困惑度和理论模型大小
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
# 打印结果：困惑度 vs 压缩后大小
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

# 收集校准数据集中每个 Linear 层的激活统计信息
"""
具体流程：
1. stat_input_max_hook — 注册到每个 nn.Linear 层的 forward hook，在该层被调用时捕获输入激活，沿 token 维度求平均绝对值（abs().mean(dim=0)），得到一个 (hidden_dim,) 的向量，存入 input_dict[name] 列表。
2. get_calib_dataset 获取校准样本，逐条送入模型前向推理。
3. 所有样本跑完后，input_dict 中每个 Linear 层名对应一个列表 [sample1_max, sample2_max, ...]，记录了该层在不同样本上的平均激活幅度。
这个结果后续用于 AWQ 中识别激活离群值所在的通道，进而决定哪些权重通道需要保护（不量化或保留更高精度）

后面如何进行判断的?
核心判断逻辑在 pseudo_quantize_model_salient_weight_fp16（第 357 行）：
importance = sum(input_feat[n]).float()           # ①
outlier_indices = topk(importance, k).indices      # ②
outlier = m.weight.data[:, outlier_indices].clone() # ③ 备份
m.weight.data = pseudo_quantize_tensor(...)         # ④ 全部量化
m.weight.data[:, outlier_indices] = outlier         # ⑤ 恢复 1%
四步：
1. 聚合重要性 — input_feat[n] 是 [num_samples, hidden_dim] 的列表，sum() 将所有样本的激活绝对值均值按通道累加，得到每个通道的标量重要性（越大表示该通道在所有 token 上激活幅度越大）。
2. 选出 top 1% — torch.topk(importance, k, largest=True).indices 取激活量最大的 k 个通道索引，这些就是"离群通道"。
3. 备份权重 — 把这些通道的列权重克隆为 FP16。
4. 全部量化 — 对整个权重做普通均匀量化。
5. 恢复保护 — 将备份的 FP16 权重覆盖回量化后的张量，让这些关键通道不损失精度。
消融实验（random_weight_fp16）把步骤 ② 换成 torch.randperm 随机选，困惑度从 ~17 飙升到 >100，证明靠激活幅度筛选是关键。
"""


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
            k = max(1, int(0.01 * importance.shape[0]))
            outlier_indices = torch.topk(importance, k, largest=True).indices

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the salient weight channels
            outlier = m.weight.data[:, outlier_indices].clone()

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 1B】恢复重要通道 — 将备份的 outlier 覆盖回量化后权重的 outlier_indices 列
            # 即 m.weight.data[:, outlier_indices] = outlier
            m.weight.data[:, outlier_indices] = outlier
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
            k = max(1, int(0.01 * importance.shape[0]))
            outlier_mask = torch.randperm(k)[:k]

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the selected weight channels
            outlier = m.weight.data[:, outlier_mask].clone()

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 2B】恢复选中通道 — 同 1B: m.weight.data[:, outlier_mask] = outlier
            m.weight.data[:, outlier_mask] = outlier

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

#### 量化误差分析

考虑一个线性层输出 y = Wx，其中某个通道为 y_i = w_i · x_i。
量化权重后，输出变为 Q(w_i) · x_i，产生的误差为：

    Err = (Q(w_i) - w_i) · x_i

均匀量化以步长 Δ = (max-min)/(2^n-1) 将连续值映射到整数。Q(w_i) - w_i 可分解为：

    Q(w_i) - w_i = Δ · round_err(w_i / Δ)

其中 round_err ∈ [-0.5, 0.5] 是取整损失。代入得：

    Err = Δ · round_err(w_i / Δ) · x_i

关键观察：**误差与激活 x_i 成正比**。重要通道的 x_i 远大于普通通道，因此同样的
权重量化误差会被 x_i 放大，造成不成比例的输出误差。

#### AWQ 的核心思想：输入缩放 = 误差缩放

对重要通道做数学等价的变换：

    w_i · x_i = (w_i · s) · (x_i / s)

即：将权重放大 s 倍，输入缩小 s 倍，结果不变。现在量化放大后的权重 Q(w_i · s)：

    Err' = (Q(w_i · s) - w_i · s) · (x_i / s)
         = Δ · round_err(w_i · s / Δ) · (x_i / s)

round_err 的分布与数值大小无关（始终在 [-0.5, 0.5]），因此：

    round_err(w_i · s / Δ) ≈ round_err(w_i / Δ)     （分布不变）
    ⇒  Err' ≈ Δ · round_err(w_i / Δ) · x_i / s = Err / s

**误差直接缩小了 s 倍！** 本质是：输入的缩小直接将 x_i 的放大效应抵消掉了。

#### 另一种直观理解：有效精度提升

放大 s 倍后再量化，等价于使用更细的量化步长（Δ/s）来量化原始权重 w_i，
从而为核心通道分配了更多量化 levels，提升了有效精度。由于这些通道的 x_i 很大，
它们对输出贡献大，也更值得分配高精度。

#### 对比混合精度方案

| 方案 | 操作 | 优缺点 |
|------|------|--------|
| 混合精度 (1.1) | 量化→1%通道恢复FP16 | 困惑度更低(~17.15)，但需要 FP16 单元推理，硬件支持复杂 |
| AWQ 缩放 (2.1) | 放大→量化→缩小 | 困惑度略高(~18.93)，但全张量 INT 推理，无混合精度开销 |

AWQ 用轻微的精度代价换来了纯整数量化推理的便利性，更适合实际部署。
"""

"""
#### 问题 2.1（20 分）
编写代码，将重要权重通道放大，然后量化，最后再缩小回去，并观察困惑度变化。

期望的困惑度为 18.93
"""


# ── 函数：pseudo_quantize_model_weight_scaleup ──
# AWQ 放大-量化-缩小策略：放大重要通道×s→全体量化→缩小÷s。无需混合精度，全 INT 推理。
# 原理：Err' = Err / s，期望困惑度≈18.93（混合精度≈17.15，纯量化≈284）。
@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    # 遍历模型所有子模块
    for n, m in model.named_modules():
        # 只处理线性层
        if isinstance(m, nn.Linear):
            # 将 input_feat[n] 中所有校准样本的激活均值累加，得到每个通道的重要性标量
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # 计算要保护的通道数 k = 1%（至少 1 个通道）
            k = max(1, int(0.01 * importance.shape[0]))
            # 选出重要性最高的 k 个通道索引（激活离群值对应的权重通道）
            outlier_mask = torch.topk(importance, k, largest=True).indices

            ############### YOUR CODE ENDS HERE #################

            # 放大阶段：将重要通道的权重乘以 scale_factor（如 s=2）
            # 等价于"把输入缩小 s 倍"的数学对偶操作
            m.weight.data[:, outlier_mask] *= scale_factor

            # 对放大后的整体权重做均匀伪量化
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )

            ############### YOUR CODE STARTS HERE ###############

            # 缩小阶段：将重要通道的权重除以 scale_factor，恢复原始数值范围
            # 这里并没有真正去缩输入（x），而是通过对偶的权重缩放来等价实现
            m.weight.data[:, outlier_mask] /= scale_factor
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
#
# 核心原因：per-group 量化中，Δ = (max-min)/max_int 由组内最大值和最小值决定。
# 当 s 增大时，重要通道的权重被放大，导致它们所在组（group_size=128）的
# 动态范围（max - min）展宽 → Δ 增大 → 组内所有通道（含重要和非重要）的
# 量化步长变粗，量化误差增大。
#
# 所以存在 s 的"最佳平衡点"：
#   s 太小（≈1）：重要通道未受充分保护，量化误差以 Err·x 形式放大
#   s 太大（≥3）：重要通道所在组的 Δ 被过度展宽，组内所有通道精度下降
#   s=2 左右：保护收益与组内精度损失取得平衡，困惑度最低

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
# 将缩放因子应用到 LayerNorm → Linear 的路径上。
# 数学等价转换：y = (LN(x) / s) · (W · s) = LN(x) · W，输出不变。
# 目的：把权重的放大"转嫁"到前一层的 LayerNorm 缩小，使得 weight 可以在推理前
#       融合进 LN 的 weight/bias，不需要运行时额外处理。
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    # 确保 fcs 是列表，方便统一遍历
    if not isinstance(fcs, list):
        fcs = [fcs]

    # 将 scales 搬到 LayerNorm 所在设备
    scales = scales.to(ln.weight.device)

    # LayerNorm 的 weight 除以 scales：等价于输出缩小
    ln.weight.div_(scales)
    # 如果有 bias，也同步缩小
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    # 后续各个 Linear 的 weight 乘以 scales：等价于权重放大，抵消 LN 的缩小
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    # 断言：缩放后所有参数不含 NaN
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


# ── 函数：scale_fc_fc ──
# 将缩放因子应用到 fc1 → fc2 的路径上（两个 Linear 之间）。
# 数学等价转换：W2 · (W1 · x / s) · s = W2 · W1 · x，输出不变。
# 应用场景：Attention 中 v_proj → out_proj 和 FFN 中 fc1 → fc2。
@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    # 将 scales 搬到 fc1 所在设备
    scales = scales.to(fc1.weight.device)

    # fc1 的输出通道权重缩小（只处理最后 scales.size(0) 个通道，即输出维度）
    # 注：注释掉的 fc1.weight.div_(scales.view(-1, 1)) 是全部缩小，
    #     但实际只缩小输出通道中与 scales 对应的部分
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    # fc1 的 bias 也同步缩小
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # fc2 的输入通道权重放大，抵消 fc1 输出的缩小
    fc2.weight.mul_(scales.view(1, -1))

    # 断言：缩放后所有参数不含 NaN
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


# ── 函数：auto_scale_block ──
# 对一个 DecoderLayer（OPTDecoderLayer）自动搜索最优缩放因子。
# 原理：在 α ∈ {0/20, 1/20, ..., 20/20} 共 21 个网格点上，
#       计算 scales = s_x^α（s_x 是激活绝对值均值），
#       对 linear 层做"放大 → 量化 → 缩小"后，
#       以 block 输出 MSE 最小为目标选出最优 α。
#       然后将最优 scales 通过 scale_ln_fcs / scale_fc_fc 固化到模型中。
# 期望困惑度 ≈ 17.92。
@torch.no_grad()
def auto_scale_block(module, name, w_bit, q_group_size, input_feat):

    # ── 内嵌函数：_search_module_scale ──
    # 对给定 block（如 self_attn 或 fc1）和需要缩放的 Linear 列表，
    # 网格搜索最优 α，返回使输出 MSE 最小的 per-channel scales。
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        # 将校准输入搬到 block 所在设备
        x = x.to(next(block.parameters()).device)
        # 记录 block 的原始输出（FP16），作为量化后对比的 ground truth
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        # 计算 per-channel 的激活绝对值均值 s_x，作为缩放基础
        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        ############### YOUR CODE STARTS HERE ###############

        # 【填空 4A】初始化搜索变量: best_error = float('inf'), best_ratio = -1, best_scales = None
        # best_error 追踪当前遇到的最小输出 MSE
        # best_ratio 记录该 MSE 对应的 alpha 值（0~1）
        # best_scales 记录对应的 per-channel 缩放向量
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        ############### YOUR CODE ENDS HERE #################

        n_grid = 20  # 网格点数（实际遍历 0/20 ~ 20/20 共 21 个点）
        history = []  # 记录每个 alpha 的 MSE，用于调试

        # 保存 block 的原始 state_dict，每次尝试后恢复
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid + 1):  # 遍历 0..20 → α = 0, 0.05, ..., 1.0
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############

            # 【填空 4B】计算 scales = s_x^alpha: scales = s_x ** ratio
            # ratio 越大，重要通道的缩放越大；ratio=0 时 scales=1（不做缩放）
            scales = s_x**ratio

            ############### YOUR CODE ENDS HERE #################

            # 归一化 scales：除以 sqrt(max * min)，保证 scales 的整体几何均值为 1
            # 避免整体偏移影响后续 LayerNorm 参数
            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            # 对列表中每个 Linear 层应用缩放 → 量化 → 恢复
            for fc in linears2scale:
                scales = scales.to(fc.weight.device)

                # 放大重要通道：权重乘以 per-channel 的 scales
                fc.weight.mul_(scales)

                # 对放大后的权重做伪量化（模拟 INT 推理）
                fc.weight.data = pseudo_quantize_tensor(
                    fc.weight.data, w_bit, q_group_size
                )

                ############### YOUR CODE STARTS HERE ###############

                # 【填空 4C】缩小回来 — fc.weight.div_(scales.view(1, -1))
                fc.weight.div_(scales.view(1, -1))

                ############### YOUR CODE ENDS HERE #################

            # 计算量化后的 block 输出
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            # MSE 损失：量化输出 vs FP16 原始输出
            loss = (org_out - out).float().pow(2).mean().item()
            history.append(loss)
            # 如果当前 loss 更小，更新最优解
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            # 恢复 block 参数，为下一个 alpha 做准备
            block.load_state_dict(org_sd)

        # 如果 best_ratio 仍然是 -1，说明所有 alpha 都未更新（不应发生）
        if best_ratio == -1:
            print(history)
            raise Exception

        # 拉平 scales 为 1D 向量，后续传给 scale_ln_fcs / scale_fc_fc 使用
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # ── 第 1 步：Attention 的 Q/K/V 投影 ──
    # 使用 self_attn 的整体输出激活作为校准信号
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    # 搜索最优 scales，使 self_attn 整体输出误差最小
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    # 将 scales 应用到 self_attn_layer_norm → Q/K/V 路径
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # ── 第 2 步：Attention 的输出投影 ──
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(
        module.self_attn.out_proj, [module.self_attn.out_proj], inp
    )
    # 将 scales 应用到 v_proj → out_proj 路径
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # ── 第 3 步：FFN 的第一层 fc1 ──
    inp = input_feat[name + ".fc1"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    # 将 scales 应用到 final_layer_norm → fc1 路径
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # ── 第 4 步：FFN 的第二层 fc2 ──
    inp = input_feat[name + ".fc2"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    # 将 scales 应用到 fc1 → fc2 路径
    scale_fc_fc(module.fc1, module.fc2, final_scales)


# ── 函数：pseudo_quantize_model_weight_auto_scale ──
# AWQ 的完整自动缩放 + 量化流程：
#   1. 遍历每个 OPTDecoderLayer，对其 4 个子路径分别搜索最优 scales
#   2. 通过 scale_ln_fcs / scale_fc_fc 将最优 scales 固化到模型参数中
#   3. 最后对所有 Linear 层执行一次伪量化
# 全程无需混合精度，所有缩放融合到 LN/FC 参数中，推理时零额外开销。
# 期望困惑度 ≈ 17.92（vs 混合精度 17.15，纯量化 284）。
@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(model, w_bit, q_group_size, input_feat):
    # 导入 OPT 的 DecoderLayer 类型，只对 DecoderLayer 做自动缩放
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    # 遍历模型所有子模块，找到 DecoderLayer 进行自动缩放
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    # 缩放融合完成后，对所有 Linear 层执行最终的伪量化
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )


# 加载模型并执行 AWQ 自动缩放 + 量化流程
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_auto_scale(
    model, w_bit=3, q_group_size=128, input_feat=input_feat
)

# 评估量化后模型的困惑度和理论模型大小
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
