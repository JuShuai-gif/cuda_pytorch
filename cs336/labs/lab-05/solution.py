"""
Lab 05 解答: Data Pipeline & Alignment

数据管线、DPO 损失和 GRPO 的完整实现。
"""

from __future__ import annotations

import re
import hashlib
import math
from collections import Counter
from typing import List, Dict, Tuple, Set

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 任务 1: 数据管线
# ══════════════════════════════════════════════════════════════════════


class TextCleaner:
    """在训练前清洗原始网页文本。"""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<[^>]+>")
    EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")

    # 激进策略: 仅保留可打印 ASCII + 常见 Unicode + 空白字符
    CLEAN_PATTERN = re.compile(r"[^\w\s.,!?;:'\"()\[\]{}@#$%^&*+=/\\\-]")

    @staticmethod
    def clean(text: str) -> str:
        # 1. 移除 HTML 标签
        text = TextCleaner.HTML_PATTERN.sub(" ", text)
        # 2. 替换 URL
        text = TextCleaner.URL_PATTERN.sub("<URL>", text)
        # 3. 替换邮箱
        text = TextCleaner.EMAIL_PATTERN.sub("<EMAIL>", text)
        # 4. 移除非 ASCII 字符（保留基本可打印字符）
        text = text.encode("ascii", errors="ignore").decode("ascii")
        # 5. 移除异常字符
        text = TextCleaner.CLEAN_PATTERN.sub(" ", text)
        # 6. 规范化空白字符
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def clean_batch(texts: List[str]) -> List[str]:
        return [TextCleaner.clean(t) for t in texts]


class QualityFilter:
    """基于规则的训练数据质量过滤。"""

    def __init__(
        self,
        min_chars: int = 100,
        max_chars: int = 100_000,
        max_word_repetition_ratio: float = 0.3,
        min_avg_word_length: float = 2.0,
        max_avg_word_length: float = 10.0,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_word_repetition_ratio = max_word_repetition_ratio
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length

    def should_keep(self, text: str) -> bool:
        # 1. 长度检查
        n_chars = len(text)
        if n_chars < self.min_chars or n_chars > self.max_chars:
            return False

        # 2. 单词重复率
        words = text.split()
        if len(words) == 0:
            return False
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        rep_ratio = most_common_count / len(words)
        if rep_ratio > self.max_word_repetition_ratio:
            return False

        # 3. 平均单词长度
        avg_word_len = sum(len(w) for w in words) / len(words)
        if (
            avg_word_len < self.min_avg_word_length
            or avg_word_len > self.max_avg_word_length
        ):
            return False

        return True

    def filter(self, texts: List[str]) -> List[str]:
        return [t for t in texts if self.should_keep(t)]


class DataMixer:
    """按指定比例混合多个数据源。"""

    def __init__(self, ratios: Dict[str, float]):
        total = sum(ratios.values())
        self.ratios = {k: v / total for k, v in ratios.items()}  # 归一化

    def sample_batch(self, sources: Dict[str, List[str]], batch_size: int) -> List[str]:
        import random

        sources_list = list(self.ratios.keys())
        probs = [self.ratios[s] for s in sources_list]
        batch: List[str] = []

        for _ in range(batch_size):
            src = random.choices(sources_list, weights=probs, k=1)[0]
            samples = sources.get(src, [])
            if samples:
                batch.append(random.choice(samples))
            else:
                batch.append("")  # 回退

        return batch

    @staticmethod
    def compute_minhash_signature(
        text: str, n_gram: int = 3, num_hashes: int = 128
    ) -> List[int]:
        """计算 MinHash 签名用于近似去重。"""
        # 生成 n-grams
        ngrams: Set[int] = set()
        for i in range(len(text) - n_gram + 1):
            ng = text[i : i + n_gram]
            # 使用简单的哈希，不同种子对应多个哈希函数
            for seed in range(num_hashes):
                h = hashlib.md5(f"{seed}:{ng}".encode()).hexdigest()
                ngrams.add(int(h[:8], 16))
            break  # 只需要一次遍历，每个种子添加所有 ngrams 一次
        # 实际上，典型的 MinHash 实现是不同的。
        # 简化版: 直接对每个 n-gram 使用哈希函数。

        if not ngrams:
            return [0] * num_hashes

        signatures = [float("inf")] * num_hashes
        for i in range(len(text) - n_gram + 1):
            ng = text[i : i + n_gram]
            for seed in range(num_hashes):
                h = int(hashlib.md5(f"{seed}:{ng}".encode()).hexdigest()[:8], 16)
                if h < signatures[seed]:
                    signatures[seed] = h
        return signatures


# ══════════════════════════════════════════════════════════════════════
# 任务 2: DPO Loss
# ══════════════════════════════════════════════════════════════════════


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 Rafailov et al. (2023) 提出的 DPO 损失。

    L_DPO = -log σ(β * Δ)

    其中 Δ = log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))
         = (log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))
    """
    # 计算 log-ratios（隐式 rewards）
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps

    # log-ratios 的差值
    logits = beta * (policy_log_ratio - reference_log_ratio)

    # Sigmoid 损失（二元分类: chosen 应具有更高的概率）
    if label_smoothing > 0:
        # 平滑的二元标签
        targets = torch.ones_like(logits) * (1.0 - label_smoothing)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
    else:
        loss = -F.logsigmoid(logits).mean()

    # 用于日志记录: 计算隐式 rewards (beta * log π/π_ref)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return loss, chosen_rewards, rejected_rewards


# ══════════════════════════════════════════════════════════════════════
# 加分项: GRPO Loss
# ══════════════════════════════════════════════════════════════════════


def grpo_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    """简化版 GRPO (Group Relative Policy Optimization)。

    与带 clipped objective 的 PPO 类似，但 advantage 是
    以 group-relative 方式计算的（在同一 prompt 的响应组内）。

    L = -E[min(r * A, clip(r, 1-ε, 1+ε) * A) + β * KL]
    """
    # 重要性采样 ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL 散度惩罚（近似）
    kl = (old_log_probs - log_probs).mean()  # 简化的 KL 估计

    loss = policy_loss.mean() + beta * kl
    return loss


# ══════════════════════════════════════════════════════════════════════
# 任务 3: RLHF vs DPO — 答案
# ══════════════════════════════════════════════════════════════════════


def answer_alignment_questions() -> str:
    return """
Q1: 解释 RLHF 的三步流程
──────────────────────────
Answer:
RLHF (Reinforcement Learning from Human Feedback) 包含三个步骤：

Step 1 — Supervised Fine-Tuning (SFT):
  在一组高质量的 instruction-response pair 数据上 fine-tune 预训练模型，
  使模型学会遵循指令格式。这步使用标准的 next-token prediction loss。

Step 2 — Reward Model (RM) Training:
  对同一 prompt 采样多个 response，人工标注偏好（A > B > C）。
  根据 Bradley-Terry model 训练 reward model r(x, y) 来预测人类偏好。
  Loss: -log σ(r(x, y_w) - r(x, y_l))

Step 3 — PPO Fine-Tuning:
  使用 reward model 作为 reward signal，通过 PPO (Proximal Policy Optimization)
  优化策略模型。同时加入 KL penalty 防止模型过度偏离 SFT 模型:
  max_θ E[r(x, y)] - β·KL(π_θ || π_SFT)

Q2: DPO 如何通过 reparameterization 简化 RLHF？
────────────────────────────────────────────────
Answer:
DPO 的核心洞察来自 RLHF 最优策略的 closed-form 解：

在 RLHF 中，最优策略满足:
  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)

反解出 reward:
  r(x,y) = β · log(π*(y|x)/π_ref(y|x)) + β · log Z(x)

将 reward 代入 Bradley-Terry preference model:
  P(y_w ≻ y_l) = σ(r(x, y_w) - r(x, y_l))

Z(x) 项抵消！得到 DPO loss：
  L_DPO = -log σ(β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)))

这样就不需要单独训练 reward model，直接用 preference data
优化策略模型。这就是 "reparameterization" 的含义。

Q3: 分析 DPO 的优缺点（何时用 RLHF，何时用 DPO）
────────────────────────────────────────────────
Answer:
DPO 的优点：
  - 简单：不需要 reward model，训练 pipeline 更短
  - 稳定：标准的 binary cross-entropy loss，比 PPO 更稳定
  - 高效：只需要 2 个模型的显存（policy + reference），vs RLHF 的 4 个

DPO 的缺点：
  - 数据效率低：需要大量 preference data（每次更新只能用离线数据）
  - 无法在线探索：DPO 是离线算法，不能像 PPO 那样在线采样新 response
  - reward model 无法复用：RLHF 的 reward model 可用于多个 downstream task

何时用 RLHF：
  - 有资源训练和维护 reward model
  - 需要在线探索（当前策略生成的 response 与 reference 差异很大）
  - reward model 需要用于多个任务

何时用 DPO：
  - 有大量现成的 preference data
  - 追求简单和训练稳定性
  - 参考模型和当前模型差距不大

Q4: GRPO 与 DPO 的区别
───────────────────────
Answer:
GRPO (Group Relative Policy Optimization, DeepSeek 2024) 与 DPO 的主要区别：

1. 无需 pairwise preference：
   DPO 需要 (prompt, chosen, rejected) 三元组
   GRPO 只需要 (prompt, response_group)，在组内根据 reward 计算相对排名

2. 无需 reference model：
   DPO 需要一个 frozen reference model 来计算 KL divergence
   GRPO 使用组内 relative advantage，消除了 reference model 的需求

3. 计算方式：
   DPO: 直接优化 preference probability
   GRPO: 类似 PPO 的 clipped objective，但 advantage 来自 group relative ranking

4. 适用场景：
   DPO 更适合有明确 pairwise preference 的场景
   GRPO 更适合可以采样多个 response 并用 verifiable reward 评估的场景
     （如数学推理、代码生成等有 ground truth 的任务）
"""
    return answers


# ══════════════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Lab 05 解答验证 ===\n")

    # --- TextCleaner ---
    raw = "<html>Hello <b>world</b>! Visit https://example.com or email a@b.com</html>"
    cleaned = TextCleaner.clean(raw)
    print(f"Cleaned: '{cleaned}'")
    assert "<URL>" in cleaned, "URL should be replaced"
    assert "<EMAIL>" in cleaned, "Email should be replaced"
    assert "<html>" not in cleaned, "HTML should be removed"
    print("  TextCleaner: PASS")

    # --- QualityFilter ---
    qf = QualityFilter(min_chars=20, max_chars=1000)
    good = "This is a good quality text with proper words and reasonable length"
    bad_short = "too short"
    bad_repeat = "hello hello hello hello hello hello hello hello hello hello"

    assert qf.should_keep(good), "Good text should pass"
    assert not qf.should_keep(bad_short), "Short text should fail"
    # 重复率测试:
    words = bad_repeat.split()
    rep = Counter(words).most_common(1)[0][1] / len(words)  # 10/10 = 1.0
    assert rep > 0.3, f"Expected high repetition ratio, got {rep}"
    assert not qf.should_keep(bad_repeat), "Repetitive text should fail"
    print("  QualityFilter: PASS")

    # --- DPO Loss ---
    torch.manual_seed(42)
    # 场景: chosen 有更高的 log prob → loss 应较小
    p_chosen = torch.tensor([-0.5, -1.0, -0.3])
    p_rejected = torch.tensor([-2.0, -3.0, -2.5])
    ref_chosen = torch.tensor([-0.8, -1.2, -0.5])
    ref_rejected = torch.tensor([-0.8, -1.2, -0.5])

    loss, rewards_c, rewards_r = dpo_loss(
        p_chosen, p_rejected, ref_chosen, ref_rejected, beta=0.5
    )
    print(f"\nDPO Loss: {loss.item():.4f}")
    print(f"  Chosen rewards (均值): {rewards_c.mean().item():.4f}")
    print(f"  Rejected rewards (均值): {rewards_r.mean().item():.4f}")

    # Chosen 应有比 rejected 更高的 reward（全部样本）
    assert (rewards_c > rewards_r).all(), "Chosen should have higher reward"
    print("  DPO: PASS")

    # 场景: rejected 有更高的 log prob → loss 应更大
    loss_bad, _, _ = dpo_loss(p_rejected, p_chosen, ref_chosen, ref_rejected, beta=0.5)
    assert loss_bad > loss, "Loss should be higher when chosen is worse"
    print("  DPO (bad case): PASS")

    # --- 知识问答 ---
    print("\n" + "=" * 60)
    print(answer_alignment_questions())
