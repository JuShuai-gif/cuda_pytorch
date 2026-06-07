"""
组合多个预训练数据集的数据混合策略。

实现了三种混合策略并可视化生成的
token 分布：

1. **按比例混合（Proportional mixing）** – 按每个数据集的大小比例采样。
2. **基于温度的采样（Temperature-based sampling）** – 将采样概率的指数调整为 ``1/T``
   以平滑或锐化分布。
3. **领域加权混合（Domain-weighted mixing）** – 为不同领域（代码、网页、书籍等）
   分配显式权重。

所有演示均使用合成数据；无需外部数据依赖。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

# Third-party
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np


# ---------------------------------------------------------------------------
# 合成数据集生成
# ---------------------------------------------------------------------------


def _generate_synthetic_tokens(
    rng: random.Random,
    vocab_size: int,
    doc_length: int,
) -> List[int]:
    """生成一个合成文档，以 token ID 列表形式返回。"""
    return [rng.randint(0, vocab_size - 1) for _ in range(doc_length)]


def create_synthetic_datasets(
    seed: int = 42,
) -> Dict[str, List[List[int]]]:
    """为多个领域创建合成数据集。

    返回一个将领域名映射到 token-ID 文档列表的字典。
    每个领域具有不同的大小和 token 分布，以使混合效果可视化。
    """
    rng = random.Random(seed)
    datasets: Dict[str, List[List[int]]] = {}

    # Web 数据：大容量，短文档
    datasets["web"] = [
        _generate_synthetic_tokens(rng, vocab_size=2000, doc_length=50)
        for _ in range(500)
    ]

    # 代码：中等容量，较长文档
    datasets["code"] = [
        _generate_synthetic_tokens(rng, vocab_size=800, doc_length=120)
        for _ in range(200)
    ]

    # 书籍：小容量，长文档
    datasets["books"] = [
        _generate_synthetic_tokens(rng, vocab_size=3000, doc_length=200)
        for _ in range(100)
    ]

    # Wikipedia：中等偏小容量
    datasets["wikipedia"] = [
        _generate_synthetic_tokens(rng, vocab_size=2500, doc_length=80)
        for _ in range(150)
    ]

    # 新闻：中等容量
    datasets["news"] = [
        _generate_synthetic_tokens(rng, vocab_size=1500, doc_length=60)
        for _ in range(200)
    ]

    return datasets


# ---------------------------------------------------------------------------
# 混合策略
# ---------------------------------------------------------------------------


def proportional_mixing(
    datasets: Dict[str, List[List[int]]],
    total_samples: int,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """按各数据集大小比例采样文档。

    返回
    -------
    (sampled_docs, counts_by_domain)
    """
    rng = random.Random(seed)
    sizes = {name: len(docs) for name, docs in datasets.items()}
    total_size = sum(sizes.values())
    probs = {name: sz / total_size for name, sz in sizes.items()}

    domain_names = list(datasets.keys())
    weights = [probs[name] for name in domain_names]

    sampled: List[List[int]] = []
    counts: Dict[str, int] = {name: 0 for name in datasets}

    for _ in range(total_samples):
        domain = rng.choices(domain_names, weights=weights, k=1)[0]
        doc = rng.choice(datasets[domain])
        sampled.append(doc)
        counts[domain] += 1

    return sampled, counts


def temperature_sampling(
    datasets: Dict[str, List[List[int]]],
    total_samples: int,
    temperature: float = 1.0,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """使用温度调整后的概率采样文档。

    每个数据集的概率为 ``p_i^{1/T} / sum_j p_j^{1/T}``，
    其中 ``p_i`` 为按比例的概率。

    - ``T < 1``：锐化分布（偏向大型数据集）。
    - ``T > 1``：平滑分布（更均匀的采样）。
    - ``T = 1``：等价于按比例混合。

    返回
    -------
    (sampled_docs, counts_by_domain)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    rng = random.Random(seed)
    sizes = {name: len(docs) for name, docs in datasets.items()}
    total_size = sum(sizes.values())

    # 计算温度调整后的权重
    inv_t = 1.0 / temperature
    raw_weights = {name: (sz / total_size) ** inv_t for name, sz in sizes.items()}
    weight_sum = sum(raw_weights.values())
    probs = {name: w / weight_sum for name, w in raw_weights.items()}

    domain_names = list(datasets.keys())
    weights = [probs[name] for name in domain_names]

    sampled: List[List[int]] = []
    counts: Dict[str, int] = {name: 0 for name in datasets}

    for _ in range(total_samples):
        domain = rng.choices(domain_names, weights=weights, k=1)[0]
        doc = rng.choice(datasets[domain])
        sampled.append(doc)
        counts[domain] += 1

    return sampled, counts


def domain_weighted_mixing(
    datasets: Dict[str, List[List[int]]],
    domain_weights: Dict[str, float],
    total_samples: int,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """按显式领域权重采样文档。

    参数
    ----------
    domain_weights : Dict[str, float]
        从领域名到期望采样权重的映射。权重内部归一化，无需总和为 1。

    返回
    -------
    (sampled_docs, counts_by_domain)
    """
    rng = random.Random(seed)

    # 归一化权重
    total_w = sum(domain_weights.values())
    probs = {name: w / total_w for name, w in domain_weights.items()}

    domain_names = list(probs.keys())
    weights = [probs[name] for name in domain_names]

    # 验证所有请求的领域都存在
    for name in probs:
        if name not in datasets:
            raise KeyError(f"Domain '{name}' not found in datasets.")

    sampled: List[List[int]] = []
    counts: Dict[str, int] = {name: 0 for name in probs}

    for _ in range(total_samples):
        domain = rng.choices(domain_names, weights=weights, k=1)[0]
        doc = rng.choice(datasets[domain])
        sampled.append(doc)
        counts[domain] += 1

    return sampled, counts


# ---------------------------------------------------------------------------
# Token 分布分析
# ---------------------------------------------------------------------------


def compute_token_distribution(
    sampled_docs: List[List[int]],
    vocab_size: int = 5000,
) -> np.ndarray:
    """统计所有采样文档中 token 的出现次数。

    返回长度为 *vocab_size* 的 1D numpy 数组，每个 token ID 对应一个计数值。
    """
    hist = np.zeros(vocab_size, dtype=np.int64)
    for doc in sampled_docs:
        for tid in doc:
            if tid < vocab_size:
                hist[tid] += 1
    return hist


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_mixing_comparison(
    datasets: Dict[str, List[List[int]]],
    total_samples: int = 2000,
) -> None:
    """绘制一个 3x3 网格，比较混合策略及其 token 分布。"""
    strategies: Dict[str, object] = {
        "Proportional (base)": proportional_mixing(datasets, total_samples),
        "Temperature T=0.5\n(sharpen, favour large)": temperature_sampling(
            datasets, total_samples, temperature=0.5
        ),
        "Temperature T=2.0\n(flatten, more equal)": temperature_sampling(
            datasets, total_samples, temperature=2.0
        ),
        "Temperature T=3.0\n(very flat)": temperature_sampling(
            datasets, total_samples, temperature=3.0
        ),
        "Domain-weighted\n(web=0.1, books=0.5)": domain_weighted_mixing(
            datasets,
            {"web": 0.1, "code": 0.2, "books": 0.5, "wikipedia": 0.1, "news": 0.1},
            total_samples,
        ),
        "Domain-weighted\n(web=0.7, code=0.2)": domain_weighted_mixing(
            datasets,
            {"web": 0.7, "code": 0.2, "books": 0.05, "wikipedia": 0.03, "news": 0.02},
            total_samples,
        ),
    }

    n_strategies = len(strategies)
    fig, axes = plt.subplots(
        n_strategies,
        2,
        figsize=(12, 3.5 * n_strategies),
        constrained_layout=True,
    )

    # 如果只有一行，将 axes 包装为二维形状
    if n_strategies == 1:
        axes = np.array([axes])

    for row_idx, (label, (sampled, counts)) in enumerate(strategies.items()):  # type: ignore[misc]
        ax_domain = axes[row_idx, 0]
        ax_token = axes[row_idx, 1]

        # --- 领域分布（饼图） ---
        domain_names = list(counts.keys())
        domain_vals = [counts[n] for n in domain_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(domain_names)))

        # 在百分比旁显示计数
        total = sum(domain_vals)
        wedges, texts, autotexts = ax_domain.pie(
            domain_vals,
            labels=domain_names,
            autopct=lambda pct: f"{pct:.1f}%\n({int(pct / 100 * total)})",
            colors=colors,
            startangle=90,
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax_domain.set_title(f"Domain Mix: {label.split(chr(10))[0]}", fontsize=11)

        # --- Token 分布（直方图） ---
        token_dist = compute_token_distribution(sampled)
        # 为视觉清晰度进行分桶（显示前 500 个 token）
        top_k = min(500, len(token_dist))
        ax_token.bar(
            range(top_k),
            token_dist[:top_k],
            width=1.0,
            color="steelblue",
            edgecolor="none",
        )
        ax_token.set_xlabel("Token ID")
        ax_token.set_ylabel("Count")
        ax_token.set_title("Token Distribution (top 500)", fontsize=11)
        ax_token.set_xlim(0, top_k)

    fig.suptitle(
        "Data Mixing Strategies - Domain & Token Distributions",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.show()

    print("Displayed mixing comparison plots.")


def print_mixing_stats(
    label: str,
    counts: Dict[str, int],
    sampled: List[List[int]],
) -> None:
    """打印混合结果的摘要统计信息。"""
    total = sum(counts.values())
    total_tokens = sum(len(doc) for doc in sampled)

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Total documents sampled: {total}")
    print(f"  Total tokens:            {total_tokens}")
    print(f"  Domain breakdown:")
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total > 0 else 0
        print(f"    {name:<12} {cnt:>5d} docs  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# 主演示
# ---------------------------------------------------------------------------


def main() -> None:
    """演示并可视化所有三种混合策略。"""
    print("Generating synthetic datasets...")
    datasets = create_synthetic_datasets(seed=42)

    print("\nDataset sizes:")
    for name, docs in datasets.items():
        total_tokens = sum(len(d) for d in docs)
        print(f"  {name:<12} {len(docs):>4d} docs  {total_tokens:>7d} tokens")

    # ---- 按比例混合 ----
    sampled_prop, counts_prop = proportional_mixing(datasets, total_samples=2000)
    print_mixing_stats("Proportional Mixing", counts_prop, sampled_prop)

    # ---- 温度采样 ----
    for T in [0.5, 1.0, 2.0, 3.0]:
        sampled_temp, counts_temp = temperature_sampling(
            datasets, total_samples=2000, temperature=T
        )
        print_mixing_stats(f"Temperature T={T}", counts_temp, sampled_temp)

    # ---- 领域加权混合 ----
    domain_weights_1: Dict[str, float] = {
        "web": 0.1,
        "code": 0.2,
        "books": 0.5,
        "wikipedia": 0.1,
        "news": 0.1,
    }
    sampled_dw1, counts_dw1 = domain_weighted_mixing(
        datasets, domain_weights_1, total_samples=2000
    )
    print_mixing_stats("Domain-Weighted (books-heavy)", counts_dw1, sampled_dw1)

    domain_weights_2: Dict[str, float] = {
        "web": 0.7,
        "code": 0.2,
        "books": 0.05,
        "wikipedia": 0.03,
        "news": 0.02,
    }
    sampled_dw2, counts_dw2 = domain_weighted_mixing(
        datasets, domain_weights_2, total_samples=2000
    )
    print_mixing_stats("Domain-Weighted (web-heavy)", counts_dw2, sampled_dw2)

    # ---- 可视化 ----
    print("\nGenerating plots...")
    plot_mixing_comparison(datasets, total_samples=2000)


if __name__ == "__main__":
    main()
