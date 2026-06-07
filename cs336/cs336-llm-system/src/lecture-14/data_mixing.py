"""
Data mixing strategies for combining multiple pretraining datasets.

Implements three mixing strategies and visualises the resulting
token distributions:

1. **Proportional mixing** – sample each dataset in proportion to its size.
2. **Temperature-based sampling** – raise sampling probabilities to ``1/T``
   to smooth or sharpen the distribution.
3. **Domain-weighted mixing** – assign explicit weights to different
   domains (code, web, books, etc.).

All demonstrations use synthetic data; no external data dependencies.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

# Third-party
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------


def _generate_synthetic_tokens(
    rng: random.Random,
    vocab_size: int,
    doc_length: int,
) -> List[int]:
    """Generate a synthetic document as a list of token IDs."""
    return [rng.randint(0, vocab_size - 1) for _ in range(doc_length)]


def create_synthetic_datasets(
    seed: int = 42,
) -> Dict[str, List[List[int]]]:
    """Create synthetic datasets for several domains.

    Returns a dict mapping domain name to a list of token-ID documents.
    Each domain has different sizes and token distributions to make the
    mixing effects visible.
    """
    rng = random.Random(seed)
    datasets: Dict[str, List[List[int]]] = {}

    # Web data: large volume, short documents
    datasets["web"] = [
        _generate_synthetic_tokens(rng, vocab_size=2000, doc_length=50)
        for _ in range(500)
    ]

    # Code: medium volume, longer documents
    datasets["code"] = [
        _generate_synthetic_tokens(rng, vocab_size=800, doc_length=120)
        for _ in range(200)
    ]

    # Books: small volume, long documents
    datasets["books"] = [
        _generate_synthetic_tokens(rng, vocab_size=3000, doc_length=200)
        for _ in range(100)
    ]

    # Wikipedia: medium-small volume
    datasets["wikipedia"] = [
        _generate_synthetic_tokens(rng, vocab_size=2500, doc_length=80)
        for _ in range(150)
    ]

    # News: medium volume
    datasets["news"] = [
        _generate_synthetic_tokens(rng, vocab_size=1500, doc_length=60)
        for _ in range(200)
    ]

    return datasets


# ---------------------------------------------------------------------------
# Mixing strategies
# ---------------------------------------------------------------------------


def proportional_mixing(
    datasets: Dict[str, List[List[int]]],
    total_samples: int,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """Sample documents proportionally to each dataset's size.

    Returns
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
    """Sample documents with temperature-adjusted probabilities.

    Each dataset's probability is ``p_i^{1/T} / sum_j p_j^{1/T}``
    where ``p_i`` is the proportional probability.

    - ``T < 1``: sharpens the distribution (favours large datasets).
    - ``T > 1``: flattens the distribution (more equal sampling).
    - ``T = 1``: equivalent to proportional mixing.

    Returns
    -------
    (sampled_docs, counts_by_domain)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    rng = random.Random(seed)
    sizes = {name: len(docs) for name, docs in datasets.items()}
    total_size = sum(sizes.values())

    # Compute temperature-adjusted weights
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
    """Sample documents according to explicit domain weights.

    Parameters
    ----------
    domain_weights : Dict[str, float]
        Mapping from domain name to desired sampling weight.  Weights
        are normalised internally; they do not need to sum to 1.

    Returns
    -------
    (sampled_docs, counts_by_domain)
    """
    rng = random.Random(seed)

    # Normalise weights
    total_w = sum(domain_weights.values())
    probs = {name: w / total_w for name, w in domain_weights.items()}

    domain_names = list(probs.keys())
    weights = [probs[name] for name in domain_names]

    # Validate that all requested domains exist
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
# Token distribution analysis
# ---------------------------------------------------------------------------


def compute_token_distribution(
    sampled_docs: List[List[int]],
    vocab_size: int = 5000,
) -> np.ndarray:
    """Count token occurrences across all sampled documents.

    Returns a 1D numpy array of length *vocab_size* with count per token ID.
    """
    hist = np.zeros(vocab_size, dtype=np.int64)
    for doc in sampled_docs:
        for tid in doc:
            if tid < vocab_size:
                hist[tid] += 1
    return hist


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_mixing_comparison(
    datasets: Dict[str, List[List[int]]],
    total_samples: int = 2000,
) -> None:
    """Plot a 3x3 grid comparing mixing strategies and their token distributions."""
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

    # If only one row, wrap axes in a 2D shape
    if n_strategies == 1:
        axes = np.array([axes])

    for row_idx, (label, (sampled, counts)) in enumerate(strategies.items()):  # type: ignore[misc]
        ax_domain = axes[row_idx, 0]
        ax_token = axes[row_idx, 1]

        # --- Domain distribution (pie chart) ---
        domain_names = list(counts.keys())
        domain_vals = [counts[n] for n in domain_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(domain_names)))

        # Display counts alongside percentages
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

        # --- Token distribution (histogram) ---
        token_dist = compute_token_distribution(sampled)
        # Bin for visual clarity (show top 500 tokens)
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
    """Print summary statistics for a mixing result."""
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
# Main demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate and visualise all three mixing strategies."""
    print("Generating synthetic datasets...")
    datasets = create_synthetic_datasets(seed=42)

    print("\nDataset sizes:")
    for name, docs in datasets.items():
        total_tokens = sum(len(d) for d in docs)
        print(f"  {name:<12} {len(docs):>4d} docs  {total_tokens:>7d} tokens")

    # ---- Proportional mixing ----
    sampled_prop, counts_prop = proportional_mixing(datasets, total_samples=2000)
    print_mixing_stats("Proportional Mixing", counts_prop, sampled_prop)

    # ---- Temperature sampling ----
    for T in [0.5, 1.0, 2.0, 3.0]:
        sampled_temp, counts_temp = temperature_sampling(
            datasets, total_samples=2000, temperature=T
        )
        print_mixing_stats(f"Temperature T={T}", counts_temp, sampled_temp)

    # ---- Domain-weighted mixing ----
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

    # ---- Visualisation ----
    print("\nGenerating plots...")
    plot_mixing_comparison(datasets, total_samples=2000)


if __name__ == "__main__":
    main()
