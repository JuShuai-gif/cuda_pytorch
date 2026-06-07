"""
Evaluation metrics for LLM benchmarks.

Provides:
  - compute_perplexity: Perplexity from logits/CE loss.
  - compute_pass_at_k: Unbiased pass@k estimator (Chen et al.).
  - compute_elo: Bradley-Terry Elo rating system.
  - compute_bleu / compute_rouge_l: Text generation metrics.
  - bootstrap_confidence_interval: Statistical CI via bootstrapping.
  - DataContaminationDetector: n-gram overlap and embedding similarity.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence

import torch
import torch.nn.functional as F


# =========================================================================
# Perplexity
# =========================================================================


def compute_perplexity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> float:
    """Compute perplexity from model logits.

    PPL = exp(cross_entropy_loss)
    Range: 1 (perfect) to vocab_size (random guessing).

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size) or
                (seq_len, vocab_size).
        target_ids: Ground truth token ids, same shape minus vocab dim.
        ignore_index: Token id to ignore in loss computation (e.g. padding).
        reduction: "mean", "sum", or "none".

    Returns:
        Perplexity value (scalar or per-token list depending on reduction).
    """
    # Handle 2D input (seq_len, vocab_size)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )

    if reduction == "none":
        return torch.exp(loss).tolist()

    return float(torch.exp(loss).item())


def compute_perplexity_from_loss(loss: float) -> float:
    """Convert cross-entropy loss to perplexity.

    Args:
        loss: Average cross-entropy loss.

    Returns:
        Perplexity.
    """
    return float(math.exp(loss))


# =========================================================================
# Pass@k (unbiased estimator from Chen et al. 2021)
# =========================================================================


def compute_pass_at_k(
    n_total: int,
    n_correct: int,
    k: int,
) -> float:
    """Compute unbiased pass@k estimator.

    From "Evaluating Large Language Models Trained on Code" (Chen et al. 2021):
      pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples, k = evaluation budget.

    Args:
        n_total: Total number of samples generated per problem.
        n_correct: Number of samples that pass all tests.
        k: Number of samples to evaluate (e.g. 1, 10, 100).

    Returns:
        pass@k probability (0.0 to 1.0).
    """
    if n_total - n_correct < k:
        return 1.0
    if n_total < k:
        return 1.0 if n_correct > 0 else 0.0

    # compute 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    # This avoids large factorials
    product = 1.0
    for i in range(k):
        product *= (n_total - n_correct - i) / (n_total - i)
    return 1.0 - product


def compute_pass_at_k_from_samples(
    samples_per_problem: list[list[bool]],
    k: int,
) -> float:
    """Compute pass@k across multiple problems.

    Args:
        samples_per_problem: List of per-problem boolean lists indicating
                             pass/fail for each generated sample.
        k: Evaluation budget.

    Returns:
        Average pass@k across all problems.
    """
    if not samples_per_problem:
        return 0.0

    pass_rates = []
    for samples in samples_per_problem:
        n = len(samples)
        c = sum(samples)
        pass_rates.append(compute_pass_at_k(n, c, k))

    return sum(pass_rates) / len(pass_rates)


# =========================================================================
# Elo Rating (Bradley-Terry model)
# =========================================================================


def compute_elo(
    wins_a: int,
    wins_b: int,
    ties: int = 0,
    rating_a: float = 1500.0,
    rating_b: float = 1500.0,
    k_factor: float = 32.0,
    scale: float = 400.0,
) -> tuple[float, float]:
    """Update Elo ratings based on a single match outcome.

    Uses standard Elo with Bradley-Terry probability model:
      P(A beats B) = 1 / (1 + 10^((R_B - R_A) / 400))

    Handles ties by treating them as half a win each.

    Args:
        wins_a: Number of wins for player A.
        wins_b: Number of wins for player B.
        ties: Number of tied games.
        rating_a: Current Elo rating for A.
        rating_b: Current Elo rating for B.
        k_factor: K-factor controlling rating volatility.
        scale: Rating scale (default 400 for chess, use 400 for LLMs).

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    total_games = wins_a + wins_b + ties
    if total_games == 0:
        return rating_a, rating_b

    # Expected scores
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))
    expected_b = 1.0 - expected_a

    # Actual scores (ties count as 0.5 for each)
    actual_a = (wins_a + 0.5 * ties) / total_games
    actual_b = (wins_b + 0.5 * ties) / total_games

    new_a = rating_a + k_factor * (actual_a - expected_a)
    new_b = rating_b + k_factor * (actual_b - expected_b)

    return new_a, new_b


def compute_elo_from_battles(
    battles: list[dict[str, Any]],
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    scale: float = 400.0,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute Elo ratings from a series of head-to-head battles.

    Implements the Chatbot Arena Elo ranking approach.

    Args:
        battles: List of dicts with keys: "model_a", "model_b", "winner"
                 where winner is "model_a", "model_b", or "tie".
        initial_rating: Starting Elo rating for all models.
        k_factor: K-factor for rating updates.
        scale: Rating scale factor.
        num_bootstrap: Number of bootstrap iterations for confidence intervals.
        seed: Random seed.

    Returns:
        Dict mapping model name to {"elo": float, "elo_lower": float, "elo_upper": float,
                                     "wins": int, "losses": int, "ties": int, "num_battles": int}.
    """
    if not battles:
        return {}

    # Bootstrap to get confidence intervals
    rng = random.Random(seed)
    n = len(battles)

    all_ratings: list[dict[str, float]] = []
    for _ in range(num_bootstrap):
        # Sample with replacement
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample = [battles[i] for i in indices]
        ratings = _compute_elo_single_run(sample, initial_rating, k_factor, scale)
        all_ratings.append(ratings)

    # Final ratings from full dataset
    final_ratings = _compute_elo_single_run(battles, initial_rating, k_factor, scale)

    # Compute confidence intervals
    result: dict[str, dict[str, float]] = {}
    for model, elo in final_ratings.items():
        elo_samples = sorted([r[model] for r in all_ratings if model in r])
        lower_idx = int(0.025 * len(elo_samples))
        upper_idx = int(0.975 * len(elo_samples))

        result[model] = {
            "elo": elo,
            "elo_lower": elo_samples[lower_idx],
            "elo_upper": elo_samples[upper_idx - 1],
        }

    # Add win/loss/tie stats
    for model in result:
        wins, losses, ties_count, num = 0, 0, 0, 0
        for b in battles:
            if model not in (b["model_a"], b["model_b"]):
                continue
            num += 1
            if b["winner"] == "tie":
                ties_count += 1
            elif b["winner"] == model:
                wins += 1
            else:
                losses += 1
        result[model]["wins"] = wins
        result[model]["losses"] = losses
        result[model]["ties"] = ties_count
        result[model]["num_battles"] = num

    return result


def _compute_elo_single_run(
    battles: list[dict[str, Any]],
    initial_rating: float,
    k_factor: float,
    scale: float,
) -> dict[str, float]:
    """Compute Elo ratings from battles in a single pass.

    Battles are processed sequentially; order does not affect convergence
    with reasonable K and enough battles, but randomized order is preferred.
    """
    ratings: dict[str, float] = {}

    for battle in battles:
        a = battle["model_a"]
        b = battle["model_b"]
        winner = battle["winner"]

        ra = ratings.get(a, initial_rating)
        rb = ratings.get(b, initial_rating)

        if winner == "model_a":
            new_ra, new_rb = compute_elo(1, 0, 0, ra, rb, k_factor, scale)
        elif winner == "model_b":
            new_ra, new_rb = compute_elo(0, 1, 0, ra, rb, k_factor, scale)
        else:  # tie
            new_ra, new_rb = compute_elo(0, 0, 1, ra, rb, k_factor, scale)

        ratings[a] = new_ra
        ratings[b] = new_rb

    return ratings


# =========================================================================
# BLEU
# =========================================================================


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from token list."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(
    candidate: str,
    references: str | list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """Compute BLEU score with brevity penalty.

    Args:
        candidate: Candidate text.
        references: Reference text(s).
        max_n: Maximum n-gram order (default 4 = BLEU-4).
        smooth: Apply smoothing to avoid zero scores.

    Returns:
        Dict with "bleu", "precisions", "brevity_penalty" keys.
    """
    cand_tokens = candidate.lower().split()

    if isinstance(references, str):
        refs_tokens = [references.lower().split()]
    else:
        refs_tokens = [r.lower().split() for r in references]

    if not cand_tokens:
        return {"bleu": 0.0, "precisions": [0.0] * max_n, "brevity_penalty": 0.0}

    ref_lengths = [len(r) for r in refs_tokens]

    # Modified n-gram precisions
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_ngrams = _ngrams(cand_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue

        cand_counts = Counter(cand_ngrams)

        # Clip counts by max reference count
        clipped = 0
        for ng, count in cand_counts.items():
            max_ref_count = 0
            for ref_tokens in refs_tokens:
                ref_ngrams = _ngrams(ref_tokens, n)
                ref_counts = Counter(ref_ngrams)
                max_ref_count = max(max_ref_count, ref_counts.get(ng, 0))
            clipped += min(count, max_ref_count)

        precision = clipped / len(cand_ngrams) if cand_ngrams else 0.0
        if smooth and precision == 0.0:
            precision = 1.0 / (2 ** (n + 1))
        precisions.append(precision)

    # Brevity penalty
    cand_len = len(cand_tokens)
    closest_ref_len = min(ref_lengths, key=lambda r: abs(r - cand_len))

    if cand_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - closest_ref_len / cand_len)

    # BLEU
    if any(p == 0.0 for p in precisions):
        bleu = 0.0
    else:
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)

    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
    }


# =========================================================================
# ROUGE-L
# =========================================================================


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (O(mn), 1D DP)."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_rouge_l(
    candidate: str,
    references: str | list[str],
) -> dict[str, float]:
    """Compute ROUGE-L (longest common subsequence) score.

    Args:
        candidate: Candidate text.
        references: Reference text(s).

    Returns:
        Dict with "recall", "precision", "f1" keys.
    """
    cand_tokens = candidate.lower().split()

    if isinstance(references, str):
        refs_tokens = [references.lower().split()]
    else:
        refs_tokens = [r.lower().split() for r in references]

    if not cand_tokens:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    best_f1, best_recall, best_precision = 0.0, 0.0, 0.0
    for ref_tokens in refs_tokens:
        lcs_len = _lcs_length(cand_tokens, ref_tokens)
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        precision = lcs_len / len(cand_tokens) if cand_tokens else 0.0
        f1 = (
            (2 * recall * precision / (recall + precision))
            if (recall + precision) > 0
            else 0.0
        )
        if f1 > best_f1:
            best_f1, best_recall, best_precision = f1, recall, precision

    return {"recall": best_recall, "precision": best_precision, "f1": best_f1}


# =========================================================================
# Bootstrap Confidence Interval
# =========================================================================


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a set of per-sample scores.

    Args:
        scores: Per-example metric values.
        confidence: Confidence level (0 to 1).
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with "mean", "lower", "upper", "confidence", "n_resamples" keys.
    """
    if not scores:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "confidence": confidence,
            "n_resamples": 0,
        }

    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []

    for _ in range(n_resamples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples)

    return {
        "mean": sum(means) / n_resamples,
        "lower": means[lower_idx],
        "upper": means[upper_idx - 1],
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


# =========================================================================
# Data Contamination Detector
# =========================================================================


class DataContaminationDetector:
    """Detect potential training data contamination in evaluation sets.

    Two detection methods:
      1. N-gram overlap: Counts overlapping n-grams between eval data and
         training corpus samples.
      2. Embedding similarity: Cosine similarity between embeddings of
         eval samples and training samples.

    High overlap does NOT prove contamination (common phrases exist),
    but is a useful diagnostic signal.
    """

    def __init__(
        self,
        ngram_sizes: tuple[int, ...] = (8, 13),
        similarity_threshold: float = 0.8,
        seed: int = 42,
    ):
        """
        Args:
            ngram_sizes: N-gram sizes for overlap detection.
            similarity_threshold: Cosine similarity threshold for flagging.
            seed: Random seed.
        """
        self.ngram_sizes = ngram_sizes
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self._train_ngrams: dict[int, set[tuple[str, ...]]] = {}
        self._train_embeddings: list[list[float]] | None = None

    def index_training_data(self, texts: list[str]) -> None:
        """Build n-gram index from training corpus samples.

        Args:
            texts: Training data text samples.
        """
        for n in self.ngram_sizes:
            ngram_set: set[tuple[str, ...]] = set()
            for text in texts:
                tokens = text.lower().split()
                for i in range(len(tokens) - n + 1):
                    ngram_set.add(tuple(tokens[i : i + n]))
            self._train_ngrams[n] = ngram_set

    def load_training_embeddings(self, embeddings: list[list[float]]) -> None:
        """Load pre-computed training data embeddings.

        Args:
            embeddings: List of embedding vectors for training samples.
        """
        self._train_embeddings = embeddings

    def check_n_gram_overlap(self, eval_text: str) -> dict[int, float]:
        """Check n-gram overlap ratio between eval text and training data.

        Args:
            eval_text: Evaluation sample text.

        Returns:
            Dict mapping n-gram size to overlap ratio (0 to 1).
        """
        tokens = eval_text.lower().split()
        result: dict[int, float] = {}

        for n in self.ngram_sizes:
            if n not in self._train_ngrams or len(tokens) < n:
                result[n] = 0.0
                continue

            eval_ngrams = {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
            if not eval_ngrams:
                result[n] = 0.0
                continue

            matched = eval_ngrams & self._train_ngrams[n]
            result[n] = len(matched) / len(eval_ngrams)

        return result

    def check_embedding_similarity(
        self, eval_embedding: list[float]
    ) -> dict[str, float]:
        """Check max cosine similarity against training embeddings.

        Args:
            eval_embedding: Embedding vector for evaluation sample.

        Returns:
            Dict with "max_similarity", "mean_top5", "flagged" keys.
        """
        if self._train_embeddings is None:
            return {"max_similarity": 0.0, "mean_top5": 0.0, "flagged": False}

        similarities = [
            self._cosine_similarity(eval_embedding, train_emb)
            for train_emb in self._train_embeddings
        ]
        similarities.sort(reverse=True)

        top5_mean = sum(similarities[:5]) / min(5, len(similarities))
        max_sim = similarities[0] if similarities else 0.0

        return {
            "max_similarity": max_sim,
            "mean_top5": top5_mean,
            "flagged": max_sim >= self.similarity_threshold,
        }

    def run_full_check(
        self,
        eval_texts: list[str],
        eval_embeddings: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Run both n-gram overlap and embedding similarity checks.

        Args:
            eval_texts: Evaluation sample texts.
            eval_embeddings: Optional pre-computed embeddings for eval samples.

        Returns:
            Summary report with per-sample and aggregate statistics.
        """
        ngram_results = [self.check_n_gram_overlap(text) for text in eval_texts]

        # Aggregate n-gram overlap
        avg_overlap: dict[int, float] = {}
        max_overlap: dict[int, float] = {}
        flagged_count_by_n: dict[int, int] = {}
        for n in self.ngram_sizes:
            overlaps = [r.get(n, 0.0) for r in ngram_results]
            avg_overlap[n] = sum(overlaps) / len(overlaps) if overlaps else 0.0
            max_overlap[n] = max(overlaps) if overlaps else 0.0
            flagged_count_by_n[n] = sum(1 for o in overlaps if o >= 0.5)

        report: dict[str, Any] = {
            "ngram_overlap": {
                "per_sample": ngram_results,
                "avg_overlap": avg_overlap,
                "max_overlap": max_overlap,
                "flagged_count": flagged_count_by_n,
            },
            "embedding_similarity": None,
        }

        if eval_embeddings is not None:
            emb_results = [
                self.check_embedding_similarity(emb) for emb in eval_embeddings
            ]
            emb_flagged = sum(r["flagged"] for r in emb_results)
            report["embedding_similarity"] = {
                "per_sample": emb_results,
                "flagged_count": emb_flagged,
                "flagged_ratio": emb_flagged / len(emb_results) if emb_results else 0.0,
            }

        return report

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# =========================================================================
# Demo
# =========================================================================


def demo_perplexity() -> None:
    """Demonstrate perplexity computation."""
    print("=" * 70)
    print("Perplexity Metrics Demo")
    print("=" * 70)

    vocab_size = 100
    seq_len = 10
    logits = torch.randn(1, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (1, seq_len))

    ppl = compute_perplexity(logits, targets)
    print(f"\n  Random model PPL: {ppl:.2f} (expect ~{vocab_size})")

    # Boost correct token logits to simulate better model
    for t in range(seq_len):
        logits[0, t, targets[0, t]] += 5.0
    ppl_good = compute_perplexity(logits, targets)
    print(f"  Good model PPL:   {ppl_good:.4f}")


def demo_pass_at_k() -> None:
    """Demonstrate unbiased pass@k estimator."""
    print("\n" + "=" * 70)
    print("Pass@k Estimator Demo")
    print("=" * 70)

    # Simulate: for one problem, generate 200 samples, 30 pass
    n_total, n_correct = 200, 30
    for k in [1, 10, 100]:
        pk = compute_pass_at_k(n_total, n_correct, k)
        print(f"  n={n_total}, c={n_correct}, pass@{k} = {pk:.4f}")

    # Multi-problem scenario
    samples = [
        [True, False, False, True, False],  # 2/5 pass
        [True, True, True, False, True],  # 4/5 pass
        [False, False, False, False, True],  # 1/5 pass
    ]
    for k in [1, 3, 5]:
        pk = compute_pass_at_k_from_samples(samples, k)
        print(f"  Multi-problem pass@{k} = {pk:.4f}")


def demo_elo() -> None:
    """Demonstrate Elo rating computation."""
    print("\n" + "=" * 70)
    print("Elo Rating (Bradley-Terry) Demo")
    print("=" * 70)

    # Manual update
    ra, rb = compute_elo(1, 0, 0, 1500, 1500)
    print(f"\n  After A beats B once: A={ra:.1f}, B={rb:.1f}")

    ra, rb = compute_elo(0, 1, 0, 1500, 1500)
    print(f"  After B beats A once: A={ra:.1f}, B={rb:.1f}")

    # Many battles
    battles = (
        [
            {"model_a": "ModelA", "model_b": "ModelB", "winner": "model_a"},
        ]
        * 30
        + [
            {"model_a": "ModelA", "model_b": "ModelB", "winner": "model_b"},
        ]
        * 15
        + [
            {"model_a": "ModelA", "model_b": "ModelB", "winner": "tie"},
        ]
        * 5
    )

    ratings = compute_elo_from_battles(battles)
    print(f"\n  Chatbot Arena-style Elo:")
    for model, r in ratings.items():
        print(
            f"    {model}: Elo={r['elo']:.1f} [{r['elo_lower']:.1f}, {r['elo_upper']:.1f}], "
            f"W={r['wins']} L={r['losses']} T={r['ties']}"
        )


def demo_bleu_rouge() -> None:
    """Demonstrate BLEU and ROUGE-L."""
    print("\n" + "=" * 70)
    print("BLEU / ROUGE-L Demo")
    print("=" * 70)

    cand = "the cat sat on the mat"
    ref = "the cat sat on the mat"
    bleu = compute_bleu(cand, ref)
    rouge = compute_rouge_l(cand, ref)
    print(f"\n  Exact match:")
    print(f"    BLEU:    {bleu['bleu']:.4f}")
    print(f"    ROUGE-L: {rouge['f1']:.4f}")

    cand2 = "the feline sits on the rug"
    refs = ["the cat sat on the mat", "a cat is sitting on the mat"]
    bleu2 = compute_bleu(cand2, refs)
    rouge2 = compute_rouge_l(cand2, refs)
    print(f"\n  Partial match:")
    print(f"    BLEU:    {bleu2['bleu']:.4f}")
    print(f"    ROUGE-L: {rouge2['f1']:.4f}")


def demo_bootstrap() -> None:
    """Demonstrate bootstrap confidence intervals."""
    print("\n" + "=" * 70)
    print("Bootstrap Confidence Interval Demo")
    print("=" * 70)

    rng = random.Random(42)
    scores = [rng.random() for _ in range(200)]
    scores = [1.0 if s > 0.28 else 0.0 for s in scores]
    actual_mean = sum(scores) / len(scores)

    ci = bootstrap_confidence_interval(scores)
    print(f"\n  True mean: {actual_mean:.4f}")
    print(f"  95% CI:    [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"  (Bootstrap mean: {ci['mean']:.4f})")


def demo_contamination() -> None:
    """Demonstrate data contamination detection."""
    print("\n" + "=" * 70)
    print("Data Contamination Detection Demo")
    print("=" * 70)

    detector = DataContaminationDetector(ngram_sizes=(5, 10))

    train_texts = [
        "the quick brown fox jumps over the lazy dog",
        "in a hole in the ground there lived a hobbit",
        "it was the best of times it was the worst of times",
    ]
    detector.index_training_data(train_texts)

    # Clean eval sample
    eval_clean = "the weather is very nice today and the sun is shining"
    overlap = detector.check_n_gram_overlap(eval_clean)
    print(f"\n  Clean eval: '{eval_clean}'")
    for n, ratio in overlap.items():
        print(f"    {n}-gram overlap: {ratio:.4f}")

    # Contaminated eval sample
    eval_dirty = "the quick brown fox jumps over the lazy dog indeed it does"
    overlap = detector.check_n_gram_overlap(eval_dirty)
    print(f"\n  Suspicious eval: '{eval_dirty}'")
    for n, ratio in overlap.items():
        flag = " *** HIGH OVERLAP ***" if ratio > 0.5 else ""
        print(f"    {n}-gram overlap: {ratio:.4f}{flag}")

    # Embedding similarity
    eval_embeddings = [[0.1, 0.9, 0.3], [0.2, 0.1, 0.8]]
    detector.load_training_embeddings(
        [[0.1, 0.85, 0.3], [0.5, 0.5, 0.5], [0.2, 0.15, 0.78]]
    )
    for i, emb in enumerate(eval_embeddings):
        result = detector.check_embedding_similarity(emb)
        print(f"\n  Eval sample {i} embedding similarity:")
        print(
            f"    Max: {result['max_similarity']:.4f}, "
            f"Mean Top-5: {result['mean_top5']:.4f}, "
            f"Flagged: {result['flagged']}"
        )


def main() -> None:
    demo_perplexity()
    demo_pass_at_k()
    demo_elo()
    demo_bleu_rouge()
    demo_bootstrap()
    demo_contamination()


if __name__ == "__main__":
    main()
