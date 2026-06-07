"""
Streaming data processing pipeline for LLM training data.

Implements a ``DataPipeline`` class that chains: reading → cleaning →
filtering → tokenization in a streaming fashion, yielding documents
one at a time to avoid loading the entire corpus into memory.
"""

from __future__ import annotations

import collections
import re
import statistics
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from text_cleaning import TextCleaner
from filtering import (
    NGramModel,
    detect_language,
    find_near_duplicates,
    quality_filter_by_perplexity,
)


# ---------------------------------------------------------------------------
# Simple word-level tokenizer (no external dependency)
# ---------------------------------------------------------------------------


def simple_tokenize(text: str) -> List[str]:
    """Split *text* into word tokens using whitespace and punctuation boundaries.

    This is a minimal tokenizer for demonstration; a production pipeline
    would use a subword tokenizer (BPE, SentencePiece, etc.).
    """
    # Keep word characters and apostrophes for contractions
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|[0-9]+|[^\s\w]", text)
    return tokens


# ---------------------------------------------------------------------------
# DataPipeline class
# ---------------------------------------------------------------------------


class DataPipeline:
    """Streaming pipeline for preprocessing LLM training data.

    Reads documents from an iterable, applies a sequence of transformation
    stages, and yields processed documents one at a time.

    Parameters
    ----------
    reader : Callable[[], Iterable[str]]
        Zero-argument callable that returns an iterable of raw document strings.
    cleaner : TextCleaner | None
        Text cleaning pipeline.  If ``None``, a default ``TextCleaner`` is used.
    filters : List[Callable[[str], Tuple[bool, Optional[str]]]] | None
        List of filter functions.  Each receives the cleaned text and returns
        ``(keep: bool, reason: str | None)``.  Documents where any filter
        returns ``keep=False`` are skipped.
    tokenizer : Callable[[str], List[str]] | None
        Tokenizer function.  If ``None``, ``simple_tokenize`` is used.
    report_interval : int
        Print a progress report every *report_interval* documents.
    """

    def __init__(
        self,
        reader: Callable[[], Iterable[str]],
        cleaner: TextCleaner | None = None,
        filters: List[Callable[[str], Tuple[bool, Optional[str]]]] | None = None,
        tokenizer: Callable[[str], List[str]] | None = None,
        report_interval: int = 100,
    ) -> None:
        self._reader = reader
        self._cleaner = cleaner if cleaner is not None else TextCleaner()
        self._filters: List[Callable[[str], Tuple[bool, Optional[str]]]] = (
            filters if filters is not None else []
        )
        self._tokenizer = tokenizer if tokenizer is not None else simple_tokenize
        self._report_interval = report_interval

        # Statistics
        self.stats: Dict[str, int] = collections.defaultdict(int)

    def process(self) -> Iterator[Dict[str, object]]:
        """Yield processed documents as dictionaries.

        Each yielded dict has the keys ``"raw"``, ``"cleaned"``, ``"tokens"``,
        and ``"doc_id"``.

        Yields
        ------
        dict
            Processed document with metadata.
        """
        start_time = time.monotonic()
        skipped_reasons: Dict[str, int] = collections.defaultdict(int)

        for doc_id, raw in enumerate(self._reader()):
            self.stats["total_read"] += 1

            # Stage 1: Clean
            cleaned = self._cleaner.clean(raw)
            if not cleaned.strip():
                skipped_reasons["empty_after_cleaning"] += 1
                continue

            # Stage 2: Filter
            keep = True
            for filt in self._filters:
                keep, reason = filt(cleaned)
                if not keep:
                    skipped_reasons[reason or "unknown_filter"] += 1
                    break
            if not keep:
                continue

            # Stage 3: Tokenize
            tokens = self._tokenizer(cleaned)
            self.stats["total_accepted"] += 1

            # Progress reporting
            if (self.stats["total_read"] % self._report_interval) == 0:
                elapsed = time.monotonic() - start_time
                rate = self.stats["total_read"] / elapsed if elapsed > 0 else 0
                print(
                    f"  [Progress] Read: {self.stats['total_read']:>6d}, "
                    f"Accepted: {self.stats['total_accepted']:>6d}, "
                    f"Rate: {rate:.0f} docs/s"
                )

            yield {
                "raw": raw,
                "cleaned": cleaned,
                "tokens": tokens,
                "doc_id": doc_id,
            }

        elapsed = time.monotonic() - start_time
        print(f"\nPipeline complete in {elapsed:.1f}s")
        print(f"  Total read:     {self.stats['total_read']}")
        print(f"  Total accepted: {self.stats['total_accepted']}")
        if skipped_reasons:
            print(f"  Skipped by reason:")
            for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")


# ---------------------------------------------------------------------------
# Pre-built filter factories
# ---------------------------------------------------------------------------


def make_language_filter(
    target_lang: str,
) -> Callable[[str], Tuple[bool, Optional[str]]]:
    """Return a filter that keeps only *target_lang* documents."""

    def _filt(text: str) -> Tuple[bool, Optional[str]]:
        lang, _conf = detect_language(text)
        if lang != target_lang:
            return (False, f"not_{target_lang}_detected_{lang}")
        return (True, None)

    return _filt


def make_perplexity_filter(
    clean_texts: List[str],
    max_ratio: float = 2.0,
) -> Callable[[str], Tuple[bool, Optional[str]]]:
    """Return a filter that removes high-perplexity (low-quality) text.

    Builds an n-gram model from *clean_texts*, computes the baseline
    perplexity, and rejects any document whose perplexity exceeds
    ``baseline * max_ratio``.
    """
    model = NGramModel(n=3, k=0.1)
    model.train(clean_texts)
    baseline = statistics.mean(model.perplexity(t) for t in clean_texts)

    def _filt(text: str) -> Tuple[bool, Optional[str]]:
        keep, _ppl = quality_filter_by_perplexity(text, model, baseline, max_ratio)
        if not keep:
            return (False, "high_perplexity")
        return (True, None)

    return _filt


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def _generate_synthetic_docs() -> List[str]:
    """Create a small set of synthetic documents."""
    clean_docs = [
        "Machine learning is a subset of artificial intelligence that "
        "enables systems to learn and improve from experience without "
        "being explicitly programmed.",
        "Deep learning uses neural networks with many layers to model "
        "complex patterns in large datasets such as images and text.",
        "Natural language processing combines linguistics and machine "
        "learning to help computers understand human language.",
        "Reinforcement learning trains agents to make sequences of "
        "decisions by rewarding desired behaviors and penalizing mistakes.",
        "Computer vision enables machines to interpret and understand "
        "visual information from the world around them.",
    ]
    noisy_docs = [
        "the the the the the the the the the the the the the the the the",
        "asdf qwer zxcv poiuy mnbvc lkjhg fdsa trew",
        "<html><body><p>Some <b>HTML</b> content here.</p></body></html>",
        "This is a normal English sentence that should pass all filters.",
        "Transformers have revolutionized NLP with the attention mechanism.",
    ]
    return clean_docs + noisy_docs


def main() -> None:
    """Demonstrate the full streaming data pipeline."""
    raw_docs = _generate_synthetic_docs()
    print(f"Generated {len(raw_docs)} synthetic documents\n")

    # Build filters
    en_filter = make_language_filter("en")
    clean_corpus = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers",
        "natural language processing combines linguistics and machine learning",
        "reinforcement learning trains agents to make decisions",
        "computer vision enables machines to interpret visual information",
        "transformers have revolutionized nlp with attention mechanism",
    ]
    perplexity_filter = make_perplexity_filter(clean_corpus, max_ratio=3.0)

    # Build pipeline
    pipeline = DataPipeline(
        reader=lambda: raw_docs,  # Return the whole list (simulates a reader)
        filters=[en_filter, perplexity_filter],
        tokenizer=simple_tokenize,
        report_interval=3,
    )

    print("=" * 60)
    print("PROCESSING...")
    print("=" * 60)

    accepted = 0
    for doc in pipeline.process():
        accepted += 1
        n_tokens = len(doc["tokens"])  # type: ignore[arg-type]
        cleaned_preview = str(doc["cleaned"])[:80]  # type: ignore[index]
        print(
            f"  Doc #{doc['doc_id']:<4d} | tokens={n_tokens:>3d} | "
            f'"{cleaned_preview}..."'
        )

    print(f"\nAccepted {accepted} / {len(raw_docs)} documents")


if __name__ == "__main__":
    main()
