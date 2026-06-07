"""
Define and compare major data sources used for LLM pretraining.

Each data source is represented with structured metadata including:
    - Approximate size (tokens)
    - Language coverage
    - Quality level
    - Typical use cases in research and production

This module uses only hard-coded reference data -- no external downloads needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DataSource:
    """Structured representation of a pretraining data source."""

    name: str
    """Short name of the source, e.g. 'CommonCrawl'."""

    description: str
    """One-sentence description."""

    estimated_size: str
    """Approximate size, e.g. '~400B tokens'."""

    language_coverage: str
    """Languages covered, e.g. 'Multilingual (100+)' or 'English-only'."""

    quality_level: str
    """Quality tier: 'Raw', 'Filtered', 'Curated', or 'High-quality'."""

    typical_use_cases: list[str] = field(default_factory=list)
    """Examples: ['general pretraining', 'multilingual models']."""

    notes: Optional[str] = None
    """Additional notes, caveats, or references."""


# ---------------------------------------------------------------------------
# Known data sources
# ---------------------------------------------------------------------------

DATA_SOURCES: list[DataSource] = [
    DataSource(
        name="CommonCrawl",
        description="Monthly crawl of the public web; the largest openly available web corpus.",
        estimated_size="~400B tokens (filtered subset ~3-6 TB compressed)",
        language_coverage="Multilingual (100+), English-dominant",
        quality_level="Raw",
        typical_use_cases=[
            "general pretraining (e.g. C4, CC100, mC4 derivatives)",
            "multilingual models",
            "domain-specific extraction",
        ],
        notes="Requires extensive filtering: language detection, dedup, quality heuristics.",
    ),
    DataSource(
        name="C4 (Colossal Clean Crawled Corpus)",
        description="Heavily filtered English subset of CommonCrawl used to train T5.",
        estimated_size="~160B tokens (en only)",
        language_coverage="English",
        quality_level="Filtered",
        typical_use_cases=[
            "T5-family models",
            "English-only pretraining benchmarks",
            "instruction-tuning data sourcing",
        ],
        notes="Filtering steps: line length, curse words, 'lorem ipsum', JS/code removal.",
    ),
    DataSource(
        name="Wikipedia",
        description="Collaboratively edited encyclopedia; high factual accuracy.",
        estimated_size="~3B tokens (English), ~10B+ (multilingual)",
        language_coverage="Multilingual (300+)",
        quality_level="High-quality",
        typical_use_cases=[
            "knowledge-intensive models",
            "multilingual pretraining warm-up",
            "benchmarking/downstream evaluation",
        ],
        notes="Small size limits its use as sole pretraining corpus; often oversampled.",
    ),
    DataSource(
        name="Books (Books3 / BookCorpus / PG-19)",
        description="Long-form narrative text from published books and public domain works.",
        estimated_size="~100B tokens (Books3), ~2B tokens (PG-19)",
        language_coverage="Primarily English",
        quality_level="High-quality",
        typical_use_cases=[
            "long-context modeling",
            "narrative generation",
            "coherence and storytelling benchmarks",
        ],
        notes="Books3 has copyright concerns; PG-19 (Project Gutenberg) is openly licensed.",
    ),
    DataSource(
        name="The Pile",
        description="Curated 800 GB dataset of 22 diverse high-quality subsets.",
        estimated_size="~300B tokens",
        language_coverage="Primarily English",
        quality_level="Curated",
        typical_use_cases=[
            "GPT-Neo, GPT-J, Pythia training",
            "diverse-domain evaluation",
            "research on data composition effects",
        ],
        notes="Includes ArXiv, PubMed, GitHub, StackExchange, HackerNews, etc.",
    ),
    DataSource(
        name="OpenWebText / OpenWebText2",
        description="Open recreation of the WebText corpus (GPT-2 training data).",
        estimated_size="~38B tokens (OWT), ~60B tokens (OWT2)",
        language_coverage="English",
        quality_level="Filtered",
        typical_use_cases=[
            "GPT-2 reproduction studies",
            "comparing architectures on same data",
        ],
        notes="Scraped Reddit outbound links with >= 3 karma; mimics GPT-2 recipe.",
    ),
    DataSource(
        name="GitHub / Code Repositories (The Stack)",
        description="Source code from public GitHub repositories with permissive licenses.",
        estimated_size="~350B tokens (The Stack v1), code subsets in The Pile ~100B",
        language_coverage="Programming languages (300+)",
        quality_level="Filtered",
        typical_use_cases=[
            "code generation models (Codex, StarCoder, CodeLlama)",
            "improving reasoning in general LLMs",
            "structured generation",
        ],
        notes="The Stack filters by license; near-deduplication applied.",
    ),
    DataSource(
        name="RedPajama",
        description="Open reproduction of LLaMA's training data (1.2T tokens).",
        estimated_size="~1.2T tokens (v1), ~30T tokens (v2)",
        language_coverage="Multilingual (5 languages: en, de, fr, it, es)",
        quality_level="Filtered",
        typical_use_cases=[
            "LLaMA reproduction",
            "open-source LLM pretraining",
            "large-scale data pipeline studies",
        ],
        notes="v1: 7 subsets mirroring LLaMA. v2: 30T tokens with improved filtering.",
    ),
    DataSource(
        name="FineWeb / FineWeb-Edu",
        description="High-quality filtered CommonCrawl subsets from HuggingFace.",
        estimated_size="~15T tokens (FineWeb), ~1.3T tokens (FineWeb-Edu)",
        language_coverage="Multilingual (FineWeb-2 covers 1000+ languages)",
        quality_level="Filtered",
        typical_use_cases=[
            "modern open-source LLM pretraining",
            "educational-quality filtering research",
        ],
        notes="FineWeb-Edu uses LLM-based quality annotation for educational content.",
    ),
    DataSource(
        name="ArXiv / Scientific Papers",
        description="Preprint papers covering physics, math, CS, and related fields.",
        estimated_size="~20B tokens",
        language_coverage="English (LaTeX source)",
        quality_level="High-quality",
        typical_use_cases=[
            "scientific reasoning",
            "math/code generation",
            "domain-specific adaptation",
        ],
        notes="Requires LaTeX-to-text conversion; often deduplicated against training data.",
    ),
    DataSource(
        name="Social Media / Reddit / Twitter",
        description="Conversational and informal text from social platforms.",
        estimated_size="~100B tokens (PushShift Reddit), variable for Twitter",
        language_coverage="Multilingual, English-dominant",
        quality_level="Raw",
        typical_use_cases=[
            "dialogue and conversational models",
            "sentiment analysis",
            "research on bias and toxicity",
        ],
        notes="Noisy, short texts; needs heavy filtering for toxicity and PII.",
    ),
]


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------


def compare_sources(
    sources: Optional[list[DataSource]] = None,
) -> None:
    """Print a formatted comparison table of data sources.

    Args:
        sources: List of sources to compare. Defaults to all known sources.
    """
    if sources is None:
        sources = DATA_SOURCES

    if not sources:
        print("No data sources to compare.")
        return

    # Column widths
    name_w = max(len(s.name) for s in sources) + 2
    quality_w = max(len(s.quality_level) for s in sources) + 2

    # Header
    header = (
        f"{'Source':<{name_w}} {'Size':<22} {'Quality':<{quality_w}} "
        f"{'Languages':<20} {'Use Cases'}"
    )
    print(header)
    print("-" * len(header))

    for s in sources:
        use_cases_str = ", ".join(s.typical_use_cases[:2])
        if len(s.typical_use_cases) > 2:
            use_cases_str += ", ..."
        print(
            f"{s.name:<{name_w}} "
            f"{s.estimated_size:<22} "
            f"{s.quality_level:<{quality_w}} "
            f"{s.language_coverage:<20} "
            f"{use_cases_str}"
        )


def get_source_by_name(name: str) -> Optional[DataSource]:
    """Look up a data source by name (case-insensitive)."""
    for s in DATA_SOURCES:
        if s.name.lower() == name.lower():
            return s
    return None


def filter_by_quality(sources: list[DataSource], quality: str) -> list[DataSource]:
    """Filter sources by their quality level (case-insensitive)."""
    q = quality.lower()
    return [s for s in sources if s.quality_level.lower() == q]


def get_total_size_range() -> tuple[str, str]:
    """Return a rough sum of token counts across all sources.

    Returns:
        A (min, max) string pair summarizing the estimated corpus sizes.
    """
    return "~17T+ tokens (sum of listed sources)", "30T+ including RedPajama v2"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Major Data Sources for LLM Pretraining")
    print("=" * 70)
    print(f"Total sources listed: {len(DATA_SOURCES)}\n")

    # Print full comparison table
    compare_sources()
    print()

    # Quality breakdown
    print("--- Quality Breakdown ---")
    all_qualities = sorted(set(s.quality_level for s in DATA_SOURCES))
    for quality in all_qualities:
        filtered = filter_by_quality(DATA_SOURCES, quality)
        names = [s.name for s in filtered]
        print(f"  {quality:<14s} ({len(filtered)}): {', '.join(names)}")
    print()

    # Size overview
    print("--- Estimated Total Corpus Size ---")
    low, high = get_total_size_range()
    print(f"  Conservative: {low}")
    print(f"  Including v2: {high}")
    print()

    # Demonstrate lookup
    print("--- Source Lookup Example ---")
    source = get_source_by_name("The Pile")
    if source:
        print(f"  Name:        {source.name}")
        print(f"  Description: {source.description}")
        print(f"  Size:        {source.estimated_size}")
        print(f"  Quality:     {source.quality_level}")
        print(f"  Languages:   {source.language_coverage}")
        print(f"  Uses:        {', '.join(source.typical_use_cases)}")
        if source.notes:
            print(f"  Notes:       {source.notes}")
    print()

    print("Demo complete. All source data is reference-based (no downloads).")


if __name__ == "__main__":
    main()
