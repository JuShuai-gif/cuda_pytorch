"""
Demonstrate web crawling and text extraction for LLM data collection.

Uses trafilatura as the primary extraction engine with BeautifulSoup as fallback.
Both libraries are imported gracefully so the module remains importable even
when they are not installed, falling back to a simulated demo mode.

Typical usage for LLM pretraining:
    - Fetch raw HTML from CommonCrawl WARC files or live URLs
    - Extract clean plain text (boilerplate removal)
    - Filter by language, quality heuristics, deduplication
"""

from __future__ import annotations

import sys
import time
from typing import Optional
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Graceful imports: try trafilatura first, then BeautifulSoup, else simulate
# ---------------------------------------------------------------------------

_trafilatura_available = False
_bs4_available = False

try:
    import trafilatura  # type: ignore[import-untyped]

    _trafilatura_available = True
except ImportError:
    pass

try:
    import requests
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]

    _bs4_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def fetch_and_extract_text(
    url: str,
    timeout: float = 15.0,
    user_agent: Optional[str] = None,
) -> str:
    """Fetch a web page and extract clean plain text from it.

    Strategy:
        1. Download raw HTML via ``requests``.
        2. Try ``trafilatura`` for best-in-class boilerplate removal.
        3. Fall back to ``BeautifulSoup`` for basic text extraction.
        4. If neither is available, return a simulated placeholder.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        user_agent: Custom User-Agent header; falls back to a reasonable default.

    Returns:
        Extracted clean text, or an error description on failure.
    """
    headers: dict[str, str] = {
        "User-Agent": user_agent
        or ("Mozilla/5.0 (compatible; CS336-LLM-Crawler/1.0; +https://example.com/bot)")
    }

    # --- Step 1: Fetch ---
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html: str = resp.text
    except requests.exceptions.Timeout:
        return f"[ERROR] Timeout after {timeout:.0f}s fetching {url}"
    except requests.exceptions.ConnectionError:
        return f"[ERROR] Connection error fetching {url}"
    except requests.exceptions.HTTPError as exc:
        return f"[ERROR] HTTP {exc.response.status_code if exc.response else '?'} for {url}"
    except (requests.exceptions.RequestException, URLError) as exc:
        return f"[ERROR] Request failed for {url}: {exc}"

    # --- Step 2: Extract with trafilatura ---
    if _trafilatura_available:
        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=False
        )
        if extracted:
            return extracted.strip()

    # --- Step 3: Fallback to BeautifulSoup ---
    if _bs4_available:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style tags before extracting text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Collapse multiple blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    # --- Step 4: Neither library available ---
    return _simulate_extraction(url)


# ---------------------------------------------------------------------------
# Simulation fallback
# ---------------------------------------------------------------------------


def _simulate_extraction(url: str) -> str:
    """Return a short simulated extraction when no library is available."""
    return (
        f"[SIMULATED] This is placeholder text extracted from {url}.\n"
        "Install trafilatura and beautifulsoup4 for real extraction.\n"
        "Run: pip install trafilatura beautifulsoup4 requests\n\n"
        "In a real LLM data pipeline, this would contain the cleaned text\n"
        "from the target URL after boilerplate removal, language filtering,\n"
        "and quality heuristics."
    )


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    # Determine available extraction method
    if _trafilatura_available:
        method = "trafilatura (primary) + BeautifulSoup (fallback)"
    elif _bs4_available:
        method = "BeautifulSoup only (trafilatura not available)"
    else:
        method = "SIMULATED (install trafilatura/beautifulsoup4 for real extraction)"
        print("=" * 60)
        print("  WARNING: trafilatura and beautifulsoup4 are not installed.")
        print("  Running in simulated mode.")
        print("  Install with: pip install trafilatura beautifulsoup4 requests")
        print("=" * 60)

    print("Web Crawler Demo for LLM Data Collection")
    print(f"Extraction method: {method}")
    print()

    # Example URLs (these are lightweight, well-known sites)
    example_urls: list[str] = [
        "https://example.com/",
        "https://httpbin.org/html",
        "https://en.wikipedia.org/wiki/Language_model",
    ]

    for i, url in enumerate(example_urls, 1):
        print(f"[{i}/{len(example_urls)}] Fetching: {url}")
        start = time.perf_counter()
        text = fetch_and_extract_text(url, timeout=10.0)
        elapsed = time.perf_counter() - start

        # Print first 500 characters as a preview
        preview = text[:500]
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Text length: {len(text)} chars")
        print(f"  Preview:\n    {preview.replace(chr(10), chr(10) + '    ')}")
        if len(text) > 500:
            print("    ... (truncated)")
        print()

    print("Demo complete. Each call returns cleaned, deduplication-ready text.")


if __name__ == "__main__":
    main()
