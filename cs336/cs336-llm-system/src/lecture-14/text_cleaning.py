"""
Text cleaning pipeline for LLM training data.

Provides a composable `TextCleaner` class that chains individual cleaning
steps (HTML tag removal, Unicode normalization, whitespace normalization)
into a single pass over raw text.
"""

import re
import unicodedata
from typing import Callable, List


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from *text* using a simple regex.

    This is a best-effort clean and does not attempt to parse HTML
    structurally.  It handles self-closing tags, attributes with quotes,
    and nested <script>/<style> blocks.
    """
    # Remove script and style blocks including their content
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalization to *text*.

    NFKC (Compatibility Composition) decomposes characters and recomposes
    them, which helps unify visually-similar forms (e.g. fullwidth Latin
    letters -> halfwidth, ligatures -> individual characters).
    """
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip leading/trailing.

    This also normalises non-breaking spaces (U+00A0) and other Unicode
    space characters to the standard ASCII space.
    """
    # Replace common Unicode whitespace with regular space
    text = re.sub(r"[\u00A0\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]+", " ", text)
    # Collapse all whitespace runs
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Composable cleaner
# ---------------------------------------------------------------------------


class TextCleaner:
    """Chain cleaning functions into a pipeline.

    Parameters
    ----------
    steps : List[Callable[[str], str]]
        Ordered list of functions (``str -> str``) to apply in sequence.
    """

    def __init__(self, steps: List[Callable[[str], str]] | None = None) -> None:
        if steps is None:
            steps = [remove_html_tags, normalize_unicode, normalize_whitespace]
        self._steps: List[Callable[[str], str]] = steps

    def clean(self, text: str) -> str:
        """Run *text* through every step in the pipeline."""
        for step in self._steps:
            text = step(text)
        return text

    @property
    def steps(self) -> List[Callable[[str], str]]:
        """Return a copy of the current step list."""
        return list(self._steps)

    def add_step(self, step: Callable[[str], str]) -> None:
        """Append a cleaning function to the end of the pipeline."""
        self._steps.append(step)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate each cleaning step and the full pipeline."""
    dirty_text = (
        "<html><head><title>Hello</title></head>"
        "<body><p>This   is   <b>dirty</b> &amp; <i>noisy</i> text.</p>"
        "<!-- a comment -->"
        "<script>alert('xss');</script>"
        "<p>M\u2003u\u2003l\u2003t\u2003i\u2002s\u2002p\u2002a\u2002c\u2002e</p>"
        "<p>\u3000fullwidth space\u3000</p>"
        "</body></html>"
        "       \t\n\r   "
    )

    print("=" * 60)
    print("ORIGINAL")
    print("=" * 60)
    print(repr(dirty_text))
    print(dirty_text)
    print()

    print("=" * 60)
    print("AFTER remove_html_tags")
    print("=" * 60)
    step1 = remove_html_tags(dirty_text)
    print(repr(step1))
    print()

    print("=" * 60)
    print("AFTER normalize_unicode")
    print("=" * 60)
    step2 = normalize_unicode(step1)
    print(repr(step2))
    print()

    print("=" * 60)
    print("AFTER normalize_whitespace")
    print("=" * 60)
    step3 = normalize_whitespace(step2)
    print(repr(step3))
    print()

    print("=" * 60)
    print("FULL PIPELINE (TextCleaner)")
    print("=" * 60)
    cleaner = TextCleaner()
    result = cleaner.clean(dirty_text)
    print(f"Output: {result!r}")
    print()

    # Custom pipeline: reverse order for demonstration
    print("=" * 60)
    print("CUSTOM PIPELINE (whitespace first, then HTML)")
    print("=" * 60)
    custom = TextCleaner([normalize_whitespace, remove_html_tags])
    print(repr(custom.clean(dirty_text)))


if __name__ == "__main__":
    main()
