"""
LLM 训练数据的流式数据处理管道。

实现了一个 ``DataPipeline`` 类，以流式方式串联：读取 → 清洗 →
过滤 → 分词，一次产出（yield）一个文档，避免将整个语料库加载到内存中。
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
# 简单的词级分词器（无外部依赖）
# ---------------------------------------------------------------------------


def simple_tokenize(text: str) -> List[str]:
    """使用空白和标点边界将 *text* 分割为词 token。

    这是一个用于演示的最小化分词器；生产管道
    会使用子词分词器（BPE、SentencePiece 等）。
    """
    # 保留单词字符和撇号以处理缩写
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|[0-9]+|[^\s\w]", text)
    return tokens


# ---------------------------------------------------------------------------
# DataPipeline 类
# ---------------------------------------------------------------------------


class DataPipeline:
    """用于预处理 LLM 训练数据的流式管道。

    从可迭代对象中读取文档，应用一系列转换阶段，
    并逐个产出（yield）处理后的文档。

    参数
    ----------
    reader : Callable[[], Iterable[str]]
        无参数的可调用对象，返回原始文档字符串的可迭代对象。
    cleaner : TextCleaner | None
        文本清洗管道。如果为 ``None``，则使用默认的 ``TextCleaner``。
    filters : List[Callable[[str], Tuple[bool, Optional[str]]]] | None
        过滤器函数列表。每个函数接收清洗后的文本并返回
        ``(keep: bool, reason: str | None)``。
        任何过滤器返回 ``keep=False`` 的文档将被跳过。
    tokenizer : Callable[[str], List[str]] | None
        分词器函数。如果为 ``None``，则使用 ``simple_tokenize``。
    report_interval : int
        每处理 *report_interval* 个文档打印一次进度报告。
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

        # 统计数据
        self.stats: Dict[str, int] = collections.defaultdict(int)

    def process(self) -> Iterator[Dict[str, object]]:
        """以字典形式产出处理后的文档。

        每个产出的字典包含 ``"raw"``、``"cleaned"``、``"tokens"``
        和 ``"doc_id"`` 键。

        产出
        ------
        dict
            带元数据的处理后文档。
        """
        start_time = time.monotonic()
        skipped_reasons: Dict[str, int] = collections.defaultdict(int)

        for doc_id, raw in enumerate(self._reader()):
            self.stats["total_read"] += 1

            # 阶段 1：清洗
            cleaned = self._cleaner.clean(raw)
            if not cleaned.strip():
                skipped_reasons["empty_after_cleaning"] += 1
                continue

            # 阶段 2：过滤
            keep = True
            for filt in self._filters:
                keep, reason = filt(cleaned)
                if not keep:
                    skipped_reasons[reason or "unknown_filter"] += 1
                    break
            if not keep:
                continue

            # 阶段 3：分词
            tokens = self._tokenizer(cleaned)
            self.stats["total_accepted"] += 1

            # 进度报告
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
# 预构建的过滤器工厂函数
# ---------------------------------------------------------------------------


def make_language_filter(
    target_lang: str,
) -> Callable[[str], Tuple[bool, Optional[str]]]:
    """返回一个仅保留 *target_lang* 语言文档的过滤器。"""

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
    """返回一个移除高 perplexity（低质量）文本的过滤器。

    从 *clean_texts* 构建一个 n-gram 模型，计算基线
    perplexity，并拒绝任何 perplexity 超过
    ``baseline * max_ratio`` 的文档。
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
# 演示
# ---------------------------------------------------------------------------


def _generate_synthetic_docs() -> List[str]:
    """创建一小组合成文档。"""
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
    """演示完整的流式数据处理管道。"""
    raw_docs = _generate_synthetic_docs()
    print(f"Generated {len(raw_docs)} synthetic documents\n")

    # 构建过滤器
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

    # 构建管道
    pipeline = DataPipeline(
        reader=lambda: raw_docs,  # 返回整个列表（模拟读取器）
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
