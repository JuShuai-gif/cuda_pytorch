"""
LLM 训练数据的文本清洗管道。

提供一个可组合的 `TextCleaner` 类，将各个清洗步骤（HTML 标签移除、Unicode 规范化、空白字符规范化）
串联成对原始文本的单次处理流程。
"""

import re
import unicodedata
from typing import Callable, List


# ---------------------------------------------------------------------------
# 单独的清洗函数
# ---------------------------------------------------------------------------


def remove_html_tags(text: str) -> str:
    """使用简单正则表达式移除 *text* 中的 HTML 标签。

    这是一个尽力而为的清洗方案，不会对 HTML 进行结构性解析。
    它处理自闭合标签、带引号的属性以及嵌套的 <script>/<style> 块。
    """
    # 移除 script 和 style 块及其内容
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 移除 HTML 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # 移除剩余的 HTML 标签
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def normalize_unicode(text: str) -> str:
    """对 *text* 应用 NFKC Unicode 规范化。

    NFKC（兼容性组合）会分解字符并重新组合，有助于统一视觉上相似的形态
    （例如全角拉丁字母 -> 半角，连字 -> 独立字符）。
    """
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """将空白字符连续序列压缩为单个空格，并去除首尾空白。

    同时将不间断空格（U+00A0）及其他 Unicode 空格字符
    规范化为标准 ASCII 空格。
    """
    # 将常见的 Unicode 空白字符替换为普通空格
    text = re.sub(r"[\u00A0\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]+", " ", text)
    # 压缩所有空白字符序列
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 可组合的清洗器
# ---------------------------------------------------------------------------


class TextCleaner:
    """将清洗函数串联为管道。

    参数
    ----------
    steps : List[Callable[[str], str]]
        按顺序应用的有序函数列表（``str -> str``）。
    """

    def __init__(self, steps: List[Callable[[str], str]] | None = None) -> None:
        if steps is None:
            steps = [remove_html_tags, normalize_unicode, normalize_whitespace]
        self._steps: List[Callable[[str], str]] = steps

    def clean(self, text: str) -> str:
        """将 *text* 逐一遍历管道中的每个步骤进行处理。"""
        for step in self._steps:
            text = step(text)
        return text

    @property
    def steps(self) -> List[Callable[[str], str]]:
        """返回当前步骤列表的副本。"""
        return list(self._steps)

    def add_step(self, step: Callable[[str], str]) -> None:
        """将清洗函数追加到管道末尾。"""
        self._steps.append(step)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    """演示每个清洗步骤和完整管道。"""
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

    # 自定义管道：为了演示，先处理空白再处理 HTML
    print("=" * 60)
    print("CUSTOM PIPELINE (whitespace first, then HTML)")
    print("=" * 60)
    custom = TextCleaner([normalize_whitespace, remove_html_tags])
    print(repr(custom.clean(dirty_text)))


if __name__ == "__main__":
    main()
