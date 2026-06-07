"""
演示 LLM 数据收集中的网页爬取和文本提取功能。

使用 trafilatura 作为主要提取引擎，BeautifulSoup 作为后备方案。
两个库均采用优雅导入方式，即使未安装模块也能正常导入，
自动回退到模拟演示模式。

LLM 预训练的典型用法：
    - 从 CommonCrawl WARC 文件或实时 URL 获取原始 HTML
    - 提取干净的纯文本（去除网页模板/广告等内容）
    - 按语言、质量启发式规则、去重进行过滤
"""

from __future__ import annotations

import sys
import time
from typing import Optional
from urllib.error import URLError


# ---------------------------------------------------------------------------
# 优雅导入：优先尝试 trafilatura，其次 BeautifulSoup，否则模拟
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
# 核心提取函数
# ---------------------------------------------------------------------------


def fetch_and_extract_text(
    url: str,
    timeout: float = 15.0,
    user_agent: Optional[str] = None,
) -> str:
    """获取网页并从中提取干净的纯文本。

    策略:
        1. 通过 ``requests`` 下载原始 HTML。
        2. 优先使用 ``trafilatura`` 进行最佳的网页模板去除。
        3. 回退到 ``BeautifulSoup`` 进行基础文本提取。
        4. 如果两者都不可用，返回模拟占位文本。

    Args:
        url: 要获取的 URL。
        timeout: 请求超时时间（秒）。
        user_agent: 自定义 User-Agent 头；未指定时使用合理的默认值。

    Returns:
        提取出的干净文本，或在失败时返回错误描述。
    """
    headers: dict[str, str] = {
        "User-Agent": user_agent
        or ("Mozilla/5.0 (compatible; CS336-LLM-Crawler/1.0; +https://example.com/bot)")
    }

    # --- 步骤 1：获取 ---
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

    # --- 步骤 2：使用 trafilatura 提取 ---
    if _trafilatura_available:
        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=False
        )
        if extracted:
            return extracted.strip()

    # --- 步骤 3：回退到 BeautifulSoup ---
    if _bs4_available:
        soup = BeautifulSoup(html, "html.parser")
        # 在提取文本之前移除 script 和 style 标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # 合并多个空行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    # --- 步骤 4：两个库都不可用 ---
    return _simulate_extraction(url)


# ---------------------------------------------------------------------------
# 模拟回退
# ---------------------------------------------------------------------------


def _simulate_extraction(url: str) -> str:
    """当没有可用的提取库时，返回一段简短的模拟提取结果。"""
    return (
        f"[SIMULATED] This is placeholder text extracted from {url}.\n"
        "Install trafilatura and beautifulsoup4 for real extraction.\n"
        "Run: pip install trafilatura beautifulsoup4 requests\n\n"
        "In a real LLM data pipeline, this would contain the cleaned text\n"
        "from the target URL after boilerplate removal, language filtering,\n"
        "and quality heuristics."
    )


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    # 判断可用的提取方法
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

    # 示例 URL（这些是轻量级的知名网站）
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

        # 打印前 500 个字符作为预览
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
