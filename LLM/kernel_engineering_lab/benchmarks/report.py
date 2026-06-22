"""
CLI entry point for benchmark report generation.

Usage:
    python benchmarks/report.py results.json          # print comparison table
    python benchmarks/report.py results.json -o report  # save CSV and Markdown
    python benchmarks/report.py results1.json results2.json -o combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.benchmark_utils import (
    compare_kernels,
    generate_report,
    load_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from JSON result files."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="One or more JSON files containing benchmark results.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Base output path (without extension) for CSV and Markdown reports.",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only print the comparison table, skip file output.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["throughput", "p50_ms", "bandwidth_gb_s", "gflops", "name"],
        default="throughput",
        help="Sort results by this field.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order.",
    )
    args = parser.parse_args()

    all_results = []
    for path in args.results:
        try:
            results = load_results(path)
            all_results.extend(results)
            print(f"Loaded {len(results)} results from {path}")
        except FileNotFoundError:
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            sys.exit(1)

    if not all_results:
        print("No results loaded.", file=sys.stderr)
        sys.exit(1)

    # Sort results
    sort_key = args.sort_by
    all_results.sort(
        key=lambda r: getattr(r, sort_key),
        reverse=not args.reverse if sort_key != "throughput" else True,
    )

    if args.table_only:
        compare_kernels(all_results)
    else:
        compare_kernels(all_results)
        print()
        if args.output:
            generate_report(all_results, args.output)
        else:
            # Print Markdown to stdout
            report = generate_report(all_results)
            print(report)


if __name__ == "__main__":
    main()
