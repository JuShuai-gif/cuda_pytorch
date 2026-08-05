#!/usr/bin/env bash
# cachegrind.sh - run valgrind --tool=cachegrind on a command.
set -u

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <command...>" >&2
    exit 2
fi

if ! command -v valgrind >/dev/null 2>&1; then
    echo "valgrind not installed." >&2
    echo "Install it, e.g.:" >&2
    echo "  sudo apt install valgrind    (Debian/Ubuntu)" >&2
    echo "  sudo dnf install valgrind    (Fedora)" >&2
    exit 1
fi

out=/tmp/cachegrind_cpu_memory.out
rm -f "$out"
valgrind --tool=cachegrind --cachegrind-out-file="$out" "$@"
echo
echo "== cg_annotate summary =="
if command -v cg_annotate >/dev/null 2>&1; then
    cg_annotate "$out" 2>/dev/null | sed -n '1,45p' || true
else
    echo "(cg_annotate not found; raw output file: $out)"
fi
