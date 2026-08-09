#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; build="${1:-$root/build}"
for demo in 05_false_sharing 26_lock_contention_bad_good 27_shared_mutex 28_thread_imbalance 12_context_switch; do [[ -x "$build/$demo" ]] && { echo "===== $demo ====="; timeout 90s "$build/$demo"; }; done
command -v strace >/dev/null && strace -f -c "$build/26_lock_contention_bad_good" || true
