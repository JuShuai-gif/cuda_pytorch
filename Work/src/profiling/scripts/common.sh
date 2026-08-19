#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

profiling_die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

profiling_require_command() {
  command -v "$1" >/dev/null 2>&1 || profiling_die "required command not found: $1"
}

profiling_safe_label() {
  local value=${1:-}
  [[ -n "$value" ]] || profiling_die "empty artifact label"
  [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || profiling_die "unsafe label: $value"
  printf '%s\n' "$value"
}

profiling_new_run_dir() {
  local root=${1:-}
  local label
  label=$(profiling_safe_label "${2:-}")
  [[ -n "$root" && "$root" != "/" ]] || profiling_die "unsafe output root: ${root:-<empty>}"
  mkdir -p -- "$root"
  local stamp candidate suffix
  stamp=$(date -u +%Y%m%dT%H%M%S)
  for suffix in $(seq 0 99); do
    candidate="${root}/${label}_${stamp}_$$_${suffix}"
    if (umask 077 && mkdir -- "$candidate") 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  profiling_die "could not allocate a unique run directory under $root"
}

profiling_write_command() {
  local output=${1:?}
  shift
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } >"$output"
}
