#!/usr/bin/env bash
# run_all.sh - run every experiment binary found in the build dir.
# Hardware-unsupported experiments print their own notices and skip.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/src/build}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Build dir not found: $BUILD_DIR (run scripts/build.sh first)" >&2
    exit 1
fi

declare -A ORDER
i=0
for b in \
    memory_latency sequential_random_access cache_line_size stride_access \
    cache_capacity cache_associativity cache_conflict write_back_behavior \
    matrix_traversal cache_blocking aos_soa pointer_chasing false_sharing \
    atomic_contention thread_affinity prefetch non_temporal_store tlb_capacity \
    page_size huge_pages page_fault memory_mapping memory_bandwidth \
    numa_local_remote numa_first_touch numa_replication instruction_cache \
    integrated_project p1_debug_vs_release p2_dead_code_elimination \
    p3_vector_bool p4_shared_ptr_contention p5_alignment p6_benchmark_noise \
    p7_atomic_memory_order p8_volatile_not_atomic p9_hugepage_verify; do
    ORDER["$b"]=$i
    i=$((i + 1))
done

list=()
for exe in "$BUILD_DIR"/*; do
    [[ -f "$exe" && -x "$exe" ]] || continue
    name="${exe##*/}"
    if [[ -n "${ORDER["$name"]+x}" ]]; then
        list+=("$name")
    fi
done

# stable order
mapfile -t sorted < <(for e in "${list[@]}"; do echo "${ORDER[$e]} $e"; done | sort -n | cut -d' ' -f2-)

echo "== running ${#sorted[@]} experiments =="
for e in "${sorted[@]}"; do
    echo
    echo "========== $e =========="
    "$BUILD_DIR/$e"
    echo "rc=$?"
done
