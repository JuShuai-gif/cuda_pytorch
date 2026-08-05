#!/usr/bin/env bash
# system_info.sh - print system memory/cache/NUMA/compiler information.
set -u

echo "== uname =="
uname -a

echo
echo "== lscpu (selected) =="
lscpu 2>/dev/null | grep -Ei 'architecture|model name|core|thread|socket|numa|hypervisor|virtualization' || true

echo
echo "== numactl --hardware =="
numactl --hardware 2>/dev/null || echo "(numactl not available)"

echo
echo "== CPU cache topology (/sys) =="
for i in /sys/devices/system/cpu/cpu0/cache/index*; do
    [[ -d "$i" ]] || continue
    printf '%-12s level=%-3s %-12s size=%-8s ways=%-4s line=%s\n' \
        "$(basename "$i")" \
        "$(cat "$i/level" 2>/dev/null)" \
        "$(cat "$i/type" 2>/dev/null)" \
        "$(cat "$i/size" 2>/dev/null)" \
        "$(cat "$i/ways_of_associativity" 2>/dev/null)" \
        "$(cat "$i/coherency_line_size" 2>/dev/null)"
done

echo
echo "== page / huge pages =="
echo "pagesize: $(getconf PAGESIZE) bytes"
grep -E 'HugePages|Hugepagesize' /proc/meminfo || true
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"

echo
echo "== memory =="
grep -E 'MemTotal|MemFree' /proc/meminfo

echo
echo "== compilers / kernel =="
uname -r
g++ --version 2>/dev/null | head -1 || true
clang++ --version 2>/dev/null | head -1 || true
cmake --version 2>/dev/null | head -1 || true
