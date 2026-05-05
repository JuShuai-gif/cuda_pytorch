# Lecture 8: Data-Parallel Thinking

**Course:** Stanford CS149 - Parallel Computing, Fall 2025
**PDF:** Lecture 8

---

## Core Concepts

### 1. The Data-Parallel Model

**Key idea:** Organize computation as operations on **sequences** of elements, not as "what workers do."

- Programs access elements through specific operations, not direct element access
- High-performance parallel implementations of these primitives exist
- Main challenge: avoid being bandwidth-bound

### 2. Core Data-Parallel Primitives

#### Map
Applies a side-effect-free unary function `f :: a -> b` to all elements.
```
map f [3,8,4,6,3,9,2,8] → [13,18,14,16,13,19,12,18]
```
- C++ equivalent: `std::transform`
- **Trivially parallelizable** because `f` is side-effect-free (pure function)

#### Fold (Reduce)
Applies binary operation to each element and accumulated value.
```
fold 10 (+) [3,8,4,6,3,9,2,8] → 53
```
- **Parallel fold** requires:
  - Binary function `f :: (b,a) -> b`
  - Combiner function `comb :: (b,b) -> b` (associative)
  - Identity element for both `f` and `comb`

#### Scan (Prefix Sum)
Outputs running totals of an associative operation.
```
scan_inclusive (+) [3,8,4,6,3,9,2,8] → [3,11,15,21,24,33,35,43]
scan_exclusive (+) [3,8,4,6,3,9,2,8] → [0,3,11,15,21,24,33,35]
```

### 3. Parallel Scan Algorithms

#### Naive Inclusive Scan
- **Work:** O(N log N) — inefficient!
- **Span:** O(log N)
- Each step: double the stride between paired elements

#### Work-Efficient Exclusive Scan (Blelloch Algorithm)
- **Work:** O(N)
- **Span:** O(log N)
- Two phases:
  - **Up-sweep** (reduce phase): build partial sums in a tree
  - **Down-sweep**: propagate partial sums downward
- Down-sweep starts by setting last element to 0 (identity), then distributes values

```
Up-sweep pseudocode:
for d=0 to (log2N - 1):
    forall k=0 to N-1 by 2^(d+1):
        a[k + 2^(d+1) - 1] = a[k + 2^d - 1] + a[k + 2^(d+1) - 1]

Down-sweep pseudocode:
x[N-1] = 0
for d=(log2N - 1) down to 0:
    forall k=0 to N-1 by 2^(d+1):
        tmp = a[k + 2^d - 1]
        a[k + 2^d - 1] = a[k + 2^(d+1) - 1]
        a[k + 2^(d+1) - 1] = tmp + a[k + 2^(d+1) - 1]
```

#### Multi-Core Scan Strategy (2 cores)
1. Core 1: sequential scan on left half
2. Core 2: sequential scan on right half
3. Core 2 adds base from Core 1 to its results
- **Work:** O(N), constant ~1.5x sequential
- **High spatial locality**

#### GPU/CUDA Scan (Multi-level)
1. **Intra-warp scan** (32 elements): use naive O(N log N) — better SIMD utilization
2. **Inter-warp scan**: apply bases from warp-level scan
3. **Inter-block scan**: kernel-level scan on block partial sums
- Different strategies at different hierarchy levels

### 4. Segmented Scan

Performs scans on contiguous partitions of input simultaneously.

```
segmented_scan_exclusive(+) [[1,2],[6],[1,2,3,4]] → [[0,1],[0],[0,1,3,6]]
```

**Flag representation:**
```
flag: 1 0 0 1 0 0 0 0
data: 1 2 3 4 5 6 7 8
```

**Modified work-efficient algorithm:**
- Up-sweep: only combine if flag indicates same segment
- Down-sweep: propagate flag information, reset at segment boundaries

### 5. Gather/Scatter

#### Gather
```
output[i] = input[index[i]]
```
- Data-parallel read: execute in any order
- AVX2 supports gathered loads (2013); GPU supports gather

#### Scatter
```
output[index[i]] = input[i]
```
- Requires synchronization when indices collide
- AVX512 supports scatter; GPU supports scatter

### 6. Sparse Matrix Multiplication via Data-Parallel Ops

```
values = [[3,1], [2], [4], [2,6,8]]
cols   = [[0,2], [1], [2], [1,2,3]]
row_starts = [0, 2, 3, 4]

Steps:
1. Gather from x based on cols
2. Map (multiply) values × gathered
3. Create flags from row_starts
4. Segmented inclusive scan on products
5. Extract last element of each segment
```

### 7. Turning Irregular Parallelism into Regular Parallelism

**Problem:** Build grid of particles (16 cells, 1M particles)

| Solution | Approach | Problem |
|---|---|---|
| 1: Thread per particle + single lock | Atomic append to shared list | Massive contention |
| 2: Per-cell locks | Fine-grained locking | ~16x less contention, still high |
| 3: Parallel over cells | For each cell, check all particles | Too few parallel tasks; work-inefficient |
| 4: Partial results + merge | N partial grids, then combine | Extra memory; extra merge work |
| 5: **Data-parallel** (sort-based) | Map → sort → find starts/ends | Extra BW; but massive parallelism, no locks |

**Data-parallel approach steps:**
1. Map: compute cell for each particle
2. Sort: by cell ID (also permutes particle index)
3. Find start/end of each cell segment

### 8. Parallel Histogram via Sort

```
Steps:
1. map f() over input → bin_ids[]
2. sort bin_ids[]
3. find starts of each bin in sorted list
4. compute bin sizes from starts and ends
```

---

## Knowledge Points → Corresponding C++ Files

| Knowledge Point | C++ File |
|---|---|
| Map, reduce, and basic data-parallel operations | `lecture8_part1.cpp` |
| Parallel scan: naive O(N log N) and work-efficient O(N) | `lecture8_part2.cpp` |
| Segmented scan, gather, scatter | `lecture8_part3.cpp` |

---

## Actionable Learning Points

1. **Think in sequences**: map, reduce, scan are the building blocks of parallel algorithms
2. **O(N log N) vs O(N) scan**: work-efficient algorithm matters for large N, but naive may be better for small SIMD (32 elements)
3. **Multi-level strategy**: use different algorithms at different levels of the machine hierarchy
4. **Segmented scan enables working with irregular data structures** (lists of lists) in regular, data-parallel ways
5. **Sort-based approaches** trade extra bandwidth for elimination of fine-grained synchronization
6. **Gather = parallel-friendly; Scatter = needs synchronization** when indices collide
7. **Key tension**: data-parallel solutions often require multiple passes (bandwidth-hungry) but expose massive parallelism
