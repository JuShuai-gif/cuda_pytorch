# Lecture 18: Transactional Memory Part II + Course Wrap Up

**PDF**: Lecture 18 - Transactional Memory Part II + Course Wrap Up
**Course**: Stanford CS149, Fall 2025

---

## Core Concepts Summary

### 1. TM Implementation Design Space (Review)

| Implementation | Versioning | Read Detection | Write Detection |
|---|---|---|---|
| Sun TL2 (SW) | Lazy | Optimistic | Optimistic |
| MS OSTM (SW) | Lazy | Optimistic | Pessimistic |
| Intel STM (SW) | Eager | Optimistic/Pessimistic | Pessimistic |
| Stanford TCC (HW) | Lazy | Optimistic | Optimistic |
| MIT LTM (HW) | Lazy | Pessimistic | Pessimistic |
| Wisconsin LogTM (HW) | Eager | Pessimistic | Pessimistic |

### 2. Software Transactional Memory (STM) Implementation

#### Runtime Data Structures
- **Transaction Descriptor** (per-thread): read set, write set, undo log or write buffer, status
- **Transaction Record** (per data item): pointer-sized record guarding shared data
  - Shared state: version number (timestamp) or shared reader lock
  - Exclusive state: writer lock pointing to owner transaction

#### Mapping Data to Transaction Records
| Language | Approach | Trade-off |
|---|---|---|
| Java/C# | Embed TxR in each object header | Low mapping overhead, false conflicts at object level |
| C/C++ | Hash fields or addresses to global table | Flexible granularity, higher mapping overhead |

#### Conflict Detection Granularity
- **Object-level**: Low overhead, false conflicts
- **Field/Word-level**: Minimum false conflicts, higher overhead
- **Cache-line-level**: Natural for HTM, hard for compiler analysis
- **Hybrid**: Per-type basis (e.g., element-level for arrays, object-level for non-arrays)

### 3. Intel McRT STM Algorithm Example
- **Versioning**: Eager (undo-log based)
- **Reads**: Optimistic (validate after read)
- **Writes**: Pessimistic (acquire lock before write)
- **Version tracking**: Global timestamp + per-object timestamp
- **STM read**: direct read → validate (check unlocked, version ≤ local timestamp) → insert in read set
- **STM write**: validate → acquire lock → create undo log → write in place
- **STM commit**: atomically increment global timestamp by 2 → validate read-set → release locks with new version

### 4. STM Challenges & Compiler Optimizations
- **Overhead**: 2-8x per-thread slowdown due to software barriers
- **Optimization technique**: decompose monolithic barriers to expose redundancies
  - Remove redundant `txnOpenForWrite` / `txnOpenForRead` calls
  - Merge consecutive undo-log entries on same object
  - Result: <40% overhead over sequential, <30% over lock-based

### 5. Hardware Transactional Memory (HTM)

#### Cache-based Implementation
- Data versioning in caches (write buffer or undo log in cache lines)
- New cache line metadata: **R bit** (read set), **W bit** (write set)
- Conflict detection through **cache coherence protocol**:
  - BusRd to W-line = read-write conflict
  - BusRdX to R-line = write-read conflict
  - BusRdX to W-line = write-write conflict

#### HTM Execution Steps
1. **Xbegin**: Initialize CPU/cache state, checkpoint registers
2. **Load**: Mark cache line with R bit
3. **Store**: Mark cache line with W bit (lazy: buffer in cache; eager: keep undo log)
4. **Xcommit**: Two-phase - validate (request exclusive access to write set), then gang-clear R/W bits
5. **Abort (conflict)**: Invalidate write set, gang-reset R/W bits, restore register checkpoint

### 6. Intel Haswell RTM (Restricted Transactional Memory)
- Instructions: `xbegin` (with fallback address), `xend`, `xabort`
- Tracks read/write set in L1 cache
- **Limitation**: cache eviction of any line in read/write set causes abort
- **Does NOT guarantee progress** → must provide lock-based fallback path
- Intel optimization guide (Chapter 12) provides guidelines for transaction success

### 7. HTM Performance
- 2x-7x over STM performance
- Within 10% of sequential for single thread
- Scales efficiently with processor count
- Near-ideal speedup on Vacation benchmark

---

## Knowledge Points → C++ File Mapping

| Knowledge Point | C++ File |
|---|---|
| STM implementation: transaction descriptor, read/write sets, per-object versioning | `lecture18_part1.cpp` |
| STM compiler optimization: barrier decomposition + redundancy elimination | `lecture18_part2.cpp` |
| HTM cache-line simulation with R/W bits and coherence conflict detection | `lecture18_part3.cpp` |

---

## Actionable Learning Points

1. **Understand the full STM read path**: validate (check version ≤ local timestamp) → if stale, validate entire read set → insert to read set → return value
2. **Know why STM commit increments by 2**: LSb used as write-lock bit, MS bits as version number (avoid conflict between lock state and version)
3. **Recognize why HTM is so much faster**: no software barriers, conflict detection piggybacks on existing coherence traffic
4. **Understand HTM limitations**: L1 cache size bounds transaction size; context switches, interrupts, page faults can spuriously abort transactions
5. **The fallback path is critical**: Intel RTM requires a lock-based fallback for when hardware transactions repeatedly abort
6. **Gang-clear optimization**: R/W bits are cleared as a group (not individually) on commit/abort for efficiency

---

## Key Architecture Insight

HTM essentially extends the cache coherence protocol:
- **Without HTM**: coherence tracks per-cache-line ownership (MESI states)
- **With HTM**: coherence additionally tracks per-cache-line transactional read/write membership
- Conflict detection becomes a by-product of normal coherence snoops

```
Cache line metadata:
  MESI state (4 bits) | R bit | W bit | Tag | Data (64 bytes)

Coherence conflict triggers:
  BusRd (shared request) to W-line  → read-write conflict → abort reader
  BusRdX (exclusive request) to R-line → write-read conflict → abort reader
  BusRdX (exclusive request) to W-line → write-write conflict → abort one writer
```
