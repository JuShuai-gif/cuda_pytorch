# Lecture 14: Cache Coherence

**PDF:** Lecture 14 — Cache Coherence  
**Course:** Stanford CS149, Fall 2025 — Parallel Computing

---

## Core Concepts Summary

### 1. The Cache Coherence Problem
- In shared-memory multiprocessors, each core has a private cache
- Without coherence, different cores can observe **different values** for the same memory location
- This is a *hardware* problem (data replication), NOT a mutual exclusion problem — locks can't fix it

### 2. Formal Definition of Coherence
A memory system is **coherent** if, for each memory location, there exists a hypothetical serial order of all operations such that:
1. Operations from any one processor appear in program order
2. A read returns the value of the last write in that serial order

### 3. Cache Coherence Invariants
- **Single-Writer, Multiple-Read (SWMR)** invariant:
  - Read-Write epoch: exactly one processor may write
  - Read-Only epoch: multiple processors may read
- **Data-Value invariant** (write serialization): value at epoch start = value at end of previous read-write epoch

### 4. MSI Protocol (Snooping, Write-Back Invalidation)
| State | Meaning |
|-------|---------|
| **M** (Modified) | Line valid in exactly one cache; dirty; can write without bus transaction |
| **S** (Shared) | Line valid in one or more caches; clean; memory is up-to-date |
| **I** (Invalid) | Line not valid in this cache |

**Bus transactions:**
| Transaction | Purpose |
|-------------|---------|
| BusRd | Obtain shared copy (no intent to modify) |
| BusRdX | Obtain exclusive copy (intent to modify); invalidates others |
| BusWB | Write dirty line back to memory |

**Key transitions:**
- I → S: PrRd triggers BusRd (miss, load shared copy)
- I → M: PrWr triggers BusRdX (miss, load with exclusive intent)
- S → M: PrWr triggers BusRdX (upgrade — hit in S, but need exclusive access)
- M → S: BusRd from another cache triggers BusWB (downgrade, writeback dirty data)
- M → I: BusRdX from another cache triggers BusWB then invalidate

### 5. MESI Protocol
- Adds **E (Exclusive Clean)** state: line is clean but only in this cache
- Eliminates unnecessary BusRdX when doing read-then-write to unshared data
- E → M upgrade: no bus transaction needed (cache knows it's the only copy)
- On BusRd, other caches signal "shared" or not → determines whether entering S or E

### 6. Snooping vs. Directory-Based Coherence
| Approach | Mechanism | Scalability |
|----------|-----------|-------------|
| **Snooping** | Broadcast to all caches; all controllers "snoop" the bus | Limited (bus bandwidth bottleneck) |
| **Directory** | Point-to-point messages; directory tracks which caches hold each line | Scalable (no broadcast) |

**Intel Core i7 example:** L3 cache serves as a centralized directory. Instead of broadcasting coherence traffic to all L2 caches, only send messages to L2s that contain the line.

### 7. False Sharing
- Two processors write to **different** addresses that map to the **same cache line**
- Cache line "ping-pongs" between caches → massive coherence traffic
- **No inherent communication** — purely artifactual
- **Fix**: pad per-thread data to cache line size (usually 64 bytes)

### 8. AMAT in Multiprocessors
```
AMAT_multiprocessor > AMAT_uniprocessor
```
Latency hierarchy (Core i7 Xeon 5500, approximate):
- L2 hit: ~10 cycles
- L3 hit, unshared: ~40 cycles
- L3 hit, shared in another core: ~65 cycles
- L3 hit, modified in another core: ~75 cycles
- Local DRAM: ~120 cycles
- Remote DRAM: ~400 cycles

---

## Knowledge Points → Corresponding C++ Files

| Knowledge Point | C++ File |
|-----------------|----------|
| MSI state machine simulation | `lecture14_part1.cpp` |
| MESI state machine + false sharing demo | `lecture14_part2.cpp` |
| Directory-based coherence simulation | `lecture14_part3.cpp` |

---

## Actionable Learning Points
1. **Coherence ≠ synchronization** — it's a hardware mechanism, not a software fix
2. **SWMR invariant** is the core guarantee of all coherence protocols
3. **MESI saves a bus transaction** on the common read-then-write-to-unshared-data pattern
4. **False sharing kills performance** — always pad per-thread data to 64 bytes
5. **Directory-based coherence scales** better than snooping by avoiding broadcasts
6. **Cache line size affects miss rate**: larger lines reduce cold/capacity misses but increase false sharing
