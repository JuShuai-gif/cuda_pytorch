# Lecture 13: Domain-Specific Programming Systems and Automatic Performance Optimization

**PDF:** Lecture 13 — Domain-Specific Programming Systems and Automatic Performance Optimization  
**Course:** Stanford CS149, Fall 2025 — Parallel Computing

---

## Core Concepts Summary

### 1. The Productivity vs. Performance Tradeoff
- C++/ISPC/CUDA offer high performance but low productivity
- Domain-Specific Languages (DSLs) aim to *raise the level of abstraction* while preserving (or improving) performance
- Key insight: constrain generality to gain both productivity and performance

### 2. Halide: A DSL for Image Processing
- **Algorithm vs. Schedule separation** — the central insight of Halide
  - *Algorithm*: declarative, side-effect-free expressions defining *what* to compute
  - *Schedule*: imperative directives specifying *how* to map computation onto hardware
- Embedded in C++; functions map integer coordinates to values (e.g., pixel colors)
- Used in production at Google (HDR+ pipeline: 2000+ Halide functions), Instagram, Adobe

### 3. Image Blur Case Study: From O(N²) to Optimal
| Approach | Work per Image | Key Idea |
|----------|---------------|----------|
| Direct 2D blur | 9 × W × H | N² filter operations per pixel |
| Two-pass (separable) | 6 × W × H | 2 × N filter ops; temporary buffer overhead |
| Chunked two-pass (v1) | ~12 × W × H | 3-row tmp buffer; excess recompute |
| Chunked two-pass (v2) | ~6.4 × W × H | Cache-sized chunk; minimal recompute |
| Fused + tiled + SIMD + parallel | optimal | Halide-generated: fusion, tiling, vectorization, threading |

### 4. Halide Scheduling Primitives
| Primitive | Purpose |
|-----------|---------|
| `tile(x, y, xi, yi, tx, ty)` | 2D blocked (tiled) iteration order |
| `vectorize(xi, N)` | SIMD vectorize innermost loop |
| `parallel(y)` | Multi-thread the loop over y |
| `compute_root()` | Pre-compute entire function (like C eager evaluation) |
| `compute_at(consumer, loop)` | Compute on-demand at the specified loop level (fusion) |
| `reorder(...)` | Reorder loop dimensions |

### 5. Auto-Scheduling
- **Problem**: Very few programmers could write effective Halide schedules
- **Solution**: Model scheduling as a sequence of choices (DAG node placement, tile sizes)
- Search over schedule space with beam search
- **ML cost model**: Simple MLP estimates cost in ~10 μs per schedule
  - Trained on randomly generated Halide programs with measured runtimes
  - Auto-scheduler matches or exceeds expert human schedules

### 6. LLM-Based Performance Optimization
- **Trial-and-error via reflection**: LLM generates CUDA → profile → reflect → revise
- **Database-driven**: Store example solutions + optimization trajectories; retrieve for new problems
- **Prompt optimization**: Summarize prior experience into refined prompts
- **Combined approach**: Exhaustive search (Halide-style) + LLM agents → best results
- DSLs like Triton, CUTLASS, TileLang help LLMs work at higher abstraction

### 7. Key Takeaways
- Good representations are productive AND enable powerful system optimizations
- Halide's schedule language makes the space of optimizations *enumerable* (enabling automated search)
- The best CS149 students of the future will work in tandem with automatic optimization agents

---

## Knowledge Points → Corresponding C++ Files

| Knowledge Point | C++ File |
|-----------------|----------|
| Single-pass, two-pass, chunked blur | `lecture13_part1.cpp` |
| Two-pass chunked blur with fusion + tiling | `lecture13_part1.cpp` |
| Halide scheduling concept simulation (tile/vectorize/parallel) | `lecture13_part2.cpp` |
| compute_root vs compute_at simulation | `lecture13_part2.cpp` |
| Auto-scheduler search concept | `lecture13_part2.cpp` |

---

## Actionable Learning Points
1. **Separate algorithm from schedule** — the Halide philosophy applies beyond image processing
2. **Use separable filters** when possible (2N vs N² work)
3. **Fuse producer-consumer loops** to eliminate intermediate buffer traffic
4. **Tile for cache** — size intermediate buffers to fit in cache
5. **Auto-scheduling works** when the optimization space is well-structured
6. **LLMs + DSLs** are more reliable than LLMs generating low-level CUDA directly
