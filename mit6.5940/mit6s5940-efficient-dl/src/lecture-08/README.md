# Lecture 08: Evolutionary Search with Latency-Aware NAS

## Overview

This code accompanies **MIT 6.5940 Lecture 08: Hardware-Aware Neural Architecture Search**.
It implements an **evolutionary NAS algorithm** that simultaneously optimises for
model accuracy **and** inference latency -- a multi-objective optimisation problem.
The results are compared against a random-search baseline and visualised as a
**Pareto frontier** (accuracy vs latency scatter plot).

Building on the random-search NAS from Lecture 07, this lecture introduces:

1. **Evolutionary operators** -- population management, tournament selection,
   one-point crossover, and mutation (kernel, channel, depth).
2. **Latency awareness** -- a **simulated latency lookup table** that models
   per-layer Conv2d latency as a function of kernel size, channel count, and
   spatial resolution. This mimics the hardware latency predictors used in
   production NAS systems (ProxylessNAS, MNasNet, FBNet).
3. **Multi-objective selection** -- **NSGA-II-style non-dominated sorting**
   to maintain a diverse Pareto front without requiring manual trade-off weights.

## Search Space

| Dimension     | Choices                | Description                          |
|---------------|------------------------|--------------------------------------|
| Kernel size   | `[3, 5, 7]`            | Per-layer Conv2d kernel size         |
| Channels      | `[16, 32, 64, 128]`    | Per-layer output channel count       |
| Depth         | `[1, 2, 3, 4]`         | Number of convolutional layers       |

Total possible architectures: **22,620**.

## Evolutionary Algorithm

### Population

A population of `POPULATION_SIZE` (default 10) individuals, each an `ArchSpec`,
is initialised randomly from the search space.

### Selection: Tournament + Pareto Rank

Individuals are ranked using **non-dominated sorting** (NSGA-II style). The
first Pareto front contains all non-dominated individuals; the second front
contains those dominated only by the first front; and so on.

**Tournament selection** (`tournament_size=3`) picks the individual with
the lowest Pareto rank, breaking ties by higher accuracy.

### Crossover: One-Point

Two parents exchange layer lists at a random crossover point, producing
two children. Operates on per-layer `kernel_sizes` and `out_channels`
simultaneously.

### Mutation: Three Operators

| Operator         | Description                                          |
|------------------|------------------------------------------------------|
| Kernel mutation  | Change the kernel size of one random layer           |
| Channel mutation | Change the output channel count of one random layer  |
| Depth mutation   | Add or remove one convolutional layer                |

### Environmental Selection

After evaluating offspring, parents + offspring are combined (2× population)
and the top `POPULATION_SIZE` individuals are retained according to Pareto rank,
guaranteeing monotonic (non-degrading) population quality across generations.

## Latency Lookup Table

Real hardware-aware NAS relies on a **latency lookup table** that maps
`(kernel, in_c, out_c, H, W)` to a measured or predicted latency in
milliseconds. Here we simulate this with the `LatencyLookupTable` class:

```python
lt = LatencyLookupTable(peak_ops_per_ms=1e5, overhead_ms=0.02)
latency_ms = lt.query(kernel=3, in_c=3, out_c=16, h=32, w=32, padding=1)
model_latency = lt.estimate_model_latency(arch_spec)
```

The latency model accounts for:
- **Compute-bound** portion: MACs / peak throughput
- **Memory-bound penalty**: +30% for very large feature maps (>100K elements)
- **Kernel-size overhead**: +15% for kernel sizes >= 5 (hardware pipeline effects)
- **Fixed launch overhead**: 0.02ms per Conv2d layer

Results are cached so repeated queries for the same key are instant.

## Prerequisites

```bash
pip install torch torchvision matplotlib
```

CIFAR-10 is downloaded automatically on first run (~170 MB). Subsequent runs
use the cached dataset in `./data/`.

## Usage

```bash
cd /path/to/mit6s5940-efficient-dl
python src/lecture-08/main.py
```

The script runs entirely on CPU and produces:

1. **Search space summary** -- total architectures, sampling budget, evolution parameters.
2. **Random search baseline** -- `NUM_RANDOM_SAMPLES` architectures evaluated.
3. **Evolutionary search** -- population initialised, then evolved over `NUM_GENERATIONS`
   with tournament selection, crossover, and mutation.
4. **Comparison table** -- random vs evolutionary on accuracy, latency, Pareto frontier size.
5. **Pareto frontier plot** -- `nas_accuracy_vs_latency.png` with both search strategies
   overlaid.

Typical runtime on a modern CPU:
- **Smoke test** (5 random + pop=4, gen=3): ~2 minutes
- **Full run** (20 random + pop=10, gen=5): ~15--25 minutes

## Key Functions

| Function | Description |
|----------|-------------|
| `LatencyLookupTable.query(kernel, in_c, out_c, h, w)` | Query simulated per-layer latency (cached) |
| `LatencyLookupTable.estimate_model_latency(spec)` | Sum per-layer latencies for full architecture |
| `random_sample_architecture(...)` | Uniformly sample from the search space |
| `mutate_kernel(spec)` | Change kernel size of one random layer |
| `mutate_channels(spec)` | Change channel count of one random layer |
| `mutate_depth(spec)` | Add or remove one convolutional layer |
| `crossover(parent1, parent2)` | One-point crossover on layer lists |
| `non_dominated_sorting(results)` | NSGA-II front partitioning (accuracy ↑, latency ↓) |
| `tournament_select(population, results, fronts)` | Tournament selection by Pareto rank |
| `run_evolutionary_search(...)` | Full evolutionary algorithm main loop |
| `compute_pareto_frontier(results)` | Identify non-dominated architectures |
| `plot_pareto_frontier(random, evo, save_path)` | Scatter plot with Pareto frontier overlay |
| `NasCNN(spec)` | Build a VGG-style CNN from an ArchSpec |

## Concepts

### Multi-Objective NAS

Unlike single-objective NAS that maximises accuracy alone, **hardware-aware NAS**
considers both accuracy **and** efficiency (latency, energy, memory). This is a
**multi-objective optimisation** problem -- there is rarely a single "best"
architecture; instead there is a **Pareto frontier** of non-dominated trade-offs.

### Evolutionary NAS (Aging Evolution / Regularized Evolution)

Evolutionary algorithms work well for NAS because:

- They maintain a **population** of diverse architectures.
- **Crossover** combines good sub-structures from different architectures.
- **Mutation** explores local neighbourhoods in the search space.
- **Environmental selection** ensures the population never degrades.

This approach powers state-of-the-art NAS systems like **AmoebaNet**
(Real et al., 2019) and the **Regularized Evolution** algorithm used
in Google's Cloud AutoML.

### Non-Dominated Sorting (NSGA-II)

Architecture **A dominates B** if:
- A has **higher or equal** accuracy **and** **lower or equal** latency than B
- At least one of these is **strictly** better.

The first Pareto front contains all non-dominated architectures. The second
front contains architectures dominated only by the first front, etc. This
ranking provides a principled way to select individuals in a multi-objective
setting without arbitrary scalarisation weights.

### Latency Lookup Table vs On-Device Measurement

In production NAS (ProxylessNAS, MNasNet, FBNet):
- Latency is **measured on the target device** (e.g., Pixel phone, Raspberry Pi)
  for every candidate (kernel, C_in, C_out, H, W) combination.
- Results are stored in a **lookup table**.
- During search, total model latency is estimated by **summing** per-layer
  latencies (additive assumption).

Our simulated table follows the same pattern but uses a parametric formula
instead of real hardware measurements.

### Evolutionary vs Random Search

| Aspect | Random Search | Evolutionary Search |
|--------|--------------|---------------------|
| Exploration | Uniform sampling | Population + mutation |
| Exploitation | None | Crossover combines good traits |
| Convergence | N/A | Improves over generations |
| Pareto diversity | Depends on sample size | Maintained via non-dominated sorting |
| Computational cost | N × train_cost | (pop_size × (1 + gen_count)) × train_cost |

While random search is a surprisingly strong baseline (Li & Talwalkar, 2019),
evolutionary search typically finds **better Pareto-optimal architectures**
and does so more **sample-efficiently** by building on previous discoveries.

## References

- Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. "Regularized Evolution for
  Image Classifier Architecture Search." AAAI 2019.
- Deb, K., et al. "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
  IEEE Transactions on Evolutionary Computation, 2002.
- Cai, H., Zhu, L., & Han, S. "ProxylessNAS: Direct Neural Architecture Search
  on Target Task and Hardware." ICLR 2019.
- Tan, M., et al. "MNasNet: Platform-Aware Neural Architecture Search for Mobile."
  CVPR 2019.
- Wu, B., et al. "FBNet: Hardware-Aware Efficient ConvNet Design via
  Differentiable Neural Architecture Search." CVPR 2019.
- Li, L., & Talwalkar, A. "Random Search and Reproducibility for Neural
  Architecture Search." UAI 2019.
- MIT 6.5940 Lecture 08: Hardware-Aware Neural Architecture Search --
  https://hanlab.mit.edu/courses/2025-fall-65940
