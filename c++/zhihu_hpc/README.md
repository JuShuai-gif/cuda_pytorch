# zhihu_hpc

Industrialized C++ performance optimization examples based on the Zhihu HPC notes.

## Scope

This project keeps the original notes under `notes/` and provides small runnable demos under `demos/`. The demos focus on measurable CPU performance topics: language construct cost, compiler optimization, memory access, SIMD, dispatch, metaprogramming, and benchmarking methodology.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON -DBUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optional ISA toggles:

```bash
cmake -S . -B build -DENABLE_AVX2=ON
```

## Engineering Notes

- Keep architecture-specific flags target-local, not global.
- Record CPU model, compiler, flags, input size, and warmup when reporting benchmark numbers.
- Use `BUILD_TESTS=ON` for smoke tests and `BUILD_DEMOS=ON` for chapter demos.
- Generated binaries should stay in `build/`, never under `src/`.
