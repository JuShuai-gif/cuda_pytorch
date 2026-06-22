#ifndef CHAPTER_18_CODE_BENCHMARK_H_
#define CHAPTER_18_CODE_BENCHMARK_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define a fixed seed for reproducibility
#define RANDOM_SEED 42

// CUDA error checking macro
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

// Function to clear L2 cache between benchmark runs
void clear_l2();

// Function to generate random atoms
void generate_atoms(float* atoms, int numatoms, float max_x, float max_y, float max_z);

// Function to compare two energy grids
int grids_allclose(const float* grid1, const float* grid2, dim3 grid_dimensions, float rtol, float atol, int verbose);

// Energy calculation function type definitions
typedef void (*EnergyCalculationFunc)(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms,
                                      int numatoms);

// Benchmark function for energy calculations
float benchmark_energy_func(EnergyCalculationFunc func, float* energygrid, dim3 grid, float gridspacing, float z,
                            const float* atoms, int numatoms, int warmup, int reps, const char* name);

#endif  // CHAPTER_18_CODE_BENCHMARK_H_
