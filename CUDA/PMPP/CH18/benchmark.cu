#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "benchmark.h"
#include "cenergy.h"

// Function to clear L2 cache between benchmark runs
void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;

    if (!gpu_scratch_l2_clear) {
        int dev;
        gpuErrchk(cudaGetDevice(&dev));
        gpuErrchk(cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, dev));
        l2_clear_size *= 2;  // Use a buffer twice the L2 size
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }

    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

// Function to generate k synthetic atoms
void generate_atoms(float* atoms, int numatoms, float max_x, float max_y, float max_z) {
    // Seed random number generator with defined seed
    srand(RANDOM_SEED);

    for (int i = 0; i < numatoms; i++) {
        // Generate random positions within the grid boundaries
        atoms[i * 4] = (float)rand() / RAND_MAX * max_x;      // x position
        atoms[i * 4 + 1] = (float)rand() / RAND_MAX * max_y;  // y position
        atoms[i * 4 + 2] = (float)rand() / RAND_MAX * max_z;  // z position

        // Generate random charge between -2.0 and 2.0
        atoms[i * 4 + 3] = ((float)rand() / RAND_MAX * 4.0) - 2.0;  // charge
    }
}

// Uses the formula: |a - b| <= atol + rtol * |b|
int grids_allclose(const float* grid1, const float* grid2, dim3 grid_dimensions, float rtol, float atol, int verbose) {
    // Calculate the total number of grid points
    int total_grid_points = grid_dimensions.x * grid_dimensions.y * grid_dimensions.z;

    int mismatches = 0;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    // Compare each grid point
    for (int i = 0; i < total_grid_points; i++) {
        // Calculate absolute difference
        float abs_diff = fabsf(grid1[i] - grid2[i]);
        // Calculate tolerance threshold
        float threshold = atol + rtol * fabsf(grid2[i]);

        if (abs_diff > threshold) {
            mismatches++;

            // Track the location of maximum difference
            if (abs_diff > max_diff) {
                max_diff = abs_diff;
                max_diff_idx = i;
            }

            // If verbose and this is one of the first few mismatches, print details
            if (verbose && mismatches <= 10) {
                int z = i / (grid_dimensions.x * grid_dimensions.y);
                int remainder = i % (grid_dimensions.x * grid_dimensions.y);
                int y = remainder / grid_dimensions.x;
                int x = remainder % grid_dimensions.x;

                printf("Mismatch at grid point (%d, %d, %d): %f vs %f (diff: %f, threshold: %f)\n", x, y, z, grid1[i],
                       grid2[i], abs_diff, threshold);
            }
        }
    }

    // Print summary
    if (mismatches > 0) {
        printf("Found %d mismatches out of %d points (%.2f%%)\n", mismatches, total_grid_points,
               (float)mismatches * 100.0f / total_grid_points);

        // Print info about maximum difference
        if (max_diff_idx >= 0) {
            int z = max_diff_idx / (grid_dimensions.x * grid_dimensions.y);
            int remainder = max_diff_idx % (grid_dimensions.x * grid_dimensions.y);
            int y = remainder / grid_dimensions.x;
            int x = remainder % grid_dimensions.x;

            printf("Maximum difference: %f at point (%d, %d, %d)\n", max_diff, x, y, z);
        }

        return 0;  // Grids don't match
    }

    printf("All %d grid points match within tolerances (rtol=%e, atol=%e)\n", total_grid_points, rtol, atol);
    return 1;  // Grids match within tolerance
}

// Generic benchmarking function for energy calculation routines
float benchmark_energy_func(EnergyCalculationFunc func, float* energygrid, dim3 grid, float gridspacing, float z,
                            const float* atoms, int numatoms, int warmup, int reps, const char* name) {
    printf("Benchmarking %s...\n", name);

    // For CPU functions
    if (name[0] == 'S') {  // Sequential implementation
        // Warmup runs
        for (int i = 0; i < warmup; i++) {
            func(energygrid, grid, gridspacing, z, atoms, numatoms);
        }

        // Timing runs
        clock_t start_time, end_time;
        double total_time_ms = 0.0;

        for (int i = 0; i < reps; i++) {
            start_time = clock();
            func(energygrid, grid, gridspacing, z, atoms, numatoms);
            end_time = clock();

            double elapsed_ms = 1000.0 * (double)(end_time - start_time) / CLOCKS_PER_SEC;
            total_time_ms += elapsed_ms;
        }

        return total_time_ms / reps;
    } else {  // For GPU functions
        // Warmup runs
        for (int i = 0; i < warmup; i++) {
            func(energygrid, grid, gridspacing, z, atoms, numatoms);
            gpuErrchk(cudaDeviceSynchronize());
        }

        // Timing runs
        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));

        float total_time_ms = 0.0f;

        for (int i = 0; i < reps; i++) {
            // Clear L2 cache to reduce inter-run effects
            clear_l2();

            gpuErrchk(cudaEventRecord(start, 0));
            func(energygrid, grid, gridspacing, z, atoms, numatoms);
            gpuErrchk(cudaEventRecord(stop, 0));
            gpuErrchk(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
            total_time_ms += elapsed_ms;
        }

        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));

        return total_time_ms / reps;
    }
}

int main() {
    // Grid configuration
    dim3 grid(100, 100, 50);
    float gridspacing = 0.5;  // 0.5 Ångström between points
    float z = 10.0;           // Computing energy for z=10 plane

    // Benchmarking parameters
    int warmup_runs = 3;
    int timing_runs = 10;

    // Number of atoms to generate
    int numatoms = 10000;  // Reduced for faster testing - adjust as needed

    printf("Energy Grid Benchmark\n");
    printf("=====================\n");
    printf("Grid size: %d x %d x %d\n", grid.x, grid.y, grid.z);
    printf("Grid spacing: %.1f Å\n", gridspacing);
    printf("Number of atoms: %d\n", numatoms);
    printf("Benchmarking for z=%.1f plane\n\n", z);

    // Allocate memory for atoms
    float* atoms = (float*)malloc(numatoms * 4 * sizeof(float));
    if (!atoms) {
        printf("Memory allocation for atoms failed\n");
        return 1;
    }

    // Generate random atoms
    float max_x = grid.x * gridspacing;
    float max_y = grid.y * gridspacing;
    float max_z = grid.z * gridspacing;
    generate_atoms(atoms, numatoms, max_x, max_y, max_z);

    // Print first few atoms for verification
    printf("First 3 atoms (using seed %d):\n", RANDOM_SEED);
    for (int i = 0; i < 3 && i < numatoms; i++) {
        printf("Atom %d: Position (%.2f, %.2f, %.2f), Charge: %.2f\n", i, atoms[i * 4], atoms[i * 4 + 1],
               atoms[i * 4 + 2], atoms[i * 4 + 3]);
    }
    printf("\n");

    // Allocate energy grids for each approach
    size_t grid_size = grid.x * grid.y * grid.z;
    float* energygrid_seq = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_gpu_scatter = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_gpu_gather = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_gpu_coarsen = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_gpu_coalescing = (float*)calloc(grid_size, sizeof(float));

    if (!energygrid_seq || !energygrid_gpu_scatter || !energygrid_gpu_gather || !energygrid_gpu_coarsen ||
        !energygrid_gpu_coalescing) {
        printf("Memory allocation for energy grids failed\n");
        free(atoms);
        if (energygrid_seq) {
            free(energygrid_seq);
        }
        if (energygrid_gpu_scatter) {
            free(energygrid_gpu_scatter);
        }
        if (energygrid_gpu_gather) {
            free(energygrid_gpu_gather);
        }
        if (energygrid_gpu_coarsen) {
            free(energygrid_gpu_coarsen);
        }
        if (energygrid_gpu_coalescing) {
            free(energygrid_gpu_coalescing);
        }
        return 1;
    }

    // Run benchmarks with the new benchmark function
    float time_seq = benchmark_energy_func(cenergySequential, energygrid_seq, grid, gridspacing, z, atoms, numatoms,
                                           warmup_runs, timing_runs, "Sequential");

    float time_scatter = benchmark_energy_func(cenergyParallelScatter, energygrid_gpu_scatter, grid, gridspacing, z,
                                               atoms, numatoms, warmup_runs, timing_runs, "GPU Scatter");

    float time_gather = benchmark_energy_func(cenergyParallelGather, energygrid_gpu_gather, grid, gridspacing, z, atoms,
                                              numatoms, warmup_runs, timing_runs, "GPU Gather");

    float time_coarsen = benchmark_energy_func(cenergyParallelCoarsen, energygrid_gpu_coarsen, grid, gridspacing, z,
                                               atoms, numatoms, warmup_runs, timing_runs, "GPU Thread Coarsening");

    float time_coalescing =
        benchmark_energy_func(cenergyParallelCoalescing, energygrid_gpu_coalescing, grid, gridspacing, z, atoms,
                              numatoms, warmup_runs, timing_runs, "GPU Memory Coalescing");

    // Compare results
    printf("\nPerformance Results:\n");
    printf("=====================\n");
    printf("Sequential Approach: %.3f ms\n", time_seq);
    printf("GPU Scatter Implementation: %.3f ms\n", time_scatter);
    printf("GPU Gather Implementation: %.3f ms\n", time_gather);
    printf("GPU Thread Coarsening Implementation: %.3f ms\n", time_coarsen);
    printf("GPU Memory Coalescing Implementation: %.3f ms\n", time_coalescing);

    printf("\nSpeedup Analysis:\n");
    printf("=================\n");
    printf("GPU Scatter vs Sequential: %.2fx\n", time_seq / time_scatter);
    printf("GPU Gather vs Sequential: %.2fx\n", time_seq / time_gather);
    printf("GPU Thread Coarsening vs Sequential: %.2fx\n", time_seq / time_coarsen);
    printf("GPU Memory Coalescing vs Sequential: %.2fx\n", time_seq / time_coalescing);

    // Print sample results from each approach for verification
    int z_index = (int)(z / gridspacing);
    int sample_idx = grid.x * grid.y * z_index + grid.x * 50 + 50;
    printf("\nEnergy at grid point (50,50,%d):\n", z_index);
    printf("Sequential: %f\n", energygrid_seq[sample_idx]);
    printf("GPU Scatter: %f\n", energygrid_gpu_scatter[sample_idx]);
    printf("GPU Gather: %f\n", energygrid_gpu_gather[sample_idx]);
    printf("GPU Thread Coarsening: %f\n", energygrid_gpu_coarsen[sample_idx]);
    printf("GPU Memory Coalescing: %f\n", energygrid_gpu_coalescing[sample_idx]);

    // Verify results match within tolerance
    printf("\nResult Validation:\n");
    printf("=================\n");
    printf("Comparing Sequential vs. GPU Memory Coalescing:\n");
    grids_allclose(energygrid_seq, energygrid_gpu_coalescing, grid, 1e-2, 1e-3, 0);

    printf("\nComparing GPU Thread Coarsening vs. GPU Memory Coalescing:\n");
    grids_allclose(energygrid_gpu_coarsen, energygrid_gpu_coalescing, grid, 1e-3, 1e-4, 0);

    // Clean up
    free(energygrid_seq);
    free(energygrid_gpu_scatter);
    free(energygrid_gpu_gather);
    free(energygrid_gpu_coarsen);
    free(energygrid_gpu_coalescing);
    free(atoms);

    return 0;
}
