#include <math.h>
#include <stdio.h>

#include "cenergy.h"

// CUDA kernel using constant memory
__constant__ float atoms[CHUNK_SIZE * 4];  // Each atom has x,y,z,charge

//==============================
// KERNEL IMPLEMENTATIONS
//==============================

__global__ void cenergyScatterKernel(float* energygrid, dim3 grid, float gridspacing, float z, int atoms_in_chunk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx * 4;

    // Only process valid atoms in the chunk
    if (idx < atoms_in_chunk) {
        float dz = z - atoms[n + 2];  // all grid points in a slice have the same
        float dz2 = dz * dz;          // z value
        // starting position of the slice in the energy grid
        int grid_slice_offset = (grid.x * grid.y * z) / gridspacing;
        float charge = atoms[n + 3];
        for (int j = 0; j < grid.y; j++) {
            float y = gridspacing * (float)j;
            float dy = y - atoms[n + 1];  // all grid points in a row have the same
            float dy2 = dy * dy;          // y value
            // starting position of the row in the energy grid
            int grid_row_offset = grid_slice_offset + grid.x * j;
            for (int i = 0; i < grid.x; i++) {
                float x = gridspacing * (float)i;
                float dx = x - atoms[n];
                atomicAdd(&energygrid[grid_row_offset + i], charge / sqrtf(dx * dx + dy2 + dz2));
            }
        }
    }
}

__global__ void cenergyGatherKernel(float* energygrid, dim3 grid_dim, float gridspacing, float z, int atoms_in_chunk,
                                    int chunk_start) {
    // Get 2D grid coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if this thread is within grid bounds
    if (i < grid_dim.x && j < grid_dim.y) {
        // Calculate actual grid point coordinates
        float x = gridspacing * (float)i;
        float y = gridspacing * (float)j;

        // Calculate z-index for the plane
        int k = (int)(z / gridspacing);

        // Grid point index in the energy grid
        int grid_point_idx = grid_dim.x * grid_dim.y * k + grid_dim.x * j + i;

        // For the first chunk, initialize the energy value
        if (chunk_start == 0) {
            energygrid[grid_point_idx] = 0.0f;
        }

        // Calculate energy contribution from all atoms in this chunk
        float energy = 0.0f;

        // Loop through all atoms in the current chunk
        for (int n = 0; n < atoms_in_chunk; n++) {
            int atom_idx = n * 4;  // Each atom has 4 values (x,y,z,charge)

            float dx = x - atoms[atom_idx];
            float dy = y - atoms[atom_idx + 1];
            float dz = z - atoms[atom_idx + 2];
            float charge = atoms[atom_idx + 3];

            energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
        }

        // Add this chunk's contribution to the energy grid (no atomic needed)
        energygrid[grid_point_idx] += energy;
    }
}

// Thread coarsening kernel - each thread processes COARSEN_FACTOR grid points
__global__ void cenergyCoarsenKernel(float* energygrid, dim3 grid, float gridspacing, float z, int atoms_in_chunk) {
    int base_i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x * COARSEN_FACTOR;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= grid.y) {
        return;
    }

    int k = z / gridspacing;
    float y = gridspacing * (float)j;

    // Array to store energy values for COARSEN_FACTOR points
    float energies[COARSEN_FACTOR];

    // Initialize all energies to 0
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        energies[c] = 0.0f;
    }

    // Calculate for all atoms
    for (int n = 0; n < atoms_in_chunk * 4; n += 4) {
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzq = dy * dy + dz * dz;  // Precalculate dy² + dz²
        float charge = atoms[n + 3];

        // Calculate energy for each of the COARSEN_FACTOR points
        for (int c = 0; c < COARSEN_FACTOR; c++) {
            int i = base_i + c;
            if (i < grid.x) {  // Check bounds
                float x = gridspacing * (float)i;
                float dx = x - atoms[n];
                energies[c] += charge / sqrtf(dx * dx + dysqdzq);
            }
        }
    }

    // Write results
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        int i = base_i + c;
        if (i < grid.x) {  // Check bounds again
            energygrid[grid.x * grid.y * k + grid.x * j + i] += energies[c];
        }
    }
}

__global__ void cenergyCoalescingKernel(float* energygrid, dim3 grid, float gridspacing, float z, int atoms_in_chunk) {
    int base_i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= grid.y) {
        return;
    }

    int k = z / gridspacing;
    float y = gridspacing * (float)j;

    // Use dynamic array to store the energy values
    float energies[COARSEN_FACTOR];

    // Initialize all energies to 0
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        energies[c] = 0.0f;
    }

    // Calculate potential contribution from all atoms
    for (int n = 0; n < atoms_in_chunk * 4; n += 4) {
        float dx_base = gridspacing * (float)base_i - atoms[n];
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzq = dy * dy + dz * dz;
        float charge = atoms[n + 3];

        // Calculate energy for each point this thread handles
        for (int c = 0; c < COARSEN_FACTOR; c++) {
            float dx = dx_base + c * blockDim.x * gridspacing;
            energies[c] += charge / sqrtf(dx * dx + dysqdzq);
        }
    }

    // Write results with bounds checking for coalesced memory access
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        int idx = base_i + c * blockDim.x;
        if (idx < grid.x) {
            energygrid[grid.x * grid.y * k + grid.x * j + idx] += energies[c];
        }
    }
}

//==============================
// CPU IMPLEMENTATIONS
//==============================

// Chunked GPU implementation with scatter approach
void cenergyParallelScatter(float* host_energygrid, dim3 grid, float gridspacing, float z, const float* host_atoms,
                            int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid.x * grid.y * grid.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk with the correct number of atoms
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (atoms_in_chunk + threadsPerBlock - 1) / threadsPerBlock;

        // Pass atoms_in_chunk to the kernel to handle the non-full last chunk
        cenergyScatterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energygrid, grid, gridspacing, z, atoms_in_chunk);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel for chunk %d (error code %s)!\n", chunk, cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for this chunk to complete
        cudaDeviceSynchronize();
    }

    // Copy the accumulated results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

// Chunked GPU implementation with gather approach
void cenergyParallelGather(float* host_energygrid, dim3 grid_dim, float gridspacing, float z, const float* host_atoms,
                           int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Define 2D thread block and grid
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((grid_dim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (grid_dim.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // For each chunk of atoms
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk
        cenergyGatherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk,
                                                                start_atom);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel for chunk %d (error code %s)!\n", chunk, cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for this chunk to complete
        cudaDeviceSynchronize();
    }

    // Copy the accumulated results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

// Chunked GPU implementation with thread coarsening approach
void cenergyParallelCoarsen(float* host_energygrid, dim3 grid_dim, float gridspacing, float z, const float* host_atoms,
                            int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Set up execution configuration with coarsening factor
    dim3 blockDim(16, 16);
    dim3 gridDim(((grid_dim.x + COARSEN_FACTOR - 1) / COARSEN_FACTOR + blockDim.x - 1) / blockDim.x,
                 (grid_dim.y + blockDim.y - 1) / blockDim.y);

    // For each chunk of atoms
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk
        cenergyCoarsenKernel<<<gridDim, blockDim>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error synchronizing: %s\n", cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }
    }

    // Copy results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

void cenergyParallelCoalescing(float* host_energygrid, dim3 grid_dim, float gridspacing, float z,
                               const float* host_atoms, int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Set up execution configuration
    dim3 blockDim(16, 16);  // 256 threads per block

    // Calculate grid dimensions based on coarsening factor
    // Each block processes blockDim.x * COARSEN_FACTOR grid points in x dimension
    dim3 gridDim((grid_dim.x + blockDim.x * COARSEN_FACTOR - 1) / (blockDim.x * COARSEN_FACTOR),
                 (grid_dim.y + blockDim.y - 1) / blockDim.y);

    // For each chunk of atoms
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk
        cenergyCoalescingKernel<<<gridDim, blockDim>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error synchronizing: %s\n", cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }
    }

    // Copy results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

void cenergySequential(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms, int numatoms) {
    int atomarrdim = numatoms * 4;  // x,y,z, and charge info for each atom
    for (int j = 0; j < grid.y; j++) {
        // calculate y coordinate of the grid point based on j
        float y = gridspacing * (float)j;
        for (int i = 0; i < grid.x; i++) {
            // calculate x coordinate based on i
            float x = gridspacing * (float)i;
            float energy = 0.0f;
            for (int n = 0; n < atomarrdim; n += 4) {
                float dx = x - atoms[n];
                float dy = y - atoms[n + 1];
                float dz = z - atoms[n + 2];
                energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
            }
            // Fixed indexing - use integer z_index
            int z_index = (int)(z / gridspacing);
            energygrid[grid.x * grid.y * z_index + grid.x * j + i] = energy;
        }
    }
}

void cenergySequentialOptimized(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms,
                                int numatoms) {
    int atomarrdim = numatoms * 4;  // x,y,z, and charge info for each atom
    // starting point of the slice in the energy grid
    int grid_slice_offset = (grid.x * grid.y * z) / gridspacing;
    // calculate potential contribution of each atom
    for (int n = 0; n < atomarrdim; n += 4) {
        float dz = z - atoms[n + 2];  // all grid points in a slice have the same
        float dz2 = dz * dz;          // z value, no recalculation in inner loops
        float charge = atoms[n + 3];
        for (int j = 0; j < grid.y; j++) {
            float y = gridspacing * (float)j;
            float dy = y - atoms[n + 1];  // all grid points in a row have the same
            float dy2 = dy * dy;          // y value
            int grid_row_offset = grid_slice_offset + grid.x * j;
            for (int i = 0; i < grid.x; i++) {
                float x = gridspacing * (float)i;
                float dx = x - atoms[n];
                energygrid[grid_row_offset + i] += charge / sqrtf(dx * dx + dy2 + dz2);
            }
        }
    }
}
