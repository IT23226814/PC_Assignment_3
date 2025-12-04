// IT23226814 - Vakesh Ranganathan
// jacobi2d_cuda.cu - CUDA implementation of Jacobi Heat Diffusion
// Build: nvcc -O2 -arch=sm_75 jacobi2d_cuda.cu -o jacobi2d_cuda
// Run:   ./jacobi2d_cuda <Nx> <Ny> <Iters> <BlockX> <BlockY>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- CUDA KERNEL ---
// Each thread calculates one cell (x,y)
__global__ void jacobi_kernel(double *u, double *un, int Nx, int Ny) {
    // Calculate global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: Only update interior points (1 to Nx-2, 1 to Ny-2)
    if (x > 0 && x < Nx - 1 && y > 0 && y < Ny - 1) {
        int idx = y * Nx + x;
        // The stencil operation
        un[idx] = 0.25 * (u[idx - 1] + u[idx + 1] + u[idx - Nx] + u[idx + Nx]);
    }
}

int main(int argc, char **argv) {
    // 1. Parameters
    int Nx = (argc > 1) ? atoi(argv[1]) : 2048;
    int Ny = (argc > 2) ? atoi(argv[2]) : 2048;
    int max_iters = (argc > 3) ? atoi(argv[3]) : 1000;
    // For CUDA benchmarking, we control block size via CLI
    int BLOCK_X = (argc > 4) ? atoi(argv[4]) : 16;
    int BLOCK_Y = (argc > 5) ? atoi(argv[5]) : 16;

    size_t N = (size_t)Nx * (size_t)Ny;
    size_t size_bytes = N * sizeof(double);

    printf("CUDA Configuration: Grid %dx%d, Block Size %dx%d\n", Nx, Ny, BLOCK_X, BLOCK_Y);

    // 2. Host Memory Allocation
    double *h_u = (double*)calloc(N, sizeof(double));
    double *h_un = (double*)calloc(N, sizeof(double));

    // Initialize Boundary Conditions (Left Wall = 1.0)
    for (int y = 0; y < Ny; ++y) {
        h_u[y * Nx] = 1.0;
        h_un[y * Nx] = 1.0; // Ensure 'un' also has BCs set
    }

    // 3. Device Memory Allocation
    double *d_u, *d_un;
    cudaCheckError(cudaMalloc((void**)&d_u, size_bytes));
    cudaCheckError(cudaMalloc((void**)&d_un, size_bytes));

    // Copy Host -> Device
    cudaCheckError(cudaMemcpy(d_u, h_u, size_bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_un, h_un, size_bytes, cudaMemcpyHostToDevice));

    // 4. Kernel Configuration
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((Nx + BLOCK_X - 1) / BLOCK_X, (Ny + BLOCK_Y - 1) / BLOCK_Y);

    // Create CUDA Events for precise timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- MAIN LOOP ---
    cudaEventRecord(start);

    for (int i = 0; i < max_iters; ++i) {
        // Launch Kernel
        jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_u, d_un, Nx, Ny);
        
        // Swap Pointers on Device (No data movement, just swap addresses)
        double *tmp = d_u; d_u = d_un; d_un = tmp;
    }
    cudaCheckError(cudaGetLastError()); // Check for launch errors

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 5. Retrieve Results
    // Copy final grid back to host (d_u is the latest because of swap)
    cudaCheckError(cudaMemcpy(h_u, d_u, size_bytes, cudaMemcpyDeviceToHost));

    // 6. Checksum Verification
    double checksum = 0.0;
    for (size_t i = 0; i < N; ++i) checksum += h_u[i];

    printf("Result: Time=%.3f s, Checksum=%.6f\n", milliseconds / 1000.0, checksum);

    // Cleanup
    cudaFree(d_u); cudaFree(d_un);
    free(h_u); free(h_un);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
