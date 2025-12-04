// IT23226814 - Vakesh Ranganathan
// jacobi2d_omp.c - OpenMP Parallel implementation
// Optimized for AMD Ryzen 7 (16 vCores) Environment
// Compile: gcc -O2 -std=c11 -fopenmp jacobi2d_omp.c -o jacobi2d_omp -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Essential for OpenMP

// Helper to calculate 1D index from 2D coordinates
static inline size_t idx(int x, int y, int Nx) { return (size_t)y * (size_t)Nx + (size_t)x; }

int main(int argc, char **argv) {
    // --- 1. Configuration ---
    int Nx = (argc > 1) ? atoi(argv[1]) : 512;
    int Ny = (argc > 2) ? atoi(argv[2]) : 512;
    int max_iters = (argc > 3) ? atoi(argv[3]) : 1000;
    double tol = (argc > 4) ? atof(argv[4]) : 1e-6;

    if (Nx < 3 || Ny < 3 || max_iters <= 0 || tol <= 0.0) {
        fprintf(stderr, "Usage: %s [Nx] [Ny] [iters] [tol]\n", argv[0]);
        return 1;
    }

    size_t N = (size_t)Nx * (size_t)Ny;
    
    // --- 2. Memory Allocation ---
    double *u  = (double*)calloc(N, sizeof(double));
    double *un = (double*)calloc(N, sizeof(double)); // New grid
    if (!u || !un) { fprintf(stderr, "Allocation failed\n"); free(u); free(un); return 1; }

    // --- 3. Initialization (First Touch Policy) ---
    // We parallelize initialization to ensure memory pages are allocated 
    // near the CPU cores that will process them (NUMA optimization).
    #pragma omp parallel for
    for (int y = 0; y < Ny; ++y) {
        // Set Left Wall to 1.0 (Dirichlet BC)
        u[idx(0, y, Nx)] = 1.0;
        un[idx(0, y, Nx)] = 1.0;
    }

    // Print config
    printf("OpenMP Settings: %d Max Threads available.\n", omp_get_max_threads());
    printf("Grid: %dx%d, Max Iters: %d\n", Nx, Ny, max_iters);

    // --- 4. Main Computation Loop ---
    double t0 = omp_get_wtime(); // OpenMP Wall clock time
    int iters = 0;
    double last_res = INFINITY;

    for (; iters < max_iters; ++iters) {
        double maxdiff = 0.0;

        // PARALLEL REGION
        // collapse(2): Flattens nested loops for better work distribution on 16 cores
        // reduction(max:maxdiff): Safely finds max error across all threads
        // schedule(static): Workload is uniform, static is most efficient
        #pragma omp parallel for collapse(2) reduction(max:maxdiff) schedule(static)
        for (int y = 1; y < Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                double v = 0.25 * (u[idx(x-1,y,Nx)] + u[idx(x+1,y,Nx)] +
                                   u[idx(x,y-1,Nx)] + u[idx(x,y+1,Nx)]);
                
                double d = fabs(v - u[idx(x,y,Nx)]);
                if (d > maxdiff) maxdiff = d; // Updates local, reduces at end
                
                un[idx(x,y,Nx)] = v;
            }
        }



        // Swap pointers (very fast, done by main thread)
        double *tmp = u; u = un; un = tmp;

        last_res = maxdiff;
        if (maxdiff < tol) break;
    }

    double t1 = omp_get_wtime();

    // --- 5. Verification ---
    double checksum = 0.0;
    // Parallel reduction for checksum to keep verification fast
    #pragma omp parallel for reduction(+:checksum)
    for (size_t i = 0; i < N; ++i) checksum += u[i];

    printf("Nx=%d Ny=%d iters=%d time=%.3f s residual=%.3e checksum=%.6f\n",
           Nx, Ny, iters, t1 - t0, last_res, checksum);

    free(u); free(un);
    return 0;
}