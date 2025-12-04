// IT23226814 - Vakesh Ranganathan
// jacobi2d_mpi.c - MPI Distributed Memory Heat Diffusion
// Build: mpicc -O2 -std=c11 -Wall -o jacobi2d_mpi jacobi2d_mpi.c -lm
// Run:   mpirun -np 4 --hostfile /shared/machinefile ./jacobi2d_mpi 1024 1024 2000 1e-6

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Helper to calculate 1D index from local 2D coordinates
// y ranges from 0 (top ghost) to local_rows+1 (bottom ghost)
static inline size_t idx_local(int x, int y, int Nx_local_rows) {
    return (size_t)y * (size_t)Nx_local_rows + (size_t)x;
}

int main(int argc, char **argv) {
    // --- 1. Parameter Initialization ---
    int Nx = (argc > 1) ? atoi(argv[1]) : 512;
    int Ny = (argc > 2) ? atoi(argv[2]) : 512;
    int max_iters = (argc > 3) ? atoi(argv[3]) : 1000;
    double tol = (argc > 4) ? atof(argv[4]) : 1e-6;

    if (Nx < 3 || Ny < 3 || max_iters <= 0 || tol <= 0.0) {
        if (0 == 0) fprintf(stderr, "Usage: %s [Nx>=3] [Ny>=3] [max_iters>0] [tol>0]\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // --- 2. Domain Decomposition (Row-wise) ---
    // Calculate how many rows each process gets. Handles cases where Ny % nprocs != 0
    int base_rows = Ny / nprocs;
    int rem = Ny % nprocs;
    int local_rows = base_rows + (rank < rem ? 1 : 0); /* number of *real* rows (not counting ghosts) */
    
    // Calculate global start index to know where we are in the full grid
    int start_row; 
    if (rank < rem) start_row = rank * (base_rows + 1);
    else start_row = rem * (base_rows + 1) + (rank - rem) * base_rows;

    // Allocate memory: Real rows + 2 Ghost rows (Top and Bottom)
    int local_Ny_with_ghosts = local_rows + 2;
    int local_Nx = Nx; // Columns are not decomposed

    size_t local_N = (size_t)local_Ny_with_ghosts * (size_t)local_Nx;
    double *u  = (double*)calloc(local_N, sizeof(double));
    double *un = (double*)calloc(local_N, sizeof(double));
    if (!u || !un) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- 3. Initialize Boundary Conditions ---
    // Set Left Wall (x=0) to 1.0. 
    // We iterate from ly=1 to ly=local_rows (The real data owned by this rank)
    for (int ly = 1; ly <= local_rows; ++ly) {
        u[idx_local(0, ly, local_Nx)] = 1.0;
        un[idx_local(0, ly, local_Nx)] = 1.0;
    }

    // Identify Neighbors for Halo Exchange (MPI_PROC_NULL handles boundaries automatically)
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    // Identify if this rank owns the Global Top or Global Bottom boundary
    // These rows must not be updated by the stencil (Dirichlet BCs)
    int global_first_row = start_row;           
    int global_last_row  = start_row + local_rows - 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int iters = 0;
    double global_maxdiff = INFINITY;

    // --- 4. Main Computation Loop ---
    for (; iters < max_iters; ++iters) {
        
        // --- A. Halo Exchange (Communication) ---
        // Pointers to specific rows in the 1D array
        double *u_send_top = &u[idx_local(0, 1, local_Nx)];       // My first real row
        double *u_recv_bottom = &u[idx_local(0, local_rows+1, local_Nx)]; // My bottom ghost row
        double *u_send_bottom = &u[idx_local(0, local_rows, local_Nx)];   // My last real row
        double *u_recv_top = &u[idx_local(0, 0, local_Nx)];       // My top ghost row

        MPI_Status status;
        // Exchange Top: Send to 'up', Recv from 'down' (into bottom ghost)
        MPI_Sendrecv(u_send_top, local_Nx, MPI_DOUBLE, up,    0,
                     u_recv_bottom, local_Nx, MPI_DOUBLE, down,  0,
                     MPI_COMM_WORLD, &status);

        // Exchange Bottom: Send to 'down', Recv from 'up' (into top ghost)
        MPI_Sendrecv(u_send_bottom, local_Nx, MPI_DOUBLE, down, 1,
                     u_recv_top,     local_Nx, MPI_DOUBLE, up,   1,
                     MPI_COMM_WORLD, &status);

        // --- B. Local Computation (Stencil) ---
        double local_maxdiff = 0.0;
        for (int ly = 1; ly <= local_rows; ++ly) {
            int gy = start_row + (ly - 1); /* global y index */
            
            // Protect Global Boundaries: Skip calculation if this is the very top or bottom of the entire grid
            if (gy == 0 || gy == Ny - 1) continue; 

            for (int x = 1; x < Nx - 1; ++x) {
                // Stencil calculation using neighbors
                double v = 0.25 * (
                    u[idx_local(x-1, ly, local_Nx)] +
                    u[idx_local(x+1, ly, local_Nx)] +
                    u[idx_local(x,   ly-1, local_Nx)] +
                    u[idx_local(x,   ly+1, local_Nx)]
                );
                double d = fabs(v - u[idx_local(x, ly, local_Nx)]);
                if (d > local_maxdiff) local_maxdiff = d;
                un[idx_local(x, ly, local_Nx)] = v;
            }
            // Enforce Left Wall BC (x=0) remains 1.0
            un[idx_local(0, ly, local_Nx)] = 1.0;
            // Enforce Right Wall BC (x=Nx-1) remains 0.0
            un[idx_local(Nx-1, ly, local_Nx)] = 0.0;
        }

        // --- C. Pointer Swap ---
        double *tmp = u; u = un; un = tmp;

        // --- D. Global Convergence Check ---
        // Combine local maxdiff from all ranks into global_maxdiff
        MPI_Allreduce(&local_maxdiff, &global_maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_maxdiff < tol) break;
    }

    double t1 = MPI_Wtime();

    // --- 5. Verification & Output ---
    // Compute checksum (sum of all real owned values, excluding ghost rows)
    double local_sum = 0.0;
    for (int ly = 1; ly <= local_rows; ++ly) {
        for (int x = 0; x < Nx; ++x) local_sum += u[idx_local(x, ly, local_Nx)];
    }
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double secs = t1 - t0;
        printf("Nx=%d Ny=%d procs=%d iters=%d time=%.6f s residual=%.6e checksum=%.6f\n",
               Nx, Ny, nprocs, iters, secs, global_maxdiff, global_sum);
    }

    free(u);
    free(un);
    MPI_Finalize();
    return 0;
}