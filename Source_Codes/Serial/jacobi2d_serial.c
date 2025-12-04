// IT23226814 - Vakesh Ranganathan
// jacobi2d_serial.c - 2D steady state heat diffusion (Laplace) via Jacobi
// Build: gcc -O2 -std=c11 jacobi2d_serial.c -o jacobi2d_serial -lm
// Run:   ./jacobi2d_serial 1024 1024 2000 1e-6
//        ^Nx   ^Ny   ^max_iters ^tolerance
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static inline size_t idx(int x, int y, int Nx) { return (size_t)y * (size_t)Nx + (size_t)x; }

int main(int argc, char **argv) {
    int Nx = (argc > 1) ? atoi(argv[1]) : 512;
    int Ny = (argc > 2) ? atoi(argv[2]) : 512;
    int max_iters = (argc > 3) ? atoi(argv[3]) : 1000;
    double tol = (argc > 4) ? atof(argv[4]) : 1e-6;

    if (Nx < 3 || Ny < 3 || max_iters <= 0 || tol <= 0.0) {
        fprintf(stderr, "Usage: %s [Nx>=3] [Ny>=3] [max_iters>0] [tol>0]\n", argv[0]);
        return 1;
    }

    size_t N = (size_t)Nx * (size_t)Ny;
    double *u  = (double*)calloc(N, sizeof(double));
    double *un = (double*)calloc(N, sizeof(double));
    if (!u || !un) { fprintf(stderr, "Allocation failed\n"); free(u); free(un); return 1; }

    // Dirichlet BCs: left wall = 1.0, other walls = 0.0
    for (int y = 0; y < Ny; ++y) {
        u[idx(0, y, Nx)] = 1.0;
        un[idx(0, y, Nx)] = 1.0;
    }

    clock_t t0 = clock();
    int iters = 0;
    double last_res = INFINITY;

    for (; iters < max_iters; ++iters) {
        double maxdiff = 0.0;

        // Jacobi interior update (independent doubly-nested loop)
        for (int y = 1; y < Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                double v = 0.25 * (u[idx(x-1,y,Nx)] + u[idx(x+1,y,Nx)] +
                                   u[idx(x,y-1,Nx)] + u[idx(x,y+1,Nx)]);
                double d = fabs(v - u[idx(x,y,Nx)]);
                if (d > maxdiff) maxdiff = d;
                un[idx(x,y,Nx)] = v;
            }
        }

        // Enforce boundary (separate loop)
        for (int y = 0; y < Ny; ++y) un[idx(0, y, Nx)] = 1.0;

        // Swap grids
        double *tmp = u; u = un; un = tmp;

        last_res = maxdiff;
        if (maxdiff < tol) break;
    }
    clock_t t1 = clock();

    // Simple checksum (helps verify & prevents dead-code elimination)
    double checksum = 0.0;
    for (size_t i = 0; i < N; ++i) checksum += u[i];

    double secs = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    printf("Nx=%d Ny=%d iters=%d time=%.3f s residual=%.3e checksum=%.6f\n",
           Nx, Ny, iters, secs, last_res, checksum);

    free(u); free(un);
    return 0;
}
