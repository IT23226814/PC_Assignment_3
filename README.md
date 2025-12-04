# Parallel Implementation of 2D Jacobi Heat Diffusion using OpenMP, MPI, and CUDA

## Overview
This repository contains implementations of the 2D Jacobi iterative method using different parallel programming paradigms: Serial, OpenMP, MPI, and CUDA.

## Project Structure
```
Source_Codes/
├── Serial/
│   └── jacobi2d_serial.c
├── OpenMP/
│   └── jacobi2d_omp.c
├── MPI/
│   └── jacobi2d_mpi.c
└── CUDA/
    ├── jacobi2d_cuda.cu
    └── jacobi2d_cuda_notebook.ipynb
```

## Prerequisites

### OpenMP
- GCC compiler with OpenMP support
- Windows: MinGW-w64 or MSYS2
- Linux: GCC (usually pre-installed)

### MPI
- MPI implementation (MPICH or OpenMPI)
- Windows: Microsoft MPI or MPICH
- Linux: OpenMPI or MPICH

### CUDA
- NVIDIA CUDA Toolkit (version 10.0 or higher)
- NVIDIA GPU with CUDA support
- nvcc compiler (comes with CUDA Toolkit)

## Compilation and Execution Instructions

### Serial Implementation

**Compilation:**
```bash
gcc jacobi2d_serial.c -o jacobi2d_serial -lm
```

**Execution:**
```bash
./jacobi2d_serial
```

### OpenMP Implementation

**Compilation:**
```bash
gcc -fopenmp jacobi2d_omp.c -o jacobi2d_omp -lm
```

**Execution:**
```bash
# Set number of threads (optional, default uses all available cores)
set OMP_NUM_THREADS=4

# Run the program
./jacobi2d_omp
```

**Note:** On Linux/Mac, use `export` instead of `set`:
```bash
export OMP_NUM_THREADS=4
./jacobi2d_omp
```

### MPI Implementation

**Compilation:**
```bash
mpicc jacobi2d_mpi.c -o jacobi2d_mpi -lm
```

**Execution:**
```bash
# Run with 4 processes
mpiexec -n 4 jacobi2d_mpi
```

**Alternative execution command:**
```bash
mpirun -np 4 jacobi2d_mpi
```

**Note:** The number after `-n` or `-np` specifies the number of MPI processes to use.

### CUDA Implementation

**Compilation:**
```bash
nvcc jacobi2d_cuda.cu -o jacobi2d_cuda
```

**Execution:**
```bash
./jacobi2d_cuda
```

**For Jupyter Notebook:**
Open `jacobi2d_cuda_notebook.ipynb` in Jupyter Notebook or Google Colab with GPU support.

## Running Instructions by Directory

### Navigate to Source Code Directory
```bash
cd Source_Codes
```

### For OpenMP:
```bash
cd OpenMP
gcc -fopenmp jacobi2d_omp.c -o jacobi2d_omp -lm
set OMP_NUM_THREADS=4
./jacobi2d_omp
```

### For MPI:
```bash
cd MPI
mpicc jacobi2d_mpi.c -o jacobi2d_mpi -lm
mpiexec -n 4 jacobi2d_mpi
```

### For CUDA:
```bash
cd CUDA
nvcc jacobi2d_cuda.cu -o jacobi2d_cuda
./jacobi2d_cuda
```

## Performance Testing

To test with different configurations:

**OpenMP - Vary thread count:**
```bash
set OMP_NUM_THREADS=1
./jacobi2d_omp

set OMP_NUM_THREADS=2
./jacobi2d_omp

set OMP_NUM_THREADS=4
./jacobi2d_omp
```

**MPI - Vary process count:**
```bash
mpiexec -n 1 jacobi2d_mpi
mpiexec -n 2 jacobi2d_mpi
mpiexec -n 4 jacobi2d_mpi
mpiexec -n 8 jacobi2d_mpi
```

## Troubleshooting

### OpenMP
- **Error:** `gcc: error: unrecognized command line option '-fopenmp'`
  - **Solution:** Install GCC with OpenMP support or use `-Xpreprocessor -fopenmp` on macOS

### MPI
- **Error:** `mpicc: command not found`
  - **Solution:** Install MPI (e.g., `sudo apt-get install mpich` on Ubuntu)
  
### CUDA
- **Error:** `nvcc: command not found`
  - **Solution:** Install NVIDIA CUDA Toolkit and add it to PATH
- **Error:** `no CUDA-capable device detected`
  - **Solution:** Ensure you have an NVIDIA GPU and updated drivers

## Expected Output

All implementations should produce similar output showing:
- Grid size
- Number of iterations
- Convergence status
- Execution time
- Final grid values or statistics

## Screenshots

Performance comparison screenshots are available in the `Screenshots/` directory.