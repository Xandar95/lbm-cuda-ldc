# GPU-Accelerated 3D LBM Solver for Lid Driven Cavity Flow

This repository contains serial and GPU-accelerated implementations of a 3D Lattice Boltzmann Method (LBM) solver for the lid driven cavity flow benchmark test.

## Features
- D3Q27 lattice
- Single Relaxation Time (SRT) Bhatnagar-Gross-Krook (BGK) model
- Push streaming on serial code with Array-of-Structures (AoS) memory layout
- Pull streaming on parallel code with Structure-of-Arrays (SoA) memory layout
- Both LBM streaming/collision step and boundary conditions embedded in CUDA kernel
- Written in Fortran (host) and CUDA C++
- Only global memory and constant memory utilized
- Convergence criterion to break the time loop velocity change between 1000 time steps reach 10^-6

## Directory Structure
- serial/ - Serial Fortran implementation
- gpu/ - CUDA kernels and interfaces
- data/ - Sample output file (.dat)
- figures/ - Post-processed plots including velocity validation plots against Ghia et al. (1982)
- post-processing/ - Post-processing script and validation data from Ghia et al. (1982)

## Compilation Requirements
- NVIDIA GPU (Compute Capability >= 6.0 for the atomicAdd function used for the convergence criterion) 
- NVIDIA HPC SDK (nvfortran, nvcc)
- CUDA 12.0 or above
- gfortran for serial code

## Build Instructions
After setting-up the environment, compile using:
- nvcc -c lbm_kernel.cu -o lbm_kernel.o -ccbin gcc-12 -arch=sm_60 
- nvfortran ldc_D3Q27_parallel.f90 lbm_kernel.o -cuda
- gfortran ldc_D3Q27_serial.f90 -o ldc_D3Q27_serial.exe
- Change the lattice domain sizes as desired
- Change the memory access patterns in the CUDA kernel as necessary (Note: 1D thread blocks results in better memory coalescence for this algorithm)
- The solver outputs velocity (u, v, w components), pressure, and streamfunction for XY plane

## Post-Processing
Post-processing of the results done using MATLAB and includes:
- Solver data loading
- 3D vector plot of the cavity
- Velocity contour and streamline plotting
- Validation plots against reference data
- Pressure contour and pressure isosurface plotting
! Post-processing of the serial computed data requires a permutation of the data matrices
