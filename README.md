# GPU-Accelerated 3D LBM Solver for Lid Driven Cavity Flow

This repository contains serial and GPU-accelerated implementations of a 3D Lattice Boltzmann Method (LBM) for the lid driven cavity flow benchmark test.

## Features
- D3Q27 lattice
- Single Relaxation Time (SRT) Bhatnagar-Gross-Krook (BGK) model
- Push streaming on serial code with Array-of-Structures (AoS) memory layout
- Pull streaming on parallel code with Structure-of-Arrays (SoA) memory layout
- Both LBM streaming/collision step and boundary conditions embedded in CUDA kernel
- Written in Fortran (host) and CUDA C++
- Only the global memory and constant memory was utilized

## Directory Structure
- serial/ - Serial Fortran implementation
- gpu/ - CUDA kernels and interfaces
- data/ - Output files (.dat)
- figures/ - Post-processed plots including velocity validation plots against Ghia et al. (1982)

## Compilation Requirements
- NVIDIA GPU (Compute Capability >= 8.0) 
- NVIDIA HPC SDK (nvfortran, nvcc)
- CUDA 12.0 or above
- gfortran for serial code

## Build Instructions
After setting-up the environment compile using:
nvcc -c lbm_kernel.cu -o lbm_kernel
nvfortran ldc_D3Q27_parallel.f90 lbm_kernel.o -cuda
