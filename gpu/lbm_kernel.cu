// LBM Kernel Implementation in CUDA
#include <cuda_runtime.h>
#include <cstdio>

typedef double real_t; // Define real_t as double to match Fortran's real(8)

// D3Q27 model parameters stored in constant memory
// D3Q27 discrete velocities 
__constant__ int d_cx[27];
__constant__ int d_cy[27];
__constant__ int d_cz[27];
__constant__ int d_opp[27]; // Opposite directions
__constant__ double d_weights[27]; // D3Q27 weights

// Persistent device memory allocation for distribution functions
static real_t *d_f_in = nullptr;
static real_t *d_f_out = nullptr;

// ----- Bulk kernel for LBM step (Pull scheme + SOA) -----
__global__ void lbm_kernel_soa(const real_t* __restrict__ f_in,
                               real_t* __restrict__ f_out,
                               int nx, int ny, int nz,
                               real_t omega)
{
    // define global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check

    int N = nx * ny * nz; // total number of lattice points
    int curr_idx = i + j * nx + k * nx * ny; // current index

    // register variables for local calculations
    real_t f_streamed[27];
    real_t rho = 0.0;
    real_t u = 0.0, v = 0.0, w = 0.0; 

    // Pull streaming & moment calculation
    #pragma unroll
    for (int l = 0; l < 27; l++){
        int ip = i - d_cx[l];
        int jp = j - d_cy[l];
        int kp = k - d_cz[l];

        // apply periodic BC for simplicity
        // Actual BC will overwrite periodicity later
        if (ip >= 0 && ip < nx && jp >= 0 && jp < ny && kp >= 0 && kp < nz) {
            int neigh_idx = ip + jp * nx + kp * nx * ny;
            f_streamed[l] = f_in[l * N + neigh_idx];
        } else {
            f_streamed[l] = f_in[l * N + curr_idx]; // boundary fallback
        }

        // accumulate moments
        rho += f_streamed[l];
        u += f_streamed[l] * d_cx[l];
        v += f_streamed[l] * d_cy[l];
        w += f_streamed[l] * d_cz[l];
        }

        if (rho > 1e-9) {u /= rho; v /= rho; w /= rho;} // avoid division by zero
        else {u = 0.0; v = 0.0; w = 0.0;}

        real_t u2 = u * u + v * v + w * w;

    // Collision step
    #pragma unroll
    for (int l = 0; l < 27; l++){
        real_t cu = d_cx[l] * u + d_cy[l] * v + d_cz[l] * w;
        real_t feq = d_weights[l] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        // write post collision distribution
        f_out[l * N + curr_idx] = (1.0 - omega) * f_streamed[l] + omega * feq;
    }
}

// ----- Boundary Condition Kernel -----
__global__ void apply_bc_kernel(real_t* __restrict__ f,
                                int nx, int ny, int nz,
                                real_t u_lid)
{
    // define global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check

    int N = nx * ny * nz;
    int idx = i + j * nx + k * nx * ny; 

    // No slip walls (bounce-back)
    // West wall (i=0)
    if (i == 0) {
        for (int l = 0; l < 27; l++) {
            if (d_cx[l] == 1) 
                f[l * N + idx] = f[d_opp[l] * N + idx];
        }
    }

    // East wall (i=nx-1)
    if (i == nx -1) {
        for (int l = 0; l < 27; l++) {
            if (d_cx[l] == -1)
                f[l * N + idx] = f[d_opp[l] * N + idx];
        }
    }

    // South wall (k=0)
    if (k == 0) {
        for (int l = 0; l < 27; l++) {
            if (d_cz[l] == 1)
                f[l * N + idx] = f[d_opp[l] * N + idx];
        }
    }

    // North wall (k=nz-1)
    if (k == nz - 1) {
        for (int l = 0; l < 27; l++) {
            if (d_cz[l] == -1)
                f[l * N + idx] = f[d_opp[l] * N + idx];
        }
    }

    // Bottom wall (j=0)
    if (j == 0) {
        for (int l = 0; l < 27; l++) {
            if (d_cy[l] == 1)
                f[l * N + idx] = f[d_opp[l] * N + idx];
        }
    }

    // Top wall (j=ny-1) - moving lid
    if (j == ny - 1 &&  i > 0 && i < nx - 1 && k > 0 && k < nz -1) { // exclude corners
        // calculate local density (at top wall)
        real_t rho_wall = 0.0;
        for (int l =0; l < 27; l++) rho_wall += f[l * N + idx]; // replace with Zou/He if needed
        
        for (int l = 0; l < 27; l++) {
            if (d_cy[l] == -1) 
                f[l* N + idx] = f[d_opp[l] * N + idx] + 6.0 * d_weights[l] * rho_wall * d_cx[l] * u_lid;
        }
    }
}

// ----- Host functions to launch kernels -----
extern "C" {
// function to initialize GPU with parameters
void lbm_init_gpu(int nx, int ny, int nz,
                  int* h_cx, int* h_cy, int* h_cz, int* h_opp, double* h_weights)
{
    size_t size = 27 * nx * ny * nz * sizeof(real_t);
    // Allocate persistent device memory for distribution funcitons
    cudaMalloc((void**)&d_f_in, size);
    cudaMalloc((void**)&d_f_out, size);
    // Copy D3Q27 parameters to GPU constant memory
    cudaMemcpyToSymbol(d_cx, h_cx, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, h_cy, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_cz, h_cz, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_opp, h_opp, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_weights, h_weights, 27 * sizeof(double));
}

// function to copy initial conditons to GPU
void lbm_copy_host_to_device(real_t* h_f_in, int nx, int ny, int nz)
{
    size_t size = 27 * nx * ny * nz * sizeof(real_t);
    cudaMemcpy(d_f_in, h_f_in, size, cudaMemcpyHostToDevice);
}

// function to perform one LBM step
void lbm_run_step_gpu(int nx, int ny, int nz, real_t omega, real_t u_lid)
{
    // define block and grid sizes
    dim3 block(8, 4, 4); // better occupancy with 1D flattened thread blocks
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);
    // launch bulk lbm kernel
    lbm_kernel_soa<<<grid, block>>>(d_f_in, d_f_out, nx, ny, nz, omega);
    // launch boundary condition kernel
    apply_bc_kernel<<<grid, block>>>(d_f_out, nx, ny, nz, u_lid);
    // synchronize
    cudaDeviceSynchronize();
    // swap pointers for next iteration
    real_t* temp = d_f_in;
    d_f_in = d_f_out;
    d_f_out = temp;
}

// function to copy results back to host
void lbm_copy_device_to_host(real_t* h_f_out, int nx, int ny, int nz)
{
    size_t size = 27 * nx * ny * nz * sizeof(real_t);
    cudaMemcpy(h_f_out, d_f_in, size, cudaMemcpyDeviceToHost);
}

// function to free GPU memory
void lbm_free_gpu()
{
    if (d_f_in) cudaFree(d_f_in);
    if (d_f_out) cudaFree(d_f_out);
}

} // extern "C"





