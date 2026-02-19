// LBM Kernel Implementation in CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath> 

// D3Q27 model parameters stored in constant memory
// D3Q27 discrete velocities 
__constant__ int d_cx[27];
__constant__ int d_cy[27];
__constant__ int d_cz[27];
__constant__ int d_opp[27]; // Opposite directions
__constant__ double d_weights[27]; // D3Q27 weights

// Persistent device memory allocation for distribution functions
static double *d_f_in = nullptr;
static double *d_f_out = nullptr;
// Device memory for residual calculation
static double *d_diff_sq = nullptr;  
static double *d_mag_sq = nullptr; 

// ----- Bulk kernel for LBM step (Pull scheme + SOA) -----
__global__ void lbm_kernel_soa(const double* __restrict__ f_in,
                               double* __restrict__ f_out,
                               int nx, int ny, int nz,
                               double omega)
{
    // define global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check

    int N = nx * ny * nz; // total number of lattice points
    int curr_idx = i + j * nx + k * nx * ny; // current index

    // register variables for local calculations
    double f_streamed[27];
    double rho = 0.0;
    double u = 0.0, v = 0.0, w = 0.0; 

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

        double u2 = u * u + v * v + w * w;

    // Collision step
    #pragma unroll
    for (int l = 0; l < 27; l++){
        double cu = d_cx[l] * u + d_cy[l] * v + d_cz[l] * w;
        double feq = d_weights[l] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        // write post collision distribution
        f_out[l * N + curr_idx] = (1.0 - omega) * f_streamed[l] + omega * feq;
    }
}

// ----- Boundary Condition Kernel -----
__global__ void apply_bc_kernel(double* __restrict__ f,
                                int nx, int ny, int nz,
                                double u_lid)
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
        double rho_wall = 0.0;
        for (int l =0; l < 27; l++) rho_wall += f[l * N + idx]; // replace with Zou/He if needed
        
        for (int l = 0; l < 27; l++) {
            if (d_cy[l] == -1) 
                f[l* N + idx] = f[d_opp[l] * N + idx] + 6.0 * d_weights[l] * rho_wall * d_cx[l] * u_lid;
        }
    }
}

// Residual kernel for convergence monitoring
// Check criterion (||u_new - u_old|| / ||u_new|| < tol) at each lattice point
__global__ void lbm_residual_kernel(const double* __restrict__ f_old,
                                    const double* __restrict__ f_new,
                                    double* diff_acc, double* mag_acc,
                                    int nx, int ny, int nz)
{
    // define global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check

    int N = nx * ny * nz; // total number of lattice points
    int idx = i + j * nx + k * nx * ny; // current index

    // calculate velocity from distribution functions
    double rho_old = 0.0, u_old = 0.0, v_old = 0.0, w_old = 0.0;
    double rho_new = 0.0, u_new = 0.0, v_new = 0.0, w_new = 0.0;

    #pragma unroll
    for (int l =0; l < 27; l++){
        double f_old_val = f_old[l * N + idx];
        double f_new_val = f_new[l * N + idx];
        rho_old += f_old_val;
        u_old += f_old_val * d_cx[l];
        v_old += f_old_val * d_cy[l];
        w_old += f_old_val * d_cz[l];

        rho_new += f_new_val;
        u_new += f_new_val * d_cx[l];
        v_new += f_new_val * d_cy[l];
        w_new += f_new_val * d_cz[l];
    }
    if (rho_old > 1e-9) {u_old /= rho_old; v_old /=rho_old; w_old /= rho_old;}
    else {u_old = 0.0; v_old = 0.0; w_old = 0.0;}

    if (rho_new > 1e-9) {u_new /= rho_new; v_new /= rho_new; w_new /= rho_new;}
    else {u_new = 0.0; v_new = 0.0; w_new = 0.0;}

    // calculate local residual
    double du = u_new - u_old; double dv = v_new - v_old; double dw = w_new - w_old;
    double diff_sq = du * du + dv * dv + dw * dw;
    double mag_sq = u_new * u_new + v_new * v_new + w_new * w_new;

    // atomic add to accumulate residual across threads
    atomicAdd(diff_acc, diff_sq); // accumulate sum of squared differences
    atomicAdd(mag_acc, mag_sq); // accumulate sum of squared magnitudes
}                               

// ----- Host functions to launch kernels -----
extern "C" {
// function to initialize GPU with parameters
void lbm_init_gpu(int nx, int ny, int nz,
                  int* h_cx, int* h_cy, int* h_cz, int* h_opp, double* h_weights)
{
    size_t size = 27 * nx * ny * nz * sizeof(double);
    // Allocate persistent device memory for distribution funcitons
    cudaMalloc((void**)&d_f_in, size);
    cudaMalloc((void**)&d_f_out, size);
    // Allocate device memory for residual
    cudaMalloc((void**)&d_diff_sq, sizeof(double));
    cudaMalloc((void**)&d_mag_sq, sizeof(double));
    // Copy D3Q27 parameters to GPU constant memory
    cudaMemcpyToSymbol(d_cx, h_cx, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, h_cy, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_cz, h_cz, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_opp, h_opp, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_weights, h_weights, 27 * sizeof(double));
}

// function to copy initial conditons to GPU
void lbm_copy_host_to_device(double* h_f_in, int nx, int ny, int nz)
{
    size_t size = 27 * nx * ny * nz * sizeof(double);
    cudaMemcpy(d_f_in, h_f_in, size, cudaMemcpyHostToDevice);
}

// function to perform one LBM step
void lbm_run_step_gpu(int nx, int ny, int nz, double omega, double u_lid)
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
    double* temp = d_f_in;
    d_f_in = d_f_out;
    d_f_out = temp;
}

// function to compute residual for convergence monitoring
double lbm_compute_residual_gpu(int nx, int ny, int nz)  
{   
    // reset residual accumulators
    double zero = 0.0; // initialize to zero for atomic add
    cudaMemcpy(d_diff_sq, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mag_sq, &zero, sizeof(double), cudaMemcpyHostToDevice);
    // define block and grid sizes
    dim3 block(8, 4, 4); // better occupancy with 1D flattened thread blocks
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);
    // launch residual kernel
    lbm_residual_kernel<<<grid, block>>>(d_f_in, d_f_out, d_diff_sq, d_mag_sq, nx, ny, nz);
    // synchronize
    cudaDeviceSynchronize();
    // copy results back to host and compute final residual
    double h_diff_sq, h_mag_sq;
    cudaMemcpy(&h_diff_sq, d_diff_sq, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mag_sq, d_mag_sq, sizeof(double), cudaMemcpyDeviceToHost);
    if (h_mag_sq < 1e-9) return 1.0; // avoid division by zero
    return sqrt(h_diff_sq) / sqrt(h_mag_sq);
}

// function to copy results back to host
void lbm_copy_device_to_host(double* h_f_out, int nx, int ny, int nz)
{
    size_t size = 27 * nx * ny * nz * sizeof(double);
    cudaMemcpy(h_f_out, d_f_in, size, cudaMemcpyDeviceToHost);
}

// function to free GPU memory
void lbm_free_gpu()
{
    if (d_f_in) cudaFree(d_f_in);
    if (d_f_out) cudaFree(d_f_out);
}

} // extern "C"





