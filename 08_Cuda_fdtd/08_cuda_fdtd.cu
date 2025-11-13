#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// ---- GPU constant memory ----
__constant__ double d_EPSILON_0;
__constant__ double d_MU_0;
__constant__ double d_c0;
__constant__ double d_frequency;
__constant__ double d_center_wavelength;
__constant__ double d_simulation_size;
__constant__ double d_step_size;
__constant__ double d_dx;
__constant__ double d_dt;
__constant__ double d_h_coeff;
__constant__ double d_e_coeff;
__constant__ double d_omega;

__global__ void update_Ex(double* Ex, const double* Hz, int Nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Nx && j >= 1 && j < Nx) {
        int idx = i * Nx + j;
        Ex[idx] += d_e_coeff * (Hz[idx] - Hz[i * Nx + j - 1]);
    }
}


__global__ void update_Ey(double* Ey, const double* Hz, int Nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx && j < Nx) {
        int idx = i * Nx + j;
        Ey[idx] -= d_e_coeff * (Hz[idx] - Hz[(i - 1) * Nx + j]);
    }
}


__global__ void update_Hz(double* Hz, const double* Ex, const double* Ey, int Nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Nx - 1 && j < Nx - 1) {
        int idx = i * Nx + j;
        Hz[idx] -= d_h_coeff * ((Ey[(i + 1) * Nx + j] - Ey[idx]) -
                                (Ex[i * Nx + j + 1] - Ex[idx]));
    }
}

__global__ void inject_source(double* Ey, int src_i, int src_j, int Nx, double time) {
    int idx = src_i * Nx + src_j;        // flatten (i,j) correctly
    Ey[idx] += sin(d_omega * time);
}




int main() {

  // --- Physical constants ---
  double EPSILON_0 = 8.8541878128e-12;
  double MU_0 = 1.256637062e-6;
  double c0 = 2.99792458e8;

  // --- Simulation parameters ---
  double frequency = 1e9;
  double center_wavelength = c0 / frequency;
  double simulation_size = 10 * center_wavelength;
  int cells_per_wavelength = 50;

  double step_size = center_wavelength / cells_per_wavelength;
  double dx = step_size;
  int Nx = static_cast<int>(simulation_size / step_size);
  double dy = dx;  // added since dt uses dy
  double dt = 0.5 / (c0 * sqrt((1.0 / (dx * dx)) + (1.0 / (dy * dy))));
  int N_time_steps = 2000;

  // Source position
  int src_i = Nx / 2;
  int src_j = 3 * Nx / 4;

  double h_coeff = dt / (sqrt(MU_0 * EPSILON_0) * dx);
  double e_coeff = dt / (sqrt(MU_0 * EPSILON_0) * dx);
  double omega = 2 * M_PI * frequency;

  // ---- Copy constants to GPU ----
  cudaMemcpyToSymbol(d_EPSILON_0, &EPSILON_0, sizeof(double));
  cudaMemcpyToSymbol(d_MU_0, &MU_0, sizeof(double));
  cudaMemcpyToSymbol(d_c0, &c0, sizeof(double));
  cudaMemcpyToSymbol(d_frequency, &frequency, sizeof(double));
  cudaMemcpyToSymbol(d_center_wavelength, &center_wavelength, sizeof(double));
  cudaMemcpyToSymbol(d_simulation_size, &simulation_size, sizeof(double));
  cudaMemcpyToSymbol(d_step_size, &step_size, sizeof(double));
  cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(double));
  cudaMemcpyToSymbol(d_h_coeff, &h_coeff, sizeof(double));
  cudaMemcpyToSymbol(d_e_coeff, &e_coeff, sizeof(double));
  cudaMemcpyToSymbol(d_omega, &omega, sizeof(double));

  cout << "All constants copied to GPU constant memory.\n";

  // --- Allocate and initialize fields on CPU ---
  vector<double> Ex(Nx * Nx, 0.0);
  vector<double> Ey(Nx * Nx, 0.0);
  vector<double> Hz(Nx * Nx, 0.0);

  // --- Allocate fields on GPU ---
  double *d_Ex, *d_Ey, *d_Hz;
  size_t field_size = Nx * Nx * sizeof(double);

  cudaMalloc((void**)&d_Ex, field_size);
  cudaMalloc((void**)&d_Ey, field_size);
  cudaMalloc((void**)&d_Hz, field_size);

  // --- Copy initial data from CPU to GPU ---
  cudaMemset(d_Ex, 0, field_size);
  cudaMemset(d_Ey, 0, field_size);
  cudaMemset(d_Hz, 0, field_size);


    // Define block and grid dimensions
  dim3 block(16, 16);  // 16x16 threads per block (good starting point)
  dim3 grid((Nx + block.x - 1) / block.x,
            (Nx + block.y - 1) / block.y);


  for (int t = 0; t < N_time_steps; ++t) {
    update_Hz<<<grid, block>>>(d_Hz, d_Ex, d_Ey, Nx);
    cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { cerr<<"Hz kernel failed: "<<cudaGetErrorString(err)<<endl; break; }

    update_Ex<<<grid, block>>>(d_Ex, d_Hz, Nx);
    err = cudaGetLastError(); if (err != cudaSuccess) { cerr<<"Ex kernel failed: "<<cudaGetErrorString(err)<<endl; break; }

    update_Ey<<<grid, block>>>(d_Ey, d_Hz, Nx);
    err = cudaGetLastError(); if (err != cudaSuccess) { cerr<<"Ey kernel failed: "<<cudaGetErrorString(err)<<endl; break; }

    double time = t * dt;
    inject_source<<<1,1>>>(d_Ey, src_i, src_j, Nx, time);
    err = cudaGetLastError(); if (err != cudaSuccess) { cerr<<"inject kernel failed: "<<cudaGetErrorString(err)<<endl; break; }

    // Optional debug sync (remove for best performance)
    cudaDeviceSynchronize();
}



// --- Copy back results ---
cudaMemcpy(Ex.data(), d_Ex, field_size, cudaMemcpyDeviceToHost);
cudaMemcpy(Ey.data(), d_Ey, field_size, cudaMemcpyDeviceToHost);
cudaMemcpy(Hz.data(), d_Hz, field_size, cudaMemcpyDeviceToHost);

// --- Free GPU memory ---
cudaFree(d_Ex);
cudaFree(d_Ey);
cudaFree(d_Hz);

  for(int i = 0;i<10;i++) {
    cout<<Hz[i]<<"\t"<<endl;
  }

  return 0;
}
