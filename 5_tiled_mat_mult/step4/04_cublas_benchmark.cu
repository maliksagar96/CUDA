/*
  Step 3: Benchmarking CPU vs cuBLAS GEMM
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>

#define CUDA_CHECK(call)                                      \
do {                                                          \
  cudaError_t err = call;                                     \
  if (err != cudaSuccess) {                                   \
    std::cerr << "CUDA error at " << __FILE__ << ":"          \
              << __LINE__ << " -> "                            \
              << cudaGetErrorString(err) << std::endl;        \
    std::exit(EXIT_FAILURE);                                  \
  }                                                           \
} while (0)

#define CUBLAS_CHECK(call)                                    \
do {                                                          \
  cublasStatus_t stat = call;                                 \
  if (stat != CUBLAS_STATUS_SUCCESS) {                        \
    std::cerr << "cuBLAS error at " << __FILE__ << ":"        \
              << __LINE__ << std::endl;                       \
    std::exit(EXIT_FAILURE);                                  \
  }                                                           \
} while (0)

using namespace std;

/* ---------------- CPU matmul (row-major) ---------------- */
void cpu_mat_mult(float *a, float *b, float *c,
                  int n1, int n2, int n3) {
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      float aij = a[i*n2 + j];
      for (int k = 0; k < n3; k++) {
        c[i*n3 + k] += aij * b[j*n3 + k];
      }
    }
  }
}

/* ---------------- main ---------------- */
int main() {

  int n1 = 4096, n2 = 4096, n3 = 4096;
  int N_runs = 10;

  size_t a_sz = n1 * n2 * sizeof(float);
  size_t b_sz = n2 * n3 * sizeof(float);
  size_t c_sz = n1 * n3 * sizeof(float);

  float *a = (float*)malloc(a_sz);
  float *b = (float*)malloc(b_sz);
  float *c = (float*)malloc(c_sz);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, a_sz));
  CUDA_CHECK(cudaMalloc(&d_b, b_sz));
  CUDA_CHECK(cudaMalloc(&d_c, c_sz));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Random initialization
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  vector<double> cpu_times, gpu_times;

  cout << "Warm-up runs...\n";
  for (int w = 0; w < 3; w++) {
    for (int i = 0; i < n1*n2; i++) a[i] = dis(gen);
    for (int i = 0; i < n2*n3; i++) b[i] = dis(gen);
    for (int i = 0; i < n1*n3; i++) c[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    // C^T = B^T * A^T
    CUBLAS_CHECK(
      cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        n3, n1, n2,
        &alpha,
        d_b, n3,
        d_a, n2,
        &beta,
        d_c, n3)
    );
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cout << "Benchmarking...\n";

  for (int r = 0; r < N_runs; r++) {

    for (int i = 0; i < n1*n2; i++) a[i] = dis(gen);
    for (int i = 0; i < n2*n3; i++) b[i] = dis(gen);
    for (int i = 0; i < n1*n3; i++) c[i] = 0.0f;

    // CPU timing
    auto cpu_start = chrono::high_resolution_clock::now();
    cpu_mat_mult(a, b, c, n1, n2, n3);
    auto cpu_end = chrono::high_resolution_clock::now();
    cpu_times.push_back(
      chrono::duration<double, milli>(cpu_end - cpu_start).count()
    );

    // GPU timing (H2D + GEMM)
    CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    auto gpu_start = chrono::high_resolution_clock::now();

    CUBLAS_CHECK(
      cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        n3, n1, n2,
        &alpha,
        d_b, n3,
        d_a, n2,
        &beta,
        d_c, n3)
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = chrono::high_resolution_clock::now();

    gpu_times.push_back(
      chrono::duration<double, milli>(gpu_end - gpu_start).count()
    );
  }

  double cpu_avg =
    accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / N_runs;
  double gpu_avg =
    accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / N_runs;

  cout << "\nAverage CPU time (ms): " << cpu_avg << endl;
  cout << "Average cuBLAS time (ms): " << gpu_avg << endl;
  cout << "Speedup: " << cpu_avg / gpu_avg << "x\n";

  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);

  return 0;
}
