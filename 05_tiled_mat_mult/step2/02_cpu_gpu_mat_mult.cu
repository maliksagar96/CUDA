#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <random>

#define CUDA_CHECK(call)                                   \
do {                                                       \
  cudaError_t err = call;                                  \
  if (err != cudaSuccess) {                                \
    std::cerr << "CUDA error at " << __FILE__ << ":"       \
              << __LINE__ << " -> "                        \
              << cudaGetErrorString(err) << std::endl;    \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
} while (0)

using namespace std;

/* ---------------- CPU reference ---------------- */
void cpu_mat_mult(const float* a, const float* b, float* c,
                  int n1, int n2, int n3)
{
  for (int i = 0; i < n1; i++) {
    for (int k = 0; k < n3; k++) {
      float sum = 0.0f;
      for (int j = 0; j < n2; j++) {
        sum += a[i*n2 + j] * b[j*n3 + k];
      }
      c[i*n3 + k] = sum;
    }
  }
}

/* ---------------- GPU kernel ---------------- */
__global__ void gpu_mat_mult(const float* a,
                             const float* b,
                             float* c,
                             int n1, int n2, int n3)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n1 && k < n3) {
    float sum = 0.0f;
    for (int j = 0; j < n2; j++) {
      sum += a[i*n2 + j] * b[j*n3 + k];
    }
    c[i*n3 + k] = sum;
  }
}

int main()
{
  const int n1 = 1024;
  const int n2 = 1024;
  const int n3 = 1024;

  const size_t a_sz = n1 * n2 * sizeof(float);
  const size_t b_sz = n2 * n3 * sizeof(float);
  const size_t c_sz = n1 * n3 * sizeof(float);

  /* -------- Host memory -------- */
  float* a = (float*)malloc(a_sz);
  float* b = (float*)malloc(b_sz);
  float* c_cpu = (float*)malloc(c_sz);
  float* c_gpu = (float*)malloc(c_sz);

  /* -------- Initialize input -------- */
  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (int i = 0; i < n1*n2; i++) a[i] = dis(gen);
  for (int i = 0; i < n2*n3; i++) b[i] = dis(gen);

  /* -------- CPU reference -------- */
  cpu_mat_mult(a, b, c_cpu, n1, n2, n3);

  /* -------- Device memory -------- */
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, a_sz));
  CUDA_CHECK(cudaMalloc(&d_b, b_sz));
  CUDA_CHECK(cudaMalloc(&d_c, c_sz));

  CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, c_sz));

  /* -------- Kernel launch -------- */
  //Launching 256 threads.
  dim3 threads(16, 16);
  dim3 blocks((n3 + threads.x - 1) / threads.x,
              (n1 + threads.y - 1) / threads.y);

  gpu_mat_mult<<<blocks, threads>>>(d_a, d_b, d_c, n1, n2, n3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(c_gpu, d_c, c_sz, cudaMemcpyDeviceToHost));

  /* -------- Compare (relative + absolute) -------- */
  bool match = true;
  for (int i = 0; i < n1*n3; i++) {
    float diff = fabs(c_cpu[i] - c_gpu[i]);
    float tol  = 1e-5f * fabs(c_cpu[i]) + 1e-5f;
    if (diff > tol) {
      cout << "Mismatch at " << i
           << " CPU=" << c_cpu[i]
           << " GPU=" << c_gpu[i]
           << " diff=" << diff << endl;
      match = false;
      break;
    }
  }

  if (match)
    cout << "Results match ✔" << endl;

  /* -------- Cleanup -------- */
  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
