/*
Compile using 
nvcc gpuFFT.cu -lcufft -o gpuFFT

*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cufft.h>

using namespace std;

int main() {

  int N = 1000;
  double frequency = 1e9;
  double omega = 2 * M_PI * frequency;
  double dt = 1 / (frequency * 20);

  // allocate host input
  vector<float> f_t(N);
  for (int n = 0; n < N; n++) {
    f_t[n] = sin(omega * n * dt);
  }

  int float_byte_size = sizeof(float) * N;
  int cufft_byte_size = sizeof(cufftComplex) * (N/2 + 1);

  // allocate device input/output
  float* d_in;
  cufftComplex* d_out;
  cudaMalloc(&d_in, float_byte_size);
  cudaMalloc(&d_out, cufft_byte_size);

  // copy input to device
  cudaMemcpy(d_in, f_t.data(), float_byte_size, cudaMemcpyHostToDevice);

  // create plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_R2C, 1);

  // execute FFT
  cufftExecR2C(plan, d_in, d_out);

  // copy output back to host
  vector<cufftComplex> X(N/2 + 1);
  cudaMemcpy(X.data(), d_out, cufft_byte_size, cudaMemcpyDeviceToHost);

  // print magnitude of non-zero bins
  for (int k = 0; k < N/2 + 1; k++) {
    float mag = sqrt(X[k].x*X[k].x + X[k].y*X[k].y);
    if (mag > 1e-6) {
      cout << "k = " << k << "  magnitude = " << mag << endl;
    }
  }

  // cleanup
  cufftDestroy(plan);
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}
