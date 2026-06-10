/*
*/

#include <iostream>
#include <vector>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>
#include <ctime>


using namespace std;
using namespace nvcuda;

void cpu_implementation(vector<int> &matA,vector<int> &matB,vector<int> &matC,int M,int K,int N)
{
  for(int i=0;i<M;i++) {
    for(int j=0;j<N;j++) {
      matC[i*N+j]=0;
      for(int k=0;k<K;k++) {
        matC[i*N+j]+=matA[i*K+k]*matB[k*N+j];
      }
    }
  }
}


/*
  A = MxK
  B = KxN
  C = MxN
*/
__global__ void mma_kernel(const half *A,const half *B,float *C,int M,int N,int K) {
  
  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  int row = tile_row*16;
  int col = tile_col*16;

  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
  wmma::fill_fragment(c_frag,0.0f);

  //
  for(int k = 0; k < K; k += 16) {

    const half *A_tile = A + row*K + k;
    const half *B_tile = B + k*N + col;

    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, B_tile, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float *C_tile = C + row * N + col;
  wmma::store_matrix_sync(C_tile,c_frag,N,wmma::mem_row_major);
}

int main() {

  const int M = 2048;
  const int N = 2048;
  const int K = 2048;

  vector<int> cpuA(M*K), cpuB(K*N), cpuC(M*N, 0);
  vector<float> gpuResult(M*N);
  srand(time(0));

  for(size_t i = 0;i<cpuA.size();i++) {
    cpuA[i] = rand()%10;
  }

  for(size_t i = 0;i<cpuB.size();i++) {
    cpuB[i] = rand()%10;
  }

  std::vector<half> h_A(M*K);
  std::vector<half> h_B(K*N);
  std::vector<float> h_C(M*N);

  for(size_t i=0;i<cpuA.size();i++)
    h_A[i] = __float2half(float(cpuA[i]));

  for(size_t i=0;i<cpuB.size();i++)
    h_B[i] = __float2half(float(cpuB[i]));

  half *d_A,*d_B;
  float *d_C;

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);

  dim3 grid(N/16,M/16);
  dim3 block(32);

  //cpu computation
  cpu_implementation(cpuA, cpuB, cpuC, M, K, N);

  //gpu computation
  mma_kernel<<<grid,block>>>(d_A, d_B, d_C, M, N, K);
  cudaDeviceSynchronize();
  cudaMemcpy(gpuResult.data(),d_C,M*N*sizeof(float),cudaMemcpyDeviceToHost);

  float maxAbsError = 0.0f;

  for(size_t i = 0;i<gpuResult.size();i++) {
    maxAbsError = max(maxAbsError, fabs(gpuResult[i] - float(cpuC[i])));
  }

  cout << "Max error = "<<maxAbsError<<".\n";
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}