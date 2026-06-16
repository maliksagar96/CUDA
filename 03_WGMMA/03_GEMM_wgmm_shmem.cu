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
  
  //can be 0,1,2,3
  int warp_id = threadIdx.x / 32;

  // can be from 0 to 31
  int lane_id = threadIdx.x % 32;

  // Each block computes a 32x32 tile of C. This is a stride of 32 steps at once. 
  int block_row = blockIdx.y * 32;
  int block_col = blockIdx.x * 32;

  // Layout of warps inside the block:
  // Warp 0 -> upper left, Warp 1 -> upper right, Warp 2 -> lower left, Warp 3 -> lower right
  //+----+----+
  //| W0 | W1 |
  //+----+----+
  //| W2 | W3 |
  //+----+----+

  //Warp row and warp column can be either 0 or 1
  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;

  // Starting location of the 16x16 tile computed by this warp. global_offset + local_offset.
  int row = block_row + warp_row * 16;
  int col = block_col + warp_col * 16;

  // Shared memory stores: A tile : 32x16, B tile : 16x32. These tiles will be reused by 4 warps.
  __shared__ half As[32*16];
  __shared__ half Bs[16*32];

  // Tensor Core fragments.
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;

  wmma::fill_fragment(c_frag,0.0f);

  // Walk through K dimension in chunks of 16.
  for(int k=0;k<K;k+=16) {
    // Load a 32x16 tile of A into shared memory. block_row : block_row+31, k:k+15
    for(int idx = threadIdx.x; idx < 32 * 16;idx += blockDim.x) {
      int r = idx / 16;
      int c = idx % 16;
      As[idx] = A[(block_row + r)*K + (k+c)];
    }

    // Load a 16x32 tile of B into shared memory. k:k+15. block_col : block_col+31
    for(int idx=threadIdx.x;idx<16*32;idx+=blockDim.x) {
      int r = idx / 32;
      int c = idx % 32;
      Bs[idx] = B[(k + r) * N + (block_col + c)];
    }

    __syncthreads();

    // Select the A and B tile needed by this warp.
    // Warp 0 and Warp 1 share the upper half. Warp 2 and Warp 3 share the lower half.
    half *A_tile = &As[warp_row * 16 * 16];

    // Warp 0 and Warp 2 use left half. Warp 1 and Warp 3 use right half.
    half *B_tile=&Bs[warp_col*16];

    // Load shared-memory tiles into Tensor Core fragments.
    wmma::load_matrix_sync(a_frag,A_tile,16);
    wmma::load_matrix_sync(b_frag,B_tile,32);

    // Tensor Core operation:C += A * B
    wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);

    __syncthreads();
  }

  // Store this warp's 16x16 result tile.
  float *C_tile=C + row*N + col;

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

  dim3 grid(N/32,M/32);
  dim3 block(128);

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