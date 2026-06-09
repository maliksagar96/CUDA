#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>  

using namespace nvcuda;

// One warp computes:
// C = A * B
//
// A : 16x16 half matrix
// B : 16x16 half matrix
// C : 16x16 float matrix

__global__ void mma_kernel(const half *A, const half *B, float *C)
{
  //Fragment that stores a tile of matrix A. matrix_a, M, N and K. All fragments are in order M, N and K.
  wmma::fragment<wmma::matrix_a,16, 16, 16, half,wmma::row_major> a_frag;
  wmma::fragment< wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment< wmma::accumulator, 16, 16, 16, float> c_frag; // Fragment that stores the result. Accumulator is usually float.

  // Initialize accumulator fragment to zero.
  wmma::fill_fragment(c_frag, 0.0f);

  // Load 16x16 matrix A from global memory into Tensor Core registers.
  wmma::load_matrix_sync(a_frag, A, 16);

  // Load 16x16 matrix B.
  wmma::load_matrix_sync(b_frag, B, 16);

  // Perform Tensor Core operation: c_frag = a_frag * b_frag + c_frag
  // Since c_frag was initialized to zero: c_frag = a_frag * b_frag
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  // Store the result back to global memory.
  wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main() {
  const int N = 16;
  half  h_A[N*N];
  half  h_B[N*N];
  float h_C[N*N];

  for(int i=0;i<N*N;i++)
  {
    h_A[i] = __float2half(1.0f);
    h_B[i] = __float2half(2.0f);
  }

  half  *d_A, *d_B;
  float *d_C;

  cudaMalloc(&d_A, N*N*sizeof(half));
  cudaMalloc(&d_B, N*N*sizeof(half));
  cudaMalloc(&d_C, N*N*sizeof(float));

  cudaMemcpy(d_A, h_A, N*N*sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N*N*sizeof(half), cudaMemcpyHostToDevice);

  // Launch one block with one warp.
  mma_kernel<<<1,32>>>(d_A, d_B, d_C);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  // Print result matrix
  std::cout << "Result Matrix C\n";

  for(int i=0;i<16;i++)
  {
    for(int j=0;j<16;j++)
    {
      std::cout << h_C[i*16+j] << " ";
    }

    std::cout << "\n";
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}