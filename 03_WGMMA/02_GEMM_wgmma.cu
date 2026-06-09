/*


*/


#include <iostream>
#include <vector>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

void cpu_implementation(vector<int> &matA, vector<int> &matB, vector<int> &matC, int P, int Q, int R) {

	//A is PxQ and B is QxR
	for(int i = 0;i<P;i++) {
		for(int j = 0;j<R;j++) {            
			for(int k = 0;k < Q;k++) {
				//cij = aik*bkj
				matC[i * R + j] += matA[Q*i + k] * matB[R * k + j]; 
			}
		}
	}
}



__global__ void mma_kernel(const half *A,const half *B,float *C,int M,int N,int K) {
  
  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  int row = tile_row*16;
  int col = tile_col*16;

  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;

  wmma::fill_fragment(c_frag,0.0f);

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

  std::vector<half> h_A(M*K);
  std::vector<half> h_B(K*N);
  std::vector<float> h_C(M*N);

  for(int i=0;i<M*K;i++)
    h_A[i]=__float2half(1.0f);

  for(int i=0;i<K*N;i++)
    h_B[i]=__float2half(2.0f);

  half *d_A,*d_B;
  float *d_C;

  cudaMalloc(&d_A,M*K*sizeof(half));
  cudaMalloc(&d_B,K*N*sizeof(half));
  cudaMalloc(&d_C,M*N*sizeof(float));

  cudaMemcpy(d_A,h_A.data(),M*K*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B.data(),K*N*sizeof(half),cudaMemcpyHostToDevice);

  dim3 grid(N/16,M/16);
  dim3 block(32);

  mma_kernel<<<grid,block>>>(d_A,d_B,d_C,M,N,K);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C.data(),d_C,M*N*sizeof(float),cudaMemcpyDeviceToHost);

  std::cout<<"Top-left 8x8 block of C\n";

  for(int i=0;i<8;i++) {
    for(int j=0;j<8;j++)
      std::cout<<h_C[i*N+j]<<" ";

    std::cout<<"\n";
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}