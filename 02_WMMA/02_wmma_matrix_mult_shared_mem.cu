#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <ctime>
#include <mma.h>
#include <cuda_fp16.h>
#include <cmath>

using namespace std;
using namespace nvcuda;

void cpuCalculation(vector<half>& A, vector<half>& B, vector<float>& C, const int M, const int K, const int N) {

	for(int i = 0;i<M;i++) {
		for(int j = 0;j<N;j++) {
			for(int k = 0;k<K;k++) {
				//cij = aik*bkj
				C[i * N + j] += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
			}
		}
	}
}

__global__ void gpuCalculation(half *A, half *B, float *C, const int M, const int K, const int N) {

	__shared__ half a_shared[64 * 16];
	__shared__ half b_shared[16 * 64];

	int tid = threadId.x;
	int warpId = tid/32;

	int warpRow = warpId/4;
	int warpN = warp;

	

	//create fragment of A, B and C
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	//declare elements of c = 0
	wmma::fill_fragment(c_frag, 0.0f);

	for(int ) {
		for(int k = 0;k<K;k += 16) {
			//load matrixA and matrixB to wmma fragments
			wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
			wmma::load_matrix_sync(b_frag, B + k*N + warpN * 16, N);
			
			//compute the matrix multiple
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
	}

	//store the result back to C
	wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, wmma::mem_row_major);

}

int main() {

	const int M = 1024;
	const int K = 1024;
	const int N = 1024;
	
	vector<half> A(M*K); 
	vector<half> B(K*N);
	vector<float> C(M*N, 0);
	vector<float> gpuResult(M*N);

	srand(time(0));

	for(int i = 0;i<A.size();i++) {
		A[i] = rand()%10;
	}

	for(int i = 0;i<B.size();i++) {
		B[i] = rand()%10;
	}

	cpuCalculation(A, B, C, M, K, N);

	half *d_A, *d_B;
	float *d_C;
	const int byteSizeA = A.size() * sizeof(half);
	const int byteSizeB = B.size() * sizeof(half);
	const int byteSizeC = C.size() * sizeof(float);

	cudaMalloc(&d_A, byteSizeA);
	cudaMalloc(&d_B, byteSizeB);
	cudaMalloc(&d_C, byteSizeC);

	cudaMemcpy(d_A, A.data(), byteSizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B.data(), byteSizeB, cudaMemcpyHostToDevice);

	dim3 threads(32);
	dim3 blocks(N/64, M/64);

	gpuCalculation<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
	cudaDeviceSynchronize();
	
	cudaMemcpy(gpuResult.data(), d_C, byteSizeC, cudaMemcpyDeviceToHost);

	float maxError = 0.0;

	for(int i = 0;i<C.size();i++) {
		maxError = max(maxError, fabs(gpuResult[i] - C[i]));
	}

	cout << "Max error = "<<maxError<<endl;				

  return 0;
}