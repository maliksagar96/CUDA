#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <ctime>

using namespace std;

void cpuCalculation(vector<float>& A, vector<float>& B, vector<float>& C, const int M, const int K, const int N) {

	for(int i = 0;i<M;i++) {
		for(int j = 0;j<N;j++) {
			for(int k = 0;k<K;k++) {
				//cij = aik*bkj
				C[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}


__global__ void gpuCalculation(float *A, float *B, float *C, const int M, const int N, const int K) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < M && col < N) {
		float sum = 0.0f;
		for(int k = 0;k<K;k++) {
			//cij = aik*bkj
			sum += A[row * K + k] * B[k*N + col];
		}
		C[row * N + col] = sum;
	}
}

int main() {

	const int M = 1024;
	const int K = 1024;
	const int N = 1024;
	
	vector<float> A(M*K); 
	vector<float> B(K*N);
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

	float *d_A, *d_B, *d_C;
	const int byteSizeA = A.size() * sizeof(float);
	const int byteSizeB = B.size() * sizeof(float);
	const int byteSizeC = C.size() * sizeof(float);

	cudaMalloc(&d_A, byteSizeA);
	cudaMalloc(&d_B, byteSizeB);
	cudaMalloc(&d_C, byteSizeC);

	cudaMemcpy(d_A, A.data(), byteSizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B.data(), byteSizeB, cudaMemcpyHostToDevice);

	dim3 threads(16,16);
	dim3 blocks((N + threads.x -1)/threads.x, (M + threads.y - 1)/threads.y);

	gpuCalculation<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

	cudaMemcpy(gpuResult.data(), d_C, byteSizeC, cudaMemcpyDeviceToHost);

	float maxError = 0.0;

	for(int i = 0;i<C.size();i++) {
		maxError = max(maxError, fabs(gpuResult[i] - C[i]));
	}

	cout << "Max error = "<<maxError<<endl;				

  return 0;
}