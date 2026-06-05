#include <iostream>
#include <ctime>
#include <cassert>
#include <cuda_runtime.h>

using namespace std;

__global__ void reduction(int *d_input, int N) {
	int gTID = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int shmem[16];

	int warpID = threadIdx.x % warpSize;
	int value = 0;
	if(threadIdx.x < blockDim.x) {
		value = input[threadIdx.x];
	}
	
	shmem[warpID] = __reduce_add_sync(0xFFFFFFFF, value);
	__syncthreads();
}

int main() {

	int N = 1 << 20;
	int byteSize = N * sizeof(int);
	int gpuResult;

	vector<int> input(N);
	srand(time(0));

	long long sum = 0;

	for(int i = 0;i<N;i++) {
		input[i] = rand()%10;
		sum += input[i];
	}

	int *d_input, *d_output;

	cudaMalloc(&d_input, byteSize);
	cudaMemcpy(d_input, input.data(), byteSize, cudaMemcpyHostToDevice);

	int block = 512;
	int grid = (N + block - 1)/block;

	reduction<<<grid, block>>>(d_input);
	cudaDeviceSynchronize();



	cudaFree(d_input);
  return 0;
}