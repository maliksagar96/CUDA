#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__ void shuffleupSyncDemo(int *input, int N) {

  int gTID = blockIdx.x * blockDim.x + threadIdx.x;

  if(gTID < N) {

    int value = input[gTID];

    // Each thread gets value from laneID - 1
    int shuffledValue = __shfl_up_sync(0xffffffff, value, 1);

    printf("Thread %d  Original = %d  Shuffled = %d\n", gTID, value, shuffledValue);
  }
}

int main() {

	int N = 32;
	int byteSize = N * sizeof(N);

	vector<int> input(N);
	for(int i = 0;i<N;i++) {
		input[i] = i;	
	}

	int *d_input;
	cudaMalloc(&d_input, byteSize);
	cudaMemcpy(d_input, input.data(), byteSize, cudaMemcpyHostToDevice);

	int block = 32;
	int grid = 1;

	shuffleupSyncDemo<<<grid, block>>>(d_input, N);


	cudaFree(d_input);
	return 0;
}