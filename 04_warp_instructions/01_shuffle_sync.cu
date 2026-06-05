/*
	__shfl_sync is a broadcast commnd. It broadcasts on a warp level. 
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__ void shuffleSyncDemo(int *input, int N) {

  int gTID = blockIdx.x * blockDim.x + threadIdx.x;

  if(gTID < N) {

    int myValue = input[gTID];

    // Alternate threads active: 101010...
    unsigned mask = 0xAAAAAAAA;

    // Read value from lane 2
    int shuffledValue = __shfl_sync(mask, myValue, 1);

    printf("Thread %d : myValue = %d , shuffledValue = %d\n",gTID, myValue, shuffledValue);
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

	shuffleSyncDemo<<<grid, block>>>(d_input, N);


	cudaFree(d_input);
	return 0;
}