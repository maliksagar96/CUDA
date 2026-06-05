#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__ void reduceAddDemo(int *input, int N) {

  int tid = threadIdx.x;

  int value = 0;

  if(tid < N) {
    value = input[tid];
  }

  // Warp-wide sum reduction
  int sum = __reduce_add_sync(0xAAAAAAAA, value);//Half threads unabled. 
//int sum = __reduce_add_sync(0xffffffff, value);

  // Print only once
  if(tid == 0) {
    printf("Warp Sum = %d\n", sum);
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

    reduceAddDemo<<<grid, block>>>(d_input, N);


	cudaFree(d_input);
	return 0;
}