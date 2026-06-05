#include <iostream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

__global__ void matchsyncDemo(int *input, int N) {

  int tid = threadIdx.x;

  if(tid >= N) {
    return;
  }

  int value = input[tid];

  // Find all threads in the warp having same value
  unsigned int mask = __match_any_sync(0xffffffff, value);

  printf("Thread %2d | Value = %2d | Match Mask = 0x%08x\n", tid, value, mask);
}

int main() {

	int N = 32;
	int byteSize = N * sizeof(N);

	srand(time(0));

	vector<int> input(N);
	for(int i = 0;i<N;i++) {
		input[i] = rand()%10;	
		cout << input[i] << " ";
	}
	cout << endl;

	

	int *d_input;
	cudaMalloc(&d_input, byteSize);
	cudaMemcpy(d_input, input.data(), byteSize, cudaMemcpyHostToDevice);

	int block = 32;
	int grid = 1;

  matchsyncDemo<<<grid, block>>>(d_input, N);


	cudaFree(d_input);
	return 0;
}