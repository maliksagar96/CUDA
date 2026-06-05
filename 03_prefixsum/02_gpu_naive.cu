#include <iostream>
#include <vector>
#include <cassert>
#include <ctime> 
#include <cuda_runtime.h>

using namespace std;

__global__ void gpuPrefixSum(int *input, int *output, int N) {
  
  int gTID = blockIdx.x *  blockDim.x + threadIdx.x;

  if(gTID < N) {
    int sum = 0;
    for(int i = 0;i<gTID;i++) {
      sum += input[i];
    }
    output[gTID] = sum;
  }
}

void cpuPrefixSum(vector<int>& input, vector<int>& output) {
  int currentSum = 0;
  for(int i = 0;i<input.size();i++) {
    int prefixSum = input[i] + currentSum;
    output[i] = currentSum;
    currentSum = prefixSum;
  }
}

int main() {
  int N = 1 << 12;
  int byteSize = N * sizeof(int);

  vector<int> input(N), prefixSum(N), gpuOutPut(N);  
  srand(time(0));

  for(int i = 0;i<N;i++) {
    input[i] = rand()%10;
  }

  cpuPrefixSum(input, prefixSum);

  int *d_input, *d_output;
  cudaMalloc(&d_input, byteSize);
  cudaMalloc(&d_output, byteSize);

  cudaMemcpy(d_input, input.data(), byteSize, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (N+block-1)/block;

  gpuPrefixSum<<<grid, block>>>(d_input, d_output, N);

  cudaMemcpy(gpuOutPut.data(), d_output, byteSize, cudaMemcpyDeviceToHost);

  for(int i = 0;i<N;i++) {
    assert(gpuOutPut[i] == prefixSum[i]);
  }

  // for(int i = 0;i<N;i++) {
  //   cout << "PrefixSum["<<i<<"] = "<<prefixSum[i]<<", gpuoutput["<<i<<"] = "<<gpuOutPut[i]<<endl;
  // }

  cout << "Result Match.\n";
  

  cudaFree(d_input); cudaFree(d_output);


  return 0;
}