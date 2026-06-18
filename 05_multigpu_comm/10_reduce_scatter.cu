#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>

using namespace std;

int main() {
  const int ngpus = 4;

  // Each GPU contributes 4 integers
  int* d_send[ngpus];

  // Each GPU receives 1 integer
  int* d_receive[ngpus];

  cudaStream_t streams[ngpus];
  ncclComm_t comms[ngpus];

  int devices[ngpus] = {0, 1, 2, 3};

  // Initialize data
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    int hostData[4] ={i * 4 + 1,i * 4 + 2,i * 4 + 3,i * 4 + 4};
    cudaMalloc(&d_send[i], 4 * sizeof(int));
    cudaMalloc(&d_receive[i], sizeof(int));
    cudaMemcpy(d_send[i],hostData,4 * sizeof(int),cudaMemcpyHostToDevice);
    cudaStreamCreate(&streams[i]);
  }

  // Create communicators
  ncclCommInitAll(comms, ngpus, devices);

  // Launch collective
  ncclGroupStart();

  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    ncclReduceScatter(d_send[i],d_receive[i],1,ncclInt,ncclSum,comms[i],streams[i]);
  }

  ncclGroupEnd();

  // Wait for completion
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

  // Print results
  for(int i = 0; i < ngpus; i++) {
    int result;
    cudaSetDevice(i);
    cudaMemcpy( &result, d_receive[i], sizeof(int), cudaMemcpyDeviceToHost);
    cout << "GPU " << i<< " : "<< result<< endl;
  }

  // Cleanup
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaFree(d_send[i]);
    cudaFree(d_receive[i]);
    cudaStreamDestroy(streams[i]);
    ncclCommDestroy(comms[i]);
  }

  return 0;
}