#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>

int main() {

  const int ngpus = 4;

  // Device buffers on each GPU
  int* d_buff[ngpus];

  // One CUDA stream per GPU
  cudaStream_t streams[ngpus];

  // One NCCL communicator per GPU
  ncclComm_t comms[ngpus];

  // GPU IDs we want to use
  int devs[ngpus] = {0, 1, 2, 3};

  // Allocate memory and create streams
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_buff[i], sizeof(int));
    cudaStreamCreate(&streams[i]);
  }

  // Create NCCL communicators
  ncclCommInitAll(comms, ngpus, devs);

  // Value that will be broadcast
  int value = 42;

  // Copy value only to GPU0
  cudaSetDevice(0);

  cudaMemcpy(d_buff[0], &value, sizeof(int), cudaMemcpyHostToDevice);

  // Launch collective operation on all GPUs
  ncclGroupStart();

  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);

    // Root GPU is GPU0
    // After completion every GPU will contain 42
    ncclBroadcast(
      d_buff[0],      // source buffer on root GPU
      d_buff[i],      // destination buffer on this GPU
      1,              // number of integers
      ncclInt,        // datatype
      0,              // root rank
      comms[i],       // communicator
      streams[i]      // stream
    );
  }

  ncclGroupEnd();

  // Wait for communication to finish
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

  // Read back and print results
  for(int i = 0; i < ngpus; i++)
  {
    int result;

    cudaSetDevice(i);

    cudaMemcpy(&result,d_buff[i],sizeof(int),cudaMemcpyDeviceToHost);

    std::cout << "GPU " << i<< " contains "<< result<< std::endl;
  }

  // Cleanup
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaFree(d_buff[i]);
    cudaStreamDestroy(streams[i]);
    ncclCommDestroy(comms[i]);
  }

  return 0;
}