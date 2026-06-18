#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>

using namespace std;

int main() {

  const int ngpus = 4;

  // Device buffers for each GPU
  int* d_buff[ngpus];

  // One CUDA stream per GPU
  cudaStream_t streams[ngpus];

  // One NCCL communicator per GPU
  ncclComm_t comms[ngpus];

  // GPUs participating in communication
  int devs[ngpus] = {0, 1, 2, 3};

  // Allocate memory and create streams
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_buff[i], sizeof(int));
    cudaStreamCreate(&streams[i]);
  }

  // Create communicators for all GPUs
  ncclCommInitAll(comms, ngpus, devs);

  // Initialize each GPU with a different value
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    int value = (i + 1) * 10;
    cudaMemcpy(d_buff[i],&value,sizeof(int),cudaMemcpyHostToDevice);
  }

  /*
    Initial state

    GPU0 = 10
    GPU1 = 20
    GPU2 = 30
    GPU3 = 40

    Communication pattern

    GPU0 ----> GPU1
    GPU1 ----> GPU2
    GPU2 ----> GPU3
  */

  // Group NCCL operations together
  ncclGroupStart();

  // GPU0 sends its value to GPU1
  ncclSend(d_buff[0],1,ncclInt,1,comms[0],streams[0]);

  // GPU1 receives value from GPU0
  ncclRecv(d_buff[1],1,ncclInt,0,comms[1],streams[1]);

  // GPU1 sends its value to GPU2
  ncclSend(d_buff[1],1,ncclInt,2,comms[1],streams[1]);

  // GPU2 receives value from GPU1
  ncclRecv(d_buff[2],1,ncclInt,1,comms[2],streams[2]);

  // GPU2 sends its value to GPU3
  ncclSend(d_buff[2],1,ncclInt,3,comms[2],streams[2]);

  // GPU3 receives value from GPU2
  ncclRecv(d_buff[3],1,ncclInt,2,comms[3],streams[3]);

  ncclGroupEnd();

  // Wait for communication to complete
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

  // Copy results back to CPU and print
  for(int i = 0; i < ngpus; i++) {
    int result;
    cudaSetDevice(i);
    cudaMemcpy(&result,d_buff[i],sizeof(int),cudaMemcpyDeviceToHost);
    std::cout << "GPU " << i << " : " << result << std::endl;
  }

  // Cleanup resources
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaFree(d_buff[i]);
    cudaStreamDestroy(streams[i]);
    ncclCommDestroy(comms[i]);
  }

  return 0;
}