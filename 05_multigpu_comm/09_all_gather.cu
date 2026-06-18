#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>

using namespace std;

int main()
{
  const int ngpus = 4;

  // Device buffers
  int *d_send[ngpus], *d_receive[ngpus];

  // Streams and communicators
  cudaStream_t streams[ngpus];
  ncclComm_t comms[ngpus];

  // GPU IDs
  int devices[ngpus] = {0, 1, 2, 3};

  // Initialize buffers and streams
  for(int i = 0; i < ngpus; i++) {
    int localValue = i * 10;

    cudaSetDevice(i);
    // Each GPU contributes one integer
    cudaMalloc(&d_send[i], sizeof(int));
    // Each GPU receives ngpus integers
    cudaMalloc(&d_receive[i], ngpus * sizeof(int));
    cudaMemcpy(d_send[i], &localValue, sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamCreate(&streams[i]);
  }

  // Create communicators
  ncclCommInitAll(comms, ngpus, devices);

  // Launch collective
  ncclGroupStart();

  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    ncclAllGather(d_send[i],d_receive[i],1,ncclInt,comms[i],streams[i]);
  }

  ncclGroupEnd();

  // Wait for completion
  for(int i = 0; i < ngpus; i++){
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

  // Print results
  int result[ngpus];

  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaMemcpy( result, d_receive[i], ngpus * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "GPU " << i << " : ";

    for(int j = 0; j < ngpus; j++) {
      cout << result[j] << "\t";
    }

    cout << endl;
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