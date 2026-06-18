#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>

using namespace std;

int main() {
	const int ngpus = 4;
	int localValue;
	int *d_value[ngpus];
	int *d_receive[ngpus];

	//setup streams
  cudaStream_t streams[ngpus];

	//setup communicator object
	ncclComm_t comms[ngpus];

	//Declare device ids
	int devices[ngpus] = {0,1,2,3};

	//declare data and start streams
	for(int i = 0;i<ngpus;i++) {
		localValue = i*10;
		cudaSetDevice(i);
		cudaMalloc(&d_value[i], sizeof(int));
		cudaMalloc(&d_receive[i], sizeof(int));
		cudaMemcpy(d_value[i], &localValue, sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamCreate(&streams[i]);
	}

	//comm init all
	ncclCommInitAll(comms, ngpus, devices);

	//groupstart command
  ncclGroupStart();

	//Reduce
	for(int i = 0;i<ngpus;i++) {
		cudaSetDevice(i);
		ncclAllReduce(d_value[i], d_receive[i], 1, ncclInt, ncclSum, comms[i], streams[i]);
	}
	
	//groupend command
	ncclGroupEnd();

	//synchronise streams	
  for(int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

	int result[ngpus];
	for(int i = 0;i<ngpus;i++) {
		cudaSetDevice(i);
		cudaMemcpy(&result[i], d_receive[i], sizeof(int), cudaMemcpyDeviceToHost);
		cout << "Result["<< i<<"] = "<<result[i]<<endl;
	}
	
	//Destroy streams, destroy comms, free up memory
	for(int i = 0;i<ngpus;i++) {
		cudaSetDevice(i);
		cudaFree(d_receive[i]);
		cudaFree(d_value[i]);
		cudaStreamDestroy(streams[i]);
		ncclCommDestroy(comms[i]);
	} 

	return 0;
}







