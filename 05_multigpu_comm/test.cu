#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>

using namespace std;

int main() {

	const int ngpus;

	cudaStream_t streams[ngpus];
	ncclComm_t comms[ngpus];

	int *d_buffer[ngpus];
	int devices = {0,1,2,3};

	//init streams
	for(int i = 0;i<ngpus;i++) {
		cudaSetDevice(i);
		cudaMalloc(&d_buff[i], sizeof(int));
		int value = i * 10;
		cudaMemcpy(&d_buffer, value, sizeof(int), cudaMemcpyHostToDevice);
		cudaStreamCreate(&streams[i]);
	}

	//init communication for all
	ncclCommInitAll(comms, ngpus, devices);
	
	ncclGroupStart();
	/*
		perform communication
	*/
	ncclGroupEnd();

	for(int i = 0;i<ngpus;i++) {
		cudaSetDevice(i);
		cudaFree(d_buffer);
		cudaStreamDestroy(&streams[i]);
		ncclCommDestroy(comms[i]);
	}

	return 0;
}