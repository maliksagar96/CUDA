/*
	Step 3 is benchmarking the code.
*/

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random> 
#include <chrono>
#include <unistd.h>

#define CUDA_CHECK(call)                                    \
do {                                                        \
  cudaError_t err = call;                                   \
  if (err != cudaSuccess) {                                 \
    std::cerr << "CUDA error at " << __FILE__ << ":"        \
              << __LINE__ << " -> "                         \
              << cudaGetErrorString(err) << std::endl;     \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while(0)


using namespace std;

void cpu_mat_mult(float *a, float *b, float *c, int n1, int n2, int n3) {

	for(int i = 0;i < n1;i++) {
		for(int j = 0;j < n2; j++) {
			float aij = a[i*n2 + j]; //a_ijth element
			for(int k = 0;k<n3;k++) {
				float bjk = b[j*n3 + k];
				//cik = summation(aij*bjk);
				c[i*n3 + k] += aij * bjk; 
			}
		}
	}
}

__global__ void gpu_mat_mult(float *a, float *b, float *c, int n1, int n2, int n3) {

  int i = blockIdx.y * blockDim.y + threadIdx.y; // row of A, C
  int k = blockIdx.x * blockDim.x + threadIdx.x; // column of B, C

  if (i < n1 && k < n3) {
    float sum = 0.0f;
    for (int j = 0; j < n2; j++) {   // j = col of A = row of B
      sum += a[i*n2 + j] * b[j*n3 + k];
    }
    c[i*n3 + k] = sum;
  }
}

int main() {

	int n1 = 4096, n2 = 4096, n3 = 4096;
	int N_runs = 10;
	int flt_sz = sizeof(float);
	int a_sz = n1*n2*flt_sz, b_sz = n2*n3*flt_sz, c_sz = n1*n3*flt_sz;

	float *a = (float*)malloc(a_sz);
	float *b = (float*)malloc(b_sz);
	float *c = (float*)malloc(c_sz);
	float *gpu_result = (float*)malloc(c_sz);

	float *d_a, *d_b, *d_c;


	CUDA_CHECK(cudaMalloc((void**)&d_a, a_sz));
	CUDA_CHECK(cudaMalloc((void**)&d_b, b_sz));
	CUDA_CHECK(cudaMalloc((void**)&d_c, c_sz));

	dim3 threadsPerBlock(16, 16);   // 256 threads per block
	dim3 numberOfBlocks(
	(n3 + threadsPerBlock.x - 1) / threadsPerBlock.x, // columns
	(n1 + threadsPerBlock.y - 1) / threadsPerBlock.y); // rows
 
	//initialise a, b and c
	std::random_device rd;  // non-deterministic seed (hardware if available)
	std::mt19937 gen(rd()); // Mersenne Twister engine
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);

	
	// Warm up runs
	cout<<"Performing warm up runs"<<endl;
	// for(int i = 0;i<3;i++) {
	// 	for(int i = 0; i < n1*n2; i++) {
	// 	a[i] = dis(gen);
	// 	}

	// 	for(int i = 0;i<n2*n3;i++) {
	// 		b[i] = dis(gen);
	// 	}

	// 	for(int i = 0;i<n1*n3;i++) {
	// 		c[i] = 0;
	// 	}
	// 	cpu_mat_mult(a, b, c, n1, n2, n3);

	// 	CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
	// 	CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));
	// 	CUDA_CHECK(cudaMemcpy(d_c, c, c_sz, cudaMemcpyHostToDevice));

	// 	gpu_mat_mult<<<numberOfBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n1, n2, n3);
	// }
	cout<<"Completed warm up runs"<<endl;

	vector<double> cpu_times, gpu_times;

	//benchmarking over 10 runs
	cout<<"Performing benchmarking runs."<<endl;
	sleep(2);
	for(int i = 0;i<N_runs;i++) {
		for(int i = 0; i < n1*n2; i++) {
		a[i] = dis(gen);
		}

		for(int i = 0;i<n2*n3;i++) {
			b[i] = dis(gen);
		}

		for(int i = 0;i<n1*n3;i++) {
			c[i] = 0;
		}


		// auto start = std::chrono::high_resolution_clock::now();
		// cpu_mat_mult(a, b, c, n1, n2, n3);
		// auto end = std::chrono::high_resolution_clock::now();
		// cpu_times.emplace_back(std::chrono::duration<double, std::milli>(end-start).count());

		auto gpu_start = std::chrono::high_resolution_clock::now();
		CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_c, c, c_sz, cudaMemcpyHostToDevice));		
		gpu_mat_mult<<<numberOfBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n1, n2, n3);

		auto gpu_end = std::chrono::high_resolution_clock::now();
		CUDA_CHECK(cudaMemcpy(gpu_result, d_c, c_sz, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaDeviceSynchronize());
		gpu_times.emplace_back(std::chrono::duration<double, std::milli>(gpu_end-gpu_start).count());
	}

	// cout<<"Finished benchmarking runs."<<endl;
	// float acceleration = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0f)/std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0f);
	// cout<<"The GPU acceleration  = "<<acceleration<<endl;
	
	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}