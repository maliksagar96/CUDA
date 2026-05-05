/*
Step 5 is using shared memory.
*/

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random> 

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
 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int k = blockIdx.x * blockDim.x + threadIdx.x; 

    // Shared memory: each block holds one tile of A (for current i, across j in tile)
    // Tile size in j-dimension = blockDim.x (we'll use x-dim for tiling j)
    extern __shared__ float tileA[];

    float sum = 0.0f;

    // Loop over tiles in the j-dimension (common dim of A and B)
    for (int tile_j = 0; tile_j < n2; tile_j += blockDim.x) {
        // Each thread in the block loads one element of A's row i, if in range
        int j = tile_j + threadIdx.x; // j index covered by this thread in this tile
        if (i < n1 && j < n2) {
            tileA[threadIdx.x] = a[i * n2 + j];
        } else {
            tileA[threadIdx.x] = 0.0f; // padding to avoid uninitialized reads
        }

        __syncthreads(); // Ensure all threads in block have loaded tileA

        // Now compute partial dot product using this tile of A and global B
        for (int j_local = 0; j_local < blockDim.x && (tile_j + j_local) < n2; ++j_local) {
            // tileA[j_local] corresponds to a[i][tile_j + j_local]
            // b[(tile_j + j_local) * n3 + k] is the corresponding element in B
            sum += tileA[j_local] * b[(tile_j + j_local) * n3 + k];
        }

        __syncthreads(); // Prevent race on next tile load (conservative but safe)
    }

    if (i < n1 && k < n3) {
        c[i * n3 + k] = sum;
    }
}

int main() {

	int n1 =256, n2 =256, n3 =256;

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
	(n1 + threadsPerBlock.y - 1) / threadsPerBlock.y  // rows
	);

	//initialise a, b and c
	std::random_device rd;  // non-deterministic seed (hardware if available)
	std::mt19937 gen(rd()); // Mersenne Twister engine
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);

	
	// Warm up runs
	for(int i = 0;i<3;i++) {
		for(int i = 0; i < n1*n2; i++) {
		a[i] = dis(gen);
		}

		for(int i = 0;i<n2*n3;i++) {
			b[i] = dis(gen);
		}

		for(int i = 0;i<n1*n3;i++) {
			c[i] = 0;
		}
		cpu_mat_mult(a, b, c, n1, n2, n3);

		CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_c, c, c_sz, cudaMemcpyHostToDevice));

		gpu_mat_mult<<<numberOfBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n1, n2, n3);
	}

	CUDA_CHECK(cudaMemcpy(d_a, a, a_sz, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b, b, b_sz, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_c, c, c_sz, cudaMemcpyHostToDevice));

	gpu_mat_mult<<<numberOfBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n1, n2, n3);

	CUDA_CHECK(cudaMemcpy(gpu_result, d_c, c_sz, cudaMemcpyDeviceToHost));

	//result comparison

	bool match = true;

	for(int i = 0;i<n1*n3;i++) {
		if(fabs(c[i] - gpu_result[i]) > 1e-6) {
			cout<<"result mismatch at i = "<<i<<endl;
			match = false;
			break;
		}
	}

	if(match) {
		cout<<"The results match"<<endl;
	}

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	return 0;
}