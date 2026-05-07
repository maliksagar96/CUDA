/*
	Profile with:
	ncu \
--kernel-name gpuTranspose \
--replay-mode application \
--metrics \
gpu__time_duration.sum,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum.per_second,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warps_eligible.avg.per_cycle,\
smsp__warps_issued.avg.per_cycle,\
smsp__inst_executed.avg.per_cycle_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
gpu__global_load_efficiency.avg.pct,\
gpu__global_store_efficiency.avg.pct \
./06_gpu_transpose

compile with 
nvcc -arch=sm_75 06_gpu_transpose.cu -o 06_gpu_transpose

*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <ctime>

using namespace std;

void matrixTranspose(vector<int>& matrix, vector<int>& transpose, int nx, int ny) {
    
	for(int iy = 0; iy < ny;iy++) {
		for(int ix = 0;ix < nx;ix++) {
			transpose[ix * ny + iy] = matrix[iy * nx + ix];
		}
	}    
}

__global__ void gpuTranspose(int *matrix, int *transpose, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny) {
        transpose[ix * nx + iy] = matrix[iy * nx + ix];
    }
}

__global__ void gpuTranspose_read_column_write_row(int *matrix, int *transpose, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny) {
        // transpose[ix * nx + iy] = matrix[iy * nx + ix];
				transpose[iy * nx + ix] = matrix[ix * nx + iy];
    }
}


int main() {

    int nx = 1024, ny = 1024;
    int NM = nx*ny;

    int block_x = 128;
    int block_y = 8;
    int size = nx * ny;
    int byteSize = size * sizeof(int);

    vector<int> matrix(NM, 0);
    vector<int> transpose(NM, 0), gpuResult(NM, 0);


    srand(time(0));

    for(int i = 0;i<NM;i++) {
	    matrix[i] = rand()%100;
    }

    matrixTranspose(matrix, transpose, nx, ny);

    dim3 block(block_x, block_y);
    dim3 grid((nx + block_x - 1 )/block_x, (ny + block_y - 1)/block_y);

    int *d_transpose, *d_matrix;
    cudaMalloc(&d_transpose, byteSize);
    cudaMalloc(&d_matrix, byteSize);
    cudaMemcpy(d_matrix, matrix.data(), byteSize, cudaMemcpyHostToDevice);

    // gpuTranspose<<<grid, block>>>(d_matrix, d_transpose, nx, ny);
		gpuTranspose_read_column_write_row<<<grid, block>>>(d_matrix, d_transpose, nx, ny);
    
    cudaMemcpy(gpuResult.data(), d_transpose, byteSize, cudaMemcpyDeviceToHost);
    
    for(int i = 0;i<NM;i++) {
        assert(transpose[i] == gpuResult[i]);
    }

    cout << "Results Match.\n";

    cudaFree(d_transpose);
    cudaFree(d_matrix);

    return 0;
}