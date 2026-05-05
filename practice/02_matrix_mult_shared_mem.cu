#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

void cpu_implementation(vector<int>& a, vector<int>& b, vector<int>& c, int N) {

    for(int i = 0;i<N;i++) {
        for(int j = 0;j<N;j++) {
            // cij = aik*bkj
            int sum = 0;
            for(int k = 0;k<N;k++) {
               sum += a[i*N + k] * b[k*N + j];
            }
             c[i*N +j] = sum;
        }
    }
}

__global__ void gpu_implementation(int *a, int *b, int *c, int N) {

    //Trace global thread ID
    int global_column = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    //trace local thread ID
    int local_thread_x = threadIdx.x;
    int local_thread_y = threadIdx.y;

    __shared__ int A_tile[16][16];
    __shared__ int B_tile[16][16];
    
    //Loop over tiles
    int sum = 0;
    int stride = 16;
    for(int tile = 0;tile < 64; tile++) {
        //load global information in shared memory.
        //Strided Columns, row will remain same while going over tiles
        A_tile[local_thread_y][local_thread_x] = a[(global_row * N) + (local_thread_x + stride * tile)];  

        //Strided Rows, column number will remain same while going over rows
        B_tile[local_thread_y][local_thread_x] = b[(tile*stride +  local_thread_y)*N + global_column];  

        __syncthreads();

        //compute sum over this tile.
        for(int k = 0;k<16;k++) {            
            //cij = aik*bkj
            sum += A_tile[local_thread_y][k] * B_tile[k][local_thread_x];            
        }

        __syncthreads();
    }
    
    if(global_row < N && global_column < N) {
        c[global_row * N + global_column] = sum;
    }
    
}

int main() {

    srand(time(nullptr));
    const int N = 1024;
    const int matrixSize = N * N;
    const size_t byteSize = static_cast<size_t>(matrixSize) * sizeof(int);

    vector<int> h_a(matrixSize);
    vector<int> h_b(matrixSize);
    vector<int> h_c(matrixSize);
    vector<int> gpu_c(matrixSize);

    std::generate(h_a.begin(), h_a.end(), [](){ return rand() % 10; });
    std::generate(h_b.begin(), h_b.end(), [](){return rand() % 10;});
    cpu_implementation(h_a,h_b,h_c,N);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, byteSize);
    cudaMalloc(&d_b, byteSize);
    cudaMalloc(&d_c, byteSize);

    cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), byteSize, cudaMemcpyHostToDevice);   

    dim3 threadPerBlock(16, 16);
    dim3 blocks((threadPerBlock.x + N - 1)/threadPerBlock.x, (threadPerBlock.y + N - 1)/threadPerBlock.y);
    // cout << "blocksidx.x = "<< blocks.x<< ", blocksidx.y = " << blocks.y << endl;
    // exit(0);
    gpu_implementation<<<blocks, threadPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize(); 

    cudaMemcpy(gpu_c.data(), d_c, byteSize, cudaMemcpyDeviceToHost);

    for(int i = 0;i<matrixSize;i++) {
        if(abs(h_c[i] - gpu_c[i]) > 1e-5) {
            cout << "Host and device results are not matching.\n";
            exit(0);
        }
    }

    cout << "Results Match.\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}