#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>

using namespace std;

int cpu_implementation (vector<int> &nums) {
    int sum = 0;
    for(int num : nums) sum += num;
    return sum;
}

__global__ void paralllel_sum_reduction(int *nums, int *nums_r) {

    __shared__ int partial_sum[256];

    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    partial_sum[threadIdx.x] = nums[i] + nums[i + blockDim.x];
    __syncthreads();

    for(int stride = blockDim.x/2;stride>0;stride>>=1) {
        if(threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        nums_r[blockIdx.x] = partial_sum[0];
    }

}

int main() {

    const int N = 1 << 16;
    const size_t byteSize = N * sizeof(int);

    vector<int> nums(N, 1);
    vector<int> nums_r(N);
    int *d_nums, *d_nums_r;

    cudaMalloc(&d_nums, byteSize);
    cudaMalloc(&d_nums_r, byteSize);
    cudaMemcpy(d_nums, nums.data(), byteSize, cudaMemcpyHostToDevice);

    const int THREADS = 256;
    const int BLOCKS =  N/(THREADS*2);

    paralllel_sum_reduction<<<BLOCKS, THREADS>>>(d_nums, d_nums_r);
    paralllel_sum_reduction<<<1, THREADS>>>(d_nums_r, d_nums_r);
    cudaDeviceSynchronize();

    cudaMemcpy(nums_r.data(), d_nums_r, byteSize, cudaMemcpyDeviceToHost);
    int sum = cpu_implementation(nums);
    assert(nums_r[0] == sum);
    cout << "Results match.\n" ;

    return 0;
}