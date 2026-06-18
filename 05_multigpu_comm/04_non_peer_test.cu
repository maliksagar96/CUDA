#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void init(int *data,int n)
{
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n) data[idx]=idx;
}

int main()
{
  const int N=1024;

  int *d_gpu0;
  int *d_gpu1;
  int *h_buf;
  int *h_check;

  h_buf=new int[N];
  h_check=new int[N];

  cudaSetDevice(0);
  cudaMalloc(&d_gpu0,N*sizeof(int));

  cudaSetDevice(1);
  cudaMalloc(&d_gpu1,N*sizeof(int));

  cudaSetDevice(0);
  init<<<(N+255)/256,256>>>(d_gpu0,N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_buf,d_gpu0,N*sizeof(int),cudaMemcpyDeviceToHost);

  cudaSetDevice(1);
  cudaMemcpy(d_gpu1,h_buf,N*sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcpy(h_check,d_gpu1,N*sizeof(int),cudaMemcpyDeviceToHost);

  bool pass=true;

  for(int i=0;i<N;i++)
  {
    if(h_check[i]!=i)
    {
      pass=false;
      cout<<"Mismatch at "<<i<<" "<<h_check[i]<<"\n";
      break;
    }
  }

  if(pass)
    cout<<"GPU0 -> Host -> GPU1 transfer successful\n";

  cudaSetDevice(0);
  cudaFree(d_gpu0);

  cudaSetDevice(1);
  cudaFree(d_gpu1);

  delete[] h_buf;
  delete[] h_check;

  return 0;
}