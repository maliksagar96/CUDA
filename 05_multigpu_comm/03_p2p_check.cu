#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
  int ngpus=0;
  cudaGetDeviceCount(&ngpus);

  cout<<"Number of GPUs : "<<ngpus<<"\n\n";

  for(int i=0;i<ngpus;i++)
  {
    for(int j=0;j<ngpus;j++)
    {
      if(i==j) continue;

      int canAccess=0;
      cudaDeviceCanAccessPeer(&canAccess,i,j);

      cout<<"GPU"<<i<<" -> GPU"<<j<<" : "<<canAccess<<"\n";
    }
  }

  return 0;
}