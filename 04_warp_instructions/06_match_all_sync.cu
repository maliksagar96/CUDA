/*
  Value in all the threads have to match.
*/

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matchAllDemo() {

  int tid = threadIdx.x;

  // Imagine each thread represents a student section
  // Threads 0-31 are inside one warp
  int section;

  // All threads belong to same section
  section = 101;
  // if(threadIdx.x == 0) section = 202; //uncomment this to see the use of match_all_sync

  int allSame;
  unsigned int mask = __match_all_sync(0xffffffff, section, &allSame);

  printf("Thread %2d | Section = %3d | All Same = %d\n", tid, section, allSame);
}


int main() {

  matchAllDemo<<<1, 32>>>();

  cudaDeviceSynchronize();

  return 0;
}