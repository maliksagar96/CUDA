/*
  The following is one way to assign a 2D matrix. But this gives out a memory which is not contiguous. 
  Hence this becomes inefficient for CUDA. 
  That's why we'll use the flattened 2D array.

  float **a = (float**)malloc(N*size(float*));
  
  for(int i = 0;i < N;i++) {
    a[i] = (float*)malloc(M*size(float));
  }

*/

/*

To insert the value in a flattened matrix we use 

a[i * M + j] = value, which puts the value in the ith row and jth column. 

*/


#include <stdio.h>

#define N 2
#define M 2

int main() {

  float *a = (float*)malloc(N*M*sizeof(float));




  return 0;
}