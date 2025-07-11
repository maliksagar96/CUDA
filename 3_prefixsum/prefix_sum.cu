#include <stdio.h>

int main() {
  int n = 10;
  int* a = (int*)malloc(sizeof(int) * n);
  int* prefix_sum = (int*)malloc(sizeof(int) * n);
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
  }

  for (int i = 0; i < n; i++) {
    for(int j = 0; j <= i;j++) {
      prefix_sum[i] += a[j];
    }
  }

  for(int i = 0;i<n;i++) {
    printf("prefix sum[%d] = %d\n", i, prefix_sum[i]);
  }

  free(a);
  free(prefix_sum);
  
  return 0;
}