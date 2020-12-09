/*
** This program finds out the transfer bandwidth for a given transfer size (cudaMemcpy host to device).
*/

#include <stdio.h>
#define PG (4*1024)

__global__ void add_kern(float *x)
{
  int current = 0;
  for (int i = 0; i < 9; i++) { 
    for (; current < (int)(1024 * (1<<(i+2))); current+=(int)(1024 * (1<<(i+2)))){
      x[i] += (int)(1024*pow(2.0,(i+2)));
    }
  }
}

int main(void)
{
  int N = 2044*1024;
  float *x;

	cudaMallocManaged( &x, N*sizeof(float) );
  add_kern<<<1,1>>>(x);

  printf("x: %f\n", x[0]);
  // Free memory
  cudaFree(x);

  return 0;
}

