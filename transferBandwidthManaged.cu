/*
** This program finds out the transfer bandwidth for a given transfer size (cudaMemcpy host to device).
*/

#include <stdio.h>
#define PG (4*1024)

__global__ void add_kern(float *x)
{
  for (int i = 0; i < 9; i++) { 
    x[i] += (int)(1024*pow(2.0,(i+2)));
  }

}

int main(void)
{
  int N = 2044*1024;
  float *x;

	cudaMallocManaged( &x, N*sizeof(float) );
  add_kern<<<10,10>>>(x);

  
  printf("x: %f\n", x[0]);
  // Free memory
  cudaFree(x);

  return 0;
}
