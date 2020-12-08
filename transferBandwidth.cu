#define PG (4*1024)
#include <stdio.h>

int main(void)
{
  int N = 2044*1024;
  float *x, *d_x;

  x = (float*)malloc(N*sizeof(float));
 
  cudaMalloc(&d_x, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 0;
  }


	cudaEvent_t start, end;
	float time;
  int current = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < 9; i++) { 
    cudaMemcpy((d_x+current), (x+current), (int)(1024*pow(2.0,(i+2))), cudaMemcpyHostToDevice);
    current += (int)(1024*pow(2.0,(i+2)));
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);

  printf("time: %f\n", time);
  
  // Free memory
  cudaFree(d_x);
  free(x);

  return 0;
}

