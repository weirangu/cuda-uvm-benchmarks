// Courtesy of https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/
// REMOVE ME: Uncommnet the code only upon full implementation or get seg-fault
 
#include <iostream>
#include <math.h>
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(int argc, char** argv)
{
  
  if(argc < 2)
    return 0;

  int N = 1<<atoi(argv[1]);
  float *x, *y;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, end;
  float time;
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  //for(int i = 0; i < 5; i++){
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    fprintf(stdout, "%0.6lf\n", time);
  //}
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}

