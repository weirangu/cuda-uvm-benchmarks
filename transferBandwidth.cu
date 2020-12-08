/*
** This program finds out the transfer bandwidth for a given transfer size (cudaMemcpy host to device).
*/

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
  float *x, *d_x;

  x = (float*)malloc(N*sizeof(float));
  cudaMalloc(&d_x, N*sizeof(float));

  for (int i = 0; i < 9; i++) { 
    cudaMemcpy((d_x+current), (x+current), (int)(1024*pow(2.0,(i+2))), cudaMemcpyHostToDevice);
    add_kern<<<10,10>>>(x);
  }
  
  // Free memory
  cudaFree(d_x);
  free(x);

  return 0;
}
