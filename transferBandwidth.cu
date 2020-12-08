#define PG (4*1024)

int main(void)
{
  int N = 2044*1024;
  float *x, *d_x;

  x = (float*)malloc(N*sizeof(float));
 
  cudaMalloc(&d_x, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 0;
  }

  int current = 0;
  for (int i = 0; i < 9; i++) { 
    cudaMemcpy((d_x+current), (x+current), (int)(1024*pow(2.0,(i+2))), cudaMemcpyHostToDevice);
    current += (int)(1024*pow(2.0,(i+2)));
  }
  
  // Free memory
  cudaFree(d_x);
  free(x);

  return 0;
}

