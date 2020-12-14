/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#include "../common/polybenchUtilFuncts.h"

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
//#define NX 4096
//#define NY 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r, int NX, int NY)
{
	int i, j;

  	for (i = 0; i < NX; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < NY; j++)
		{
      			A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
	
	for (i = 0; i < NY; i++)
	{
    		p[i] = i * M_PI;
	}
}



void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	cudaSetDevice( GPU_DEVICE );
}


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, int NX, int NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < NX; i++)
		{
			s[j] += A[i * NY + j] * r[i];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q, int NX, int NY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}



void bicgCuda(DATA_TYPE* A_gpu, DATA_TYPE* r_gpu, DATA_TYPE* s_gpu, DATA_TYPE* p_gpu, DATA_TYPE* q_gpu,
			 int NX, int NY)
{
  //double t_start, t_end;
  cudaEvent_t start, end;
  float time;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  //t_start = rtclock();
	bicg_kernel1<<< grid1, block >>>(A_gpu, r_gpu, s_gpu, NX, NY);
  cudaDeviceSynchronize();
	bicg_kernel2<<< grid2, block >>>(A_gpu, p_gpu, q_gpu, NX, NY);
  cudaDeviceSynchronize();
  //t_end = rtclock();

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);

  fprintf(stdout, "%0.6lf\n", time);
}


int main(int argc, char** argv)
{
  if(argc < 2){
    printf("please no troll\n");
    return 1;
  }

  int NX = atoi(argv[1]); 
  int NY = atoi(argv[1]);
  
	DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;
 	
	cudaMallocManaged(&A, NX*NY*sizeof(DATA_TYPE));
	cudaMallocManaged(&r, NX*sizeof(DATA_TYPE));
	cudaMallocManaged(&s, NY*sizeof(DATA_TYPE));
	cudaMallocManaged(&p, NY*sizeof(DATA_TYPE));
	cudaMallocManaged(&q, NX*sizeof(DATA_TYPE));

	init_array(A, p, r, NX, NY);
	GPU_argv_init();
	bicgCuda(A, r, s, p, q, NX, NY);

	cudaFree(A);
	cudaFree(r);
	cudaFree(s);
	cudaFree(p);
	cudaFree(q);

  	return 0;
}

