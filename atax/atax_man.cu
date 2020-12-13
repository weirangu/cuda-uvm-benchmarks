/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
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



void init_array(DATA_TYPE *x, DATA_TYPE *A, int NX, int NY)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp, int NX, int NY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp, int NX, int NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}


void ataxGpu(DATA_TYPE* A_gpu, DATA_TYPE* x_gpu, DATA_TYPE* y_gpu, DATA_TYPE* tmp_gpu, DATA_TYPE* y_outputFromGpu, int NX, int NY)
{
  //cudaEvent_t start, end;
  //float time;
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

  double t_start, t_end;


  //cudaEventCreate(&start);
  //cudaEventCreate(&end);
  //cudaEventRecord(start);

  t_start = rtclock();
	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu, NX, NY);
  cudaDeviceSynchronize();
	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu, NX, NY);
  cudaDeviceSynchronize();
  t_end = rtclock();
	fprintf(stdout, "%0.6lfs\n", t_end - t_start);

  //cudaEventRecord(end);
  //cudaEventSynchronize(end);
  //cudaEventElapsedTime(&time, start, end);

  //fprintf(stdout, "%0.6lf\n", time);

	
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);
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
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;

	cudaMallocManaged(&A, NX*NY*sizeof(DATA_TYPE));
	cudaMallocManaged(&x, NY*sizeof(DATA_TYPE));
	cudaMallocManaged(&y, NY*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	init_array(x, A, NX, NY);

	GPU_argv_init();
	ataxGpu(A, x, y, tmp, y_outputFromGpu, NX, NY);
	

	cudaFree(A);
	cudaFree(x);
	cudaFree(y);
	free(y_outputFromGpu);
	free(tmp);

  	return 0;
}

