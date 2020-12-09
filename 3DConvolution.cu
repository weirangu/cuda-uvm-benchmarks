/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

/* Problem size */
#define NI 512l
#define NJ 512l
#define NK 512l

#define NUM_ITERATIONS 10

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE* A)
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


__global__ void convolution3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int i)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;


	if ((i < (NI-1)) && (j < (NJ-1)) &&  (k < (NK-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
	}
}


void convolution3DCuda(DATA_TYPE* A, DATA_TYPE* B)
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NK) / ((float)block.x) )), (size_t)(ceil( ((float)NJ) / ((float)block.y) )));

	int i;
	for (i = 1; i < NI - 1; ++i) // 0
	{
		convolution3D_kernel<<< grid, block >>>(A, B, i);
	}

	cudaDeviceSynchronize();
}


int main(int argc, char *argv[])
{
	DATA_TYPE* A;
	DATA_TYPE* B;  

	float average_time = 0;

	cudaEvent_t start, end;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
#ifndef UNMANAGED
	cudaMallocManaged( &A, NI*NJ*NK*sizeof(DATA_TYPE) );
	cudaMallocManaged( &B, NI*NJ*NK*sizeof(DATA_TYPE) );
	//initialize the arrays
	init(A);
	for (int i = 0; i < NUM_ITERATIONS + 1; ++i) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		convolution3DCuda(A, B);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		if (i > 0) {
			// first iteration warms up the GPU
			average_time += time / NUM_ITERATIONS;
		}
	}
#else
	DATA_TYPE *gA, *gB;
	cudaMalloc( &gA, NI*NJ*NK*sizeof(DATA_TYPE) );
	cudaMalloc( &gB, NI*NJ*NK*sizeof(DATA_TYPE) );
	A = (DATA_TYPE *) malloc( NI*NJ*NK*sizeof(DATA_TYPE) );
	B = (DATA_TYPE *) malloc( NI*NJ*NK*sizeof(DATA_TYPE) );
	//initialize the arrays
	init(A);
	cudaMemcpy(gA, A, NI*NJ*NK*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	for (int i = 0; i < NUM_ITERATIONS + 1; ++i) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		convolution3DCuda(gA, gB);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		if (i > 0) {
			// first iteration warms up the GPU
			average_time += time / NUM_ITERATIONS;
		}
	}
	cudaMemcpy(B, gB, NI*NJ*NK*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#endif
	printf("%f\n", average_time);
#ifndef UNMANAGED
	cudaFree(A);
	cudaFree(B);
#else
	cudaFree(gA);
	cudaFree(gB);
	free(A);
	free(B);
#endif
	return 0;
}

