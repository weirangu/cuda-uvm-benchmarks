/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define NUM_ITERATIONS 10

/* Problem size */
#define NI 15000l
#define NJ 15000l

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

void init(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			A[i*NJ + j] = (DATA_TYPE)rand()/RAND_MAX;
        	}
    	}
}

__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
	{
		B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
			+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
			+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
	}
}


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B)
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
	
	Convolution2D_kernel<<<grid, block>>>(A, B);

	// Wait for GPU to finish before accessing on host
	// mock synchronization of memory specific to stream
	cudaDeviceSynchronize();
}


int main(int argc, char *argv[])
{
	DATA_TYPE* A;
	DATA_TYPE* B;  

	float average_time = 0;

	cudaEvent_t start, end;
	float time;
#ifndef UNMANAGED
	cudaMallocManaged( &A, NI*NJ*sizeof(DATA_TYPE) );
	cudaMallocManaged( &B, NI*NJ*sizeof(DATA_TYPE) );
	//initialize the arrays
	init(A);
	for (int i = 0; i < NUM_ITERATIONS + 1; ++i) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		convolution2DCuda(A, B);
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
	cudaMalloc( &gA, NI*NJ*sizeof(DATA_TYPE) );
	cudaMalloc( &gB, NI*NJ*sizeof(DATA_TYPE) );
	A = (DATA_TYPE *) malloc( NI*NJ*sizeof(DATA_TYPE) );
	B = (DATA_TYPE *) malloc( NI*NJ*sizeof(DATA_TYPE) );
	//initialize the arrays
	init(A);
	cudaMemcpy(gA, A, NI*NJ*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	for (int i = 0; i < NUM_ITERATIONS + 1; ++i) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		convolution2DCuda(gA, gB);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		if (i > 0) {
			// first iteration warms up the GPU
			average_time += time / NUM_ITERATIONS;
		}
	}
	cudaMemcpy(B, gB, NI*NJ*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
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

