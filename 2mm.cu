/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
# define NI 2048
# define NJ 2048
# define NK 2048
# define NL 2048

#define NUM_ITERATIONS 10

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NI + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < NL; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}

__global__ void mm2_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{ 
		int k;
		for (k = 0; k < NK; k++)
		{
			C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

__global__ void mm2_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{ 
		int k;
		for (k = 0; k < NJ; k++)
		{
			E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
		}
	}
}

void mm2Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	mm2_kernel1<<<grid1,block>>>(A, B, C);
	cudaDeviceSynchronize();
	mm2_kernel2<<<grid2,block>>>(C, D, E);
	cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{	
	DATA_TYPE* C;
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* D;
	DATA_TYPE* E;
	cudaEvent_t start, end;
	float time, average_time = 0;

#ifndef UNMANAGED
	cudaMallocManaged( &A, NI*NJ*sizeof(DATA_TYPE) );
	cudaMallocManaged( &B, NI*NJ*sizeof(DATA_TYPE) );
	cudaMallocManaged( &C, NI*NJ*sizeof(DATA_TYPE) );
	cudaMallocManaged( &D, NI*NJ*sizeof(DATA_TYPE) );
	cudaMallocManaged( &E, NI*NJ*sizeof(DATA_TYPE) );
	//initialize the arrays
  	init_array(A, B, C, D);

	for (int i = 0; i < NUM_ITERATIONS + 1; ++i) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		mm2Cuda(A, B, C, D, E);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		if (i > 0) {
			// first iteration warms up the GPU
			average_time += time / NUM_ITERATIONS;
		}
	}
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
	cudaFree(E);
#else
	DATA_TYPE *gA, *gB, *gC, *gD, *gE;
	cudaMalloc((void **)&gA, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&gB, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&gC, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&gD, sizeof(DATA_TYPE) * NJ * NL);
	cudaMalloc((void **)&gE, sizeof(DATA_TYPE) * NI * NL);
	
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	DATA_TYPE* E_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));


	//initialize the arrays
  	init_array(A, B, C, D);

	cudaMemcpy(gA, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(gB, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(gC, C, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(gD, D, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(gE, E, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);

	for (int i = 0; i < NUM_ITERATIONS + 1; ++i) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		mm2Cuda(gA, gB, gC, gD, gE);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		if (i > 0) {
			// first iteration warms up the GPU
			average_time += time / NUM_ITERATIONS;
		}
	}
	cudaMemcpy(E_outputFromGpu, gE, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

	cudaFree(gA);
	cudaFree(gB);
	cudaFree(gC);
	cudaFree(gD);
	cudaFree(gE);
	free(C);
	free(A);
	free(B);
	free(D);
	free(E);
	free(E_outputFromGpu);
#endif
	printf("%f\n", average_time);
  	return 0;
}

