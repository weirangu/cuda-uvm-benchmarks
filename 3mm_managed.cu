/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
//# define NI 512
//# define NJ 512
//# define NK 512
//# define NL 512
//# define NM 512

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

int NI;  
int NJ; 
int NK; 
int NL; 
int NM; 

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}
	
__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E,int NI, int NJ, int NK, int NL, int NM)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{
		int k;
		for(k=0; k < NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

	
__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F,int NI, int NJ, int NK, int NL, int NM)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NJ) && (j < NL))
	{
		int k;
		for(k=0; k < NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G, int NI, int NJ, int NK, int NL, int NM)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{
		int k;
		for(k=0; k < NJ; k++)
		{
			G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
		}
	}
}


void mm3Cuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu, DATA_TYPE* E_gpu, DATA_TYPE* F_gpu, 
		DATA_TYPE* G_gpu, int NI, int NJ, int NK, int NL, int NM)
{
	cudaEvent_t start, end;
	float time;

	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu,NI, NJ, NK, NL, NM);
	cudaDeviceSynchronize();
	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu,NI, NJ, NK, NL, NM);
	cudaDeviceSynchronize();
	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu,NI, NJ, NK, NL, NM);
	cudaDeviceSynchronize();

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

  NI = atoi(argv[1]); 
  NJ = atoi(argv[1]);
  NK = atoi(argv[1]);
  NL = atoi(argv[1]);  
  NM = atoi(argv[1]);

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;

	cudaMallocManaged(&A, NI*NK*sizeof(DATA_TYPE));
	cudaMallocManaged(&B, NK*NJ*sizeof(DATA_TYPE));
	cudaMallocManaged(&C, NJ*NM*sizeof(DATA_TYPE));
	cudaMallocManaged(&D, NM*NL*sizeof(DATA_TYPE));
	cudaMallocManaged(&E, NI*NJ*sizeof(DATA_TYPE));
	cudaMallocManaged(&F, NJ*NL*sizeof(DATA_TYPE));
	cudaMallocManaged(&G, NI*NL*sizeof(DATA_TYPE));

	init_array(A, B, C, D);

	mm3Cuda(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM); 

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
	cudaFree(E);
	cudaFree(F);
	cudaFree(G);

	return 0;
}

