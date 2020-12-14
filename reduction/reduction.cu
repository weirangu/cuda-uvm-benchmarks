// Reductions based off of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#define CEIL(x,y) (1 + (((x) - 1) / (y)))

void init(double* A, int size, int dims)
{
	for (int i = 0; i < size; ++i) {
		A[i] = rand();
	}
}

__global__ void reduce_kernel(double *input, double *output, int num_arr) {
	extern __shared__ double shared_mem[];
	int dim = blockIdx.y;
	input += dim * gridDim.x * 2;

	int thread_id = threadIdx.x;
	int i = blockIdx.x * blockDim.x * 2 + thread_id;

	if (i < num_arr) {
		shared_mem[thread_id] = input[i];
	} else {
		shared_mem[thread_id] = 0;
	}
	__syncthreads();

	if (i + blockDim.x < num_arr) {
		shared_mem[thread_id] += input[i + blockDim.x];
	}
	__syncthreads();

	for (int j = blockDim.x / 2; j > 0; j /= 2) {
		if (thread_id < j) {
			shared_mem[thread_id] += shared_mem[thread_id + j];
		}
		__syncthreads();
	}

	if (thread_id == 0) {
		output[dim] += shared_mem[0];
	}
}

void reduce_cuda(int num_dims, int num_arr)
{
	int total_size = num_dims * num_arr * sizeof(double);
	double *arr;
	double *sum;

#ifdef UNMANAGED
	arr = (double *) malloc(total_size);
	sum = (double *) malloc(sizeof(double) * num_dims);
#else
	cudaMallocManaged(&arr, total_size);
	cudaMallocManaged(&sum, sizeof(double) * num_dims);
	cudaMemset(sum, 0, sizeof(double) * num_dims);
#endif

	init(arr, num_dims * num_arr, num_arr);

	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 grid(CEIL(num_arr, 512), num_dims);

#ifdef UNMANAGED
	double *arr_gpu;
	double *sum_gpu;
	cudaMalloc(&arr_gpu, total_size);
	cudaMalloc(&sum_gpu, sizeof(double) * num_dims);
	cudaEventRecord(start);
	cudaMemcpy(arr_gpu, arr, total_size, cudaMemcpyHostToDevice);
	cudaMemset(sum_gpu, 0, sizeof(double) * num_dims);
	reduce_kernel<<<grid, 512, sizeof(double) * 512>>>(arr_gpu, sum_gpu, num_arr);
	cudaEventRecord(end);

	cudaMemcpy(sum, sum_gpu, sizeof(double) * num_dims, cudaMemcpyDeviceToHost);
#else
	cudaEventRecord(start);
	reduce_kernel<<<grid, 512, sizeof(double) * 512>>>(arr, sum, num_arr);
	cudaEventRecord(end);
#endif

	cudaEventSynchronize(end);
	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, end);
	printf("%f \n", elapsed);

#ifdef UNMANAGED
	free(arr);
	free(sum);
	cudaFree(arr_gpu);
	cudaFree(sum_gpu);
#else
	cudaFree(arr);
	cudaFree(sum);
#endif
}


int main(int argc, char *argv[])
{
	// handle command line arguments
	if (argc != 2) {
		printf("Incorrect command line arguments! Need to provide num_arr.\n");
		return -1;
	}
	// int num_dims = strtol(argv[1], NULL, 10);
	int num_dims = 1;
	int num_arr = strtol(argv[1], NULL, 10);

	reduce_cuda(num_dims, num_arr);
	return 0;
}
