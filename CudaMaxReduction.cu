#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <vector>

std::vector<float> resVector;

__global__ void gpuFindMax(float* vector, long int n, float* res) {
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	sdata[tid] = vector[0];
	__syncthreads();

	for (unsigned int s = blockDim.x; s != 0; s /= 2) {
		if (tid < s) {
			if (sdata[tid + s] > sdata[tid]) {
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		res[blockIdx.x] = sdata[0];
	}
}

float cpuFindMax(float* vector, long int n) {
	float max = vector[0];
	for (long int i = 1; i < n; i++) {
		if (vector[i] > max) {
			max = vector[i];
		}
	}
	return max;
}

float cpuParallelMax(float* vector, long int n) {
	float max = vector[0];

#pragma omp parallel private(max)
	{
#pragma omp for
		for (long int i = 1; i < n; i++) {
			if (vector[i] > max) {
#pragma omp critical
				max = vector[i];
			}
		}
	}

	return max;
}

int main()
{
	clock_t timer;
	double elapsedTime;
	long int n;
	float* vector;
	float res;

	float* resArray;
	float* newRes;

	resArray = (float*)malloc(65336 * sizeof(float));
	newRes = (float*)malloc(65336 * sizeof(float));

	omp_set_num_threads(16);

	srand(time(0));

	printf("Enter a vector size\n");
	scanf("%d", &n);
	vector = (float*)malloc(sizeof(float)*n);

	for (long int i = 0; i < n; i++) {
		vector[i] = (float)rand();
		vector[i] = vector[i] / 3.317f;
	}
	printf("\n");

	timer = clock();
	res = cpuFindMax(vector, n);
	timer = clock() - timer;
	elapsedTime = (double)(timer) / CLOCKS_PER_SEC;
	printf("Max: %9.6f. Sequential CPU took %f seconds\n", res, elapsedTime);

	timer = clock();
	res = cpuParallelMax(vector, n);
	timer = clock() - timer;
	elapsedTime = (double)(timer) / CLOCKS_PER_SEC;
	printf("Max: %9.6f. Parallel CPU took %f seconds\n", res, elapsedTime);

	res = 0;
	timer = clock();
	gpuFindMax<<<1, n>>>(vector, n, resArray);
	timer = clock() - timer;
	elapsedTime = (double)(timer) / CLOCKS_PER_SEC;

	cudaMemcpy(newRes, resArray, 65336, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 65336; i++) {
		if (newRes[i] > res) {
			res = newRes[i];
		}
	}
	printf("Max: %9.6f. GPU took %f seconds\n", res, elapsedTime);

    return 0;
}