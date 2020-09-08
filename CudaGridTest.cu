#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <vector>

__global__ void simulateBoard(int* grid, int m, int n) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	int count = 0;

	// Upper left corner
	if (id == 0) {
		count += grid[id + 1];
		count += grid[id + n];
		count += grid[id + n + 1];
	}
	// Upper right corner
	else if (id == n - 1) {
		count += grid[id - 1];
		count += grid[id + n - 1];
		count += grid[id + n];
	}
	// Bottom left corner
	else if (id == n * (m - 1)) {
		count += grid[id - n];
		count += grid[id - n + 1];
		count += grid[id + 1];
	}
	// Bottom right corner
	else if (id == (n*m) - 1) {
		count += grid[id - n - 1];
		count += grid[id - n];
		count += grid[id - 1];
	}
	// Top edge
	else if (id < n) {
		count += grid[id - 1];
		count += grid[id + 1];
		count += grid[id + n - 1];
		count += grid[id + n];
		count += grid[id + n + 1];
	}
	// Right edge
	else if ((id + 1) % n == 0) {
		count += grid[id - n - 1];
		count += grid[id - n];
		count += grid[id - 1];
		count += grid[id + n - 1];
		count += grid[id + n];
	}
	// Left edge
	else if (id % n == 0) {
		count += grid[id - n];
		count += grid[id - n + 1];
		count += grid[id + 1];
		count += grid[id + n];
		count += grid[id + n + 1];
	}
	// Bottom edge
	else if (id > n*(m - 1)) {
		count += grid[id - 1];
		count += grid[id + 1];
		count += grid[id - n - 1];
		count += grid[id - n];
		count += grid[id - n + 1];
	}
	// Middle
	else {
		count += grid[id - n - 1];
		count += grid[id - n];
		count += grid[id - n + 1];
		count += grid[id - 1];
		count += grid[id + 1];
		count += grid[id + n - 1];
		count += grid[id + n];
		count += grid[id + n + 1];
	}

	if (grid[id] == 1) {
		if (count < 2) {
			grid[id] = 0;
		}
		else if (count >= 4) {
			grid[id] = 0;
		}
	}
	else {
		if (count == 2 || count == 3) {
			grid[id] = 1;
		}
	}

}

int main()
{
	clock_t timer;
	int M, N, K;
	printf("Enter M, N, and K\n");
	scanf("%d", &M);
	scanf("%d", &N);
	scanf("%d", &K);

	srand(time(0));

	int* board = (int*)malloc(sizeof(int)*M*N);

	int* d_board;
	cudaMalloc(&d_board, sizeof(int)*M*N);

	for (int i = 0; i < M*N; i++) {
		board[i] = rand() % 3 == 0;
	}
	for (int i = 0; i < M*N; i++) {
		if (i % N == 0) {
			printf("\n");
		}
		printf("%d ", board[i]);
	}
	printf("\n");

	timer = clock();
	for (int i = 0; i < K; i++) {
		cudaMemcpy(d_board, board, sizeof(int)*M*N, cudaMemcpyHostToDevice);
		simulateBoard << <1, N*M >> > (d_board, M, N);

		cudaMemcpy(board, d_board, sizeof(int)*M*N, cudaMemcpyDeviceToHost);
		for (int i = 0; i < M*N; i++) {
			if (i % N == 0) {
				printf("\n");
			}
			printf("%d ", board[i]);
		}
		printf("\n");
	}
	timer = clock() - timer;
	printf("Took %9.6f seconds\n", (float)timer / CLOCKS_PER_SEC);
	
    return 0;
}