#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int sqrtP;

// An incomplete implementation of parallel matrix multiplication using MPI

void multiply(int** m_A, int** m_B, int** m_C, int rank, int n, int procs){
    int** recvMatrix;

    for (int k = 0; k < sqrtP; k++){

    }
    
    MPI_Send(&val, 1, MPI_INT, (rank+1)%numprocs, 1, MPI_COMM_WORLD);
    MPI_Recv(&val, 1, MPI_INT, (rank-1+numprocs)%numprocs, 1, MPI_COMM_WORLD, &status);
}

int main(int argc, char* argv[]){
    int numprocs, rank;
    int n;
    int** A;
    int** B;
    int** C;

    if (argc > 1){
        n = atoi(argv[1]);
    } else {
        n = 40;
    }

    srand(time(0));

    A = (int**)malloc(n*sizeof(int*));
    B = (int**)malloc(n*sizeof(int*));
    C = (int**)malloc(n*sizeof(int*));

    for (int i = 0 ; i < n; i++){
        A[i] = (int*)malloc(n*sizeof(int));
        B[i] = (int*)malloc(n*sizeof(int));
        C[i] = (int*)malloc(n*sizeof(int));
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = rand() % 150;
            B[i][j] = rand() % 150;
            C[i][j] = 0;
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int square = 0;
    sqrtP = numprocs/2;
    while (sqrtP != square){
        square = sqrtP;
        sqrtP = (numprocs/square + square)/2;
    }

    int row = rank / sqrtP;
    int col = rank % sqrtP;

    int* P_A = (int*)malloc(((int)(n*n/numprocs/numprocs))*sizeof(int));
    int* P_B = (int*)malloc(((int)(n*n/numprocs/numprocs))*sizeof(int));

    for (int i = 0; i < n/numprocs; i++){
        for (int j = 0; j < n/numprocs; j++){
            P_A[i*n/numprocs+j] = A[i+n/numprocs*row][j+n/numprocs*col];
            P_B[i*n/numprocs+j] = B[i+n/numprocs*row][j+n/numprocs*col];
        }
    }

    for (int block = 0; block < sqrtP; block++){
        MPI_Send(P_A, n*n/numprocs/numprocs, MPI_INT, col*sqrtP+block, 1, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}
