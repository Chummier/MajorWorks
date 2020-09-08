#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

/* Parallel program using MPI that takes a set of communication nodes,
Gives each one a unique id in a grid layout,
And then uses those ids to send data between nodes
*/

void modifyData(int* data, int n){
    for (int i = 0; i < n; i++){
        data[i] = i;
    }
}

int main(int argc, char* argv[]){
    int numprocs, rank;

    int dim[2], period[2];
    dim[0] = 4; dim[1] = 4; // 4x4 topology
    period[0] = 1; period[1] = 1; // Whether the grid wraps in each dim
    int reorder = 1; // rankings can be reordered
    int coordinates[2];
    int id;

    MPI_Comm newComm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &newComm);

    MPI_Cart_coords(newComm, rank, 2, coordinates);
    MPI_Cart_rank(newComm, coordinates, &id);

    int left, right;
    int top, bottom;

    MPI_Cart_shift(newComm, 1, 1, &left, &right);
    MPI_Cart_shift(newComm, 0, 1, &bottom, &top);
    double average = (left+right+top+bottom+4*id)/8.0;
    //printf("For process %d\n Averages are: %0.1f %0.1f %0.1f %0.1f\n", id, (left+id)/2.0, (right+id)/2.0, (top+id)/2.0, (bottom+id)/2.0);
    printf("For process %d      total average:%0.1f\n", id, average);
    //printf("Global rank: %d     Local rank: %d at (%d, %d)\n", rank, id, coordinates[0], coordinates[1]);

    MPI_Finalize();
    return 0;
}
