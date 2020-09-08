#include <stdio.h>
#include <stdlib.h>

// Declare printSudoku function
void printSudoku(int***);
// Declare solveSudoku function
int solveSudoku(int***);

//Helper functions. You can define any functions that can help your solve the problem

/*
The main program reads a text file containing the block values of the Sudoku grid.
It then saves each 3x3 block into a 2D array. The Sudoku grid is composed of 9 3x3 blocks.
DO NOT MODIFY THE MAIN FUNTION!!!
*/
int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <file name>\n", argv[0]);
		return 2;
	}
    int i, j;
    FILE *fptr;
    int ***blocks = (int***)malloc(9 * sizeof(int**));

    // Open file for reading
    fptr = fopen(argv[1], "r");
    if (fptr == NULL) {
        printf("Cannot Open File!\n");
        return 0;
    }

	// Read 9x9 blocks into 2D arrays
    for(i=0; i<9; i++)
    {
        *(blocks+i) = (int**)malloc(3 * sizeof(int*));
        printf("Reading numbers in block %d... \n", i+1);
        for(j=0; j<3; j++)
        {
            *(*(blocks+i)+j) = (int*)malloc(3 * sizeof(int));

                fscanf(fptr, "%d %d %d", *(*(blocks+i)+j), *(*(blocks+i)+j)+1, *(*(blocks+i)+j)+2);
                printf("%d %d %d\n", *(*(*(blocks+i)+j)), *(*(*(blocks+i)+j)+1), *(*(*(blocks+i)+j)+2));
        }
    }
	
	// Print out original Sudoku grid
    printf("Printing Sudoku before being solved:\n");
    printSudoku(blocks);

	// Call solveSudoku and print out result
    printf("\nSolving Sudoku...\n\n");
    if(solveSudoku(blocks)){
        printf("Sudoku solved!\n");
        printSudoku(blocks);
    }
    else
        printf("This Sudoku cannot be solved!\n");

    return 0;
}


void printSudoku(int*** arr){
	// This function will print out the complete Sudoku grid (arr). It must produce the output in the SAME format as the samples in the instructions. 	
	
	// Your implementation here
    	
	int i, j, k, t;
	for (t = 0; t < 3; t++){
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				printf("%d ", *(*(*(arr+j+(3*t))+i)+k));
			}
			printf(" | ");
		}
		printf("\n");
	}
	if (t != 2){printf("---------------------------\n");}
	}
}

int checkSquare(int*** grid, int square, int row, int col, int value){
	int i, j;
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			if (*(*(grid+square)+i)+j != *(*(grid+square)+row)+col){
				if (*(*(*(grid+square)+i)+j) == value){
					return 0;
				}
			}
		}
	}
	return 1;
}

int checkRow(int*** grid, int square, int row, int col, int value){
	int i, j, k;
	int r;
	if (square > 5){
		r = 6;
	} else if (square > 2){
		r = 3;
	} else {
		r = 0;
	}
	for (i = 0; i < 3; i++){
		for (j = r; j < r+3; j++){
			for (k = 0; k < 3; k++){
				if (*(*(grid+j)+row)+k != *(*(grid+square)+row)+col){
					if (*(*(*(grid+j)+row)+k) == value){
						return 0;
					}
				}
			}
		}
	}
	return 1;
}

int checkColumn(int*** grid, int square, int row, int col, int value){
	int i, j, s;
	if (square > 5) {
		s = square - 6;
	} else if (square > 2){
		s = square - 3;
	} else {
		s = square;
	}
	for (i = s; i < 9; i+=3){
		for (j = 0; j < 3; j++){
			if (*(*(grid+i)+j)+col != *(*(grid+square)+row)+col){
				if (*(*(*(grid+i)+j)+col) == value){
					return 0;
				}
			}
		}
	}
	return 1;
}

int checkValid(int*** grid, int square, int row, int col, int value){
	if (value == 0 || value > 9){
		return 0;
	}
	if (checkSquare(grid, square, row, col, value)){ 
		if (checkRow(grid, square, row, col, value)){
			if (checkColumn(grid, square, row, col, value)){
				return 1;
			}
		}
	}
	return 0;
}

int isSolved(int*** grid){
	int i, j, k;
	for (i = 0; i < 9; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if (!checkValid(grid, i, j, k, *(*(*(grid+i)+j)+k))){
					return 0;
				}
			}
		}
	}
	return 1;
}

int* findNextEmpty(int*** grid, int* square, int* row, int* col){
	int i, j, k;
	if (*square == 8 && *row == 2 && *col == 2){
		return 0;
	}
	for (i = 0; i < 9; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if (*(*(grid+i)+j)+k > *(*(grid+*square)+*row)+*col){
					if (*(*(*(grid+i)+j)+k) == 0){
						*square = i;
						*row = j;
						*col = k;
						return *(*(grid+i)+j)+k;
					}
				}
			}
		}
	}
	return 0;
}

int solveGrid(int*** grid, int square, int row, int col){
	if (isSolved(grid)){
		return 1;
	}
	int s = square;
	int r = row;
	int c = col;
	int* current = findNextEmpty(grid, &s, &r, &c);
	if (!current){
		return 1;
	}

	for (int i = 1; i < 10; i++){
		if (checkValid(grid, s, r, c, i)){
			*current = i;
			if (solveGrid(grid, s, r, c)){
				return 1;
			} 
			*current = 0;
		} 
	}
	return 0;
}

int solveSudoku(int*** blocks){
	// This is the function to solve the Sudoku (blocks). Feel free to use any helper functions.
	// YOU MUST NOT USE ANY ARRAY NOTATION ([])!
	
	//Your implementation here
	
	// Collaborated with Ashish Panjwani
	int*** array = blocks;	
	
	solveGrid(array, 0, 0, -1);
	
	return 1;
}
