#include <stdio.h>
#include <stdlib.h>

/* I've removed functions that I didn't write, like the main function to run the program
This program solves a sudoku board stored as an int***:
The board is stored as 9 3x3 blocks

My implementation is a depth-first solution
It calls a function that recursively finds the next empty space on the grid,
Trys each number 1-9, checking the entire row then column, until it finds the
correct number
*/

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
