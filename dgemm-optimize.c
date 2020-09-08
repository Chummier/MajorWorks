#include <emmintrin.h>
#include <stdio.h>

// Loop unrolling
void dgemm( int m, int n, float *A, float *C )
{
  int j;
  for(int i = 0; i < m; i++ ){
    for(int k = 0; k < n; k++ ){
        for(j = 0; j < m-5; j+=5 ){
            C[i+j*m] += A[i+k*m] * A[j+k*m];
            C[i+(j+1)*m] += A[i+k*m] * A[j+1+k*m];
            C[i+(j+2)*m] += A[i+k*m] * A[j+2+k*m];
            C[i+(j+3)*m] += A[i+k*m] * A[j+3+k*m];
            C[i+(j+4)*m] += A[i+k*m] * A[j+4+k*m];
        } 
        for (j = j; j < m; j++){
          C[i+j*m] += A[i+k*m] * A[j+k*m];
        }
    }
  }
}

// SSE Instructions
//
// Mine doesn't work
/*void dgemm( int m, int n, float *A, float *C )
{
  __m128* x;
  __m128* y;
  __m128 res;
  float* temp = (float*)malloc(4*sizeof(float));
  float* temp2 = (float*)malloc(4*sizeof(float));

  for( int i = 0; i < m; i++ ){
    for( int k = 0; k < n; k++ ){
      
      for (int t = 0; t < 4; t++){ temp[t] = A[i+k*m]; }
      x = (__m128*)temp;              // x -> {A[i+k*m], A[i+k*m], A[i+k*m], A[i+k*m]}

      int j;
      for(j = 0; j+4 < m; j+=4 ){
          y = (__m128*)(&A[j+k*m]);   // y -> {A[j+k*m], A[j+1+k*m], A[j+2+k*m], A[j+3+k*m]}
          res = _mm_mul_ps(*x, *y);
          _mm_storer_ps(temp2, res);
          

          for (int jj = 0; jj < 4; jj++){
            C[i+(j+jj)*m] += temp2[jj]; // access the 4 elements of the result __m128
          }
      }

      for (j = j-4; j < m; j++){
        C[i+j*m] += A[i+k*m] * A[j+k*m]; // do remaining elements, since __m128 only does increments of 4
      }
    }
  }
  free(temp);
}*/

// Pre-fetching
/*void dgemm( int m, int n, float *A, float *C )
{
  for( int i = 0; i < m; i++ ){
    for( int k = 0; k < n; k++ ){
      for( int j = 0; j < m; j++ ){
          C[i+j*m] += A[i+k*m] * A[j+k*m];
      } 
      __builtin_prefetch(&A[(k+1)*m]);
    }
  }
}*/