#include <stdio.h>

__global__ void cuAdd(int *a,int *b,int *c, int N)
{
	// global index
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if(offset < N)
    {
        c[offset] = a[offset] + b[offset];
    }
}

#define N (1<<20)

int main()
{
	// host 
    int *a, *b, *c;

	// device
	int *_a, *_b, *_c;

    const int length = N * sizeof( int );

    cudaMalloc( (void **) &_a, length );
    cudaMalloc( (void **) &_b, length );
    cudaMalloc( (void **) &_c, length );

    a = (int *) malloc(length);
    b = (int *) malloc(length);
    c = (int *) malloc(length);

    for(int i=0; i < N; i++)
    {
        a[i]=1;
		b[i]=2;
    }

    cudaMemcpy(_a, a, length, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, length, cudaMemcpyHostToDevice);

    //int blocks = length/THREADS_PER_BLOCK;
	size_t blockSize = 1024; 
	size_t gridSize  = (N + blockSize - 1)/blockSize;
    cuAdd<<< gridSize, blockSize>>>(_a, _b, _c, length);

    cudaMemcpy(c, _c, length, cudaMemcpyDeviceToHost);

    printf("Start: %d. Finish: %d.\n",c[0], c[N-1]);

    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    free(a);
    free(b);
    free(c);


    return 0;
}
