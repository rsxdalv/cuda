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
    const int length = N * sizeof( int );

	// host 
    int *a, *b, *c;
    a = (int *) malloc(length);
    b = (int *) malloc(length);
    c = (int *) malloc(length);

	// device
	int *_a, *_b, *_c;
    cudaMalloc( (void **) &_a, length );
    cudaMalloc( (void **) &_b, length );
    cudaMalloc( (void **) &_c, length );

	// initialize data on the cpu
    for(int i=0; i < N; i++)
    {
        a[i]=1;
		b[i]=2;
    }

	// copy data to gpu
    cudaMemcpy(_a, a, length, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, length, cudaMemcpyHostToDevice);

	size_t blockSize = 1024; 
	size_t gridSize  = (N + blockSize - 1)/blockSize;

	// kernel execution
    cuAdd<<< gridSize, blockSize>>>(_a, _b, _c, length);

	// copy data back to cpu
    cudaMemcpy(c, _c, length, cudaMemcpyDeviceToHost);

    printf("Start: %d. Finish: %d.\n",c[0], c[N-1]);

	// release resources
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
