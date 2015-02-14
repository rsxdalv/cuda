#include <stdio.h>

#define THREADS_PER_BLOCK   1024

__global__
void cuAdd(int *a,int *b,int *c, int N)
{
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if(offset < N)
    {
        c[offset] = a[offset] + b[offset];
    }
}

#define N (1<<20)

int main()
{
    int *a, *b, *c,
        *_a, *_b, *_c;

    const int length = N * sizeof( int );

    cudaMalloc( (void **) &_a, length );
    cudaMalloc( (void **) &_b, length );
    cudaMalloc( (void **) &_c, length );

    a = (int *) malloc(length);
    b = (int *) malloc(length);
    c = (int *) malloc(length);

    for(int i=0; i < N; i++)
    {
        a[i]=b[i]=i;
        c[i]=-1;
    }

    cudaMemcpy(_a, a, length, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, length, cudaMemcpyHostToDevice);

    //int blocks = length/THREADS_PER_BLOCK;
    cuAdd<<<128, THREADS_PER_BLOCK>>>(_a,_b,_c, length);

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
