#include <stdio.h>

/**
 * KERNEL cuAdd() - Takes 2 input arrays of same size N and adds them into C.
 * Locations are found by computing the global index of each thread.
 * @return 
 */
__global__ void cuAdd(int *a,int *b,int *c, int N)
{
	// global index
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if(offset < N)
    {
        c[offset] = a[offset] + b[offset];
    }
}

/**
 * KERNL cuMult() - Takes two 2D matrices and multiplies them
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param p - depth of matrix A and C
 * @param n - length of matrix B and C
 * @param m - length of A and depth of B
 */
__global__ void cuMult(int *a, int *b, int *c, int p, int n, int m)
{
	// global index
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
        // i and j depth
    int i = offset / n;
    int j = offset % n;
    if(i < p)
    {
        int sum = 0;
        for(int k=0;k<m;k++)
        {
            sum += a[i][k]*b[k][j];
        }
    } 
}

#define N (1<<5)

/**
 * ENTRY main() - Tests <<<>>>cuMult() kernel: Initializes memory and data on
 * the host, then memory on the device. Copies the data from host to device,
 * executes kernel with memory device pointers, copies result back to host,
 * displays results for error checking and frees allocated memory.
 * @return 
 */
int main()
{
    const int depth_a = N,
            length_b = N,
            cosize_ab = N,
            size_a = depth_a * cosize_ab * sizeof( int ),
            size_b = cosize_ab * length_b * sizeof( int ),
            size_c = depth_a * length_b * sizeof( int );
    
    const int length = N * sizeof( int );

	// host 
    int *a, *b, *c;
    a = (int *) malloc(size_a);
    b = (int *) malloc(size_b);
    c = (int *) malloc(size_c);

	// device
	int *_a, *_b, *_c;
    cudaMalloc( (void **) &_a, size_a );
    cudaMalloc( (void **) &_b, size_b );
    cudaMalloc( (void **) &_c, size_c );

	// initialize data on the cpu
    for(int i=0; i < N; i++)
    {
        a[i]=1;
		b[i]=2;
    }

	// copy data to gpu
    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);

	size_t blockSize = 1024; 
	size_t gridSize  = (N + blockSize - 1)/blockSize;

	// kernel execution
    cuAdd<<< gridSize, blockSize>>>(_a, _b, _c, length);
    cuMult<<< gridSize, blockSize>>>(_a, _b, _c, depth_a, length_b, cosize_ab);

	// copy data back to cpu
    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);

    printf("Start: %d. Finish: %d.\n",c[0], c[depth_a * length_b - 1]);

	// release resources
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
