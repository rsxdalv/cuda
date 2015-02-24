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

    
    
    
__global__ void cuMult(int *a, int *b, int *c, int wA, int wB)
{
    // global index
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // col
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;   // row
    
    int sum = 0;
    for(int k=0;k<wA;k++)
    {
        sum += a[gidy*wA + k] * b[k*wB +gidx];
    }
    // c [gidy][gidx]
    c[gidy * wB + gidx] = sum;
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
    
    // width A
    int wA = 320;
    // height A
    int hA = 640;
    
    // width B
    int wB = 320;
    // height B
    int hB = 320;
    
    int wC = wB;
    int hC = hA;

    size_t size_a = sizeof(int) * wA * hA;
    size_t size_b = sizeof(int) * wB * hB;
    size_t size_b = sizeof(int) * wA * hA;
	
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

    // initializ A
    for(int i=0; i < hA * wA; i++)
    {
        a[i] = ;
    }
    
    // initializ B
    for(int i=0; i < hB * wB; i++)
    {
        b[i] = ;
    }
    

	// copy data to gpu
    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);

    // x : col , y: row
    dim3 blockSize(16,16); 
    dim3 gridSize((wC+15)/16, (hC+15)/16);

                //(N + blockSize - 1)/blockSize;

	// kernel execution
    //cuAdd<<< gridSize, blockSize>>>(_a, _b, _c, length);
    cuMult<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB);

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
