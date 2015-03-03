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

/**
 * KERNL cuMultOpti() - Takes two 2D matrices and multiplies them optimally
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
__global__ void cuMultOpti(int *a, int *b, int *c, int wA, int wB, int hA)
{
    /* Split the work over chunks of 16 */
    // global index
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // col
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;   // row
    
    /* Check if gidx is in within limits */
    if(gidx < hA, gidy < wB)
    
    int sum = 0;
    for(int k=0;k<wA;k++)
    {
        sum += a[gidy*wA + k] * b[k*wB + gidx];
    }
    // c [gidy][gidx]
    c[gidy * wB + gidx] = sum;
}

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
    size_t size_c = sizeof(int) * wC * hC;
	
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

    // initialize A
    for(int i=0; i < hA * wA; i++)
    {
        a[i] = 1;
    }
    
    // initialize B
    for(int i=0; i < hB * wB; i++)
    {
        b[i] = 2;
    }
    

    // copy data to GPU
    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);

    // x : col , y: row
    dim3 blockSize(16,16); 
    // (N.x + blockSize.x - 1)/blockSize.x, (N.y + blockSize.y -1)/blockSize.y)
    dim3 gridSize((wC+15)/16, (hC+15)/16);
        

    // kernel execution
    cuMultOpti<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB);

    // copy data back to CPU
    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);

    // Check first and last memory location
    printf("Start: %d. Finish: %d.\n",c[0], c[wC * hC - 1]);

    // release resources
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
