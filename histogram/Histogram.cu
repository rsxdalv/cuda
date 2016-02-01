/*
 * File: Histogram
 * Author: Roberts Slisans
 * Date: 03/24/2015 16:00
 * Last updated: 03/24/2014 17:32
 */

#include <stdio.h>
#include <assert.h>



/**
 * KERNEL cuAdd() - Takes 2 input arrays of same size N and adds them into C.
 * Locations are found by computing the global index of each thread.
 * @return 
 */
__global__ void cuAdd(int *a,int *b,int *c, int N)
{
	// 1D global index
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if(offset < N)
    {
        c[offset] = a[offset] + b[offset];
    }
}

/**
 * KERNEL cuMult() - Takes two 2D matrices and multiplies them
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
__global__ void cuMult(int *a, int *b, int *c, int wA, int wB, int hA)
{
    // global index
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // col
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;  // row
    
    if(gidx < wB && gidy < hA)
    {
        int sum = 0;
        for(int k=0; k<wA; k++)
        {
            // Multiply row of A by column of B
            sum += a[gidy*wA + k] * b[k*wB +gidx];
        }
        c[gidy * wB + gidx] = sum;
    }
}

/**
 * KERNEL cuMultOpti() - Takes two 2D matrices and multiplies them optimally
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
__global__ void cuMultOpti(
        int *a,
        int *b,
        int *c,
        int wA,
        int wB,
        int hA)
{
#define blockTile 16
    /* Blocksize is 16x16 */
    /* Allocate shared memory */
    __shared__ int aBlock[blockTile][blockTile];
    __shared__ int bBlock[blockTile][blockTile];
    
    /* Calculate global index X, Y*/
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // column
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;   // row
    
    /* Assign shared memory and sync  */
    /* Warning, wA*gidy may be out of bounds */
    aBlock[threadIdx.x][threadIdx.y] = a[gidy*wA + threadIdx.x];
    bBlock[threadIdx.x][threadIdx.y] = b[threadIdx.y*wB + gidx];
    
    /* Make sure all of the threads have cached the memory */
    __syncthreads();
    
    /* Check if global IDs are within limits */
    if(gidx < wB && gidy < hA)
    {
        int sum = 0;
        for(int k=0; k<wA; k++)
        {
            sum += aBlock[threadIdx.y][k] * bBlock[k][threadIdx.x];
        }
        // c [gidy][gidx]
        c[gidy * wB + gidx] = sum;
        
    }
}

/**
 * HOST h_MatrixMult_Naive() - Takes two 2D matrices and multiplies them naively
 * @param a wA.hA - 1st Matrix
 * @param b wB.wA - 2nd Matrix
 * @param c hA.wB - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
void h_MatrixMult_Naive(
        int *a,
        int *b,
        int *c,
        int wA,
        int wB,
        int hA)
{
    // Iterate through all rows of a
    for(int i=0; i<hA; i++)
    {
        // Iterate through all columns of b
        for(int j=0; j<wB; j++)
        {
            // Calculate all of c[i][j] products
            int sum = 0;
            for(int k=0; k<wA; k++)
            {
                sum += a[i*wA + k] * b[k*wB + j];
            }
            assert(i*wB + j < hA*wB);
            // Index - row i of column j with column width of wB
            c[i * wB + j] = sum;
        }
    }
}



/**
 * ENTRY main() - Tests <<<>>>cuMult() kernel: Initializes memory and data on
 * the host, then memory on the device. Copies the data from host to device,
 * executes kernel with memory device pointers, copies result back to host,
 * displays results for error checking and frees allocated memory.
 * @return 
 */
int main(int argc, char ** argv)
{
    // width A
    int wA = 320;
    // height A
    int hA = 640;
    
    // width B
    int wB = 320;
    // height B
    int hB = wA;
    
    // value A
    int aValue = 1;
    // value B
    int bValue = 2;
    
    /* Fetch the test parameters */
    if(argc < 6)
    {
        printf("Using default parameters: 320 640 320 1 2\n");
    }
    else
    {
        wA = atoi(argv[1]);
        hA = atoi(argv[2]);
        wB = atoi(argv[3]);
        hB = wA;
        aValue = atoi(argv[4]);
        bValue = atoi(argv[5]);
    }
    /**
     *  Neutral - both for host and device */
    
    int wC = wB;
    int hC = hA;

    size_t size_a = sizeof(int) * wA * hA;
    size_t size_b = sizeof(int) * wB * hB;
    size_t size_c = sizeof(int) * wC * hC;
	
    
    // host 
    int *a, *b, *c, *hh_c;
    a = (int *) malloc(size_a);
    b = (int *) malloc(size_b);
    c = (int *) malloc(size_c);
    /* Host test memory */
    hh_c = (int *) malloc(size_c);
    
    assert(hh_c != NULL);
    
    /**
     *  Device specific */
    
    // device
    int *_a, *_b, *_c;
    cudaMalloc( (void **) &_a, size_a );
    cudaMalloc( (void **) &_b, size_b );
    cudaMalloc( (void **) &_c, size_c );

    /**
     Neutral */
    // initialize A
    for(int i=0; i < hA * wA; i++)
    {
        a[i] = aValue;
    }
    
    // initialize B
    for(int i=0; i < hB * wB; i++)
    {
        b[i] = bValue;
    }
    
    /**
     Device*/
    
    // copy data to GPU
    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);

    // x : col , y: row
    dim3 blockSize(16,16);
    // (N.x + blockSize.x - 1)/blockSize.x, (N.y + blockSize.y -1)/blockSize.y)
    dim3 gridSize((wC+15)/16, (hC+15)/16);
        
    // kernel execution
    cuMult<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
    //cuMultOpti<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);

    // copy data back to CPU
    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);
    
    // compare with cpu results
    /**
     Host*/
    
    h_MatrixMult_Naive(a, b, hh_c, wA, wB, hA);
    
    // Check first and last memory location
    printf("Start: %d. Finish: %d.\n",c[2], c[wC * hC - 1]);
    
    /* Check */
    // Naive check
    int k = 0;
    while(c[k] == c[k+1])
        k++;
    printf("EQ Test: Breakpoint @ %d\n",k);
    // Device - Host check
    k = 0;
    while(c[k] == hh_c[k])
        k++;
    printf("H2D Test: Breakpoint @ %d\n",k);

    // release resources
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    free(a);
    free(b);
    free(c);
    free(hh_c);

    return 0;
}
