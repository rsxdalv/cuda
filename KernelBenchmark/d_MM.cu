
/**
 * KERNEL d_MM() - Takes two 2D matrices and multiplies them
 * Result is divided into threads, each thread iterating over datasets
 * to obtain the final answer. C[Thread] = Sum { A column * B Row }
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
__global__ void d_MM(float *a, float *b, float *c, int wA, int wB, int hA)
{
    // global index
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // col
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;  // row
    
    if(gidx < wB && gidy < hA)
    {
        float sum = 0.f;
        for(int k=0; k<wA; k++)
        {
            // Multiply row of A by column of B
            sum += a[gidy*wA + k] * b[k*wB +gidx];
        }
        c[gidy * wB + gidx] = sum;
    }
}

/**
 * KERNEL d_MM_OPT() - Takes two 2D matrices and multiplies them optimally
 * Output set is divided into blocks of powers of two. 
 * RENAME: d_MM_OPT
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
__global__ void d_MM_OPT(
        float *a,
        float *b,
        float *c,
        int wA,
        int wB,
        int hA)
{
#define blockTile 16
    /* Blocksize is 16x16 */
    /* Allocate shared memory */
    __shared__ float aBlock[blockTile][blockTile];
    __shared__ float bBlock[blockTile][blockTile];
    
    /* Calculate global index X, Y */
      
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int gx = blockDim.x * bx + tx;  // column
    int gy = blockDim.y * by + ty;   // row
  
    /* Compute offset idx for A & B */
    // First A index (row shift) Block.row * Block.width * A.width
    int a0 = wA * 16 * by;
    // aBegin -> last element in row -> + width - 1
    int aZ = a0 + wA - 1;
    // Column block iteration = blockDim.x
    int aD = 16;
    // b_0 -> Column Shift
    int b0 = 16 * bx;
    // Row block iteration = blockDim.y * width B
    int bD = 16 * wB;

    float sum = 0.f;
    
    for(int aI = a0, bI = b0; aI <= aZ; aI += aD, bI += bD)
    {
        
        /* Assign shared memory and sync  */
        /* Warning, wA*gidy may be out of bounds */
        aBlock[ty][tx] = a[aI + ty*wA + tx];
        bBlock[ty][tx] = b[bI + ty*wB + tx];

        /* Make sure all of the threads have updated the memory cache */
        __syncthreads();
        
        /* Sum over NK */
        for(int k=0; k < 16; k++)
        {
            /* C = (A x B) */
            sum += aBlock[ty][k] * bBlock[k][tx];
        }
    }
    
    c[gy*wB + gx] = sum;
    //c[i * NJ + j] = ALPHA*sum + BETA*c[i * NJ + j];
}