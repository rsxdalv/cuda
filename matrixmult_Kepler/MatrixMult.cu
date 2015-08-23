#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

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
__global__ void cuMult(float *a, float *b, float *c, int wA, int wB, int hA)
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
 * KERNEL cuMultOpti() - Takes two 2D matrices and multiplies them optimally
 * @param a - 1st Matrix
 * @param b - 2nd Matrix
 * @param c - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
__global__ void cuMultOpti(
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
    
    /* Calculate global index X, Y*/
    int gx = blockDim.x * blockIdx.x + threadIdx.x;  // column
    int gy = blockDim.y * blockIdx.y + threadIdx.y;   // row
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* Compute offset idx for A & B */
    // First A index (row shift) BlockRow*BlockWidth*Width-A
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

        /* Make sure all of the threads have cached the memory */
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
        float *a,
        float *b,
        float *c,
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

double microSeconds()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
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
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d (%s) has compute capability %d.%d.\nWarp Size: %d",
               device, deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.warpSize);
    }

    return 0;
//    
//    // width A
//    int wA = 512;
//    // height A
//    int hA = 512;
//    
//    // width B
//    int wB = 512;
//    // height B
//    int hB = wA;
//    
//    // value A
//    float aValue = 1.0;
//    // value B
//    float bValue = 2.0;
//    
//    /* Fetch the test parameters */
//    if(argc < 6)
//    {
//        printf("Using default parameters: 320 640 320 1 2\n");
//    }
//    else
//    {
//        wA = atoi(argv[1]);
//        hA = atoi(argv[2]);
//        wB = atoi(argv[3]);
//        hB = wA;
//        aValue = atoi(argv[4]);
//        bValue = atoi(argv[5]);
//    }
//    /**
//     *  Neutral - both for host and device */
//    
//    int wC = wB;
//    int hC = hA;
//
//    size_t size_a = sizeof(float) * wA * hA;
//    size_t size_b = sizeof(float) * wB * hB;
//    size_t size_c = sizeof(float) * wC * hC;
//	
//    
//    // host 
//    float *a, *b, *c, *hh_c;
//    a = (float *) malloc(size_a);
//    b = (float *) malloc(size_b);
//    c = (float *) malloc(size_c);
//    /* Host test memory */
//    hh_c = (float *) malloc(size_c);
//    
//    assert(hh_c != NULL);
//    
//    /**
//     *  Device specific */
//    
//    // device
//    float *_a, *_b, *_c;
//    cudaMalloc( (void **) &_a, size_a );
//    cudaMalloc( (void **) &_b, size_b );
//    cudaMalloc( (void **) &_c, size_c );
//
//    /**
//     Neutral */
//    // initialize A
//    for(int i=0; i < hA * wA; i++)
//    {
//        a[i] = aValue;
//    }
//    
//    // initialize B
//    for(int i=0; i < hB * wB; i++)
//    {
//        b[i] = bValue;
//    }
//    
//    /**
//     Device*/
//    
//    // copy data to GPU
//    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
//    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);
//
//    // x : col , y: row
//    dim3 blockSize(16,16);
//    // (N.x + blockSize.x - 1)/blockSize.x, (N.y + blockSize.y -1)/blockSize.y)
//    dim3 gridSize((wC+15)/16, (hC+15)/16);
//        
//    cudaError_t error;
//
//    cudaEvent_t start, stop;
//
//    ///////////////////////////////////////////////////
//    // OPTIMIZED
//    
//    error = cudaEventCreate(&start);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to create start event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    error = cudaEventCreate(&stop);
//
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to create stop event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    // Record the start event
//    error = cudaEventRecord(start, NULL);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to record start event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//
//    // kernel execution
//    cuMult<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
//
//    // Record the stop event
//    error = cudaEventRecord(stop, NULL);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to record stop event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    // Wait for the stop event to complete
//    error = cudaEventSynchronize(stop);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    float sgemm_msec = 0.f;
//    error = cudaEventElapsedTime(&sgemm_msec, start, stop);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//
//
//    // C := alpha*op( A )*op( B ) + beta*C
//    // GEMM performs 4 floating point operations for one data output
//    //double flops_sgemm = 4.0 * (double) NI * (double) NJ * (double) NK;
//
//    //double gigaFlops = (flops_sgemm * 1.0e-9f) / (sgemm_msec / 1000.f);
//
//    printf("%.4f\t", sgemm_msec);
////    printf("N_Time: %.3f\n, WorkgroupSize= %u threads/block\n",
////                    //gigaFlops,
////                    sgemm_msec,
////                    //flops_sgemm,
////                    blockSize.x * blockSize.y);
//    
//    
//    // copy data back to CPU
//    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);
//    
//    /////////////////////////////////////////////////
//    // OPTIMIZED
//    
//    error = cudaEventCreate(&start);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to create start event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    error = cudaEventCreate(&stop);
//
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to create stop event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    // Record the start event
//    error = cudaEventRecord(start, NULL);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to record start event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//
//    // kernel execution
//    cuMultOpti<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
//
//    // Record the stop event
//    error = cudaEventRecord(stop, NULL);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to record stop event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    // Wait for the stop event to complete
//    error = cudaEventSynchronize(stop);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//
//    sgemm_msec = 0.f;
//    error = cudaEventElapsedTime(&sgemm_msec, start, stop);
//    if (error != cudaSuccess)
//    {
//            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
//            cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//    }
//    
////    printf("O_Time: %.3f\nWorkgroupSize= %u threads/block\n",
////                    //gigaFlops,
////                    sgemm_msec,
////                    //flops_sgemm,
////                    blockSize.x * blockSize.y);
//    printf("%.4f\t", sgemm_msec);
//    
//    
//    
//    // copy data back to CPU
//    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);
//    
//    //////////////////////////////////////////////////
//    // HOST
//    
//    // compare with cpu results
//    /**
//     Host*/
//    double h_start, h_end;
//    h_start = microSeconds();
//    h_MatrixMult_Naive(a, b, hh_c, wA, wB, hA);
//    h_end = microSeconds();
//    
//    printf("%4.4f\t", (h_end - h_start) * 1000);
//    
//    // Check first and last memory location
//    //printf("Start: %d. Finish: %d.\n",c[2], c[wC * hC - 1]);
//    
//    /* Check */
////    // Naive check
////    int k = 0;
////    while(c[k] == c[k+1])
////        k++;
////    printf("EQ Test: Breakpoint @ %d\n",k);
//    
//    int fail = 0;
//    for( int k = 0; k< wB*hA; k++)
//    {
//        if(abs(c[k] - hh_c[k]) > 1e-5)
//            fail++;
//    }
//    printf("\nWorkgroup: %d Data: %d Failures: %d\n", blockSize.x*blockSize.y, wC, fail);
//
//    // release resources
//    cudaFree(_a);
//    cudaFree(_b);
//    cudaFree(_c);
//
//    free(a);
//    free(b);
//    free(c);
//    free(hh_c);
//
//    return 0;
}
