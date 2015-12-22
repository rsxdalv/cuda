/**
 * d_HG256_A_GM -
 *      Calculate a 256 bin histogram in global memory atomically.
 * @param a - Input data 1xN
 * @param N - Number of data points
 * @param H - Histogram vector, initiated at 0, 1x256
 */
__global__ void d_HG256_A_GM(int *a, int N, int *H)
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize the data
    if(gidx < 256)
        H[gidx] = 0;
    
    if(gidx < N)
    {
        // Atomically adds to result, without access conflict
        // location = a%256, current approximation for histogram, division is the
        // true result; however.
        atomicAdd(  H[a[gidx] % 256], 1  );
    }
}

/**
 * d_HG256_A_SM -
 *      Calculate a 256 bin histogram in shared memory atomically.
 * @param a - Input data 1xN
 * @param N - Number of data points
 * @param H - Histogram vector, initiated at 0, 1x256
 */
__global__ void d_HG256_A_SM(int* data, int N, int* histogram)
{
    __shared__ int histogram_SM[256];
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize the data
    if(gidx < 256)
        histogram_SM[gidx] = 0;
    
    if(gidx < N)
    {
        // Atomically adds to result, without access conflict
        // location = a%256, current approximation for histogram, division is the
        // true result; however.
        atomicAdd(  histogram_SM[a[gidx] % 256], 1  );
    }
    
    // Copy data over to the global memory
    if(gidx < 256)
        histogram[gidx] = histogram_SM[gidx];
}