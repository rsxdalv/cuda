/**
 * Calculates the histogram 256 with the CPU
 * @param a - Input Data (1xN)
 * @param H - Output 256x1 Histogram
 * @param N - Length of a
 */
void hHistogram256(int *a, int *H, int N)
{
    /* Set the data to 0 before cumulative sum */
    for(int i = 0; i < 256; ++i)
    {
        H[i] = 0;
    }
    /* Accumulate the sum for each data bin */
    for(int i = 0; i < N; ++i)
    {
        /* Calculate the brightness value ( a % 256 placeholder) */
        ++H[ a[i] % 256 ];
    }
}
/**
 * Create a naive histogram using the atomic addition in global memory
 * @param a - Input data 1xN
 * @param N - Number of data points
 * @param H - Histogram vector, initiated at 0, 1x256
 */
__global__ void dHistogram256_Atomic(int *a, int N, int *H)
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if(gidx < 256)
        H[gidx] = 0;
    if(gidx < N)
    {
        // location = a%256;
        atomicAdd(H[a[gidx] % 256], 1);
    }
}
