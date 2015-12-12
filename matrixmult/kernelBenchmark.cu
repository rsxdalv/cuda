// Kernel pointer typedef
typedef void (*MatrixMult)(float *a, float *b, float *c, int wA, int wB, int hA);

// 
struct cudaLaunchError_t {
    cudaError_t error;
    const char * action;
};

float d_Benchmark_MM(
        MatrixMult kernel, // Kernel pointer
        const char *name, // Name of kernel for identification
        dim3 gridSize, dim3 blockSize, // common launch parameters for all kernels
        float * _a, float * _b, float * _c, int wA, int wB, int hA) // kernel arguments
{
    cudaError_t error;
    cudaEvent_t start, stop;
    
    float GEMM_ms = 0.f;
    
    try
    {
        error = cudaEventCreate(&start);
        if (error != cudaSuccess)
            throw (struct cudaLaunchError_t){error, "create start event"};

        error = cudaEventCreate(&stop);
        if (error != cudaSuccess)
            throw (struct cudaLaunchError_t){error, "create stop event"};
     
        // Record the start event
        error = cudaEventRecord(start, NULL);
        if (error != cudaSuccess)
            throw (struct cudaLaunchError_t){error, "record start event"};

        // Kernel invocation
        kernel<<<gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);

        // Record the stop event
        error = cudaEventRecord(stop, NULL);
        if (error != cudaSuccess)
            throw (struct cudaLaunchError_t){error, "record stop event"};

        // Wait for the stop event to complete
        error = cudaEventSynchronize(stop);
        if (error != cudaSuccess)
            throw (struct cudaLaunchError_t){error, "synchronize on the stop event"};
        
        error = cudaEventElapsedTime(&GEMM_ms, start, stop);
        if (error != cudaSuccess)
            throw (struct cudaLaunchError_t){error, "get time elapsed between events"};
    }
    catch(struct cudaLaunchError_t e)
    {
        fprintf(stderr, "Failed to %s (error code %s)!\n", e.action, cudaGetErrorString(e.error));
        exit(EXIT_FAILURE);
    }
        
    int wC = wB;
    int hC = hA;
    
    // Calculate the number of FLOP
    const double FLOP_GEMM = 1.0 * wC * hC * wA;
    // Calculate the gigaflops per second
    double gigaFLOPS = (FLOP_GEMM * 1.0e-9f) / (GEMM_ms / 1000.f);
    
    // Print the results in a table
    fprintf(stderr, "Benchmark of %s results:\n"
            "%4.4f GFLOPS \t%4.4fms \t WorkgroupSize= %u threads/block\n",
                                name,
            gigaFLOPS,      GEMM_ms,    blockSize.x * blockSize.y);
    
    return GEMM_ms;
}
