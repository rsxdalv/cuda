enum KernelCode {
    k_MM,
    k_MM_OPT
};

float d_Benchmark_MM(enum KernelCode kid,  // kernel specifier
        //cudaError_t & error, cudaEvent_t & start, cudaEvent_t & stop,
        dim3 gridSize, dim3 blockSize, // common launch parameters for all kernels
        float * _a, float * _b, float * _c, int wA, int wB, int hA) // kernel arguments
{
    cudaError_t error;
    cudaEvent_t start, stop;
    
    error = cudaEventCreate(&start);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to create start event (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }

    error = cudaEventCreate(&stop);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to create stop event (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to record start event (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }

    // kernel call
    switch(kid)
    {
        case k_MM:
            fprintf(stderr, "Benchmark of d_MM \n");
            d_MM<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
            break;
        case k_MM_OPT:
            fprintf(stderr, "Benchmark of d_MM_OPT \n");
            d_MM_OPT<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
            break;
        default:
            fprintf(stderr, "No Kernel Code!\n");
            return 0.f;
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to record stop event (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }

    float GEMM_ms = 0.f;
    error = cudaEventElapsedTime(&GEMM_ms, start, stop);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }
    
    int wC = wB;
    int hC = hA;
    
    // Calculate the number of FLOP
    const double FLOP_GEMM = 1.0 * wC * hC * wA;
    // Calculate the giga flops per second
    double gigaFLOPS = (FLOP_GEMM * 1.0e-9f) / (GEMM_ms / 1000.f);
    
    // Print the results in a table
    printf("Results:\n %4.4f GFLOPS \t%4.4fms \t WorkgroupSize= %u threads/block\n",
                    gigaFLOPS,
                    GEMM_ms,
                    blockSize.x * blockSize.y);
    
    return GEMM_ms;
}
