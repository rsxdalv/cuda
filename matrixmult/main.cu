/**
 * Project details:
 *      Host: kepler1 (Ubuntu 14.04 LTS)
 *      Hardware: GPUs k40c k20c
 *      IDE: Netbeans 8.0.2
 *      Goal: Benchmark highly optimized matrix multiplication with parallelization
 * 
 * TODO: 
 *      Description including all the framework (Launch pad, Testing suite, Glue, Theory)
 *      Add function-lists to includes
 *      Create register pressure aware kernels
 *      Improve test suite code and output
 */
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

// double microseconds();
#include "utils.cu"

#include "kernels.cu"

#include "hostKernels.cu"

enum KernelCode {
    k_MM,
    k_MM_OPT
};

void d_Benchmark_MM(enum KernelCode kid = k_MM,  // kernel specifier
        gridSize, blockSize, // common launch parameters for all kernels
        float *a, float *b, float *c, int wA, int wB, int hA) // kernel arguments
{
    
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
            d_MM<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
            break;
        case k_MM_OPT:
            d_MM_OPT<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);
            break;
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

    float sgemm_msec = 0.f;
    error = cudaEventElapsedTime(&sgemm_msec, start, stop);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
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
    int wA = 512;
    // height A
    int hA = 512;
    
    // width B
    int wB = 512;
    // height B
    int hB = wA;
    
    // value A
    float aValue = 1.0;
    // value B
    float bValue = 2.0;
    
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

    size_t size_a = sizeof(float) * wA * hA;
    size_t size_b = sizeof(float) * wB * hB;
    size_t size_c = sizeof(float) * wC * hC;
	
    
    /* Host memory initialization */
    float *a, *b, *c, *hh_c;
    a = (float *) malloc(size_a);
    b = (float *) malloc(size_b);
    c = (float *) malloc(size_c);
    /* Host testing memory */
    hh_c = (float *) malloc(size_c);
    
    assert(hh_c != NULL);
    
    /* Device Memory Initialization */
    float *_a, *_b, *_c;
    cudaMalloc( (void **) &_a, size_a );
    cudaMalloc( (void **) &_b, size_b );
    cudaMalloc( (void **) &_c, size_c );

    /* Input initialization */
    for(int i=0; i < hA * wA; i++)
    {
        a[i] = aValue;
    }
    
    for(int i=0; i < hB * wB; i++)
    {
        b[i] = bValue;
    }
    
    /* 
     * Device 
     */
    
    // copy data to GPU
    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);

    // x : col , y: row
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define gridRound(width, blocksize) (width + blocksize - 1)/blocksize
    dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
    // (N.x + blockSize.x - 1)/blockSize.x, (N.y + blockSize.y -1)/blockSize.y)
    
    dim3 gridSize(gridRound(wC, BLOCKSIZE_X),
            gridRound(hC, BLOCKSIZE_Y));
        
    // Side by side test of the new kernel benchmark.
    // Event functions might have trouble with this.
    d_Benchmark_MM(KernelCode.d_MM,
            blockSize, gridSize,
            _a, _b, _c, wA, wB, hA);
    
    d_Benchmark_MM(KernelCode.d_MM_OPT,
            blockSize, gridSize,
            _a, _b, _c, wA, wB, hA);
        
    cudaError_t error;

    cudaEvent_t start, stop;

    ///////////////////////////////////////////////////
    // OPTIMIZED (What is? CudaEvent ?)
    
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


    // kernel execution
    d_MM<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);

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

    float sgemm_msec = 0.f;
    error = cudaEventElapsedTime(&sgemm_msec, start, stop);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }



    /* Comments about GEMM of the benchmark toolkit */
    // C := alpha*op( A )*op( B ) + beta*C
    // GEMM performs 4 floating point operations for one data output
    //double flops_sgemm = 4.0 * (double) NI * (double) NJ * (double) NK;

    // Proposed flops for the d_MM:
    // C = op( A )*op( B ), which is 1 per data output
    double flops_sgemm = 1.0 * wC * hC * wA;
    double gigaFlops = (flops_sgemm * 1.0e-9f) / (sgemm_msec / 1000.f);
    

    printf("Naive Results:\n %4.4f GFLOPS \t%4.4fms \t WorkgroupSize= %u threads/block\n",
                    gigaFlops,
                    sgemm_msec,
                    blockSize.x * blockSize.y);
    
    
    // copy data back to CPU
    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);
    
    /////////////////////////////////////////////////
    // OPTIMIZED
    
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

    // kernel execution
    d_MM_OPT<<< gridSize, blockSize >>>(_a, _b, _c, wA, wB, hA);

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

    sgemm_msec = 0.f;
    error = cudaEventElapsedTime(&sgemm_msec, start, stop);
    if (error != cudaSuccess)
    {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
            cudaGetErrorString(error));
            exit(EXIT_FAILURE);
    }
    
    gigaFlops = (flops_sgemm * 1.0e-9f) / (sgemm_msec / 1000.f);
    
    printf("Optimized Results:\n %4.4f GFLOPS \t%4.4fms \t WorkgroupSize= %u threads/block\n",
                    gigaFlops,
                    sgemm_msec,
                    blockSize.x * blockSize.y);
    
    
    // copy data back to CPU
    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);
    
    //////////////////////////////////////////////////
    // HOST
    
    // compare with cpu results
    /**
     Host*/
    /* Timing based on sys/time microseconds */
    double h_start, h_end;
    h_start = microSeconds();
    h_MM(a, b, hh_c, wA, wB, hA);
    h_end = microSeconds();
    
    printf("Host time: %4.4fs\n", (h_end - h_start) * 1000);
    
    /* TODO: Create test function */
    int errors = 0;
    for( int k = 0; k < wB*hA; k++)
    {
        /* Make sure absolute difference is below a threshold */
        if(abs(c[k] - hh_c[k]) > 1e-5)
            errors++;
    }
    printf("\nWorkgroup Size: %d Data Width: %d Errors: %d\n", blockSize.x*blockSize.y, wC, errors);

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
