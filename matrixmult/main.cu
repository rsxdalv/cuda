/**
 * Project details:
 *      Host: kepler1 (Ubuntu 14.04 LTS)
 *      Hardware: GPUs k40c k20c
 *      IDE: Netbeans 8.0.2
 *      Goal: Benchmark highly optimized matrix multiplication with parallelization
 * 
 * TODO: 
 *      Create register pressure aware kernels
 */
// Some standard libraries are included as per CMake configuration.
// TODO: Add the list of them /usr/local/cuda/lib64

// printf() - Text output via fprintf()
#include <stdio.h>
// getopt() - Command line argument parsing
#include <unistd.h>

// d_MM(), d_MM_OPT()
#include "kernels.cu"
// h_MM()
#include "hostKernels.cu"

// d_Benchmark()
#include "kernelBenchmark.cu"
// h_Benchmark()
#include "hostBenchmark.cu"

// VerifyCalculation()
#include "verificator.cu"

/**
 * Tests matrix multiplication on 2 kernels and 1 host algorithm, by setting up
 * the variables, allocating and initializing the memory, measuring the time,
 * showing the results, and cleaning up.
 */
int main(int argc, char ** argv)
{
    // width 'a', height 'h', width 'b'
    int     wA = 512,
            hA = 512,
            wB = 512;
    
    // value 'x', value 'y'
    float   aValue = 1.0,
            bValue = 2.0;
    
    // Disable getopt() error handling
    opterr = 0;
    int getopt_state = 0;
    while ((getopt_state = getopt (argc, argv, "a:h:b:x:y:")) != -1)
        switch (getopt_state)
        {
            case 'a':
                wA = atoi(optarg);
                break;
            case 'h':
                hA = atoi(optarg);
                break;
            case 'b':
                wB = atoi(optarg);
                break;
            case 'x':
                aValue = atoi(optarg);
                break;
            case 'y':
                bValue = atoi(optarg);
                break;
            case '?':
                fprintf(stderr, "Invalid Option or Missing argument for: -%c\n", optopt);
                break;
            default:
                fprintf(stderr, "GetOpt failure or uncaught option!\n");
                break;
        }
        
    printf("wA\thA\twB\ta\tb\n%d\t%d\t%d\t%1.2f\t%1.2f\n", wA, hA, wB, aValue, bValue);
    
    /**
     *  Neutral - both for host and device */
    
    int hB = wA;
    
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
    
    //assert(hh_c != NULL);
    
    /* Device Memory Initialization */
    float *_a, *_b, *_c;
    cudaMalloc( (void **) &_a, size_a );
    cudaMalloc( (void **) &_b, size_b );
    cudaMalloc( (void **) &_c, size_c );

    /* Input initialization */
    for(int i = 0; i < hA * wA; i++)
        a[i] = aValue;
    
    for(int i = 0; i < hB * wB; i++)
        b[i] = bValue;
    
    /* 
     * Device Specific Routine
     */
    
    // copy initialized data to GPU
    cudaMemcpy(_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_b, cudaMemcpyHostToDevice);

    // x : columns , y: rows
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
    
    // Shorthand for int rounding
#define gridRound(width, blocksize) (width + blocksize - 1)/blocksize
    // (N.x + blockSize.x - 1)/blockSize.x, (N.y + blockSize.y -1)/blockSize.y)
    
    dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
    
    dim3 gridSize(gridRound(wC, BLOCKSIZE_X),
            gridRound(hC, BLOCKSIZE_Y));
    
    // Benchmark Matrix Multiplication Naive kernel
    d_Benchmark_MM(k_MM,
            //error, start, stop,
            gridSize, blockSize,
            _a, _b, _c, wA, wB, hA);
    
    // Benchmark Matrix Multiplication Optimized kernel
    d_Benchmark_MM(k_MM_OPT,
            //error, start, stop,
            gridSize, blockSize,
            _a, _b, _c, wA, wB, hA);

    // copy data back to CPU
    cudaMemcpy(c, _c, size_c, cudaMemcpyDeviceToHost);
    
    //////////////////////////////////////////////////
    h_Benchmark(a, b, hh_c, wA, wB, hA);

    VerifyCalculation(c, hh_c, wB*hA, 1e-5);
    
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
