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
// Some standard libraries are included as per CMake configuration

// printf() - Text output
#include <stdio.h>

// getopt() - Command line argument parsing
#include <unistd.h>

// double microseconds();
#include "utils.cu"

#include "kernels.cu"

#include "hostKernels.cu"

#include "kernelBenchmark.cu"

/**
 * Tests matrix multiplication on 2 kernels and 1 host algorithm, by setting up
 * the variables, allocating and initializing the memory, measuring the time,
 * showing the results, and cleaning up.
 */
int main(int argc, char ** argv)
{
    // TODO: CREATE TESTING PARAMETER PARSER THAT ALLOWS FOR PARTIAL DEFAULT VALUES
    
//    int aflag = 0;
//    int bflag = 0;
//    char *cvalue = NULL;
//    int index;
//    int c;
//
//    opterr = 0;
//
//    while ((c = getopt (argc, argv, "abc:")) != -1)
//        switch (c)
//        {
//        case 'a':
//            aflag = 1;
//            break;
//        case 'b':
//            bflag = 1;
//            break;
//        case 'c':
//            cvalue = optarg;
//            break;
//        case '?':
//            if (optopt == 'c')
//              fprintf (stderr, "Option -%c requires an argument.\n", optopt);
//            else if (isprint (optopt))
//              fprintf (stderr, "Unknown option `-%c'.\n", optopt);
//            else
//              fprintf (stderr,
//                       "Unknown option character `\\x%x'.\n",
//                       optopt);
//            return 1;
//        default:
//          abort ();
//        }
//
//    printf ("aflag = %d, bflag = %d, cvalue = %s\n",
//            aflag, bflag, cvalue);
    
    
    // width A - a
    int wA = 512;
    // height A - h
    int hA = 512;
    
    // width B - b
    int wB = 512;
    
    // value A - x
    float aValue = 1.0;
    // value B - y
    float bValue = 2.0;
    
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
    for(int i=0; i < hA * wA; i++)
    {
        a[i] = aValue;
    }
    
    for(int i=0; i < hB * wB; i++)
    {
        b[i] = bValue;
    }
    
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
    // HOST Benchmark \w Timing based on sys/time microseconds
    // Record starting time
    double h_start = microSeconds();
    // Execute kernel
    h_MM(a, b, hh_c, wA, wB, hA);
    // Record ending time
    double h_end = microSeconds();
    double h_MM_ms = (h_end - h_start) * 1000;
  
    // Calculate the number of FLOP
    const double FLOP_GEMM = 1.0 * wC * hC * wA;
    // Calculate the giga flops per second
    double gigaFLOPS = (FLOP_GEMM * 1.0e-9f) / (h_MM_ms / 1000.f);
    
    // Print the results in a table
    printf("Benchmark of h_MM\nResults:\n %4.4f GFLOPS \t%4.4fms\n",
                    gigaFLOPS,
                    h_MM_ms);
    
    /* TODO: Create test function */
    int errors = 0;
    for( int k = 0; k < wB*hA; k++)
    {
        /* Make sure absolute difference is below a threshold */
        if(abs(c[k] - hh_c[k]) > 1e-5)
            errors++;
    }
    printf("Errors: %d\n", errors);

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
