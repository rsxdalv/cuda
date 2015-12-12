// gettimeofday()
#include <sys/time.h>

// Kernel pointer typedef
typedef void (*MatrixMult)(float *a, float *b, float *c, int wA, int wB, int hA);

// Returns current number of microseconds
double uSeconds()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    // Needs verification
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// HOST Benchmark \w Timing based on sys/time microseconds
// Record starting time
void h_Benchmark(MatrixMult h_MM_p, float *a, float *b, float *hh_c, int wA, int wB, int hA)
{
    double h_start = uSeconds();
    // Execute kernel
    h_MM_p(a, b, hh_c, wA, wB, hA);
    // Record ending time
    double h_end = uSeconds();
    double h_MM_ms = (h_end - h_start) * 1000;
   
    int wC = wB;
    int hC = hA;
    
    // Calculate the number of FLOP
    const double FLOP_GEMM = 1.0 * wC * hC * wA;
    // Calculate the gigaflops per second
    double gigaFLOPS = (FLOP_GEMM * 1.0e-9f) / (h_MM_ms / 1000.f);
    
    // Print the results in a table
    printf("Benchmark of h_MM\n"
            "Results:\n"
            "%4.4f GFLOPS \t%4.4fms\n",
            gigaFLOPS,  h_MM_ms);
    
}
