// HOST Benchmark \w Timing based on sys/time microseconds
// Record starting time
void h_Benchmark(float *a, float *b, float *hh_c, int wA, int wB, int hA)
{
    double h_start = microSeconds();
    // Execute kernel
    h_MM(a, b, hh_c, wA, wB, hA);
    // Record ending time
    double h_end = microSeconds();
    double h_MM_ms = (h_end - h_start) * 1000;
   
    int wC = wB;
    int hC = hA;
    
    // Calculate the number of FLOP
    const double FLOP_GEMM = 1.0 * wC * hC * wA;
    // Calculate the giga flops per second
    double gigaFLOPS = (FLOP_GEMM * 1.0e-9f) / (h_MM_ms / 1000.f);
    
    // Print the results in a table
    printf("Benchmark of h_MM\nResults:\n %4.4f GFLOPS \t%4.4fms\n",
                    gigaFLOPS,
                    h_MM_ms);
    
}
