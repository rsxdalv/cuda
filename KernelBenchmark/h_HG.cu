/**
 * Calculates the histogram 256 with the CPU
 * @param a - Input Data (1xN)
 * @param H - Output 256x1 Histogram
 * @param N - Length of a
 */
void h_HG(int* a, int N, int* H)
{
    /* Set the data to 0 before cumulative sum */
    for(int i = 0; i < 256; i++)
    {
        H[i] = 0;
    }
    /* Accumulate the sum for each data bin */
    for(int i = 0; i < N; i++)
    {
        /* Calculate the brightness value ( a % 256 placeholder) */
        H[ a[i] % 256 ]++;
    }
}