/* h_HG - Host histogram kernel
 * data - array of 0-255 values of intensity
 * N - number of data points
 * @return output - 256 bins with number of data points in each
 */
void h_HG(int* data, int N, int* output)
{
    // Initialize data to 0 for cumulative sum
    for(int i = 0; i < N; i++)
    {
        output[i] = 0;
    }
    // Sum each bin
    for(int i = 0; i < N; i++)
    {
        output[ data[i] ]++;
    }
}
