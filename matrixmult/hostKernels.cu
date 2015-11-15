/**
 * HOST h_MatrixMult_Naive() - Takes two 2D matrices and multiplies them naively
 * @param a wA.hA - 1st Matrix
 * @param b wB.wA - 2nd Matrix
 * @param c hA.wB - Result Matrix
 * @param wA - length of A and depth of B
 * @param wB - length of matrix B and C
 * @param hA - depth of matrix A and C
 */
void h_MM(
        float *a,
        float *b,
        float *c,
        int wA,
        int wB,
        int hA)
{
    // Iterate through all rows of a
    for(int i=0; i<hA; i++)
    {
        // Iterate through all columns of b
        for(int j=0; j<wB; j++)
        {
            // Calculate all of c[i][j] products
            int sum = 0;
            for(int k=0; k<wA; k++)
            {
                sum += a[i*wA + k] * b[k*wB + j];
            }
            assert(i*wB + j < hA*wB);
            // Index - row i of column j with column width of wB
            c[i * wB + j] = sum;
        }
    }
}
