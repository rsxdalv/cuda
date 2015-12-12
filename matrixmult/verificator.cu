int VerifyCalculation(float *c, float *hh_c, int size, float threshold)
{
    int errors = 0;
    for( int k = 0; k < size; k++)
    {
        /* Make sure absolute difference is below a threshold */
        if(abs(c[k] - hh_c[k]) > threshold)
            errors++;
    }
    printf("Errors: %d\n", errors);
    return errors;
}