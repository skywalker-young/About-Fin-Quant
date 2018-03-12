double horner(double p[], int n, double x)  
{  
    double sum;  
    sum = p[--n];  
    while ( n > 0 )  
    {  
        sum = p[--n] + sum * x;  
    }  
    return sum;  
}  

