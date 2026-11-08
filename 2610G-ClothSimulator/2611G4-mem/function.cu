
#include "function.h"

__global__ void add(int a, int b, int* result)
{
    *result = a + b;
}

__global__ void sub(int a, int b, int* result)
{
    *result = a - b;
}