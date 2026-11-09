
#include "function.h"

__global__ void modify(int* a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = idx;
    }
}

__global__ void add(int a, int b, int* result)
{
    *result = a + b;
}

__global__ void sub(int a, int b, int* result)
{
    *result = a - b;
}
