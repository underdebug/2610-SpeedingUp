#include "function.h"
#include <cstdio>

int main()
{
    int* result = nullptr;
    cudaMallocManaged(&result, sizeof(int));
    *result = 0;

    add<<<1, 1>>>(2, 3, result);
    cudaDeviceSynchronize();

    cudaFree(result);
    return 1;
}
