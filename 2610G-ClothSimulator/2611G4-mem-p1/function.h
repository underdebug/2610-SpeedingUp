
#pragma once
#include <cuda_runtime.h>

__global__ void modify(int* a, int n);
__global__ void add(int a, int b, int* result);
__global__ void sub(int a, int b, int* result);
