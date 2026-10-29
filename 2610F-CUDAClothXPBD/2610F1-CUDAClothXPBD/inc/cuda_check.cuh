#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

inline void check_cuda(cudaError_t result, const char* operation,
                       const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ':' << line << " ("
                  << operation << "): " << cudaGetErrorString(result)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) check_cuda((operation), #operation, __FILE__, __LINE__)
