// Build and run: cmake --build build --target NrcTest && ./build/NrcTest
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include "Evaluate.cuh"

__global__ void Nrc_Test(double* knot, double* results) {
    results[0] = Nabla(knot, 1, 1, 2);
    results[1] = Nrc(knot, 0, 1, 1, 1, 1, 1);
    results[2] = Nrc(knot, 0, 1, 2, 1, 1, 1);

    for (int r = 1; r <= 2; ++r) {
        for (int c = 1; c <= 2; ++c) {
            results[3 + (r - 1) * 2 + c - 1] = Nrc(knot, 1, r, c, 1, 1, 1);
        }
    }
}

int Main() {
    const double knot[] = {0, 1, 2, 3, 4};
    const double expected[] = {2, 0, 1, 1, 0, -1, 1};
    double results[7] = {};
    double* device_knot = nullptr;
    double* device_results = nullptr;
    cudaError_t error = cudaMalloc(&device_knot, sizeof(knot));
    if (error == cudaSuccess) error = cudaMalloc(&device_results, sizeof(results));
    if (error == cudaSuccess) error = cudaMemcpy(device_knot, knot, sizeof(knot), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
        Nrc_Test<<<1, 1>>>(device_knot, device_results);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) error = cudaMemcpy(results, device_results, sizeof(results), cudaMemcpyDeviceToHost);
    if (device_results) cudaFree(device_results);
    if (device_knot) cudaFree(device_knot);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    for (int i = 0; i < 7; ++i) {
        if (std::fabs(results[i] - expected[i]) > 1e-10) {
            std::fprintf(stderr, "result[%d]: expected %g, got %g\n", i, expected[i], results[i]);
            return 1;
        }
    }
    std::puts("Nrc test passed");
    return 0;
}


// TODO, fix _.cu _.cc test action,using cuda header,
