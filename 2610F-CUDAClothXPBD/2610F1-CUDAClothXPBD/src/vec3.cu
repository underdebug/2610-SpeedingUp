
#include "vec3.cuh"

__device__ float dot(
    const float& a_x, const float& a_y, const float& a_z,
    const float& b_x, const float& b_y, const float& b_z) {
    return a_x * b_x + a_y * b_y + a_z * b_z;
}

__device__ Vec3<float> cross(
    const float& a_x, const float& a_y, const float& a_z,
    const float& b_x, const float& b_y, const float& b_z) {
    return Vec3<float>(
        a_y * b_z - a_z * b_y,
        a_z * b_x - a_x * b_z,
        a_x * b_y - a_y * b_x
        );
}

__global__ void accumulate(float* array, int* indices, float* values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        atomicAdd(&array[indices[idx]], values[idx]);
    }
}

__device__ void atomic_add(Vec3<float>* address, const Vec3<float>& value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
    atomicAdd(&(address->z), value.z);
}

__device__ void atomic_sub(Vec3<float>* address, const Vec3<float>& value) {
    atomicAdd(&(address->x), -value.x);
    atomicAdd(&(address->y), -value.y);
    atomicAdd(&(address->z), -value.z);
}

