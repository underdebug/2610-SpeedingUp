#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>

template<typename T>
struct Vec3 {
    T x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}

    __host__ __device__ Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

    // Copy constructor
    __host__ __device__ Vec3(const Vec3<T>& other) : x(other.x), y(other.y), z(other.z) {}

    __host__ __device__ Vec3<T> operator+(const Vec3<T>& other) const {
        return Vec3<T>(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Vec3<T> operator-(const Vec3<T>& other) const {
        return Vec3<T>(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Vec3<T> operator*(T scalar) const {
        return Vec3<T>(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ Vec3<T>& operator+=(const Vec3<T>& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ Vec3<T> operator/(T scalar) const {
        return Vec3<T>(x / scalar, y / scalar, z / scalar);
    }

    // Equality operator
    __host__ __device__ bool operator==(const Vec3<T>& other) const {
        return (x == other.x && y == other.y && z == other.z);
    }

    // Inequality operator (optional, but often useful)
    __host__ __device__ bool operator!=(const Vec3<T>& other) const {
        return !(*this == other);
    }

    __host__ __device__ float length() const {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
            return sqrtf(x * x + y * y + z * z); // Use sqrtf for float
        }
        else {
            return std::sqrt(static_cast<float>(x * x + y * y + z * z)); // Convert to float for integers
        }
    }

    __host__ __device__ Vec3<T> normalized() const {
        float len = length();
        return Vec3<T>(x / len, y / len, z / len);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3<T>& vec) {
        os << "Vec3(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
        return os;
    }
};

__device__ float dot(
    const float& a_x, const float& a_y, const float& a_z,
    const float& b_x, const float& b_y, const float& b_z);
__device__ Vec3<float> cross(
    const float& a_x, const float& a_y, const float& a_z,
    const float& b_x, const float& b_y, const float& b_z);

__device__ void atomic_add(Vec3<float>* address, const Vec3<float>& value);
__device__ void atomic_sub(Vec3<float>* address, const Vec3<float>& value);

__global__ void accumulate(float* array, int* indices, float* values, int numElements);

