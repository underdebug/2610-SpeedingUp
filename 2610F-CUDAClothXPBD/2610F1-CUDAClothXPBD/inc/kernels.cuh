#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "vec3.cuh"

__device__ float length(const float& v_x, const float& v_y, const float& v_z);

__global__ void compute_rest_lengths(const float* pos_x, const float* pos_y, const float* pos_z, 
    const int* constIds, float* restLengths, int numConstraints);
__global__ void integrate(float dt, Vec3<float> gravity, float* invMass, 
    float* prev_pos_x, float* prev_pos_y, float* prev_pos_z,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    Vec3<float> sphereCenter, float sphereRadius, int numParticles);
__global__ void solve_distance_constraints(int solveType, int firstConstraint, float* invMass,
    float* pos_x, float* pos_y, float* pos_z, Vec3<float>* corr, int* constIds,
    float* restLengths, int numConstraints);
__global__ void add_corrections(float* pos_x, float* pos_y, float* pos_z, Vec3<float>* corr, float scale, int numParticles);
__global__ void update_vel(float dt, 
    float* prev_pos_x, float* prev_pos_y, float* prev_pos_z, 
    float* pos_x, float* pos_y, float* pos_z, 
    float* vel_x, float* vel_y, float* vel_z, int num_particles);
__global__ void add_normals(float* pos_x, float* pos_y, float* pos_z, int* triIds, Vec3<float>* normals, int numTriangles);
__global__ void normalize_normals(Vec3<float>* normals, int numNormals);

__global__ void raycast_triangle(Vec3<float> orig, Vec3<float> dir, float* pos_x, float* pos_y, float* pos_z, 
    int* tri_ids, float* dist, int num_tris);

