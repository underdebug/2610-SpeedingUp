
#include "device_atomic_functions.h"
#include "kernels.cuh"
#include "vec3.cuh"
#include <cassert>

__device__ float length(const float& v_x, const float& v_y, const float& v_z) {
    return sqrtf(v_x * v_x + v_y * v_y + v_z * v_z);
}

__global__ void compute_rest_lengths(const float* pos_x, const float* pos_y, const float* pos_z,
    const int* const_ids, float* rest_lengths, int num_constraints) {
    int c_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_nr >= num_constraints) return;

    float p0_x = pos_x[const_ids[2 * c_nr]];
    float p0_y = pos_y[const_ids[2 * c_nr]];
    float p0_z = pos_z[const_ids[2 * c_nr]];
    float p1_x = pos_x[const_ids[2 * c_nr + 1]];
    float p1_y = pos_y[const_ids[2 * c_nr + 1]];
    float p1_z = pos_z[const_ids[2 * c_nr + 1]];
    rest_lengths[c_nr] = length(p1_x-p0_x, p1_y-p0_y, p1_z-p0_z);
}


__global__ void integrate(
    float dt, Vec3<float> gravity, float* inv_mass, 
    float* prev_pos_x, float* prev_pos_y, float* prev_pos_z,
    float* pos_x, float* pos_y, float* pos_z, 
    float* vel_x, float* vel_y, float* vel_z, 
    Vec3<float> sphere_center, float sphere_radius, int num_particles) {

    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr >= num_particles) return;

    if (inv_mass[p_nr] == 0.0f) {
        return;
    }
    prev_pos_x[p_nr] = pos_x[p_nr];
    prev_pos_y[p_nr] = pos_y[p_nr];
    prev_pos_z[p_nr] = pos_z[p_nr];
    vel_x[p_nr] = vel_x[p_nr] + gravity.x * dt;
    vel_y[p_nr] = vel_y[p_nr] + gravity.y * dt;
    vel_z[p_nr] = vel_z[p_nr] + gravity.z * dt;
    pos_x[p_nr] = pos_x[p_nr] + vel_x[p_nr] * dt;
    pos_y[p_nr] = pos_y[p_nr] + vel_y[p_nr] * dt;
    pos_z[p_nr] = pos_z[p_nr] + vel_z[p_nr] * dt;

    // collisions
    constexpr float thickness = 0.001f;
    constexpr float friction = 0.01f;

    float d = length(pos_x[p_nr]-sphere_center.x, pos_y[p_nr]-sphere_center.y, pos_z[p_nr]-sphere_center.z);
    if (d < (sphere_radius + thickness)) {
        float p_x = pos_x[p_nr] * (1.0f - friction) + prev_pos_x[p_nr] * friction;
        float p_y = pos_y[p_nr] * (1.0f - friction) + prev_pos_y[p_nr] * friction;
        float p_z = pos_z[p_nr] * (1.0f - friction) + prev_pos_z[p_nr] * friction;
        float r_x = p_x - sphere_center.x;
        float r_y = p_y - sphere_center.y;
        float r_z = p_z - sphere_center.z;
        d = length(r_x, r_y, r_z);
        pos_x[p_nr] = sphere_center.x + r_x * ((sphere_radius + thickness) / d);
        pos_y[p_nr] = sphere_center.y + r_y * ((sphere_radius + thickness) / d);
        pos_z[p_nr] = sphere_center.z + r_z * ((sphere_radius + thickness) / d);
    }

    float p_x = pos_x[p_nr];
    float p_y = pos_y[p_nr];
    float p_z = pos_z[p_nr];
    if (p_y < thickness) {
        p_x = pos_x[p_nr] * (1.0f - friction) + prev_pos_x[p_nr] * friction;
        p_y = pos_y[p_nr] * (1.0f - friction) + prev_pos_y[p_nr] * friction;
        p_z = pos_z[p_nr] * (1.0f - friction) + prev_pos_z[p_nr] * friction;
        pos_x[p_nr] = p_x; 
        pos_y[p_nr] = thickness; 
        pos_z[p_nr] = p_z;
    }
}


__global__ void solve_distance_constraints(
    int solve_type, int first_constraint, float* inv_mass, 
    float* pos_x, float* pos_y, float* pos_z,
    Vec3<float>* corr, int* const_ids, float* rest_lengths, int num_constraints) {

    int cNr = first_constraint + blockIdx.x * blockDim.x + threadIdx.x;
    if (cNr >= first_constraint + num_constraints) return;

    int id0 = const_ids[2 * cNr];
    int id1 = const_ids[2 * cNr + 1];
    float w0 = inv_mass[id0];
    float w1 = inv_mass[id1];
    float w = w0 + w1;
    if (w == 0.0f) {
        return;
    }
    float p0_x = pos_x[id0];
    float p0_y = pos_y[id0];
    float p0_z = pos_z[id0];
    float p1_x = pos_x[id1];
    float p1_y = pos_y[id1];
    float p1_z = pos_z[id1];
    float d_x = p1_x - p0_x;
    float d_y = p1_y - p0_y;
    float d_z = p1_z - p0_z;
    float l = length(d_x, d_y, d_z);
    float n_x = d_x / l;
    float n_y = d_y / l;
    float n_z = d_z / l;
    float l0 = rest_lengths[cNr];
    float dP_x = n_x * (l - l0) / w;
    float dP_y = n_y * (l - l0) / w;
    float dP_z = n_z * (l - l0) / w;

    if (solve_type == 1) {
        atomicAdd(&(corr[id0].x), dP_x * w0);
        atomicAdd(&(corr[id0].y), dP_y * w0);
        atomicAdd(&(corr[id0].z), dP_z * w0);

        atomicAdd(&(corr[id1].x), -dP_x * w1);
        atomicAdd(&(corr[id1].y), -dP_y * w1);
        atomicAdd(&(corr[id1].z), -dP_z * w1);
    }
    else {
        atomicAdd(&pos_x[id0], dP_x * w0);
        atomicAdd(&pos_y[id0], dP_y * w0);
        atomicAdd(&pos_z[id0], dP_z * w0);
        
        atomicAdd(&pos_x[id1], -dP_x * w1);
        atomicAdd(&pos_y[id1], -dP_y * w1);
        atomicAdd(&pos_z[id1], -dP_z * w1);
    }
}


__global__ void add_normals(
    float* pos_x, float* pos_y, float* pos_z,
    int* tri_ids, Vec3<float>* normals, int num_triangles) {

    int tri_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_nr >= num_triangles) return;

    int id0 = tri_ids[3 * tri_nr];
    int id1 = tri_ids[3 * tri_nr + 1];
    int id2 = tri_ids[3 * tri_nr + 2];

    float a_x = pos_x[id1] - pos_x[id0];
    float a_y = pos_y[id1] - pos_y[id0];
    float a_z = pos_z[id1] - pos_z[id0];
    float b_x = pos_x[id2] - pos_x[id0];
    float b_y = pos_y[id2] - pos_y[id0];
    float b_z = pos_z[id2] - pos_z[id0];
    Vec3<float> normal = cross(a_x, a_y, a_z, b_x, b_y, b_z);

    atomic_add(&normals[id0], normal);
    atomic_add(&normals[id1], normal);
    atomic_add(&normals[id2], normal);
}


__global__ void normalize_normals(Vec3<float>* normals, int num_normals) {
    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr < num_normals) {
        normals[p_nr] = normals[p_nr].normalized();
    }
}

__global__ void add_corrections(float* pos_x, float* pos_y, float* pos_z, Vec3<float>* corr, float scale, int num_particles) {
    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr >= num_particles) return;

    pos_x[p_nr] = pos_x[p_nr] + corr[p_nr].x * scale;
    pos_y[p_nr] = pos_y[p_nr] + corr[p_nr].y * scale;
    pos_z[p_nr] = pos_z[p_nr] + corr[p_nr].z * scale;
}

__global__ void update_vel(float dt,
    float* prev_pos_x, float* prev_pos_y, float* prev_pos_z,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z, int num_particles) {
    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr >= num_particles) return;

    vel_x[p_nr] = (pos_x[p_nr] - prev_pos_x[p_nr]) / dt;
    vel_y[p_nr] = (pos_y[p_nr] - prev_pos_y[p_nr]) / dt;
    vel_z[p_nr] = (pos_z[p_nr] - prev_pos_z[p_nr]) / dt;
}


__global__ void raycast_triangle(
    Vec3<float> orig, Vec3<float> dir, 
    float* pos_x, float* pos_y, float* pos_z,
    int* tri_ids, float* dist, int num_tris) {

    int tri_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_nr >= num_tris) return;

    float no_hit = 1.0e6;

    int id0 = tri_ids[3 * tri_nr];
    int id1 = tri_ids[3 * tri_nr + 1];
    int id2 = tri_ids[3 * tri_nr + 2];

    float edge1_x = pos_x[id1] - pos_x[id0];
    float edge1_y = pos_y[id1] - pos_y[id0];
    float edge1_z = pos_z[id1] - pos_z[id0];
    float edge2_x = pos_x[id2] - pos_x[id0];
    float edge2_y = pos_y[id2] - pos_y[id0];
    float edge2_z = pos_z[id2] - pos_z[id0];
    Vec3<float> pvec = cross(dir.x, dir.y, dir.z, edge2_x, edge2_y, edge2_z);
    float det = dot(edge1_x, edge1_y, edge1_z, pvec.x, pvec.y, pvec.z);

    if (det == 0.0f) {
        dist[tri_nr] = no_hit;
        return;
    }

    float inv_det = 1.0f / det;
    float tvec_x = orig.x - pos_x[id0];
    float tvec_y = orig.y - pos_y[id0];
    float tvec_z = orig.z - pos_z[id0];
    float u = dot(tvec_x, tvec_y, tvec_z, pvec.x, pvec.y, pvec.z) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        dist[tri_nr] = no_hit;
        return;
    }

    Vec3<float> qvec = cross(tvec_x, tvec_y, tvec_z, edge1_x, edge1_y, edge1_z);
    float v = dot(dir.x, dir.y, dir.z, qvec.x, qvec.y, qvec.z) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        dist[tri_nr] = no_hit;
        return;
    }

    dist[tri_nr] = dot(edge2_x, edge2_y, edge2_z, qvec.x, qvec.y, qvec.z) * inv_det;
}



#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

