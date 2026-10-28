#include "cloth.cuh"
#include "app_config.h"
#include "cuda_check.cuh"

Cloth::Cloth(int block_size, int num_substeps, float y_offset, int num_x, int num_y, float spacing, 
    const Vec3<float>& sphere_center, float sphere_radius)
    :   block_size(block_size), num_substeps(num_substeps), sphere_center(sphere_center), 
        sphere_radius(sphere_radius), spacing(spacing) {

    init_params(num_x, num_y);
    alloc_host_buffers(y_offset, num_x, num_y);
    alloc_device_buffers();

    build_constraints(num_x, num_y);
    build_triangles(num_x, num_y);
}

// Destructor
Cloth::~Cloth() {
    // Free host memory
    delete[] h_pos_x;
    delete[] h_pos_y;
    delete[] h_pos_z;
    delete[] h_inv_mass;
    delete[] h_corr;
    delete[] h_normals;
    delete[] h_tri_ids;

    // Free device memory
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_prev_pos_x));
    CUDA_CHECK(cudaFree(d_prev_pos_y));
    CUDA_CHECK(cudaFree(d_prev_pos_z));
    CUDA_CHECK(cudaFree(d_rest_pos_x));
    CUDA_CHECK(cudaFree(d_rest_pos_y));
    CUDA_CHECK(cudaFree(d_rest_pos_z));
    CUDA_CHECK(cudaFree(d_inv_mass));
    CUDA_CHECK(cudaFree(d_corr));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));
    CUDA_CHECK(cudaFree(d_normals));
    CUDA_CHECK(cudaFree(d_dist_const_ids));
    CUDA_CHECK(cudaFree(d_const_rest_lengths));
    CUDA_CHECK(cudaFree(d_tri_ids));
}

void Cloth::init_params(int& num_x, int& num_y) {

    time_step = 1.0f / 60.0f;
    solve_type = 0;
    jacobi_scale = 0.2f;
    gravity = Vec3<float>(0.0f, -10.0f, 0.0f);

    drag_particle_nr = -1;
    drag_depth = 0.0f;
    drag_inv_mass = 0.0f;

    num_particles = (num_x + 1) * (num_y + 1);
}

void Cloth::alloc_host_buffers(float y_offset, int num_x, int num_y) {
    h_pos_x = new float[num_particles];
    h_pos_y = new float[num_particles];
    h_pos_z = new float[num_particles];
    h_normals = new Vec3<float>[num_particles];
    h_inv_mass = new float[num_particles];
    h_corr = new Vec3<float>[num_particles];

    for (int i = 0; i < num_particles; i++) {
        h_inv_mass[i] = 1.0f;
        h_corr[i] = Vec3<float>(0.0f, 0.0f, 0.0f);
        h_normals[i] = Vec3<float>(0.0f, 0.0f, 0.0f);
    }

    // initialize positions
    for (int xi = 0; xi <= num_x; ++xi) {
        for (int yi = 0; yi <= num_y; ++yi) {
            int id = xi * (num_y + 1) + yi;
            h_pos_x[id] = ((-num_x * 0.5f + xi) * spacing);
            h_pos_y[id] = y_offset;
            h_pos_z[id] = ((-num_y * 0.5f + yi) * spacing);
        }
    }
}

void Cloth::alloc_device_buffers() {
    CUDA_CHECK(cudaMalloc(&d_pos_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, num_particles * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_prev_pos_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_pos_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_pos_z, num_particles * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_rest_pos_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rest_pos_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rest_pos_z, num_particles * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_inv_mass, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr, num_particles * sizeof(Vec3<float>)));
    
    CUDA_CHECK(cudaMalloc(&d_vel_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, num_particles * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_vel_x, 0, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_y, 0, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_z, 0, num_particles * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_normals, num_particles * sizeof(Vec3<float>)));

    CUDA_CHECK(cudaMemcpy(d_corr, h_corr, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z, num_particles * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_prev_pos_x, h_pos_x, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev_pos_y, h_pos_y, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev_pos_z, h_pos_z, num_particles * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_rest_pos_x, h_pos_x, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rest_pos_y, h_pos_y, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rest_pos_z, h_pos_z, num_particles * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_inv_mass, h_inv_mass, num_particles * sizeof(float), cudaMemcpyHostToDevice));
}

void Cloth::build_constraints(int num_x, int num_y) {
    pass_sizes = {
        (num_x + 1) * ((num_y + 1) / 2),
        (num_x + 1) * int(floor(num_y / 2)),
        ((num_x + 1) / 2) * (num_y + 1),
        int(floor(num_x / 2)) * (num_y + 1),
        2 * num_x * num_y + (num_x + 1) * (num_y - 1) + (num_y + 1) * (num_x - 1)
    };

    pass_independent = { true, true, true, true, false };

    num_dist_constraints = 0;
    for (int pass_size : pass_sizes) {
        num_dist_constraints += pass_size;
    }

    std::vector<int> dist_const_ids(2 * num_dist_constraints);

    int i = 0;

    for (int pass_nr = 0; pass_nr < 2; ++pass_nr) {
        for (int xi = 0; xi <= num_x; ++xi) {
            for (int yi = 0; yi < (num_y + 1 - pass_nr) / 2; ++yi) {
                dist_const_ids[2 * i] = xi * (num_y + 1) + 2 * yi + pass_nr;
                dist_const_ids[2 * i + 1] = xi * (num_y + 1) + 2 * yi + pass_nr + 1;
                ++i;
            }
        }
    }
    // 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  1   2   5   6   9  10  13  14  17  18

    for (int pass_nr = 0; pass_nr < 2; ++pass_nr) {
        for (int xi = 0; xi < (num_x + 1 - pass_nr) / 2; ++xi) {
            for (int yi = 0; yi <= num_y; ++yi) {
                dist_const_ids[2 * i] = (2 * xi + pass_nr) * (num_y + 1) + yi;
                dist_const_ids[2 * i + 1] = (2 * xi + pass_nr + 1) * (num_y + 1) + yi;
                ++i;
            }
        }
    }

    // 0   4   1   5   2   6   3   7   8  12   9  13  10  14  11  15   4   8   5   9   6  10   7  11  12  16  13  17  14  18  15  19 

    for (int xi = 0; xi < num_x; ++xi) {
        for (int yi = 0; yi < num_y; ++yi) {
            dist_const_ids[2 * i] = xi * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = (xi + 1) * (num_y + 1) + yi + 1;
            ++i;

            dist_const_ids[2 * i] = (xi + 1) * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = xi * (num_y + 1) + yi + 1;
            ++i;
        }
    }

    for (int xi = 0; xi <= num_x; ++xi) {
        for (int yi = 0; yi < num_y - 1; ++yi) {
            dist_const_ids[2 * i] = xi * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = xi * (num_y + 1) + yi + 2;
            ++i;
        }
    }

    for (int xi = 0; xi < num_x - 1; ++xi) {
        for (int yi = 0; yi <= num_y; ++yi) {
            dist_const_ids[2 * i] = xi * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = (xi + 2) * (num_y + 1) + yi;
            ++i;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_dist_const_ids, 2 * num_dist_constraints * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dist_const_ids, dist_const_ids.data(),
        2 * num_dist_constraints * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_const_rest_lengths, num_dist_constraints * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_const_rest_lengths, 0, num_dist_constraints * sizeof(float)));

    int threads_per_block = block_size;
    int blocks_per_grid = (num_dist_constraints + threads_per_block - 1) / threads_per_block;
    compute_rest_lengths << <blocks_per_grid, threads_per_block >> > (
        d_pos_x, d_pos_y, d_pos_z, d_dist_const_ids, d_const_rest_lengths, num_dist_constraints
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    // -exec source cloth.cu.py build_constraints
}

void Cloth::build_triangles(int num_x, int num_y) {
    num_tris = 2 * num_x * num_y;

    h_tri_ids = new int[3 * num_tris];
    for (int i = 0; i < 3 * num_tris; ++i) {
        h_tri_ids[i] = 0;
    }

    int i = 0;
    for (int xi = 0; xi < num_x; ++xi) {
        for (int yi = 0; yi < num_y; ++yi) {
            int id0 = xi * (num_y + 1) + yi;
            int id1 = (xi + 1) * (num_y + 1) + yi;
            int id2 = (xi + 1) * (num_y + 1) + yi + 1;
            int id3 = xi * (num_y + 1) + yi + 1;

            h_tri_ids[i] = id0;
            h_tri_ids[i + 1] = id1;
            h_tri_ids[i + 2] = id2;

            h_tri_ids[i + 3] = id0;
            h_tri_ids[i + 4] = id2;
            h_tri_ids[i + 5] = id3;

            i += 6;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_tri_ids, 3 * num_tris * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tri_ids, h_tri_ids, 3 * num_tris * sizeof(int), cudaMemcpyHostToDevice));

    h_tri_dist = new float[num_tris];
    std::fill(h_tri_dist, h_tri_dist + num_tris, no_hit);

    CUDA_CHECK(cudaMalloc(&d_tri_dist, num_tris * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tri_dist, h_tri_dist, num_tris * sizeof(float), cudaMemcpyHostToDevice));

    // -exec source cloth.cu.py build_triangles Agg NODELETE
    // -exec source cloth.cu.py build_triangles QtAgg
}

void Cloth::simulate_step() {
    // -exec source cloth.cu.py simulate_step
    // -exec break if times == 30
    static int times = 0; times++;    

    float dt = time_step / num_substeps;
    int num_passes = pass_sizes.size();

    for (int step = 0; step < num_substeps; ++step) {
        // Step 1: Integrate positions and velocities
        integrate << <(num_particles + block_size-1) / block_size, block_size >> > (dt, gravity, d_inv_mass,
            d_prev_pos_x, d_prev_pos_y, d_prev_pos_z, 
            d_pos_x, d_pos_y, d_pos_z, 
            d_vel_x, d_vel_y, d_vel_z, 
            sphere_center, sphere_radius, num_particles);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Solve constraints
        if (solve_type == 0) {
            int first_constraint = 0;
            for (int pass_nr = 0; pass_nr < num_passes; ++pass_nr) {
                // -exec break if times == 30
                int num_constraints = pass_sizes[pass_nr];

                if (pass_independent[pass_nr]) {
                    // -exec break if times == 15
                    solve_distance_constraints << <(num_constraints + block_size-1) / block_size, block_size >> > (
                        0, first_constraint, d_inv_mass, 
                        d_pos_x, d_pos_y, d_pos_z, d_corr,
                        d_dist_const_ids, d_const_rest_lengths, num_constraints);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                else {
                    CUDA_CHECK(cudaMemset(d_corr, 0, num_particles * sizeof(Vec3<float>)));
                    solve_distance_constraints << <(num_constraints + block_size-1) / block_size, block_size >> > (
                        1, first_constraint, d_inv_mass, 
                        d_pos_x, d_pos_y, d_pos_z, d_corr,
                        d_dist_const_ids, d_const_rest_lengths, num_constraints);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    add_corrections << <(num_particles + block_size-1) / block_size, block_size >> > (
                        d_pos_x, d_pos_y, d_pos_z, d_corr, jacobi_scale, num_particles);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                first_constraint += num_constraints;
            }
        }
        else if (solve_type == 1) {
            CUDA_CHECK(cudaMemset(d_corr, 0, num_particles * sizeof(Vec3<float>)));
            solve_distance_constraints << <(num_dist_constraints + block_size-1) / block_size, block_size >> > (
                1, 0, d_inv_mass, d_pos_x, d_pos_y, d_pos_z, d_corr,
                d_dist_const_ids, d_const_rest_lengths, num_dist_constraints);
            CUDA_CHECK(cudaDeviceSynchronize());
            add_corrections << <(num_particles + block_size-1) / block_size, block_size >> > (
                d_pos_x, d_pos_y, d_pos_z, d_corr, jacobi_scale, num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step 3: Update velocities based on the new positions
        update_vel << <(num_particles + block_size-1) / block_size, block_size >> > (dt, 
            d_prev_pos_x, d_prev_pos_y, d_prev_pos_z, 
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, num_particles);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Step 4: Copy the updated positions back to the host
    CUDA_CHECK(cudaMemcpy(h_pos_x, d_pos_x, num_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_y, d_pos_y, num_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_z, d_pos_z, num_particles * sizeof(float), cudaMemcpyDeviceToHost));
}

void Cloth::update_mesh() {
    // Set the normals array to zero
    CUDA_CHECK(cudaMemset(d_normals, 0, num_particles * sizeof(Vec3<float>)));

    int threads_per_block = block_size;
    int blocks_per_grid = (num_tris + threads_per_block - 1) / threads_per_block;
    add_normals << <blocks_per_grid, threads_per_block >> > (d_pos_x, d_pos_y, d_pos_z, d_tri_ids, d_normals, num_tris);
    CUDA_CHECK(cudaDeviceSynchronize());

    blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;
    normalize_normals << <blocks_per_grid, threads_per_block >> > (d_normals, num_particles);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_normals, d_normals, num_particles * sizeof(Vec3<float>), cudaMemcpyDeviceToHost));
}

void Cloth::reset() {
    // Set velocity array to zero
    CUDA_CHECK(cudaMemset(d_vel_x, 0, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_y, 0, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_z, 0, num_particles * sizeof(float)));

    // Copy rest positions to current positions
    CUDA_CHECK(cudaMemcpy(d_pos_x, d_rest_pos_x, num_particles * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, d_rest_pos_y, num_particles * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, d_rest_pos_z, num_particles * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Cloth::raycast_triangle_launch(const Vec3<float>& orig, const Vec3<float>& dir) {
    // initialize distances on device (no_hit)
    std::fill(h_tri_dist, h_tri_dist + num_tris, no_hit);
    CUDA_CHECK(cudaMemcpy(d_tri_dist, h_tri_dist, num_tris * sizeof(float), cudaMemcpyHostToDevice));

    int threads_per_block = block_size;
    int blocks_per_grid = (num_tris + threads_per_block - 1) / threads_per_block;

    raycast_triangle << <blocks_per_grid, threads_per_block >> > (
        orig, dir, d_pos_x, d_pos_y, d_pos_z, d_tri_ids, d_tri_dist, num_tris);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_tri_dist, d_tri_dist, num_tris * sizeof(float), cudaMemcpyDeviceToHost));
}


void Cloth::start_drag(const Vec3<float>& orig, const Vec3<float>& dir) {

    // Launch the raycast_triangle kernel
    raycast_triangle_launch(orig, dir);

    // Initialize drag_depth
    drag_depth = 0.0f;

    // Find the triangle with the minimum distance to the ray
    int min_tri_nr = std::min_element(h_tri_dist, h_tri_dist + num_tris) - h_tri_dist;

    // Check if the closest triangle was hit
    if (h_tri_dist[min_tri_nr] < 1.0e6) {
        // Identify the particle associated with the closest triangle
        drag_particle_nr = h_tri_ids[3 * min_tri_nr];
        drag_depth = h_tri_dist[min_tri_nr];
        drag_inv_mass = h_inv_mass[drag_particle_nr];

        // Lock the particle by setting its inverse mass to 0
        h_inv_mass[drag_particle_nr] = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_inv_mass, h_inv_mass, num_particles * sizeof(float), cudaMemcpyHostToDevice));

        // Calculate the drag position based on the ray origin and direction
        Vec3<float> drag_pos = orig + dir * drag_depth;
        h_pos_x[drag_particle_nr] = drag_pos.x;
        h_pos_y[drag_particle_nr] = drag_pos.y;
        h_pos_z[drag_particle_nr] = drag_pos.z;

        // Copy the updated particle position back to the device
        CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x, num_particles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y, num_particles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Cloth::drag(const Vec3<float>& orig, const Vec3<float>& dir) {
    if (drag_particle_nr >= 0) {
        Vec3<float> drag_pos = orig + dir * drag_depth;
        h_pos_x[drag_particle_nr] = drag_pos.x;
        h_pos_y[drag_particle_nr] = drag_pos.y;
        h_pos_z[drag_particle_nr] = drag_pos.z;
        CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x, num_particles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y, num_particles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z, num_particles * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Cloth::end_drag() {
    if (drag_particle_nr >= 0) {
        h_inv_mass[drag_particle_nr] = drag_inv_mass;
        CUDA_CHECK(cudaMemcpy(d_inv_mass, h_inv_mass, num_particles * sizeof(float), cudaMemcpyHostToDevice));
        drag_particle_nr = -1;
    }
}


