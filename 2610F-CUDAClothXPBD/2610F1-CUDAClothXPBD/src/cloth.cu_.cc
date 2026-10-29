// CPU test of the original build_triangles implementation in cloth.cu.
// Build: python3 mem/extension/kernel.py src/cloth.cu.cc --output build/cloth
// Break at printf, open cloth.cu.py, and press F8 to plot using mem().
#include <algorithm>
#include <cstdio>
#include <vector>

#include "cloth.cuh"

void build_triangles() {
    constexpr int num_x = 2;
    constexpr int num_y = 4;
    constexpr float height = 2.2f;
    constexpr float spacing = 0.1f;
    constexpr int num_particles = (num_x + 1) * (num_y + 1);

    // Call the original function directly, with only CUDA transfers disabled.
    // The original constructor calls build_triangles with these dimensions.
    Cloth cloth(256, 1, height, num_x, num_y, spacing, Vec3<float>(0.0f, 1.5f, 0.0f), 0.4f);

    // Keep host arrays in this frame for Python's mem().
    std::vector<int> h_tri_ids(cloth.get_host_tri_ids(), cloth.get_host_tri_ids() + 3 * cloth.get_num_tris());
    std::vector<float> h_pos_x(num_particles);
    std::vector<float> h_pos_y(num_particles, height);
    std::vector<float> h_pos_z(num_particles);
    for (int id = 0; id < num_particles; ++id) {
        const int xi = id / (num_y + 1);
        const int yi = id % (num_y + 1);
        h_pos_x[id] = (-num_x * 0.5f + xi) * spacing;
        h_pos_z[id] = (-num_y * 0.5f + yi) * spacing;
    }

    // -exec source /src/cloth.cu.py build_triangles_cc NODELETE
    // -exec source cloth.cu.py simulate_step_cc
    printf("here\n");
}

int main() {
    build_triangles();
    return 0;
}
