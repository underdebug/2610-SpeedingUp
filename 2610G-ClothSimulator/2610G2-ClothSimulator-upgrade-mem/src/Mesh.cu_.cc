// F7 imports only these stateless methods and preserves their Mesh.cu locations.
#include <iostream>
#include <stdexcept>
#include <vector>
#include "MathHelper.cuh"

int triangulate() {
    const std::vector<Vector3f> x = {{0,0,0}, {1,0,0}, {2,1,0}, {1,2,0}, {0,2,0}, {-1,1,0}};
    const auto* name = "hexagon";
    std::vector<int> polygon, uv, triangles, uvTriangles;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        polygon.push_back(i);
        uv.push_back(100 + i);
    }
    Mesh mesh;
    mesh.triangulate(x, polygon, uv, triangles, uvTriangles);
    if (triangles.size() != 3 * (x.size() - 2) || triangles.size() != uvTriangles.size())
        throw std::runtime_error("Wrong triangle count");
    float area = 0;
    for (std::size_t i = 0; i < triangles.size(); i += 3) {
        const auto a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
        if (a < 0 || b < 0 || c < 0 || a >= x.size() || b >= x.size() || c >= x.size())
            throw std::runtime_error("Invalid vertex index");
        const float cross = (x[b](0)-x[a](0))*(x[c](1)-x[a](1))
                         - (x[b](1)-x[a](1))*(x[c](0)-x[a](0));
        if (cross <= 0) throw std::runtime_error("Degenerate or reversed triangle");
        area += cross / 2;
        for (int j = 0; j < 3; ++j)
            if (uvTriangles[i+j] != 100 + triangles[i+j])
                throw std::runtime_error("UV indices do not match vertex indices");
    }
    float polygonArea = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto& a = x[i]; const auto& b = x[(i+1) % x.size()];
        polygonArea += (a(0)*b(1) - b(0)*a(1)) / 2;
    }
    if (std::fabs(area - polygonArea) > 1e-5f)
        throw std::runtime_error("Triangle areas do not cover polygon");
    std::cout << name << ": passed (" << triangles.size()/3 << " triangles)\n";

    // -exec source Mesh.cu_.py triangulate
    return 0;
}

int main() {
    triangulate();
}
