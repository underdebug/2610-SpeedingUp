// F6 replays the CUDA kernels on the CPU and keeps their original source lines.
#include <iostream>
#include <stdexcept>
#include <vector>
#include "kernel.h"
#include "RemeshingHelper.cu"

int main() {
    Node storage[5];
    const Node* nodes[5];
    Vector3f positions[5];
    std::vector<NearPoint> points(5, NearPoint(INFINITY, Vector3f()));
    for (int i = 0; i < 5; ++i) {
        storage[i].x = Vector3f(float(i), float(i + 1), float(i + 2));
        nodes[i] = &storage[i];
    }
    // Four replay threads also exercise the grid-stride loop's second pass.
    setX<<<2, 2>>>(5, nodes, positions);
    initializeNearPoints<<<2, 2>>>(5, positions, points.data());
    for (int i = 0; i < 5; ++i) {
        if ((positions[i] - storage[i].x).norm2() != 0.0f ||
            (points[i].x - storage[i].x).norm2() != 0.0f)
            throw std::runtime_error("Remeshing position initialization failed");
    }
    const auto matrix = diagonal(Vector2f(2.0f, 3.0f));
    if (matrix(0, 0) != 2.0f || matrix(1, 1) != 3.0f ||
        matrix(0, 1) != 0.0f || matrix(1, 0) != 0.0f)
        throw std::runtime_error("Remeshing diagonal failed");
    std::cout << "remeshing: passed (5 nodes, diagonal)\n";
    return 0;
}
