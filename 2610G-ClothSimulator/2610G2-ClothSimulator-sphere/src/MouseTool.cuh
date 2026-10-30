#ifndef MOUSE_TOOL_CUH
#define MOUSE_TOOL_CUH

#include "Vector.cuh"
#include "Matrix.cuh"

// GLFW equivalent of the Qt navigation tools, dispatched axis -> pan -> rotate.
class MouseTool {
    bool left = false, right = false, middle = false, hasPosition = false;
    double lastX = 0, lastY = 0;
    float spin = 0, tilt = 0;
    Matrix4x4f rotate(const Vector3f& axis, float angle) const;
    static Vector3f direction(const Matrix4x4f& matrix, const Vector3f& v);
public:
    float scaling = 1.0f;
    Vector3f translation, pivotOffset;
    Matrix4x4f rotation = Matrix4x4f(1.0f);
    void reset();
    void anchor(double x, double y);
    void button(int button, int action, double x, double y);
    // Preserve the displayed scene when changing the orbit origin.
    void setPivotOffset(const Vector3f& offset);
    void move(double x, double y, int width, int height, const Vector3f& eye, int mods);
    void scroll(double xAmount, double yAmount, double x, double y,
                int width, int height, const Vector3f& eye, int mods);
};
#endif
