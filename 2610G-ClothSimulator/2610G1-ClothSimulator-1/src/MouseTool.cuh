#ifndef MOUSE_TOOL_CUH
#define MOUSE_TOOL_CUH

#include <algorithm>
#include <cmath>
#include <GLFW/glfw3.h>
#include "Vector.cuh"
#include "Matrix.cuh"

// GLFW equivalent of the Qt navigation tools, dispatched axis -> pan -> rotate.
class MouseTool {
    bool left = false, right = false, middle = false, hasPosition = false;
    double lastX = 0, lastY = 0;
    float spin = 0, tilt = 0;
    Matrix4x4f rotate(const Vector3f& axis, float angle) const;
    static Vector3f direction(const Matrix4x4f& matrix, const Vector3f& v) {
        const Vector4f r = matrix * Vector4f(v(0), v(1), v(2), 0.0f);
        return Vector3f(r(0), r(1), r(2));
    }
public:
    float scaling = 1.0f;
    Vector3f translation, pivotOffset;
    Matrix4x4f rotation = Matrix4x4f(1.0f);
    void reset() {
        scaling = 1.0f;
        translation = pivotOffset = Vector3f();
        rotation = Matrix4x4f(1.0f);
        spin = tilt = 0;
        left = right = middle = hasPosition = false;
    }
    void anchor(double x, double y) {
        lastX = x; lastY = y; hasPosition = true;
    }
    void button(int button, int action, double x, double y) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) left = action == GLFW_PRESS;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) right = action == GLFW_PRESS;
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) middle = action == GLFW_PRESS;
        anchor(x, y);
    }
    // Preserve the displayed scene when changing the orbit origin.
    void setPivotOffset(const Vector3f& offset) {
        const Vector3f delta = offset - pivotOffset;
        translation += scaling * direction(rotation, delta) - delta;
        pivotOffset = offset;
    }
    void move(double x, double y, int width, int height, const Vector3f& eye, int mods) {
        const float dx = hasPosition ? static_cast<float>(x - lastX) : 0;
        const float dy = hasPosition ? static_cast<float>(y - lastY) : 0;
        anchor(x, y);
        if (width <= 0 || height <= 0 || (dx == 0 && dy == 0)) return;
        const Vector3f forward = (-1.0f * eye).normalized();
        const Vector3f side = forward.cross(Vector3f(0, 0, 1)).normalized();
        const Vector3f up = side.cross(forward).normalized();
        const float pixelScale = 2.0f * eye.norm() * std::tan(3.14159265f / 8.0f) / height;
        const bool shift = (mods & GLFW_MOD_SHIFT) != 0;
        const bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
        if ((mods & (GLFW_MOD_ALT | GLFW_MOD_SUPER)) && !shift && !ctrl) {
            const Vector3f delta = (dx * side - dy * up) * pixelScale;
            setPivotOffset(pivotOffset + direction(rotation.transpose(), delta) / scaling);
        } else if (shift || (!ctrl && (left || middle))) {
            translation += (dx * side - dy * up) * pixelScale;
        } else if (ctrl || right) {
            spin = std::remainder(spin + dx * 0.5f, 360.0f);
            tilt = std::clamp(tilt + dy * 0.5f, -90.0f, 90.0f);
            const float radians = 3.14159265f / 180.0f;
            rotation = rotate(up, spin * radians) * rotate(side, tilt * radians);
        }
    }
    void scroll(double xAmount, double yAmount, double x, double y,
                int width, int height, const Vector3f& eye, int mods) {
        if (width <= 0 || height <= 0) return;
        const Vector3f forward = (-1.0f * eye).normalized();
        const Vector3f side = forward.cross(Vector3f(0, 0, 1)).normalized();
        const Vector3f up = side.cross(forward).normalized();
        const float pixelScale = 2.0f * eye.norm() * std::tan(3.14159265f / 8.0f) / height;
        if (mods & GLFW_MOD_ALT) {
            const float amount = static_cast<float>(yAmount != 0 ? yAmount : xAmount);
            const Vector3f delta = -6.0f * amount * pixelScale * forward;
            setPivotOffset(pivotOffset + direction(rotation.transpose(), delta) / scaling);
            return;
        }
        if (yAmount == 0) return;
        const float oldScale = scaling;
        scaling = std::clamp(scaling * static_cast<float>(std::pow(1.0 / 0.85, yAmount)), 0.01f, 100.0f);
        const float ratio = scaling / oldScale;
        const Vector3f cursor = ((static_cast<float>(x) - width * 0.5f) * side
                              + (height * 0.5f - static_cast<float>(y)) * up) * pixelScale;
        // Scale around the cursor's point on the original camera target plane.
        translation = ratio * translation + (1.0f - ratio) * (cursor - pivotOffset);
    }
};
#endif
