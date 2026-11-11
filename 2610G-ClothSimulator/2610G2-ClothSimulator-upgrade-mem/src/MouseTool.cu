#include "MouseTool.cuh"

#include <algorithm>
#include <cmath>
#include <GLFW/glfw3.h>

Matrix4x4f MouseTool::rotate(const Vector3f& v, float angle) const {
    Vector3f axis = v.normalized();
    float s = sin(angle);
    float c = cos(angle);
    Matrix4x4f ans;

    ans(0, 0) = (1.0f - c) * axis(0) * axis(0) + c;
    ans(0, 1) = (1.0f - c) * axis(1) * axis(0) - s * axis(2);
    ans(0, 2) = (1.0f - c) * axis(2) * axis(0) + s * axis(1);

    ans(1, 0) = (1.0f - c) * axis(0) * axis(1) + s * axis(2);
    ans(1, 1) = (1.0f - c) * axis(1) * axis(1) + c;
    ans(1, 2) = (1.0f - c) * axis(2) * axis(1) - s * axis(0);

    ans(2, 0) = (1.0f - c) * axis(0) * axis(2) - s * axis(1);
    ans(2, 1) = (1.0f - c) * axis(1) * axis(2) + s * axis(0);
    ans(2, 2) = (1.0f - c) * axis(2) * axis(2) + c;

    ans(3, 3) = 1.0f;

    return ans;
}

Vector3f MouseTool::direction(const Matrix4x4f& matrix, const Vector3f& v) {
    const Vector4f r = matrix * Vector4f(v(0), v(1), v(2), 0.0f);
    return Vector3f(r(0), r(1), r(2));
}

void MouseTool::reset() {
    scaling = 1.0f;
    translation = pivotOffset = Vector3f();
    rotation = Matrix4x4f(1.0f);
    spin = tilt = 0;
    left = right = middle = hasPosition = false;
}

void MouseTool::anchor(double x, double y) {
    lastX = x; lastY = y; hasPosition = true;
}

void MouseTool::button(int button, int action, double x, double y) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) middle = action == GLFW_PRESS;
    anchor(x, y);
}

void MouseTool::setPivotOffset(const Vector3f& offset) {
    const Vector3f delta = offset - pivotOffset;
    translation += scaling * direction(rotation, delta) - delta;
    pivotOffset = offset;
}

void MouseTool::move(double x, double y, int width, int height, const Vector3f& eye, int mods) {
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

void MouseTool::scroll(double xAmount, double yAmount, double x, double y,
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
