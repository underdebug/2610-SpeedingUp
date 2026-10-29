
#include "camera.h"

Camera::Camera() {
    speed = 0.1;
    right = glm::normalize(glm::cross(forward, up));
    key_down.assign(256, false);
    special_key_down.assign(256, false);
}

void Camera::set_view() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, aspect, z_near, z_far);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(
        pos.x, pos.y, pos.z,
        pos.x + forward.x, pos.y + forward.y, pos.z + forward.z,
        up.x, up.y, up.z
    );
}

void Camera::look_at(const glm::dvec3& position, const glm::dvec3& target) {
    pos = position;
    forward = glm::normalize(target - position);
    up = glm::dvec3(0.0, 1.0, 0.0);
    right = glm::normalize(glm::cross(forward, up));
    up = glm::normalize(glm::cross(right, forward));
}

void Camera::handle_mouse_translate(double dx, double dy) {
    double s = glm::length(pos) * 0.001;
    pos -= right * (s * dx);
    pos += up * (s * dy);
}

void Camera::handle_wheel(int direction) {
    pos += forward * (direction * speed);
}

void Camera::handle_mouse_view(double dx, double dy) {
    double s = 0.005;
    glm::dquat qYaw = glm::angleAxis(-dx * s, glm::normalize(up));
    glm::dquat qPitch = glm::angleAxis(-dy * s, glm::normalize(right));

    forward = glm::normalize(qPitch * (qYaw * forward));

    right = glm::normalize(glm::cross(forward, up));
    right.y = 0.0;
    right = glm::normalize(right);
    up = glm::normalize(glm::cross(right, forward));
    forward = glm::normalize(glm::cross(up, right));
}

void Camera::handle_key_down(unsigned char key, int x, int y) {
    key_down[key] = true;
}

void Camera::handle_key_up(unsigned char key, int x, int y) {
    key_down[key] = false;
}

void Camera::handle_special_key_down(int key, int x, int y) {
    special_key_down[key] = true;
}

void Camera::handle_special_key_up(int key, int x, int y) {
    special_key_down[key] = false;
}

void Camera::handle_special_key() {
    if (special_key_down[GLUT_KEY_UP])    pos += forward * speed;
    if (special_key_down[GLUT_KEY_DOWN])  pos -= forward * speed;
    if (special_key_down[GLUT_KEY_LEFT])  pos -= right * speed;
    if (special_key_down[GLUT_KEY_RIGHT]) pos += right * speed;
}

void Camera::handle_keys() {
    if (key_down['w']) pos += forward * speed;
    if (key_down['s']) pos -= forward * speed;
    if (key_down['a']) pos -= right * speed;
    if (key_down['d']) pos += right * speed;
    if (key_down['q']) pos += up * speed;
    if (key_down['e']) pos -= up * speed;
}

void Camera::handle_mouse_orbit(double dx, double dy, const glm::dvec3& center) {
    glm::dvec3 offset = pos - center;

    double s = 0.01;
    glm::dquat qYaw = glm::angleAxis(-dx * s, glm::normalize(up));
    glm::dquat qPitch = glm::angleAxis(-dy * s, glm::normalize(right));

    offset = qPitch * (qYaw * offset);
    pos = center + offset;

    forward = glm::normalize(center - pos);
    right = glm::normalize(glm::cross(forward, glm::dvec3(0.0, 1.0, 0.0)));
    up = glm::normalize(glm::cross(right, forward));
}