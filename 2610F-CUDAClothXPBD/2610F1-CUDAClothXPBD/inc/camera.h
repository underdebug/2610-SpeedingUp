#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>


class Camera {
public:
    Camera();

    void set_view();
    void look_at(const glm::dvec3& position, const glm::dvec3& target);

    void handle_mouse_translate(double dx, double dy);
    void handle_wheel(int direction);
    void handle_mouse_view(double dx, double dy);
    void handle_key_down(unsigned char key, int x, int y);
    void handle_key_up(unsigned char key, int x, int y);
    void handle_special_key_down(int key, int x, int y);
    void handle_special_key_up(int key, int x, int y);
    void handle_special_key();
    void handle_keys();
    void handle_mouse_orbit(double dx, double dy, const glm::dvec3& center);

private:
    glm::dvec3 pos{ 0.0, 1.0, 5.0 };
    glm::dvec3 forward{ 0.0, 0.0, -1.0 };
    glm::dvec3 up{ 0.0, 1.0, 0.0 };
    glm::dvec3 right{ 1.0, 0.0, 0.0 };

    double fov = 60.0;   // degrees
    double aspect = 1.0;  // width / height
    double z_near = 0.1;
    double z_far = 1000.0;

    double speed;
    std::vector<bool> key_down;
    std::vector<bool> special_key_down;

};
