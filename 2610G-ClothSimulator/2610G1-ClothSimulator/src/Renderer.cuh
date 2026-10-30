#ifndef RENDERER_CUH
#define RENDERER_CUH

#include <cmath>
#include <iostream>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "MathHelper.cuh"
#include "MouseTool.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Shader.cuh"

class Renderer {
private:
    Shader* axesShader = nullptr;
    GLuint axesVao = 0, axesVbo = 0;
    bool pause;
    int width, height;
    MouseTool mouseTool;
    double lastLeftPress = -1.0, lastPressX = 0, lastPressY = 0;
    int navigationModifiers() const;
    void pickPivot(double x, double y);
    Vector3f center, lightDirection, cameraPosition;
    GLFWwindow* window;
    Matrix4x4f scale(float scaling) const;
    Matrix4x4f translate(const Vector3f& v) const;
    Matrix4x4f lookAt(const Vector3f& position, const Vector3f& center, const Vector3f& up) const;
    Matrix4x4f perspective(float fovy, float aspect, float zNear, float zFar) const;
    void framebufferSizeCallback(int width, int height);
    void mouseButtonCallback(int button, int action, int mods);
    void cursorPosCallback(double x, double y);
    void scrollCallback(double x, double y);
    void keyCallback(int key, int scancode, int action, int mods);

public:
    Renderer(int width, int height);
    ~Renderer();
    void setCenter(const Vector3f& center);
    void renderAxes() const;
    GLFWwindow* getWindow() const;
    bool getPause() const;
    std::function<void()> drawForPicking;
    Vector3f getLightDirection() const;
    Vector3f getCameraPosition() const;
    Matrix4x4f getModel() const;
    Matrix4x4f getView() const;
    Matrix4x4f getProjection() const;
};

#endif