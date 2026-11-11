#include "Renderer.cuh"

Renderer::Renderer(int width, int height) :
    width(width),
    height(height),
    pause(false), // NODELETE
    lightDirection(0.0f, 0.0f, 1.0f),
    cameraPosition(0.8f, -2.0f, 0.45f) { // NODELETE
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    window = glfwCreateWindow(width, height, "ClothSimulator", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->framebufferSizeCallback(width, height);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->mouseButtonCallback(button, action, mods);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->cursorPosCallback(x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->scrollCallback(x, y);
    });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->keyCallback(key, scancode, action, mods);
    });

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);

    // Position + RGB: X red, Y blue, Z green, all starting at the orbit pivot.
    const float axes[] = {
        0, 0, 0, 1, 0, 0,  1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1,  0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0,  0, 0, 1, 0, 1, 0
    };
    axesShader = new Shader("shader/AxisVertex.glsl", "shader/AxisFragment.glsl");
    glGenVertexArrays(1, &axesVao);
    glGenBuffers(1, &axesVbo);
    glBindVertexArray(axesVao);
    glBindBuffer(GL_ARRAY_BUFFER, axesVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axes), axes, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glBindVertexArray(0);

}

Renderer::~Renderer() {
    glDeleteBuffers(1, &axesVbo);
    glDeleteVertexArrays(1, &axesVao);
    delete axesShader;
    glfwTerminate();
}

Matrix4x4f Renderer::scale(float scaling) const {
    Matrix4x4f ans;

    ans(0, 0) = ans(1, 1) = ans(2, 2) = scaling;
    ans(3, 3) = 1.0f;

    return ans;
}

Matrix4x4f Renderer::translate(const Vector3f& v) const {
    Matrix4x4f ans(1.0f);

    ans(0, 3) = v(0);
    ans(1, 3) = v(1);
    ans(2, 3) = v(2);

    return ans;
}

Matrix4x4f Renderer::lookAt(const Vector3f& position, const Vector3f& center, const Vector3f& up) const {
    Vector3f f = (center - position).normalized();
    Vector3f s = f.cross(up).normalized();
    Vector3f u = s.cross(f);
    Matrix4x4f ans;

    ans(0, 0) = s(0);
    ans(0, 1) = s(1);
    ans(0, 2) = s(2);
    ans(0, 3) = -s.dot(position);

    ans(1, 0) = u(0);
    ans(1, 1) = u(1);
    ans(1, 2) = u(2);
    ans(1, 3) = -u.dot(position);

    ans(2, 0) = -f(0);
    ans(2, 1) = -f(1);
    ans(2, 2) = -f(2);
    ans(2, 3) = f.dot(position);

    ans(3, 3) = 1.0f;

    return ans;
}

Matrix4x4f Renderer::perspective(float fovy, float aspect, float zNear, float zFar) const {
    float t = tan(fovy * 0.5f);
    Matrix4x4f ans;

    ans(0, 0) = 1.0f / (aspect * t);
    ans(1, 1) = 1.0f / t;
    ans(2, 2) = -(zNear + zFar) / (zFar - zNear);
    ans(2, 3) = -2.0f * zNear * zFar / (zFar - zNear);
    ans(3, 2) = -1.0f;

    return ans;
}

void Renderer::framebufferSizeCallback(int width, int height) {
    this->width = width;
    this->height = height;
    glViewport(0, 0, width, height);
}

void Renderer::mouseButtonCallback(int button, int action, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    mouseTool.button(button, action, x, y);
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        const double now = glfwGetTime();
        if ((mods & GLFW_MOD_ALT) && lastLeftPress >= 0 && now - lastLeftPress < 0.4
            && std::hypot(x - lastPressX, y - lastPressY) < 5.0) {
            pickPivot(x, y);
            lastLeftPress = -1.0;
        } else {
            lastLeftPress = now;
            lastPressX = x; lastPressY = y;
        }
    }
}

void Renderer::cursorPosCallback(double x, double y) {
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    mouseTool.move(x, y, w, h, cameraPosition - center, navigationModifiers());
}

void Renderer::scrollCallback(double x, double y) {
    double cursorX, cursorY;
    int w, h;
    glfwGetCursorPos(window, &cursorX, &cursorY);
    glfwGetWindowSize(window, &w, &h);
    mouseTool.scroll(x, y, cursorX, cursorY, w, h, cameraPosition - center, navigationModifiers());
}

void Renderer::setCenter(const Vector3f& value) {
    center = value;
    cameraPosition = center + Vector3f(0.8f, -2.0f, 0.6f);
    mouseTool.reset();
}

void Renderer::keyCallback(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS && (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT
        || key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL
        || key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT
        || key == GLFW_KEY_LEFT_SUPER || key == GLFW_KEY_RIGHT_SUPER)) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        mouseTool.anchor(x, y);
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    else if (key == GLFW_KEY_HOME && action == GLFW_PRESS)
        mouseTool.reset();
    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        pause = !pause;
}

GLFWwindow* Renderer::getWindow() const {
    return window;
}

bool Renderer::getPause() const {
    return pause;
}

Vector3f Renderer::getLightDirection() const {
    return lightDirection;
}

Vector3f Renderer::getCameraPosition() const {
    return cameraPosition;
}

Matrix4x4f Renderer::getModel() const {
    const Vector3f pivot = center + mouseTool.pivotOffset;
    return translate(pivot + mouseTool.translation) * scale(mouseTool.scaling) * mouseTool.rotation * translate(-1.0f * pivot);
}

Matrix4x4f Renderer::getView() const {
    // Frame the standing character from the front at a slight angle (Z is up).
    return lookAt(cameraPosition, center, Vector3f(0.0f, 0.0f, 1.0f));
}

Matrix4x4f Renderer::getProjection() const {
    return perspective(45.0f * 3.14159265f / 180.0f, static_cast<float>(width) / max(height, 1), 0.1f, 100.0f);
}

int Renderer::navigationModifiers() const {
    int mods = 0;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) mods |= GLFW_MOD_SHIFT;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) mods |= GLFW_MOD_CONTROL;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS) mods |= GLFW_MOD_ALT;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS) mods |= GLFW_MOD_SUPER;
    return mods;
}

void Renderer::pickPivot(double x, double y) {
    int w, h, fw, fh;
    glfwGetWindowSize(window, &w, &h);
    glfwGetFramebufferSize(window, &fw, &fh);
    if (w <= 0 || h <= 0 || fw <= 0 || fh <= 0) return;
    const int px = static_cast<int>(x * fw / w);
    const int py = fh - 1 - static_cast<int>(y * fh / h);
    if (px < 0 || px >= fw || py < 0 || py >= fh) return;
    if (!drawForPicking) return;
    drawForPicking();
    // Resolve the multisampled back-buffer depth before reading a single pixel.
    GLuint fbo, depth;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    GLint depthBits = 24;
    glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_DEPTH,
                                          GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE, &depthBits);
    glRenderbufferStorage(GL_RENDERBUFFER, depthBits == 32 ? GL_DEPTH_COMPONENT32 :
                          depthBits == 16 ? GL_DEPTH_COMPONENT16 : GL_DEPTH_COMPONENT24, fw, fh);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_BACK);
    glBlitFramebuffer(0, 0, fw, fh, 0, 0, fw, fh, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadBuffer(GL_NONE);
    float value = 1.0f;
    glReadPixels(px, py, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &value);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glDeleteRenderbuffers(1, &depth);
    glDeleteFramebuffers(1, &fbo);
    if (!(value >= 0.0f && value < 1.0f)) return;
    const Vector4f clip(2.0f * (px + 0.5f) / fw - 1.0f,
                        2.0f * (py + 0.5f) / fh - 1.0f, 2.0f * value - 1.0f, 1.0f);
    // Solve the inverse projection with pivoting; Matrix::inverse supports only 2x2.
    const Matrix4x4f transform = getProjection() * getView() * getModel();
    double system[4][5];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) system[r][c] = transform(r, c);
        system[r][4] = clip(r);
    }
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::abs(system[r][c]) > std::abs(system[pivot][c])) pivot = r;
        if (std::abs(system[pivot][c]) < 1e-12) return;
        for (int j = c; j < 5; ++j) std::swap(system[c][j], system[pivot][j]);
        const double divisor = system[c][c];
        for (int j = c; j < 5; ++j) system[c][j] /= divisor;
        for (int r = 0; r < 4; ++r) if (r != c) {
            const double factor = system[r][c];
            for (int j = c; j < 5; ++j) system[r][j] -= factor * system[c][j];
        }
    }
    const Vector4f point(system[0][4], system[1][4], system[2][4], system[3][4]);
    if (!std::isfinite(point(3)) || std::abs(point(3)) < 1e-8f) return;
    const Vector3f position(point(0) / point(3), point(1) / point(3), point(2) / point(3));
    mouseTool.setPivotOffset(position - center);
}

void Renderer::renderAxes() const {
    if (!axesShader) return;
    GLint previousProgram, previousVao;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVao);
    const GLboolean depthTest = glIsEnabled(GL_DEPTH_TEST);
    // Draw over the body so the interaction pivot remains visible.
    glDisable(GL_DEPTH_TEST);
    axesShader->use();
    const Matrix4x4f pivotModel = getModel() * translate(center + mouseTool.pivotOffset);
    const Vector4f pivotInView = getView() * pivotModel * Vector4f(0, 0, 0, 1);
    const float depth = -pivotInView(2);
    // A view-plane axis spans one third of the viewport width at the pivot.
    // Cancel scene zoom so the guide keeps its display size as the cloth scales.
    const float aspect = static_cast<float>(width) / max(height, 1);
    const float axisLength = 2.0f * max(depth, 0.1f) * std::tan(3.14159265f / 8.0f) * aspect / 3.0f;
    axesShader->setMat4("model", pivotModel * scale(axisLength / mouseTool.scaling));
    axesShader->setMat4("view", getView());
    axesShader->setMat4("projection", getProjection());
    glBindVertexArray(axesVao);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(previousVao);
    glUseProgram(previousProgram);
    if (depthTest) glEnable(GL_DEPTH_TEST);
}
