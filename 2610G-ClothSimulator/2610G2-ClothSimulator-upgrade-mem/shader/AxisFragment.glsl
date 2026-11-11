#version 330 core
in vec3 axisColor;
out vec4 fragmentColor;
void main() {
    fragmentColor = vec4(axisColor, 1.0);
}
