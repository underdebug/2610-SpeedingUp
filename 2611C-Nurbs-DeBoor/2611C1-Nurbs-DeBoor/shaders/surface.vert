#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(push_constant) uniform Push {
    mat4 mvp;
} pc;
layout(location = 0) out vec3 fragColor;
layout(location = 1) noperspective out vec3 barycentric;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragColor = inColor;
    barycentric = vec3(0.0);
    barycentric[gl_VertexIndex % 3] = 1.0;
}
