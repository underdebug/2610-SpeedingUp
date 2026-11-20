#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(push_constant) uniform Push {
    mat4 mvp;
    vec2 clipPerPixel;
} pc;
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 markerPosition;

const vec2 corners[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
);

void main() {
    fragColor = inColor;
    markerPosition = corners[gl_VertexIndex];
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    gl_Position.xy += markerPosition * 5.5 * pc.clipPerPixel * gl_Position.w;
}
