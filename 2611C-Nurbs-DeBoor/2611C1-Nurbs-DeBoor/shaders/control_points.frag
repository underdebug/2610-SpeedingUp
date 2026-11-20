#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 markerPosition;
layout(location = 0) out vec4 outColor;

void main() {
    float radius = length(markerPosition);
    if (radius > 1.0) {
        discard;
    }
    vec3 rimColor = vec3(0.10, 0.025, 0.008);
    outColor = vec4(mix(fragColor, rimColor, smoothstep(0.65, 0.85, radius)), 1.0);
}
