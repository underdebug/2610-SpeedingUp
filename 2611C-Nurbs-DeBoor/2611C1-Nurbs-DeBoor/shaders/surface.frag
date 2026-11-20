#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) noperspective in vec3 barycentric;
layout(location = 0) out vec4 outColor;

void main() {
    // Convert barycentric coordinates to screen-space distances from each edge.
    vec3 dx = dFdx(barycentric);
    vec3 dy = dFdy(barycentric);
    vec3 gradient = sqrt(dx * dx + dy * dy);
    vec3 distanceToEdge = barycentric / max(gradient, vec3(1e-6));
    float nearest = min(distanceToEdge.x, min(distanceToEdge.y, distanceToEdge.z));
    float edge = 1.0 - smoothstep(0.25, 0.9, nearest);
    vec3 edgeColor = vec3(0.008, 0.016, 0.035);
    outColor = vec4(mix(fragColor, edgeColor, 0.78 * edge), 1.0);
}
