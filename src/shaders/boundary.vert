#version 330 core

// Vertex attributes
layout (location = 0) in vec3 position;

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform vec3 boundary_min;
uniform vec3 boundary_max;

// Outputs
out vec3 world_position;

void main() {
    // Scale and translate to match boundary box
    vec3 world_pos = mix(boundary_min, boundary_max, position);
    world_position = world_pos;
    
    gl_Position = projection * view * vec4(world_pos, 1.0);
}
