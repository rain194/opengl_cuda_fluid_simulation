#version 330 core

// Vertex attributes
layout (location = 0) in vec3 position;

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform float grid_scale;
uniform vec3 grid_offset;

// Outputs
out float distance_from_center;

void main() {
    vec3 world_pos = position * grid_scale + grid_offset;
    
    // Calculate distance from center for fading
    distance_from_center = length(world_pos.xz);
    
    gl_Position = projection * view * vec4(world_pos, 1.0);
}
