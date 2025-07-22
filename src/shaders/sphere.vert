#version 330 core

// Vertex attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

// Instance data (for instanced rendering)
layout (location = 2) in vec3 instance_position;
layout (location = 3) in vec3 instance_color;
layout (location = 4) in float instance_scale;

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 normal_matrix;
uniform float particle_radius;
uniform bool use_instancing;

// Outputs to fragment shader
out vec3 world_pos;
out vec3 world_normal;
out vec3 particle_color;
out vec3 view_dir;

void main() {
    vec3 pos = position;
    vec3 norm = normal;
    
    // Handle instanced rendering
    if (use_instancing) {
        pos = pos * instance_scale * particle_radius + instance_position;
        particle_color = instance_color;
    } else {
        pos = pos * particle_radius;
        particle_color = vec3(0.3, 0.6, 1.0); // Default blue
    }
    
    // Transform to world space
    world_pos = (model * vec4(pos, 1.0)).xyz;
    world_normal = normalize((normal_matrix * vec4(norm, 0.0)).xyz);
    
    // Calculate view direction
    vec4 view_pos = view * vec4(world_pos, 1.0);
    view_dir = -normalize(view_pos.xyz);
    
    // Final position
    gl_Position = projection * view_pos;
}
