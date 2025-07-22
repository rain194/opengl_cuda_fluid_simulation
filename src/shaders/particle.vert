#version 330 core

// Vertex attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 velocity;

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform float particle_size;
uniform float time;
uniform bool velocity_based_size;
uniform float max_velocity;

// Outputs to fragment shader
out vec3 particle_color;
out float particle_alpha;
out float velocity_magnitude;

void main() {
    // Transform position to clip space
    vec4 world_pos = vec4(position, 1.0);
    vec4 view_pos = view * world_pos;
    gl_Position = projection * view_pos;
    
    // Calculate velocity magnitude for effects
    velocity_magnitude = length(velocity);
    
    // Dynamic particle size based on velocity or constant
    float size = particle_size;
    if (velocity_based_size && max_velocity > 0.0) {
        float vel_factor = velocity_magnitude / max_velocity;
        size = particle_size * (0.5 + vel_factor * 1.5); // Size range: 0.5x to 2.0x
    }
    
    // Set point size (for GL_POINTS rendering)
    gl_PointSize = size;
    
    // Pass data to fragment shader
    particle_color = color;
    
    // Alpha based on distance from camera (optional fade effect)
    float distance_to_camera = length(view_pos.xyz);
    particle_alpha = 1.0 - smoothstep(10.0, 50.0, distance_to_camera);
    particle_alpha = clamp(particle_alpha, 0.1, 1.0);
}
