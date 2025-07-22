#version 330 core

// Vertex attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 normal_matrix;
uniform float time;
uniform vec3 camera_position;

// Wave simulation parameters
uniform bool enable_waves;
uniform float wave_amplitude;
uniform float wave_frequency;
uniform vec2 wave_direction;

// Outputs to fragment shader
out vec3 world_pos;
out vec3 world_normal;
out vec3 view_dir;
out vec2 uv;
out float wave_height;

// Wave function
float wave(vec2 pos, float time, float amplitude, float frequency, vec2 direction) {
    vec2 wave_pos = dot(pos, normalize(direction)) * normalize(direction);
    return amplitude * sin(frequency * length(wave_pos) - time * 2.0);
}

void main() {
    vec3 pos = position;
    vec3 norm = normal;
    
    // Apply wave deformation if enabled
    wave_height = 0.0;
    if (enable_waves) {
        // Multiple wave components for realistic water
        wave_height += wave(position.xz, time, wave_amplitude, wave_frequency, wave_direction);
        wave_height += wave(position.xz, time, wave_amplitude * 0.5, wave_frequency * 2.0, vec2(-wave_direction.y, wave_direction.x));
        wave_height += wave(position.xz, time, wave_amplitude * 0.25, wave_frequency * 4.0, wave_direction * 0.7);
        
        pos.y += wave_height;
        
        // Recalculate normal for wave deformation
        float epsilon = 0.01;
        vec3 tangent = vec3(epsilon, 0.0, 0.0);
        vec3 bitangent = vec3(0.0, 0.0, epsilon);
        
        float h_x = wave(position.xz + tangent.xz, time, wave_amplitude, wave_frequency, wave_direction);
        float h_z = wave(position.xz + bitangent.xz, time, wave_amplitude, wave_frequency, wave_direction);
        
        vec3 wave_tangent = normalize(vec3(1.0, (h_x - wave_height) / epsilon, 0.0));
        vec3 wave_bitangent = normalize(vec3(0.0, (h_z - wave_height) / epsilon, 1.0));
        norm = normalize(cross(wave_tangent, wave_bitangent));
    }
    
    // Transform to world space
    world_pos = (model * vec4(pos, 1.0)).xyz;
    world_normal = normalize((normal_matrix * vec4(norm, 0.0)).xyz);
    
    // Calculate view direction
    view_dir = normalize(camera_position - world_pos);
    
    // Pass through texture coordinates
    uv = tex_coord;
    
    // Final position
    gl_Position = projection * view * vec4(world_pos, 1.0);
}
