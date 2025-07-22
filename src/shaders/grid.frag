#version 330 core

// Inputs
in float distance_from_center;

// Uniforms
uniform vec3 grid_color;
uniform float fade_distance;
uniform float line_width;

// Output
out vec4 FragColor;

void main() {
    // Fade grid lines based on distance
    float alpha = 1.0 - smoothstep(0.0, fade_distance, distance_from_center);
    alpha = clamp(alpha, 0.0, 0.8); // Never fully opaque
    
    // Add some variation to line intensity
    float line_intensity = 1.0 - smoothstep(0.5, 1.0, distance_from_center / fade_distance);
    
    vec3 final_color = grid_color * line_intensity;
    
    FragColor = vec4(final_color, alpha);
}
