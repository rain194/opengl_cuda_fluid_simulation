#version 330 core

// Inputs
in vec3 world_position;

// Uniforms
uniform vec3 boundary_color;
uniform float pulse_frequency;
uniform float time;
uniform bool show_violations;

// Output
out vec4 FragColor;

void main() {
    // Pulsing effect for boundary visibility
    float pulse = sin(time * pulse_frequency) * 0.3 + 0.7;
    
    // Different colors for different faces
    vec3 color = boundary_color;
    
    // Y-axis boundaries (floor/ceiling) in different color
    if (abs(world_position.y - (-5.0)) < 0.1 || abs(world_position.y - 5.0) < 0.1) {
        color = vec3(1.0, 0.5, 0.0); // Orange for floor/ceiling
    }
    
    vec3 final_color = color * pulse;
    float alpha = 0.6 * pulse;
    
    FragColor = vec4(final_color, alpha);
}