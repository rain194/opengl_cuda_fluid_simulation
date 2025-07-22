#version 330 core

// Inputs from vertex shader
in vec3 particle_color;
in float particle_alpha;
in float velocity_magnitude;

// Uniforms
uniform vec3 base_color;
uniform float color_intensity;
uniform bool enable_glow;
uniform bool circular_particles;
uniform float time;

// Output
out vec4 FragColor;

void main() {
    vec2 tex_coord = gl_PointCoord; // This is correct here - only available in fragment shader
    vec2 center = vec2(0.5, 0.5);
    float distance_from_center = length(tex_coord - center);
    
    // Circular particle rendering
    if (circular_particles) {
        // Create smooth circular particles
        float circle_radius = 0.45;
        float edge_softness = 0.05;
        
        if (distance_from_center > circle_radius + edge_softness) {
            discard; // Outside the circle
        }
        
        // Smooth falloff at edges
        float alpha_factor = 1.0 - smoothstep(circle_radius - edge_softness, 
                                             circle_radius + edge_softness, 
                                             distance_from_center);
        
        // Add some internal structure for visual interest
        float inner_glow = 1.0 - smoothstep(0.1, 0.3, distance_from_center);
        alpha_factor = max(alpha_factor * 0.8, inner_glow * 0.4);
        
        // Final alpha
        float final_alpha = alpha_factor * particle_alpha;
        
        // Color composition
        vec3 final_color = particle_color * base_color * color_intensity;
        
        // Add glow effect
        if (enable_glow) {
            float glow_factor = 1.0 - distance_from_center;
            final_color += vec3(0.2, 0.4, 0.8) * glow_factor * 0.3;
        }
        
        FragColor = vec4(final_color, final_alpha);
    } else {
        // Square particle rendering with some effects
        vec3 final_color = particle_color * base_color * color_intensity;
        
        // Add some texture variation
        float noise = sin(tex_coord.x * 10.0) * sin(tex_coord.y * 10.0) * 0.1;
        final_color += noise;
        
        FragColor = vec4(final_color, particle_alpha);
    }
    
    // Ensure we don't output completely transparent pixels
    if (FragColor.a < 0.01) {
        discard;
    }
}