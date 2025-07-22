#version 330 core

// Inputs from vertex shader
in vec3 world_pos;
in vec3 world_normal;
in vec3 particle_color;
in vec3 view_dir;

// Lighting uniforms
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 ambient_color;
uniform vec3 camera_position;

// Material properties
uniform float shininess;
uniform float transparency;
uniform bool enable_subsurface_scattering;

// Output
out vec4 FragColor;

// Simple subsurface scattering approximation
vec3 subsurfaceScattering(vec3 normal, vec3 light_dir, vec3 view_dir, vec3 color) {
    float backlight = max(0.0, dot(-normal, light_dir));
    float scatter = pow(backlight, 2.0) * 0.5;
    return color * scatter;
}

void main() {
    vec3 normal = normalize(world_normal);
    vec3 view = normalize(view_dir);
    vec3 light_dir = normalize(-light_direction);
    
    // Ambient lighting
    vec3 ambient = ambient_color * particle_color;
    
    // Diffuse lighting (Lambertian)
    float NdotL = max(0.0, dot(normal, light_dir));
    vec3 diffuse = light_color * particle_color * NdotL;
    
    // Specular lighting (Blinn-Phong)
    vec3 half_dir = normalize(light_dir + view);
    float NdotH = max(0.0, dot(normal, half_dir));
    vec3 specular = light_color * pow(NdotH, shininess) * 0.3;
    
    // Rim lighting for better particle definition
    float rim_factor = 1.0 - max(0.0, dot(normal, view));
    vec3 rim_light = light_color * pow(rim_factor, 2.0) * 0.2;
    
    // Subsurface scattering for organic feel
    vec3 subsurface = vec3(0.0);
    if (enable_subsurface_scattering) {
        subsurface = subsurfaceScattering(normal, light_dir, view, particle_color);
    }
    
    // Combine all lighting components
    vec3 final_color = ambient + diffuse + specular + rim_light + subsurface;
    
    // Depth-based color variation
    float depth = gl_FragCoord.z;
    final_color = mix(final_color, final_color * 0.8, depth * 0.3);
    
    // Final alpha with transparency
    float alpha = mix(transparency, 1.0, NdotL * 0.5 + 0.5);
    
    FragColor = vec4(final_color, alpha);
}
