#version 330 core

// Inputs from vertex shader
in vec3 world_pos;
in vec3 world_normal;
in vec3 view_dir;
in vec2 uv;
in float wave_height;

// Lighting uniforms
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 ambient_color;
uniform vec3 camera_position;

// Material properties
uniform vec3 water_color;
uniform float transparency;
uniform float refractive_index;
uniform float roughness;
uniform float metallic;

// Environment
uniform samplerCube skybox;
uniform bool enable_reflections;
uniform bool enable_refractions;
uniform float time;

// Output
out vec4 FragColor;

// Fresnel calculation
float fresnel(vec3 view_dir, vec3 normal, float n1, float n2) {
    float cos_theta = max(0.0, dot(view_dir, normal));
    float r0 = pow((n1 - n2) / (n1 + n2), 2.0);
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// Physically based lighting
vec3 calculatePBR(vec3 albedo, vec3 normal, vec3 view_dir, vec3 light_dir, vec3 light_color) {
    // Lambertian diffuse
    float NdotL = max(0.0, dot(normal, light_dir));
    vec3 diffuse = albedo * light_color * NdotL;
    
    // Blinn-Phong specular (simplified)
    vec3 half_dir = normalize(light_dir + view_dir);
    float NdotH = max(0.0, dot(normal, half_dir));
    float shininess = mix(1.0, 128.0, 1.0 - roughness);
    vec3 specular = light_color * pow(NdotH, shininess) * (1.0 - roughness);
    
    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(world_normal);
    vec3 view = normalize(view_dir);
    vec3 light_dir = normalize(-light_direction);
    
    // Base water color
    vec3 base_color = water_color;
    
    // Add some depth-based color variation
    float depth_factor = clamp(wave_height * 2.0 + 0.5, 0.0, 1.0);
    base_color = mix(water_color * 0.6, water_color, depth_factor);
    
    // Calculate lighting
    vec3 lighting = calculatePBR(base_color, normal, view, light_dir, light_color);
    lighting += ambient_color * base_color;
    
    // Reflection
    vec3 reflection_color = vec3(0.0);
    if (enable_reflections) {
        vec3 reflection_dir = reflect(-view, normal);
        reflection_color = texture(skybox, reflection_dir).rgb;
    }
    
    // Refraction
    vec3 refraction_color = vec3(0.0);
    if (enable_refractions) {
        vec3 refraction_dir = refract(-view, normal, 1.0 / refractive_index);
        refraction_color = texture(skybox, refraction_dir).rgb;
    }
    
    // Fresnel blending
    float fresnel_factor = fresnel(view, normal, 1.0, refractive_index);
    
    // Combine lighting, reflection, and refraction
    vec3 final_color = lighting;
    
    if (enable_reflections || enable_refractions) {
        vec3 env_color = mix(refraction_color, reflection_color, fresnel_factor);
        final_color = mix(lighting, env_color, 0.7);
    }
    
    // Add some foam/highlights on wave peaks
    float foam_factor = smoothstep(0.1, 0.3, wave_height);
    final_color = mix(final_color, vec3(1.0), foam_factor * 0.3);
    
    // Calculate final alpha
    float alpha = mix(transparency, 1.0, fresnel_factor);
    alpha = clamp(alpha, 0.1, 0.95); // Keep some transparency
    
    FragColor = vec4(final_color, alpha);
}
