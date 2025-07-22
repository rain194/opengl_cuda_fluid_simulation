#pragma once
#include <glm/glm.hpp>
#include <string>

namespace FluidSim
{

/**
 * Global simulation configuration
 * Contains all parameters for physics, rendering, and performance
 */
struct SimulationConfig
{
    // === PHYSICS PARAMETERS ===

    // SPH Core Parameters
    float time_step = 0.016f;        // 60 FPS default
    float rest_density = 1000.0f;    // kg/m³ (water density)
    float gas_constant = 200.0f;     // Pressure stiffness
    float viscosity = 0.001f;        // Pa·s (water viscosity)
    float surface_tension = 0.0728f; // N/m (water surface tension)
    float smoothing_radius = 0.1f;   // SPH kernel radius

    // Forces
    glm::vec3 gravity = glm::vec3(0.0f, -9.81f, 0.0f); // m/s²
    float damping_factor = 0.99f;                      // Velocity damping
    float buoyancy_coefficient = 1.0f;                 // Buoyancy strength

    // Collision Parameters
    float restitution = 0.5f; // Bounce factor (0-1)
    float friction = 0.1f;    // Surface friction (0-1)

    // Simulation Bounds
    glm::vec3 boundary_min = glm::vec3(-5.0f, -5.0f, -5.0f);
    glm::vec3 boundary_max = glm::vec3(5.0f, 5.0f, 5.0f);

    // === PERFORMANCE PARAMETERS ===

    // Particle Counts
    size_t initial_particle_count = 1000;
    size_t max_particles = 100000;
    bool dynamic_particle_allocation = true;

    // Optimization Settings
    bool use_cuda = true;
    bool use_spatial_grid = true;
    float spatial_grid_cell_size = 0.2f;
    int max_neighbors_per_particle = 64;

    // Threading
    int cpu_thread_count = 4; // -1 for auto-detect
    bool enable_parallel_processing = true;

    // === RENDERING PARAMETERS ===

    // Window Settings
    int window_width = 1280;
    int window_height = 720;
    std::string window_title = "Fluid Simulation";
    bool fullscreen = false;
    bool vsync = true;

    // Camera Settings
    glm::vec3 camera_position = glm::vec3(0.0f, 2.0f, 5.0f);
    glm::vec3 camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
    float camera_fov = 45.0f;
    float camera_near = 0.1f;
    float camera_far = 100.0f;
    float camera_speed = 5.0f;
    float mouse_sensitivity = 0.1f;

    // Particle Rendering
    float particle_size = 0.02f;
    bool render_as_spheres = false; // true = high quality, false = fast points
    bool enable_instancing = true;
    bool enable_depth_sorting = false;

    // Visual Effects
    bool enable_lighting = true;
    bool enable_shadows = false;
    bool enable_reflections = false;
    glm::vec3 light_direction = glm::normalize(glm::vec3(1.0f, -1.0f, -1.0f));
    glm::vec3 light_color = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 ambient_color = glm::vec3(0.3f, 0.3f, 0.3f);

    // Color Scheme
    enum ColorMode
    {
        SOLID_COLOR,
        VELOCITY_BASED,
        DENSITY_BASED,
        PRESSURE_BASED,
        TEMPERATURE_BASED
    } color_mode = VELOCITY_BASED;

    glm::vec3 base_color = glm::vec3(0.3f, 0.6f, 1.0f);
    float color_intensity = 1.0f;

    // Background
    glm::vec3 background_color = glm::vec3(0.1f, 0.1f, 0.2f);
    bool show_grid = true;
    bool show_boundaries = true;
    bool show_statistics = true;

    // === DEBUG PARAMETERS ===

    bool debug_mode = false;
    bool show_neighbor_connections = false;
    bool show_force_vectors = false;
    bool show_velocity_vectors = false;
    bool print_performance_stats = true;
    int stats_update_frequency = 60; // frames

    // === PRESETS ===

    static SimulationConfig getDefaultConfig()
    {
        return SimulationConfig{};
    }

    static SimulationConfig getHighPerformanceConfig()
    {
        SimulationConfig config;
        config.max_particles = 50000;
        config.render_as_spheres = false;
        config.enable_lighting = false;
        config.enable_depth_sorting = false;
        config.use_cuda = true;
        return config;
    }

    static SimulationConfig getHighQualityConfig()
    {
        SimulationConfig config;
        config.max_particles = 10000;
        config.render_as_spheres = true;
        config.enable_lighting = true;
        config.enable_shadows = true;
        config.enable_reflections = true;
        config.particle_size = 0.03f;
        return config;
    }

    static SimulationConfig getDebugConfig()
    {
        SimulationConfig config;
        config.max_particles = 1000;
        config.debug_mode = true;
        config.show_neighbor_connections = true;
        config.show_force_vectors = true;
        config.print_performance_stats = true;
        return config;
    }

    // Configuration validation
    bool validate() const
    {
        if (time_step <= 0.0f || time_step > 0.1f)
            return false;
        if (rest_density <= 0.0f)
            return false;
        if (smoothing_radius <= 0.0f)
            return false;
        if (max_particles == 0)
            return false;
        if (window_width <= 0 || window_height <= 0)
            return false;
        return true;
    }

    // Configuration I/O
    bool loadFromFile(const std::string &filename);
    bool saveToFile(const std::string &filename) const;
    void printConfiguration() const;
};

/**
 * Runtime constants and derived parameters
 */
struct RuntimeConstants
{
    // SPH kernel normalization constants
    float poly6_constant;
    float spiky_constant;
    float viscosity_constant;

    // Simulation timing
    float accumulated_time;
    int total_frames;
    float average_frame_time;

    // Performance metrics
    float physics_time_ms;
    float rendering_time_ms;
    float total_frame_time_ms;
    size_t memory_usage_mb;

    // Derived physics parameters
    float particle_mass;
    float time_step_squared;
    float smoothing_radius_squared;

    RuntimeConstants()
    {
        reset();
    }

    void reset()
    {
        accumulated_time = 0.0f;
        total_frames = 0;
        average_frame_time = 0.0f;
        physics_time_ms = 0.0f;
        rendering_time_ms = 0.0f;
        total_frame_time_ms = 0.0f;
        memory_usage_mb = 0;
    }

    void updateFromConfig(const SimulationConfig &config)
    {
        // Calculate SPH kernel constants
        float h = config.smoothing_radius;
        float h2 = h * h;
        float h6 = h2 * h2 * h2;
        float h9 = h6 * h2 * h;

        poly6_constant = 315.0f / (64.0f * 3.14159265358979323846 * h9);
        spiky_constant = -45.0f / (3.14159265358979323846 * h6);
        viscosity_constant = 45.0f / (3.14159265358979323846 * h6);

        // Derived parameters
        time_step_squared = config.time_step * config.time_step;
        smoothing_radius_squared = h2;

        // Estimate particle mass based on rest density and particle volume
        float particle_volume = (4.0f / 3.0f) * 3.14159265358979323846 * h2 * h / 8.0f; // Rough estimate
        particle_mass = config.rest_density * particle_volume;
    }
};

/**
 * Application-wide configuration manager
 */
class ConfigManager
{
  private:
    static SimulationConfig current_config;
    static RuntimeConstants runtime_constants;
    static bool initialized;

  public:
    // Configuration access
    static const SimulationConfig &getConfig()
    {
        return current_config;
    }
    static SimulationConfig &getMutableConfig()
    {
        return current_config;
    }
    static const RuntimeConstants &getRuntimeConstants()
    {
        return runtime_constants;
    }
    static RuntimeConstants &getMutableRuntimeConstants()
    {
        return runtime_constants;
    }

    // Configuration management
    static bool initialize(const SimulationConfig &config = SimulationConfig::getDefaultConfig());
    static bool loadConfiguration(const std::string &filename);
    static bool saveConfiguration(const std::string &filename);
    static void updateRuntimeConstants();
    static void resetRuntimeStats();

    // Configuration presets
    static void applyPreset(const std::string &preset_name);
    static void createCustomPreset(const std::string &name, const SimulationConfig &config);

    // Validation and diagnostics
    static bool validateCurrentConfig();
    static void printConfigurationSummary();
    static void printPerformanceStats();

    // Runtime updates
    static void updatePerformanceMetrics(float physics_time, float rendering_time, float total_time);
    static void updateFrameStats(float frame_time);

  private:
    static std::string getConfigDirectory();
    static std::string getDefaultConfigPath();
};

// Global configuration macros for convenience
#define GET_CONFIG() FluidSim::ConfigManager::getConfig()
#define GET_RUNTIME() FluidSim::ConfigManager::getRuntimeConstants()
#define UPDATE_RUNTIME() FluidSim::ConfigManager::updateRuntimeConstants()

} // namespace FluidSim
