#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <vector>
#include <string> 

namespace FluidSim
{

// Forward declarations
class ParticleSystem;
class Shader;
class BufferManager;

/**
 * Camera system for 3D navigation
 */
class Camera
{
  private:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;

    // Euler angles
    float yaw;
    float pitch;

    // Camera options
    float movement_speed;
    float mouse_sensitivity;
    float zoom;

  public:
    Camera(glm::vec3 pos = glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f,
           float pitch = 0.0f);

    // Camera matrices
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect_ratio, float near_plane = 0.1f, float far_plane = 100.0f) const;

    // Camera movement
    void processKeyboard(int direction, float delta_time);
    void processMouseMovement(float x_offset, float y_offset, bool constrain_pitch = true);
    void processMouseScroll(float y_offset);

    // Getters
    glm::vec3 getPosition() const
    {
        return position;
    }
    glm::vec3 getFront() const
    {
        return front;
    }
    float getZoom() const
    {
        return zoom;
    }

    // Setters
    void setPosition(glm::vec3 pos)
    {
        position = pos;
    }
    void lookAt(glm::vec3 target);

  private:
    void updateCameraVectors();
};

/**
 * Rendering configuration and settings
 */
struct RenderSettings
{
    // Particle rendering
    float particle_size;
    bool render_as_spheres;
    bool enable_depth_sorting;
    bool enable_instancing;

    // Visual effects
    bool enable_lighting;
    bool enable_shadows;
    bool enable_reflections;
    glm::vec3 light_direction;
    glm::vec3 light_color;
    glm::vec3 ambient_color;

    // Color schemes
    enum ColorMode
    {
        SOLID_COLOR,
        VELOCITY_BASED,
        DENSITY_BASED,
        PRESSURE_BASED,
        TEMPERATURE_BASED
    } color_mode;

    glm::vec3 base_color;
    float color_intensity;

    // Performance settings
    int max_rendered_particles;
    bool use_level_of_detail;
    float lod_distance_threshold;

    // Background
    glm::vec3 background_color;
    bool show_grid;
    bool show_boundaries;

    RenderSettings()
        : particle_size(0.02f), render_as_spheres(true), enable_depth_sorting(false), enable_instancing(true),
          enable_lighting(true), enable_shadows(false), enable_reflections(false),
          light_direction(glm::normalize(glm::vec3(1.0f, -1.0f, -1.0f))), light_color(1.0f, 1.0f, 1.0f),
          ambient_color(0.3f, 0.3f, 0.3f), color_mode(VELOCITY_BASED), base_color(0.3f, 0.6f, 1.0f),
          color_intensity(1.0f), max_rendered_particles(100000), use_level_of_detail(false),
          lod_distance_threshold(10.0f), background_color(0.1f, 0.1f, 0.2f), show_grid(true), show_boundaries(true)
    {
    }
};

/**
 * Main renderer class for fluid simulation visualization
 */
class Renderer
{
  private:
    // OpenGL context
    GLFWwindow *window;
    int window_width;
    int window_height;

    // Camera system
    std::unique_ptr<Camera> camera;

    // Shader programs
    std::unique_ptr<Shader> particle_shader;
    std::unique_ptr<Shader> sphere_shader;
    std::unique_ptr<Shader> grid_shader;
    std::unique_ptr<Shader> boundary_shader;

    // Buffer management
    std::unique_ptr<BufferManager> buffer_manager;

    // Rendering settings
    RenderSettings settings;

    // Frame timing
    float last_frame_time;
    float delta_time;
    int frame_count;
    float fps;

    // Mouse/keyboard state
    bool first_mouse;
    float last_mouse_x;
    float last_mouse_y;
    bool keys_pressed[1024];

  public:
    Renderer(GLFWwindow *window);
    ~Renderer();

    // Initialization
    bool initialize();
    void cleanup();

    // Main rendering
    void render(const ParticleSystem *particle_system);
    void renderFrame(const ParticleSystem *particle_system);

    // Particle rendering methods
    void renderParticles(const ParticleSystem *particle_system);
    void renderParticlesAsPoints(const ParticleSystem *particle_system);
    void renderParticlesAsSpheres(const ParticleSystem *particle_system);
    void renderParticlesInstanced(const ParticleSystem *particle_system);

    // Scene rendering
    void renderBackground();
    void renderGrid();
    void renderBoundaries(glm::vec3 min_bound, glm::vec3 max_bound);
    void renderUI();

    // Input handling
    void processInput();
    void handleKeyboard(int key, int action);
    void handleMouseMovement(double x_pos, double y_pos);
    void handleMouseScroll(double x_offset, double y_offset);
    void handleWindowResize(int width, int height);

    // Settings
    RenderSettings &getRenderSettings()
    {
        return settings;
    }
    const RenderSettings &getRenderSettings() const
    {
        return settings;
    }
    void updateRenderSettings(const RenderSettings &new_settings);

    // Camera access
    Camera *getCamera()
    {
        return camera.get();
    }

    // Performance monitoring
    float getFPS() const
    {
        return fps;
    }
    float getDeltaTime() const
    {
        return delta_time;
    }
    void updateTiming();

    // Utility methods
    glm::vec3 screenToWorld(glm::vec2 screen_pos, float depth = 0.0f);
    glm::vec2 worldToScreen(glm::vec3 world_pos);

  private:
    // Internal rendering helpers
    void setupMatrices();
    void calculateParticleColors(const ParticleSystem *particle_system, std::vector<glm::vec3> &colors);
    void sortParticlesByDepth(const ParticleSystem *particle_system, std::vector<size_t> &indices);

    // OpenGL state management
    void setDefaultRenderState();
    void enableBlending();
    void enableDepthTesting();

    // Error checking
    void checkGLError(const std::string &operation);
};

/**
 * Specialized renderers for different visualization modes
 */
class ParticleVisualizer
{
  public:
    // Density field visualization
    static void renderDensityField(const ParticleSystem *system, Shader *shader);

    // Velocity field visualization
    static void renderVelocityField(const ParticleSystem *system, Shader *shader);

    // Pressure field visualization
    static void renderPressureField(const ParticleSystem *system, Shader *shader);

    // Force vector visualization
    static void renderForceVectors(const ParticleSystem *system, Shader *shader);

    // Streamlines visualization
    static void renderStreamlines(const ParticleSystem *system, Shader *shader);

    // Metaball/isosurface rendering
    static void renderFluidSurface(const ParticleSystem *system, Shader *shader);
};

// Input handling constants
enum CameraMovement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

} // namespace FluidSim
