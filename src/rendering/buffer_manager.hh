#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>
#include <string> 

namespace FluidSim
{

// Forward declarations
struct Particle;

/**
 * OpenGL buffer management for efficient particle rendering
 * Handles VBOs, VAOs, and dynamic buffer updates
 */
class BufferManager
{
  private:
    // Particle rendering buffers
    GLuint particle_vao;
    GLuint particle_position_vbo;
    GLuint particle_color_vbo;
    GLuint particle_velocity_vbo; // For velocity-based effects

    // Sphere rendering (for high-quality particles)
    GLuint sphere_vao;
    GLuint sphere_vbo;
    GLuint sphere_ebo;
    int sphere_index_count;

    // Grid rendering
    GLuint grid_vao;
    GLuint grid_vbo;
    int grid_vertex_count;

    // Boundary box rendering
    GLuint boundary_vao;
    GLuint boundary_vbo;
    GLuint boundary_ebo;

    // Buffer capacities
    size_t max_particles;
    size_t current_particle_count;

    // Dynamic buffer management
    bool buffers_initialized;
    bool dynamic_buffer_resize;

  public:
    BufferManager();
    ~BufferManager();

    // Initialization and cleanup
    bool initialize(size_t max_particle_count = 100000);
    void cleanup();

    // Particle data management
    void updateParticleData(const std::vector<Particle> &particles, const std::vector<glm::vec3> &colors, size_t count);
    void updateParticlePositions(const std::vector<glm::vec3> &positions);
    void updateParticleColors(const std::vector<glm::vec3> &colors);
    void updateParticleVelocities(const std::vector<glm::vec3> &velocities);

    // Rendering methods
    void renderParticles();
    void renderParticlesInstanced(size_t instance_count);
    void renderSphere();
    void renderGrid();
    void renderBoundaryBox(glm::vec3 min_bound, glm::vec3 max_bound);

    // Buffer state queries
    size_t getMaxParticles() const
    {
        return max_particles;
    }
    size_t getCurrentParticleCount() const
    {
        return current_particle_count;
    }
    bool isInitialized() const
    {
        return buffers_initialized;
    }

    // Performance optimization
    void resizeBuffers(size_t new_max_particles);
    void setDynamicResize(bool enable)
    {
        dynamic_buffer_resize = enable;
    }

    // OpenGL buffer access (for advanced usage)
    GLuint getParticleVAO() const
    {
        return particle_vao;
    }
    GLuint getPositionVBO() const
    {
        return particle_position_vbo;
    }
    GLuint getColorVBO() const
    {
        return particle_color_vbo;
    }

  private:
    // Buffer creation helpers
    bool createParticleBuffers();
    bool createSphereBuffers();
    bool createGridBuffers();
    bool createBoundaryBuffers();

    // Geometry generation
    void generateSphereGeometry(std::vector<float> &vertices, std::vector<unsigned int> &indices, int stacks = 20,
                                int slices = 20);
    void generateGridGeometry(std::vector<float> &vertices, float size = 10.0f, int divisions = 20);
    void generateBoundaryGeometry(std::vector<float> &vertices, std::vector<unsigned int> &indices);

    // Buffer utilities
    void bindParticleBuffers();
    void unbindBuffers();
    void checkBufferSize(size_t required_size);

    // Error checking
    void checkGLError(const std::string &operation);
};

/**
 * Advanced buffer techniques for high-performance rendering
 */
class AdvancedBufferManager
{
  public:
    // Transform feedback for GPU-based particle updates
    static bool setupTransformFeedback(GLuint &transform_feedback_object, GLuint &feedback_buffer, size_t buffer_size);

    // Compute shader buffer management
    static bool setupComputeBuffers(GLuint &ssbo_positions, GLuint &ssbo_velocities, GLuint &ssbo_properties,
                                    size_t particle_count);

    // Instanced rendering setup
    static bool setupInstancedRendering(GLuint &instance_vbo, const std::vector<glm::mat4> &instance_matrices);

    // Multi-draw indirect setup
    static bool setupIndirectDrawing(GLuint &indirect_buffer, const std::vector<GLuint> &draw_commands);

    // Buffer streaming for large datasets
    static void *mapBufferRange(GLuint buffer, size_t offset, size_t size, GLbitfield access);
    static void unmapBuffer(GLuint buffer);
    static void invalidateBuffer(GLuint buffer);
};

} // namespace FluidSim
