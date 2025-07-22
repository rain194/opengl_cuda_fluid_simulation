#include "buffer_manager.hh"
#include "../core/particle.hh"
#include <cmath>
#include <iostream>

namespace FluidSim
{

BufferManager::BufferManager()
    : particle_vao(0), particle_position_vbo(0), particle_color_vbo(0), particle_velocity_vbo(0), sphere_vao(0),
      sphere_vbo(0), sphere_ebo(0), sphere_index_count(0), grid_vao(0), grid_vbo(0), grid_vertex_count(0),
      boundary_vao(0), boundary_vbo(0), boundary_ebo(0), max_particles(0), current_particle_count(0),
      buffers_initialized(false), dynamic_buffer_resize(true)
{
}

BufferManager::~BufferManager()
{
    cleanup();
}

bool BufferManager::initialize(size_t max_particle_count)
{
    max_particles = max_particle_count;

    // Create all buffer objects
    if (!createParticleBuffers())
    {
        std::cerr << "Failed to create particle buffers" << std::endl;
        return false;
    }

    if (!createSphereBuffers())
    {
        std::cerr << "Failed to create sphere buffers" << std::endl;
        return false;
    }

    if (!createGridBuffers())
    {
        std::cerr << "Failed to create grid buffers" << std::endl;
        return false;
    }

    if (!createBoundaryBuffers())
    {
        std::cerr << "Failed to create boundary buffers" << std::endl;
        return false;
    }

    buffers_initialized = true;
    std::cout << "BufferManager initialized for " << max_particles << " particles" << std::endl;

    return true;
}

void BufferManager::cleanup()
{
    if (!buffers_initialized)
        return;

    // Delete particle buffers
    if (particle_vao)
        glDeleteVertexArrays(1, &particle_vao);
    if (particle_position_vbo)
        glDeleteBuffers(1, &particle_position_vbo);
    if (particle_color_vbo)
        glDeleteBuffers(1, &particle_color_vbo);
    if (particle_velocity_vbo)
        glDeleteBuffers(1, &particle_velocity_vbo);

    // Delete sphere buffers
    if (sphere_vao)
        glDeleteVertexArrays(1, &sphere_vao);
    if (sphere_vbo)
        glDeleteBuffers(1, &sphere_vbo);
    if (sphere_ebo)
        glDeleteBuffers(1, &sphere_ebo);

    // Delete grid buffers
    if (grid_vao)
        glDeleteVertexArrays(1, &grid_vao);
    if (grid_vbo)
        glDeleteBuffers(1, &grid_vbo);

    // Delete boundary buffers
    if (boundary_vao)
        glDeleteVertexArrays(1, &boundary_vao);
    if (boundary_vbo)
        glDeleteBuffers(1, &boundary_vbo);
    if (boundary_ebo)
        glDeleteBuffers(1, &boundary_ebo);

    buffers_initialized = false;
}

bool BufferManager::createParticleBuffers()
{
    // Generate vertex array object
    glGenVertexArrays(1, &particle_vao);
    glBindVertexArray(particle_vao);

    // Generate vertex buffer objects
    glGenBuffers(1, &particle_position_vbo);
    glGenBuffers(1, &particle_color_vbo);
    glGenBuffers(1, &particle_velocity_vbo);

    // Setup position buffer (location 0)
    glBindBuffer(GL_ARRAY_BUFFER, particle_position_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Setup color buffer (location 1)
    glBindBuffer(GL_ARRAY_BUFFER, particle_color_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);

    // Setup velocity buffer (location 2) - for velocity-based effects
    glBindBuffer(GL_ARRAY_BUFFER, particle_velocity_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(2);

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    checkGLError("createParticleBuffers");
    return true;
}

bool BufferManager::createSphereBuffers()
{
    // Generate sphere geometry
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    generateSphereGeometry(vertices, indices, 20, 20);

    sphere_index_count = indices.size();

    // Create VAO
    glGenVertexArrays(1, &sphere_vao);
    glBindVertexArray(sphere_vao);

    // Create VBO
    glGenBuffers(1, &sphere_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Create EBO
    glGenBuffers(1, &sphere_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Setup vertex attributes (position + normal)
    // Position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Normal (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    checkGLError("createSphereBuffers");
    return true;
}

bool BufferManager::createGridBuffers()
{
    // Generate grid geometry
    std::vector<float> vertices;
    generateGridGeometry(vertices, 10.0f, 20);

    grid_vertex_count = vertices.size() / 3;

    // Create VAO
    glGenVertexArrays(1, &grid_vao);
    glBindVertexArray(grid_vao);

    // Create VBO
    glGenBuffers(1, &grid_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, grid_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Setup vertex attributes (position only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    checkGLError("createGridBuffers");
    return true;
}

bool BufferManager::createBoundaryBuffers()
{
    // Generate boundary box geometry
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    generateBoundaryGeometry(vertices, indices);

    // Create VAO
    glGenVertexArrays(1, &boundary_vao);
    glBindVertexArray(boundary_vao);

    // Create VBO
    glGenBuffers(1, &boundary_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, boundary_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    // Create EBO
    glGenBuffers(1, &boundary_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundary_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Setup vertex attributes (position only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    checkGLError("createBoundaryBuffers");
    return true;
}

void BufferManager::updateParticleData(const std::vector<Particle> &particles, const std::vector<glm::vec3> &colors,
                                       size_t count)
{
    if (!buffers_initialized || count == 0)
        return;

    // Check if we need to resize buffers
    checkBufferSize(count);

    current_particle_count = std::min(count, max_particles);

    // Prepare position data
    std::vector<float> positions(current_particle_count * 3);
    for (size_t i = 0; i < current_particle_count; i++)
    {
        positions[i * 3 + 0] = particles[i].position.x;
        positions[i * 3 + 1] = particles[i].position.y;
        positions[i * 3 + 2] = particles[i].position.z;
    }

    // Prepare color data
    std::vector<float> color_data(current_particle_count * 3);
    for (size_t i = 0; i < current_particle_count; i++)
    {
        if (i < colors.size())
        {
            color_data[i * 3 + 0] = colors[i].r;
            color_data[i * 3 + 1] = colors[i].g;
            color_data[i * 3 + 2] = colors[i].b;
        }
        else
        {
            // Fallback to white
            color_data[i * 3 + 0] = 1.0f;
            color_data[i * 3 + 1] = 1.0f;
            color_data[i * 3 + 2] = 1.0f;
        }
    }

    // Prepare velocity data
    std::vector<float> velocities(current_particle_count * 3);
    for (size_t i = 0; i < current_particle_count; i++)
    {
        velocities[i * 3 + 0] = particles[i].velocity.x;
        velocities[i * 3 + 1] = particles[i].velocity.y;
        velocities[i * 3 + 2] = particles[i].velocity.z;
    }

    // Update position buffer
    glBindBuffer(GL_ARRAY_BUFFER, particle_position_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, current_particle_count * 3 * sizeof(float), positions.data());

    // Update color buffer
    glBindBuffer(GL_ARRAY_BUFFER, particle_color_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, current_particle_count * 3 * sizeof(float), color_data.data());

    // Update velocity buffer
    glBindBuffer(GL_ARRAY_BUFFER, particle_velocity_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, current_particle_count * 3 * sizeof(float), velocities.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkGLError("updateParticleData");
}

void BufferManager::updateParticlePositions(const std::vector<glm::vec3> &positions)
{
    if (!buffers_initialized || positions.empty())
        return;

    size_t count = std::min(positions.size(), max_particles);
    std::vector<float> position_data(count * 3);

    for (size_t i = 0; i < count; i++)
    {
        position_data[i * 3 + 0] = positions[i].x;
        position_data[i * 3 + 1] = positions[i].y;
        position_data[i * 3 + 2] = positions[i].z;
    }

    glBindBuffer(GL_ARRAY_BUFFER, particle_position_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * 3 * sizeof(float), position_data.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void BufferManager::renderParticles()
{
    if (!buffers_initialized || current_particle_count == 0)
        return;

    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, current_particle_count);
    glBindVertexArray(0);

    checkGLError("renderParticles");
}

void BufferManager::renderSphere()
{
    if (!buffers_initialized || sphere_index_count == 0)
        return;

    glBindVertexArray(sphere_vao);
    glDrawElements(GL_TRIANGLES, sphere_index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    checkGLError("renderSphere");
}

void BufferManager::renderGrid()
{
    if (!buffers_initialized || grid_vertex_count == 0)
        return;

    glBindVertexArray(grid_vao);
    glDrawArrays(GL_LINES, 0, grid_vertex_count);
    glBindVertexArray(0);

    checkGLError("renderGrid");
}

void BufferManager::renderBoundaryBox(glm::vec3 min_bound, glm::vec3 max_bound)
{
    if (!buffers_initialized)
        return;

    // Update boundary geometry with new bounds
    std::vector<float> vertices = {// Bottom face
                                   min_bound.x, min_bound.y, min_bound.z, max_bound.x, min_bound.y, min_bound.z,
                                   max_bound.x, min_bound.y, max_bound.z, min_bound.x, min_bound.y, max_bound.z,

                                   // Top face
                                   min_bound.x, max_bound.y, min_bound.z, max_bound.x, max_bound.y, min_bound.z,
                                   max_bound.x, max_bound.y, max_bound.z, min_bound.x, max_bound.y, max_bound.z};

    // Update buffer data
    glBindBuffer(GL_ARRAY_BUFFER, boundary_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Render as wireframe
    glBindVertexArray(boundary_vao);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0); // 12 edges * 2 vertices each
    glBindVertexArray(0);

    checkGLError("renderBoundaryBox");
}

// Geometry generation methods
void BufferManager::generateSphereGeometry(std::vector<float> &vertices, std::vector<unsigned int> &indices, int stacks,
                                           int slices)
{
    vertices.clear();
    indices.clear();

    // Generate vertices
    for (int i = 0; i <= stacks; i++)
    {
        float phi = 3.14159265358979323846 * i / stacks; // 0 to PI
        float y = cos(phi);
        float radius_at_y = sin(phi);

        for (int j = 0; j <= slices; j++)
        {
            float theta = 2 * 3.14159265358979323846 * j / slices; // 0 to 2*PI
            float x = radius_at_y * cos(theta);
            float z = radius_at_y * sin(theta);

            // Position
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // Normal (same as position for unit sphere)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    // Generate indices
    for (int i = 0; i < stacks; i++)
    {
        for (int j = 0; j < slices; j++)
        {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            // First triangle
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            // Second triangle
            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
}

void BufferManager::generateGridGeometry(std::vector<float> &vertices, float size, int divisions)
{
    vertices.clear();

    float step = size / divisions;
    float half_size = size * 0.5f;

    // Generate horizontal lines
    for (int i = 0; i <= divisions; i++)
    {
        float z = -half_size + i * step;

        // Line from left to right
        vertices.push_back(-half_size);
        vertices.push_back(0.0f);
        vertices.push_back(z);
        vertices.push_back(half_size);
        vertices.push_back(0.0f);
        vertices.push_back(z);
    }

    // Generate vertical lines
    for (int i = 0; i <= divisions; i++)
    {
        float x = -half_size + i * step;

        // Line from back to front
        vertices.push_back(x);
        vertices.push_back(0.0f);
        vertices.push_back(-half_size);
        vertices.push_back(x);
        vertices.push_back(0.0f);
        vertices.push_back(half_size);
    }
}

void BufferManager::generateBoundaryGeometry(std::vector<float> &vertices, std::vector<unsigned int> &indices)
{
    vertices = {
        // Bottom face vertices (y = min)
        0.0f, 0.0f, 0.0f, // 0
        1.0f, 0.0f, 0.0f, // 1
        1.0f, 0.0f, 1.0f, // 2
        0.0f, 0.0f, 1.0f, // 3

        // Top face vertices (y = max)
        0.0f, 1.0f, 0.0f, // 4
        1.0f, 1.0f, 0.0f, // 5
        1.0f, 1.0f, 1.0f, // 6
        0.0f, 1.0f, 1.0f  // 7
    };

    // Edge indices for wireframe rendering
    indices = {// Bottom face edges
               0, 1, 1, 2, 2, 3, 3, 0,
               // Top face edges
               4, 5, 5, 6, 6, 7, 7, 4,
               // Vertical edges
               0, 4, 1, 5, 2, 6, 3, 7};
}

void BufferManager::checkBufferSize(size_t required_size)
{
    if (required_size <= max_particles)
        return;

    if (dynamic_buffer_resize)
    {
        std::cout << "Resizing buffers from " << max_particles << " to " << required_size << " particles" << std::endl;
        resizeBuffers(required_size);
    }
    else
    {
        std::cerr << "Buffer overflow: need " << required_size << " particles but max is " << max_particles
                  << std::endl;
    }
}

void BufferManager::resizeBuffers(size_t new_max_particles)
{
    max_particles = new_max_particles;

    // Resize position buffer
    glBindBuffer(GL_ARRAY_BUFFER, particle_position_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Resize color buffer
    glBindBuffer(GL_ARRAY_BUFFER, particle_color_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Resize velocity buffer
    glBindBuffer(GL_ARRAY_BUFFER, particle_velocity_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkGLError("resizeBuffers");
}

void BufferManager::checkGLError(const std::string &operation)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        std::cerr << "OpenGL error in BufferManager::" << operation << ": " << error << std::endl;
    }
}

// AdvancedBufferManager static methods
bool AdvancedBufferManager::setupTransformFeedback(GLuint &transform_feedback_object, GLuint &feedback_buffer,
                                                   size_t buffer_size)
{
    // Generate transform feedback object
    glGenTransformFeedbacks(1, &transform_feedback_object);

    // Generate feedback buffer
    glGenBuffers(1, &feedback_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, feedback_buffer);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_READ);

    // Bind feedback buffer to transform feedback
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, transform_feedback_object);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, feedback_buffer);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}

bool AdvancedBufferManager::setupComputeBuffers(GLuint &ssbo_positions, GLuint &ssbo_velocities,
                                                GLuint &ssbo_properties, size_t particle_count)
{
    // Position SSBO
    glGenBuffers(1, &ssbo_positions);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_positions);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particle_count * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_positions);

    // Velocity SSBO
    glGenBuffers(1, &ssbo_velocities);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_velocities);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particle_count * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_velocities);

    // Properties SSBO (mass, density, pressure, etc.)
    glGenBuffers(1, &ssbo_properties);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_properties);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particle_count * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_properties);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return true;
}

} // namespace FluidSim
