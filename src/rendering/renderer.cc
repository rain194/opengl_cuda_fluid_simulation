#include "renderer.hh"
#include "../core/particle.hh"
#include "buffer_manager.hh"
#include "shader.hh"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace FluidSim
{

// Camera Implementation
Camera::Camera(glm::vec3 pos, glm::vec3 up, float yaw, float pitch)
    : position(pos), world_up(up), yaw(yaw), pitch(pitch), movement_speed(5.0f), mouse_sensitivity(0.1f), zoom(45.0f)
{
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjectionMatrix(float aspect_ratio, float near_plane, float far_plane) const
{
    return glm::perspective(glm::radians(zoom), aspect_ratio, near_plane, far_plane);
}

void Camera::processKeyboard(int direction, float delta_time)
{
    float velocity = movement_speed * delta_time;

    switch (direction)
    {
    case FORWARD:
        position += front * velocity;
        break;
    case BACKWARD:
        position -= front * velocity;
        break;
    case LEFT:
        position -= right * velocity;
        break;
    case RIGHT:
        position += right * velocity;
        break;
    case UP:
        position += world_up * velocity;
        break;
    case DOWN:
        position -= world_up * velocity;
        break;
    }
}

void Camera::processMouseMovement(float x_offset, float y_offset, bool constrain_pitch)
{
    x_offset *= mouse_sensitivity;
    y_offset *= mouse_sensitivity;

    yaw += x_offset;
    pitch += y_offset;

    if (constrain_pitch)
    {
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }

    updateCameraVectors();
}

void Camera::processMouseScroll(float y_offset)
{
    zoom -= y_offset;
    zoom = std::clamp(zoom, 1.0f, 45.0f);
}

void Camera::lookAt(glm::vec3 target)
{
    front = glm::normalize(target - position);
    updateCameraVectors();
}

void Camera::updateCameraVectors()
{
    glm::vec3 new_front;
    new_front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    new_front.y = sin(glm::radians(pitch));
    new_front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    front = glm::normalize(new_front);
    right = glm::normalize(glm::cross(front, world_up));
    up = glm::normalize(glm::cross(right, front));
}

// Renderer Implementation
Renderer::Renderer(GLFWwindow *window)
    : window(window), window_width(800), window_height(600), last_frame_time(0.0f), delta_time(0.0f), frame_count(0),
      fps(0.0f), first_mouse(true), last_mouse_x(400.0f), last_mouse_y(300.0f)
{

    // Initialize key states
    for (int i = 0; i < 1024; i++)
    {
        keys_pressed[i] = false;
    }

    // Get window size
    glfwGetFramebufferSize(window, &window_width, &window_height);

    std::cout << "Renderer created for window " << window_width << "x" << window_height << std::endl;
}

Renderer::~Renderer()
{
    cleanup();
}

bool Renderer::initialize()
{
    // Initialize OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set viewport
    glViewport(0, 0, window_width, window_height);

    // Create camera
    camera = std::make_unique<Camera>(glm::vec3(0.0f, 2.0f, 5.0f));

    // Create buffer manager
    buffer_manager = std::make_unique<BufferManager>();
    if (!buffer_manager->initialize())
    {
        std::cerr << "Failed to initialize buffer manager" << std::endl;
        return false;
    }

    // Load shaders
    particle_shader = std::make_unique<Shader>();
    if (!particle_shader->loadFromFiles("shaders/particle.vert", "shaders/particle.frag"))
    {
        std::cerr << "Failed to load particle shaders - using fallback" << std::endl;
        // Create a basic shader instead of failing
        particle_shader = ShaderLibrary::createParticleShader();
        if (!particle_shader)
        {
            std::cerr << "Failed to create fallback shader" << std::endl;
            return false;
        }
    }

    sphere_shader = std::make_unique<Shader>();
    if (!sphere_shader->loadFromFiles("shaders/sphere.vert", "shaders/sphere.frag"))
    {
        std::cout << "Sphere shader not found, using particle shader for spheres" << std::endl;
        sphere_shader = nullptr; // Will fallback to particle shader
    }

    // Create grid shader for debugging
    grid_shader = std::make_unique<Shader>();
    if (!grid_shader->loadFromSource(
            // Grid vertex shader
            R"(
        #version 330 core
        layout (location = 0) in vec3 position;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * vec4(position, 1.0);
        }
        )",
            // Grid fragment shader
            R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 color;
        void main() {
            FragColor = vec4(color, 0.3);
        }
        )"))
    {
        std::cout << "Grid shader compilation failed, grid rendering disabled" << std::endl;
        grid_shader = nullptr;
    }

    // Set default render state
    setDefaultRenderState();

    std::cout << "Renderer initialized successfully" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    return true;
}

void Renderer::cleanup()
{
    particle_shader.reset();
    sphere_shader.reset();
    grid_shader.reset();
    boundary_shader.reset();
    buffer_manager.reset();
    camera.reset();
}

void Renderer::renderFrame(const ParticleSystem *particle_system)
{
    updateTiming();

    // Clear buffers
    glClearColor(settings.background_color.r, settings.background_color.g, settings.background_color.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup camera matrices
    setupMatrices();

    // Render scene elements
    if (settings.show_grid)
    {
        renderGrid();
    }

    if (settings.show_boundaries && particle_system)
    {
        renderBoundaries(particle_system->boundary_min, particle_system->boundary_max);
    }

    // Render particles
    if (particle_system && particle_system->getActiveCount() > 0)
    {
        renderParticles(particle_system);
    }

    // Render UI overlay
    renderUI();

    // Check for OpenGL errors
    checkGLError("renderFrame");
}

void Renderer::render(const ParticleSystem *particle_system)
{
    renderFrame(particle_system);
}

void Renderer::renderParticles(const ParticleSystem *particle_system)
{
    if (!particle_system || particle_system->getActiveCount() == 0)
        return;

    if (settings.render_as_spheres && sphere_shader)
    {
        renderParticlesAsSpheres(particle_system);
    }
    else if (settings.enable_instancing)
    {
        renderParticlesInstanced(particle_system);
    }
    else
    {
        renderParticlesAsPoints(particle_system);
    }
}

void Renderer::renderParticlesAsPoints(const ParticleSystem *particle_system)
{
    if (!particle_shader)
        return;

    particle_shader->use();

    // Set uniforms
    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 projection = camera->getProjectionMatrix((float)window_width / window_height);

    particle_shader->setMat4("view", view);
    particle_shader->setMat4("projection", projection);
    particle_shader->setFloat("particle_size", 5.0f);
    particle_shader->setVec3("base_color", glm::vec3(1.0f, 1.0f, 1.0f));

    // Prepare particle data
    const auto &particles = particle_system->getParticles();
    std::vector<glm::vec3> colors(particle_system->getActiveCount(), glm::vec3(0.2f, 0.6f, 1.0f));

    // Update buffer data
    buffer_manager->updateParticleData(particles, colors, particle_system->getActiveCount());

    // Render
    glEnable(GL_PROGRAM_POINT_SIZE);
    buffer_manager->renderParticles();
    glDisable(GL_PROGRAM_POINT_SIZE);
}

void Renderer::renderParticlesAsSpheres(const ParticleSystem *particle_system)
{
    if (!sphere_shader)
    {
        // Fallback to point rendering
        renderParticlesAsPoints(particle_system);
        return;
    }

    sphere_shader->use();

    // Set matrices
    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 projection = camera->getProjectionMatrix((float)window_width / window_height);

    sphere_shader->setMat4("view", view);
    sphere_shader->setMat4("projection", projection);

    // Set lighting parameters
    if (settings.enable_lighting)
    {
        sphere_shader->setVec3("light_direction", settings.light_direction);
        sphere_shader->setVec3("light_color", settings.light_color);
        sphere_shader->setVec3("ambient_color", settings.ambient_color);
        sphere_shader->setVec3("camera_position", camera->getPosition());
    }

    sphere_shader->setFloat("particle_radius", settings.particle_size);

    // Render each particle as a sphere (expensive, but high quality)
    const auto &particles = particle_system->getParticles();
    for (size_t i = 0; i < particle_system->getActiveCount(); i++)
    {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, particles[i].position);
        model = glm::scale(model, glm::vec3(settings.particle_size));

        sphere_shader->setMat4("model", model);
        sphere_shader->setVec3("particle_color", particles[i].color);

        buffer_manager->renderSphere();
    }
}

void Renderer::renderParticlesInstanced(const ParticleSystem *particle_system)
{
    // TODO: Implement instanced rendering for better performance
    // This would render all particles in a single draw call
    renderParticlesAsPoints(particle_system);
}

void Renderer::renderGrid()
{
    if (!grid_shader)
        return;

    grid_shader->use();

    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 projection = camera->getProjectionMatrix((float)window_width / window_height);

    grid_shader->setMat4("view", view);
    grid_shader->setMat4("projection", projection);
    grid_shader->setVec3("color", glm::vec3(0.3f, 0.3f, 0.3f));

    // Render simple grid lines
    buffer_manager->renderGrid();
}

void Renderer::renderBoundaries(glm::vec3 min_bound, glm::vec3 max_bound)
{
    if (!grid_shader)
        return;

    grid_shader->use();

    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 projection = camera->getProjectionMatrix((float)window_width / window_height);

    grid_shader->setMat4("view", view);
    grid_shader->setMat4("projection", projection);
    grid_shader->setVec3("color", glm::vec3(1.0f, 0.0f, 0.0f)); // Red boundaries

    // Render boundary box
    buffer_manager->renderBoundaryBox(min_bound, max_bound);
}

void Renderer::renderUI()
{
    // Simple text overlay showing FPS and particle count
    // For now, just print to console every 60 frames
    static int ui_frame_counter = 0;
    ui_frame_counter++;

    if (ui_frame_counter % 60 == 0)
    {
        std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps << " | Frame time: " << delta_time * 1000.0f
                  << "ms" << std::endl;
    }
}

void Renderer::setupMatrices()
{
    // Matrices are set in individual render functions
    // This could be optimized to set them once per frame
}

void Renderer::calculateParticleColors(const ParticleSystem *particle_system, std::vector<glm::vec3> &colors)
{
    const auto &particles = particle_system->getParticles();
    size_t count = particle_system->getActiveCount();
    colors.resize(count);

    switch (settings.color_mode)
    {
    case RenderSettings::SOLID_COLOR:
        for (size_t i = 0; i < count; i++)
        {
            colors[i] = settings.base_color;
        }
        break;

    case RenderSettings::VELOCITY_BASED: {
        // Color based on velocity magnitude
        float max_velocity = 0.0f;
        for (size_t i = 0; i < count; i++)
        {
            max_velocity = std::max(max_velocity, glm::length(particles[i].velocity));
        }

        for (size_t i = 0; i < count; i++)
        {
            float vel_magnitude = glm::length(particles[i].velocity);
            float normalized_vel = max_velocity > 0 ? vel_magnitude / max_velocity : 0.0f;

            // Blue (slow) to red (fast)
            colors[i] = glm::vec3(normalized_vel, 0.2f, 1.0f - normalized_vel);
        }
        break;
    }

    case RenderSettings::DENSITY_BASED: {
        // Color based on density
        float max_density = 0.0f;
        float min_density = FLT_MAX;
        for (size_t i = 0; i < count; i++)
        {
            max_density = std::max(max_density, particles[i].density);
            min_density = std::min(min_density, particles[i].density);
        }

        for (size_t i = 0; i < count; i++)
        {
            float normalized_density =
                (max_density > min_density) ? (particles[i].density - min_density) / (max_density - min_density) : 0.0f;

            // Green (low density) to blue (high density)
            colors[i] = glm::vec3(0.1f, 1.0f - normalized_density, normalized_density);
        }
        break;
    }

    case RenderSettings::PRESSURE_BASED: {
        // Color based on pressure
        float max_pressure = 0.0f;
        for (size_t i = 0; i < count; i++)
        {
            max_pressure = std::max(max_pressure, particles[i].pressure);
        }

        for (size_t i = 0; i < count; i++)
        {
            float normalized_pressure = max_pressure > 0 ? particles[i].pressure / max_pressure : 0.0f;

            // Yellow (low pressure) to red (high pressure)
            colors[i] = glm::vec3(1.0f, 1.0f - normalized_pressure * 0.5f, 0.1f);
        }
        break;
    }

    default:
        // Use particle's own color
        for (size_t i = 0; i < count; i++)
        {
            colors[i] = particles[i].color;
        }
        break;
    }
}

void Renderer::processInput()
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera->processKeyboard(FORWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera->processKeyboard(BACKWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera->processKeyboard(LEFT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera->processKeyboard(RIGHT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera->processKeyboard(UP, delta_time);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera->processKeyboard(DOWN, delta_time);
}

void Renderer::handleMouseMovement(double x_pos, double y_pos)
{
    if (first_mouse)
    {
        last_mouse_x = x_pos;
        last_mouse_y = y_pos;
        first_mouse = false;
    }

    float x_offset = x_pos - last_mouse_x;
    float y_offset = last_mouse_y - y_pos; // Reversed since y-coordinates go from bottom to top

    last_mouse_x = x_pos;
    last_mouse_y = y_pos;

    camera->processMouseMovement(x_offset, y_offset);
}

void Renderer::handleMouseScroll(double x_offset, double y_offset)
{
    camera->processMouseScroll(y_offset);
}

void Renderer::handleWindowResize(int width, int height)
{
    window_width = width;
    window_height = height;
    glViewport(0, 0, width, height);
}

void Renderer::updateTiming()
{
    float current_time = glfwGetTime();
    delta_time = current_time - last_frame_time;
    last_frame_time = current_time;

    frame_count++;

    // Calculate FPS every second
    static float fps_timer = 0.0f;
    static int fps_frame_count = 0;

    fps_timer += delta_time;
    fps_frame_count++;

    if (fps_timer >= 1.0f)
    {
        fps = fps_frame_count / fps_timer;
        fps_timer = 0.0f;
        fps_frame_count = 0;
    }
}

void Renderer::setDefaultRenderState()
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
}

void Renderer::checkGLError(const std::string &operation)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        std::cerr << "OpenGL error in " << operation << ": " << error << std::endl;
    }
}

} // namespace FluidSim
