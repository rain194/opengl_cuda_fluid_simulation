#define GLM_ENABLE_EXPERIMENTAL
#include "fluid_sim.hh"
#include "../utils/math_utils.hh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace FluidSim
{

// ParticleSystem Implementation
ParticleSystem::ParticleSystem(size_t max_count)
    : max_particles(max_count), active_count(0), rest_density(1000.0f), // Water density kg/m³
      gas_constant(200.0f),                                             // Pressure stiffness
      viscosity(0.001f),                                                // Water viscosity
      surface_tension(0.0728f),                                         // Water surface tension N/m
      smoothing_radius(0.1f),                                           // SPH kernel radius
      boundary_min(-5.0f, -5.0f, -5.0f), boundary_max(5.0f, 5.0f, 5.0f)
{

    particles.reserve(max_particles);
    std::cout << "ParticleSystem initialized with max " << max_particles << " particles\n";
}

void ParticleSystem::addParticle(const Particle &particle)
{
    if (active_count < max_particles)
    {
        particles.push_back(particle);
        particles.back().id = static_cast<int>(active_count);
        active_count++;
    }
}

void ParticleSystem::addParticle(glm::vec3 position, glm::vec3 velocity)
{
    if (active_count < max_particles)
    {
        Particle p(position, velocity);
        addParticle(p);
    }
}

void ParticleSystem::createParticleGrid(glm::vec3 start, glm::vec3 end, float spacing)
{
    std::cout << "Creating particle grid from " << start.x << "," << start.y << "," << start.z << " to " << end.x << ","
              << end.y << "," << end.z << "\n";

    for (float x = start.x; x <= end.x; x += spacing)
    {
        for (float y = start.y; y <= end.y; y += spacing)
        {
            for (float z = start.z; z <= end.z; z += spacing)
            {
                if (active_count >= max_particles)
                    break;

                // Add small random offset for more natural distribution
                float jitter = spacing * 0.1f;
                glm::vec3 pos = glm::vec3(x, y, z) + glm::vec3(((rand() % 200) - 100) / 100.0f * jitter,
                                                               ((rand() % 200) - 100) / 100.0f * jitter,
                                                               ((rand() % 200) - 100) / 100.0f * jitter);

                addParticle(pos);
            }
        }
    }

    std::cout << "Created " << active_count << " particles in grid\n";
}

void ParticleSystem::clear()
{
    particles.clear();
    active_count = 0;
}

// FluidSimulator Implementation
FluidSimulator::FluidSimulator(bool enable_cuda)
    : time_step(0.016f),                                               // ~60 FPS
      current_time(0.0f), frame_count(0), gravity(0.0f, -9.81f, 0.0f), // Earth gravity
      damping_factor(0.99f), use_cuda(enable_cuda), is_running(false), is_paused(false)
{

    particle_system = std::make_unique<ParticleSystem>(10000);

    if (use_cuda)
    {
        initializeCUDA();
    }

    std::cout << "FluidSimulator initialized (CUDA: " << (use_cuda ? "enabled" : "disabled") << ")\n";
}

FluidSimulator::~FluidSimulator()
{
    if (use_cuda)
    {
        cleanupCUDA();
    }
}

void FluidSimulator::initialize(const SimulationConfig &config)
{
    // Apply configuration
    time_step = config.time_step;
    gravity = config.gravity;
    use_cuda = config.use_cuda;

    // Set particle system parameters
    particle_system->rest_density = config.rest_density;
    particle_system->gas_constant = config.gas_constant;
    particle_system->viscosity = config.viscosity;
    particle_system->smoothing_radius = config.smoothing_radius;
    particle_system->boundary_min = config.boundary_min;
    particle_system->boundary_max = config.boundary_max;

    // Create initial particles
    if (config.initial_particle_count > 0)
    {
        particle_system->createParticleGrid(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 0.2f);
    }

    is_running = true;
    std::cout << "Simulation initialized with " << particle_system->getActiveCount() << " particles\n";
}

void FluidSimulator::update(float delta_time)
{
    if (!is_running || is_paused)
        return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Use fixed time step for stability
    updatePhysics(time_step);

    current_time += time_step;
    frame_count++;

    // Print progress every 60 frames
    if (frame_count % 60 == 0)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Frame " << frame_count << " - Physics update: " << duration.count() / 1000.0f << "ms\n";
    }
}

void FluidSimulator::updatePhysics(float dt)
{
    // Simple physics update - don't call CUDA version
    findNeighbors();
    calculateDensities();
    calculatePressures();
    calculateForces();
    integrateMotion(dt);
    handleBoundaryCollisions();
}

// SPH Physics Implementation (CPU)
void FluidSimulator::findNeighbors()
{
    size_t particle_count = particle_system->getActiveCount();
    neighbor_lists.clear();
    neighbor_lists.resize(particle_count);

    float h = particle_system->smoothing_radius;
    float h_squared = h * h;

    // Simple O(n²) neighbor search (we'll optimize this later)
    for (size_t i = 0; i < particle_count; i++)
    {
        neighbor_lists[i].clear();

        for (size_t j = 0; j < particle_count; j++)
        {
            if (i == j)
                continue;

            glm::vec3 diff = particle_system->particles[i].position - particle_system->particles[j].position;
            float dist_squared = glm::dot(diff, diff);

            if (dist_squared < h_squared)
            {
                neighbor_lists[i].push_back(j);
            }
        }
    }
}

void FluidSimulator::calculateDensities()
{
    size_t particle_count = particle_system->getActiveCount();
    densities.resize(particle_count);

    float h = particle_system->smoothing_radius;

    // SPH density calculation: ρᵢ = Σⱼ mⱼ * W(|rᵢ - rⱼ|, h)
    for (size_t i = 0; i < particle_count; i++)
    {
        float density = 0.0f;

        // Self contribution
        density += particle_system->particles[i].mass * kernelPoly6(0.0f, h);

        // Neighbor contributions
        for (size_t j : neighbor_lists[i])
        {
            glm::vec3 diff = particle_system->particles[i].position - particle_system->particles[j].position;
            float distance = glm::length(diff);

            density += particle_system->particles[j].mass * kernelPoly6(distance, h);
        }

        densities[i] = density;
        particle_system->particles[i].density = density;
    }
}

void FluidSimulator::calculatePressures()
{
    size_t particle_count = particle_system->getActiveCount();

    // Ideal gas state equation: P = k * (ρ - ρ₀)
    float k = particle_system->gas_constant;
    float rho0 = particle_system->rest_density;

    for (size_t i = 0; i < particle_count; i++)
    {
        float pressure = k * (densities[i] - rho0);
        particle_system->particles[i].pressure = std::max(pressure, 0.0f); // Prevent negative pressure
    }
}

void FluidSimulator::integrateMotion(float dt)
{
    size_t particle_count = particle_system->getActiveCount();

    // Euler integration: v = v + a*dt, x = x + v*dt
    for (size_t i = 0; i < particle_count; i++)
    {
        Particle &p = particle_system->particles[i];

        // Add gravity
        p.acceleration += gravity;

        // Update velocity and position
        p.velocity += p.acceleration * dt;
        p.velocity *= damping_factor; // Apply damping
        p.position += p.velocity * dt;

        // Reset acceleration for next frame
        p.acceleration = glm::vec3(0.0f);
    }
}

void FluidSimulator::handleBoundaryCollisions()
{
    glm::vec3 min_bound = particle_system->boundary_min;
    glm::vec3 max_bound = particle_system->boundary_max;
    float restitution = 0.5f; // Bounce factor

    for (auto &particle : particle_system->particles)
    {
        // X boundaries
        if (particle.position.x < min_bound.x)
        {
            particle.position.x = min_bound.x;
            particle.velocity.x = -particle.velocity.x * restitution;
        }
        else if (particle.position.x > max_bound.x)
        {
            particle.position.x = max_bound.x;
            particle.velocity.x = -particle.velocity.x * restitution;
        }

        // Y boundaries
        if (particle.position.y < min_bound.y)
        {
            particle.position.y = min_bound.y;
            particle.velocity.y = -particle.velocity.y * restitution;
        }
        else if (particle.position.y > max_bound.y)
        {
            particle.position.y = max_bound.y;
            particle.velocity.y = -particle.velocity.y * restitution;
        }

        // Z boundaries
        if (particle.position.z < min_bound.z)
        {
            particle.position.z = min_bound.z;
            particle.velocity.z = -particle.velocity.z * restitution;
        }
        else if (particle.position.z > max_bound.z)
        {
            particle.position.z = max_bound.z;
            particle.velocity.z = -particle.velocity.z * restitution;
        }
    }
}

// SPH Kernel Functions
float FluidSimulator::kernelPoly6(float r, float h)
{
    if (r > h)
        return 0.0f;

    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float diff = h2 - r * r;

    return (315.0f / (64.0f * 3.14159265358979323846 * h9)) * diff * diff * diff;
}

// Placeholder methods (will implement in next phases)
void FluidSimulator::calculateForces()
{
    auto &particles = particle_system->getParticles();
    size_t particle_count = particle_system->getActiveCount();

    // Simple gravity application
    for (size_t i = 0; i < particle_count; i++)
    {
        particles[i].acceleration = gravity;
    }
}

void FluidSimulator::reset()
{
    particle_system->clear();
    current_time = 0.0f;
    frame_count = 0;
    is_running = false;
}

void FluidSimulator::pause()
{
    is_paused = true;
}
void FluidSimulator::resume()
{
    is_paused = false;
}

// CUDA placeholder methods (implemented in fluid_sim.cu)
void FluidSimulator::initializeCUDA()
{
    std::cout << "CUDA initialization - placeholder\n";
}

void FluidSimulator::cleanupCUDA()
{
    std::cout << "CUDA cleanup - placeholder\n";
}

void FluidSimulator::updatePhysicsCUDA(float dt)
{
    // For now, fall back to CPU
    std::cout << "CUDA physics update - using CPU fallback\n";
    updatePhysics(dt);
}

} // namespace FluidSim
