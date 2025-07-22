#pragma once
#include <glm/glm.hpp>
#include <vector>

namespace FluidSim
{

/**
 * Individual particle representation for SPH fluid simulation
 * Uses Structure of Arrays (SoA) approach for better CUDA performance
 */
struct Particle
{
    glm::vec3 position;     // Current position (x, y, z)
    glm::vec3 velocity;     // Current velocity vector
    glm::vec3 acceleration; // Current acceleration (forces/mass)

    float mass;     // Particle mass (typically constant)
    float density;  // Current density (calculated via SPH)
    float pressure; // Current pressure (derived from density)

    glm::vec3 color; // RGB color for rendering
    float life;      // Particle lifetime (0.0 to 1.0)

    int id;      // Unique particle identifier
    bool active; // Is particle active in simulation

    // Constructor
    Particle()
        : position(0.0f), velocity(0.0f), acceleration(0.0f), mass(1.0f), density(1000.0f), // Water density kg/m³
          pressure(0.0f), color(0.3f, 0.6f, 1.0f),                                          // Blue water color
          life(1.0f), id(0), active(true)
    {
    }

    Particle(glm::vec3 pos, glm::vec3 vel = glm::vec3(0.0f))
        : position(pos), velocity(vel), acceleration(0.0f), mass(1.0f), density(1000.0f), pressure(0.0f),
          color(0.3f, 0.6f, 1.0f), life(1.0f), id(0), active(true)
    {
    }
};

/**
 * Container for managing large numbers of particles
 * Optimized for both CPU and GPU operations
 */
class ParticleSystem
{
  public:
    std::vector<Particle> particles;
    size_t max_particles;
    size_t active_count;

    // SPH simulation parameters
    float rest_density;     // ρ₀ - target density
    float gas_constant;     // k - pressure stiffness
    float viscosity;        // μ - viscosity coefficient
    float surface_tension;  // σ - surface tension coefficient
    float smoothing_radius; // h - SPH kernel radius

    // Simulation bounds
    glm::vec3 boundary_min;
    glm::vec3 boundary_max;

    ParticleSystem(size_t max_count = 10000);
    ~ParticleSystem() = default;

    // Particle management
    void addParticle(const Particle &particle);
    void addParticle(glm::vec3 position, glm::vec3 velocity = glm::vec3(0.0f));
    void removeParticle(size_t index);
    void clear();

    // Bulk particle creation
    void createParticleGrid(glm::vec3 start, glm::vec3 end, float spacing);
    void createParticleSphere(glm::vec3 center, float radius, float spacing);

    // Getters
    size_t getActiveCount() const
    {
        return active_count;
    }
    size_t getMaxParticles() const
    {
        return max_particles;
    }
    const std::vector<Particle> &getParticles() const
    {
        return particles;
    }
    std::vector<Particle> &getParticles()
    {
        return particles;
    }

    // GPU data preparation
    void *getPositionData();
    void *getColorData();
    size_t getDataSize() const;
};

} // namespace FluidSim
