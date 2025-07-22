#pragma once
#include <glm/glm.hpp>
#include <vector>

namespace FluidSim
{

// Forward declarations
struct Particle;
class ParticleSystem;

/**
 * Force calculation utilities for SPH fluid simulation
 * Implements pressure, viscosity, and surface tension forces
 */
class ForceCalculator
{
  private:
    // SPH parameters (set by ParticleSystem)
    float smoothing_radius;
    float rest_density;
    float gas_constant;
    float viscosity_coefficient;
    float surface_tension_coefficient;

    // Temporary buffers for calculations
    std::vector<float> densities;
    std::vector<float> pressures;
    std::vector<glm::vec3> pressure_forces;
    std::vector<glm::vec3> viscosity_forces;
    std::vector<glm::vec3> surface_forces;

  public:
    ForceCalculator();
    ~ForceCalculator() = default;

    // Initialize with simulation parameters
    void initialize(float h, float rho0, float k, float mu, float sigma);

    // Main force calculation entry points
    void calculateAllForces(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists);

    // Individual force calculations
    void calculateDensities(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists);

    void calculatePressureForces(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists);

    void calculateViscosityForces(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists);

    void calculateSurfaceTensionForces(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists);

    // SPH kernel functions
    static float kernelPoly6(float r, float h);
    static glm::vec3 kernelSpikyGradient(glm::vec3 r_vec, float h);
    static float kernelViscosityLaplacian(float r, float h);

    // External forces
    static glm::vec3 calculateGravityForce(float mass, glm::vec3 gravity);
    static glm::vec3 calculateBuoyancyForce(float density, float rest_density, glm::vec3 gravity,
                                            float buoyancy_coeff = 1.0f);

    // Utility functions
    void resizeBuffers(size_t particle_count);
    void clearForces(ParticleSystem *system);

    // Getters for debugging
    const std::vector<float> &getDensities() const
    {
        return densities;
    }
    const std::vector<float> &getPressures() const
    {
        return pressures;
    }
};

/**
 * External force generators (gravity, wind, etc.)
 */
class ExternalForces
{
  public:
    // Environmental forces
    static void applyGravity(ParticleSystem *system, glm::vec3 gravity);
    static void applyWind(ParticleSystem *system, glm::vec3 wind_velocity, float strength);
    static void applyVortex(ParticleSystem *system, glm::vec3 center, float strength);

    // User interaction forces
    static void applyPointAttraction(ParticleSystem *system, glm::vec3 point, float strength, float max_distance);
    static void applyPointRepulsion(ParticleSystem *system, glm::vec3 point, float strength, float max_distance);
};

} // namespace FluidSim
