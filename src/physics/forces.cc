#define GLM_ENABLE_EXPERIMENTAL
#include "forces.hh"
#include "../core/particle.hh"
#include "../utils/math_utils.hh"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace FluidSim
{

ForceCalculator::ForceCalculator()
    : smoothing_radius(0.1f), rest_density(1000.0f), gas_constant(200.0f), viscosity_coefficient(0.001f),
      surface_tension_coefficient(0.0728f)
{
}

void ForceCalculator::initialize(float h, float rho0, float k, float mu, float sigma)
{
    smoothing_radius = h;
    rest_density = rho0;
    gas_constant = k;
    viscosity_coefficient = mu;
    surface_tension_coefficient = sigma;

    std::cout << "ForceCalculator initialized:\n";
    std::cout << "  Smoothing radius: " << h << "m\n";
    std::cout << "  Rest density: " << rho0 << " kg/m³\n";
    std::cout << "  Gas constant: " << k << "\n";
    std::cout << "  Viscosity: " << mu << " Pa·s\n";
}

void ForceCalculator::calculateAllForces(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists)
{
    if (!system)
        return;

    size_t particle_count = system->getActiveCount();
    if (particle_count == 0)
        return;

    // Resize buffers if needed
    resizeBuffers(particle_count);

    // Clear previous forces
    clearForces(system);

    // Calculate in order: density -> pressure -> forces
    calculateDensities(system, neighbor_lists);
    calculatePressureForces(system, neighbor_lists);
    calculateViscosityForces(system, neighbor_lists);

    // Apply forces to particles
    auto &particles = system->getParticles();
    for (size_t i = 0; i < particle_count; i++)
    {
        particles[i].acceleration += pressure_forces[i] / particles[i].mass;
        particles[i].acceleration += viscosity_forces[i] / particles[i].mass;
    }
}

void ForceCalculator::calculateDensities(ParticleSystem *system, const std::vector<std::vector<size_t>> &neighbor_lists)
{
    auto &particles = system->getParticles();
    size_t particle_count = system->getActiveCount();

    // SPH density calculation: ρᵢ = Σⱼ mⱼ * W(|rᵢ - rⱼ|, h)
    for (size_t i = 0; i < particle_count; i++)
    {
        float density = 0.0f;

        // Self contribution (particle contributes to its own density)
        density += particles[i].mass * kernelPoly6(0.0f, smoothing_radius);

        // Neighbor contributions
        for (size_t j : neighbor_lists[i])
        {
            glm::vec3 r_ij = particles[i].position - particles[j].position;
            float distance = glm::length(r_ij);

            density += particles[j].mass * kernelPoly6(distance, smoothing_radius);
        }

        densities[i] = density;
        particles[i].density = density;

        // Calculate pressure from density using ideal gas law
        // P = k * (ρ - ρ₀) where k is gas constant, ρ₀ is rest density
        pressures[i] = gas_constant * (density - rest_density);
        particles[i].pressure = std::max(pressures[i], 0.0f); // Prevent negative pressure
    }
}

void ForceCalculator::calculatePressureForces(ParticleSystem *system,
                                              const std::vector<std::vector<size_t>> &neighbor_lists)
{
    auto &particles = system->getParticles();
    size_t particle_count = system->getActiveCount();

    // SPH pressure force: F_pressure = -mᵢ * Σⱼ mⱼ * (Pᵢ + Pⱼ)/(2 * ρⱼ) * ∇W(rᵢⱼ, h)
    for (size_t i = 0; i < particle_count; i++)
    {
        glm::vec3 pressure_force(0.0f);

        for (size_t j : neighbor_lists[i])
        {
            glm::vec3 r_ij = particles[i].position - particles[j].position;
            float distance = glm::length(r_ij);

            if (distance > 0.0f && distance < smoothing_radius)
            {
                // Symmetric pressure force formulation
                float pressure_term = (pressures[i] + pressures[j]) / (2.0f * densities[j]);

                glm::vec3 gradient = kernelSpikyGradient(r_ij, smoothing_radius);
                pressure_force -= particles[j].mass * pressure_term * gradient;
            }
        }

        pressure_forces[i] = particles[i].mass * pressure_force;
    }
}

void ForceCalculator::calculateViscosityForces(ParticleSystem *system,
                                               const std::vector<std::vector<size_t>> &neighbor_lists)
{
    auto &particles = system->getParticles();
    size_t particle_count = system->getActiveCount();

    // SPH viscosity force: F_viscosity = μ * mᵢ * Σⱼ mⱼ * (vⱼ - vᵢ)/ρⱼ * ∇²W(rᵢⱼ, h)
    for (size_t i = 0; i < particle_count; i++)
    {
        glm::vec3 viscosity_force(0.0f);

        for (size_t j : neighbor_lists[i])
        {
            glm::vec3 r_ij = particles[i].position - particles[j].position;
            float distance = glm::length(r_ij);

            if (distance > 0.0f && distance < smoothing_radius)
            {
                glm::vec3 velocity_diff = particles[j].velocity - particles[i].velocity;
                float laplacian = kernelViscosityLaplacian(distance, smoothing_radius);

                viscosity_force += particles[j].mass * (velocity_diff / densities[j]) * laplacian;
            }
        }

        viscosity_forces[i] = viscosity_coefficient * particles[i].mass * viscosity_force;
    }
}

void ForceCalculator::calculateSurfaceTensionForces(ParticleSystem *system,
                                                    const std::vector<std::vector<size_t>> &neighbor_lists)
{
    // TODO: Implement surface tension forces
    // This is more advanced and can be added later
    auto &particles = system->getParticles();
    size_t particle_count = system->getActiveCount();

    for (size_t i = 0; i < particle_count; i++)
    {
        surface_forces[i] = glm::vec3(0.0f); // No surface tension for now
    }
}

// SPH Kernel Functions Implementation
float ForceCalculator::kernelPoly6(float r, float h)
{
    if (r > h || r < 0)
        return 0.0f;

    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h; // h^9
    float diff = h2 - r * r;

    // Poly6 kernel: W(r,h) = (315/(64π*h^9)) * (h² - r²)³
    return (315.0f / (64.0f * 3.14159265358979323846 * h9)) * diff * diff * diff;
}

glm::vec3 ForceCalculator::kernelSpikyGradient(glm::vec3 r_vec, float h)
{
    float r = glm::length(r_vec);
    if (r > h || r <= 0.0f)
        return glm::vec3(0.0f);

    float h6 = h * h * h * h * h * h; // h^6
    float diff = h - r;

    // Spiky gradient: ∇W(r,h) = (-45/(π*h^6)) * (h-r)² * (r/|r|)
    float coefficient = -45.0f / (3.14159265358979323846 * h6) * diff * diff;
    return coefficient * (r_vec / r);
}

float ForceCalculator::kernelViscosityLaplacian(float r, float h)
{
    if (r > h || r < 0)
        return 0.0f;

    float h6 = h * h * h * h * h * h; // h^6

    // Viscosity laplacian: ∇²W(r,h) = (45/(π*h^6)) * (h-r)
    return (45.0f / (3.14159265358979323846 * h6)) * (h - r);
}

// External Forces Implementation
glm::vec3 ForceCalculator::calculateGravityForce(float mass, glm::vec3 gravity)
{
    return mass * gravity; // F = mg
}

glm::vec3 ForceCalculator::calculateBuoyancyForce(float density, float rest_density, glm::vec3 gravity,
                                                  float buoyancy_coeff)
{
    // Buoyant force opposes gravity based on density difference
    float density_ratio = (rest_density - density) / rest_density;
    return buoyancy_coeff * density_ratio * (-gravity); // Opposite to gravity
}

void ForceCalculator::resizeBuffers(size_t particle_count)
{
    densities.resize(particle_count);
    pressures.resize(particle_count);
    pressure_forces.resize(particle_count);
    viscosity_forces.resize(particle_count);
    surface_forces.resize(particle_count);
}

void ForceCalculator::clearForces(ParticleSystem *system)
{
    auto &particles = system->getParticles();
    size_t particle_count = system->getActiveCount();

    for (size_t i = 0; i < particle_count; i++)
    {
        particles[i].acceleration = glm::vec3(0.0f);
        pressure_forces[i] = glm::vec3(0.0f);
        viscosity_forces[i] = glm::vec3(0.0f);
        surface_forces[i] = glm::vec3(0.0f);
    }
}

// External Forces Implementation
void ExternalForces::applyGravity(ParticleSystem *system, glm::vec3 gravity)
{
    auto &particles = system->getParticles();

    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        particles[i].acceleration += gravity;
    }
}

void ExternalForces::applyWind(ParticleSystem *system, glm::vec3 wind_velocity, float strength)
{
    auto &particles = system->getParticles();

    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        glm::vec3 relative_velocity = wind_velocity - particles[i].velocity;
        glm::vec3 wind_force = strength * relative_velocity;
        particles[i].acceleration += wind_force / particles[i].mass;
    }
}

void ExternalForces::applyPointAttraction(ParticleSystem *system, glm::vec3 point, float strength, float max_distance)
{
    auto &particles = system->getParticles();

    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        glm::vec3 direction = point - particles[i].position;
        float distance = glm::length(direction);

        if (distance > 0.0f && distance < max_distance)
        {
            // F = k * m / r² (inverse square law)
            direction /= distance; // normalize
            float force_magnitude = strength * particles[i].mass / (distance * distance);

            particles[i].acceleration += direction * force_magnitude / particles[i].mass;
        }
    }
}

void ExternalForces::applyPointRepulsion(ParticleSystem *system, glm::vec3 point, float strength, float max_distance)
{
    // Same as attraction but with negative strength
    applyPointAttraction(system, point, -strength, max_distance);
}

} // namespace FluidSim
