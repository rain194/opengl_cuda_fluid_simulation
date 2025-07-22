#include "../physics/cuda_kernels.hh"
#include "../utils/cuda_helper.hh"
#include "fluid_sim.hh"
#include <cuda_runtime.h>
#include <iostream>

namespace FluidSim
{

void FluidSimulator::initializeCUDA()
{
    if (!use_cuda)
        return;

    // Check CUDA availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found, falling back to CPU" << std::endl;
        use_cuda = false;
        return;
    }

    // Set device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA initialized on: " << prop.name << std::endl;

    // Initialize CUDA physics system
    if (!initializeCudaPhysics(static_cast<int>(particle_system->getMaxParticles())))
    {
        std::cerr << "Failed to initialize CUDA physics, falling back to CPU" << std::endl;
        use_cuda = false;
        return;
    }

    // Setup physics parameters
    PhysicsParams params;
    params.gravity = make_float3(gravity.x, gravity.y, gravity.z);
    params.boundary_min =
        make_float3(particle_system->boundary_min.x, particle_system->boundary_min.y, particle_system->boundary_min.z);
    params.boundary_max =
        make_float3(particle_system->boundary_max.x, particle_system->boundary_max.y, particle_system->boundary_max.z);
    params.smoothing_radius = particle_system->smoothing_radius;
    params.rest_density = particle_system->rest_density;
    params.gas_constant = particle_system->gas_constant;
    params.viscosity = particle_system->viscosity;
    params.damping_factor = damping_factor;
    params.restitution = 0.5f; // Default restitution
    params.time_step = time_step;
    params.max_neighbors = 64;
    params.particle_count = static_cast<int>(particle_system->getActiveCount());

    updatePhysicsParams(params);

    std::cout << "CUDA physics system initialized successfully" << std::endl;
}

void FluidSimulator::cleanupCUDA()
{
    if (use_cuda)
    {
        cleanupCudaPhysics();
        std::cout << "CUDA physics system cleaned up" << std::endl;
    }
}

void FluidSimulator::updatePhysicsCUDA(float dt)
{
    if (!use_cuda)
    {
        // Fallback to CPU implementation
        updatePhysics(dt);
        return;
    }

    auto &particles = particle_system->getParticles();
    size_t particle_count = particle_system->getActiveCount();

    if (particle_count == 0)
        return;

    // Prepare data for GPU
    std::vector<float> positions(particle_count * 3);
    std::vector<float> velocities(particle_count * 3);
    std::vector<float> masses(particle_count);

    // Copy particle data to arrays
    for (size_t i = 0; i < particle_count; i++)
    {
        positions[i * 3 + 0] = particles[i].position.x;
        positions[i * 3 + 1] = particles[i].position.y;
        positions[i * 3 + 2] = particles[i].position.z;

        velocities[i * 3 + 0] = particles[i].velocity.x;
        velocities[i * 3 + 1] = particles[i].velocity.y;
        velocities[i * 3 + 2] = particles[i].velocity.z;

        masses[i] = particles[i].mass;
    }

    // Copy to GPU
    if (!copyParticlesToDevice(positions.data(), velocities.data(), masses.data(), static_cast<int>(particle_count)))
    {
        std::cerr << "Failed to copy particles to GPU, using CPU fallback" << std::endl;
        updatePhysics(dt);
        return;
    }

    // Update physics parameters if needed
    PhysicsParams params;
    params.gravity = make_float3(gravity.x, gravity.y, gravity.z);
    params.boundary_min =
        make_float3(particle_system->boundary_min.x, particle_system->boundary_min.y, particle_system->boundary_min.z);
    params.boundary_max =
        make_float3(particle_system->boundary_max.x, particle_system->boundary_max.y, particle_system->boundary_max.z);
    params.smoothing_radius = particle_system->smoothing_radius;
    params.rest_density = particle_system->rest_density;
    params.gas_constant = particle_system->gas_constant;
    params.viscosity = particle_system->viscosity;
    params.damping_factor = damping_factor;
    params.time_step = dt;
    params.particle_count = static_cast<int>(particle_count);

    updatePhysicsParams(params);

    // Run CUDA simulation
    runPhysicsSimulationCUDA(static_cast<int>(particle_count), dt);

    // Copy results back to CPU
    if (!copyParticlesFromDevice(positions.data(), velocities.data(), static_cast<int>(particle_count)))
    {
        std::cerr << "Failed to copy particles from GPU" << std::endl;
        return;
    }

    // Update particle data
    for (size_t i = 0; i < particle_count; i++)
    {
        particles[i].position.x = positions[i * 3 + 0];
        particles[i].position.y = positions[i * 3 + 1];
        particles[i].position.z = positions[i * 3 + 2];

        particles[i].velocity.x = velocities[i * 3 + 0];
        particles[i].velocity.y = velocities[i * 3 + 1];
        particles[i].velocity.z = velocities[i * 3 + 2];
    }
}

} // namespace FluidSim
