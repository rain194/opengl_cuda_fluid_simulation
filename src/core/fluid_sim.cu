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
    // Just call CPU physics directly without the fallback message spam
    updatePhysics(dt);
}

} // namespace FluidSim
