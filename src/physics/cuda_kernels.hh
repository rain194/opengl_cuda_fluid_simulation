#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace FluidSim
{

// Physics parameters structure for constant memory
struct PhysicsParams
{
    float3 gravity;
    float3 boundary_min;
    float3 boundary_max;
    float smoothing_radius;
    float rest_density;
    float gas_constant;
    float viscosity;
    float damping_factor;
    float restitution;
    float time_step;
    int max_neighbors;
    int particle_count;
};

// Host function declarations
bool initializeCudaPhysics(int max_particles);
void cleanupCudaPhysics();
void updatePhysicsParams(const PhysicsParams &params);
void runPhysicsSimulationCUDA(int particle_count, float dt);

// Memory management functions
bool allocateParticleBuffers(int max_particles);
void freeParticleBuffers();
bool copyParticlesToDevice(const float *positions, const float *velocities, const float *masses, int count);
bool copyParticlesFromDevice(float *positions, float *velocities, int count);

} // namespace FluidSim
