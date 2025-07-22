#include "../utils/cuda_helper.hh"
#include "cuda_kernels.hh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace FluidSim
{

// CUDA device memory pointers
static float3 *d_positions = nullptr;
static float3 *d_velocities = nullptr;
static float3 *d_accelerations = nullptr;
static float *d_densities = nullptr;
static float *d_pressures = nullptr;
static float *d_masses = nullptr;
static int *d_neighbor_indices = nullptr;
static int *d_neighbor_counts = nullptr;
static int max_allocated_particles = 0;

// Simulation parameters on device
__constant__ PhysicsParams d_params;

// CUDA kernel implementations
__device__ float kernelPoly6(float r, float h)
{
    if (r > h || r < 0)
        return 0.0f;

    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float diff = h2 - r * r;

    return (315.0f / (64.0f * M_PI * h9)) * diff * diff * diff;
}

__device__ float3 kernelSpikyGradient(float3 r_vec, float h)
{
    float r = sqrtf(r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z);
    if (r > h || r <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);

    float h6 = h * h * h * h * h * h;
    float diff = h - r;
    float coefficient = -45.0f / (M_PI * h6) * diff * diff;

    return make_float3(coefficient * r_vec.x / r, coefficient * r_vec.y / r, coefficient * r_vec.z / r);
}

__device__ float kernelViscosityLaplacian(float r, float h)
{
    if (r > h || r < 0)
        return 0.0f;

    float h6 = h * h * h * h * h * h;
    return (45.0f / (M_PI * h6)) * (h - r);
}

// Simple neighbor finding kernel (O(n²) - can be optimized with spatial grid)
__global__ void findNeighborsKernel(float3 *positions, int *neighbor_indices, int *neighbor_counts, int particle_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count)
        return;

    float3 pos_i = positions[idx];
    int neighbor_count = 0;
    int neighbor_start = idx * d_params.max_neighbors;

    for (int j = 0; j < particle_count; j++)
    {
        if (j == idx)
            continue;

        float3 diff = make_float3(pos_i.x - positions[j].x, pos_i.y - positions[j].y, pos_i.z - positions[j].z);
        float distance_squared = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

        if (distance_squared < d_params.smoothing_radius * d_params.smoothing_radius)
        {
            if (neighbor_count < d_params.max_neighbors)
            {
                neighbor_indices[neighbor_start + neighbor_count] = j;
                neighbor_count++;
            }
        }
    }

    neighbor_counts[idx] = neighbor_count;
}

// Density calculation kernel
__global__ void calculateDensityKernel(float3 *positions, float *densities, float *masses, int *neighbor_indices,
                                       int *neighbor_counts, int particle_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count)
        return;

    float density = 0.0f;
    float3 pos_i = positions[idx];

    // Self contribution
    density += masses[idx] * kernelPoly6(0.0f, d_params.smoothing_radius);

    // Neighbor contributions
    int neighbor_start = idx * d_params.max_neighbors;
    int neighbor_count = neighbor_counts[idx];

    for (int i = 0; i < neighbor_count; i++)
    {
        int j = neighbor_indices[neighbor_start + i];
        if (j >= 0 && j < particle_count)
        {
            float3 diff = make_float3(pos_i.x - positions[j].x, pos_i.y - positions[j].y, pos_i.z - positions[j].z);
            float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

            density += masses[j] * kernelPoly6(distance, d_params.smoothing_radius);
        }
    }

    densities[idx] = density;
}

// Pressure calculation kernel
__global__ void calculatePressureKernel(float *densities, float *pressures, int particle_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count)
        return;

    float pressure = d_params.gas_constant * (densities[idx] - d_params.rest_density);
    pressures[idx] = fmaxf(pressure, 0.0f); // Prevent negative pressure
}

// Force calculation kernel
__global__ void calculateForcesKernel(float3 *positions, float3 *velocities, float3 *accelerations, float *densities,
                                      float *pressures, float *masses, int *neighbor_indices, int *neighbor_counts,
                                      int particle_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count)
        return;

    float3 pos_i = positions[idx];
    float3 vel_i = velocities[idx];
    float density_i = densities[idx];
    float pressure_i = pressures[idx];
    float mass_i = masses[idx];

    float3 pressure_force = make_float3(0.0f, 0.0f, 0.0f);
    float3 viscosity_force = make_float3(0.0f, 0.0f, 0.0f);

    // Calculate forces from neighbors
    int neighbor_start = idx * d_params.max_neighbors;
    int neighbor_count = neighbor_counts[idx];

    for (int i = 0; i < neighbor_count; i++)
    {
        int j = neighbor_indices[neighbor_start + i];
        if (j >= 0 && j < particle_count)
        {
            float3 pos_j = positions[j];
            float3 vel_j = velocities[j];
            float density_j = densities[j];
            float pressure_j = pressures[j];
            float mass_j = masses[j];

            float3 r_ij = make_float3(pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z);
            float distance = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

            if (distance > 0.0f && distance < d_params.smoothing_radius)
            {
                // Pressure force
                float pressure_term = (pressure_i + pressure_j) / (2.0f * density_j);
                float3 gradient = kernelSpikyGradient(r_ij, d_params.smoothing_radius);
                pressure_force = make_float3(pressure_force.x - mass_j * pressure_term * gradient.x,
                                             pressure_force.y - mass_j * pressure_term * gradient.y,
                                             pressure_force.z - mass_j * pressure_term * gradient.z);

                // Viscosity force
                float3 vel_diff = make_float3(vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z - vel_i.z);
                float laplacian = kernelViscosityLaplacian(distance, d_params.smoothing_radius);
                viscosity_force = make_float3(viscosity_force.x + mass_j * (vel_diff.x / density_j) * laplacian,
                                              viscosity_force.y + mass_j * (vel_diff.y / density_j) * laplacian,
                                              viscosity_force.z + mass_j * (vel_diff.z / density_j) * laplacian);
            }
        }
    }

    // Combine forces
    float3 total_force = make_float3(
        mass_i * pressure_force.x + d_params.viscosity * mass_i * viscosity_force.x + mass_i * d_params.gravity.x,
        mass_i * pressure_force.y + d_params.viscosity * mass_i * viscosity_force.y + mass_i * d_params.gravity.y,
        mass_i * pressure_force.z + d_params.viscosity * mass_i * viscosity_force.z + mass_i * d_params.gravity.z);

    accelerations[idx] = make_float3(total_force.x / mass_i, total_force.y / mass_i, total_force.z / mass_i);
}

// Motion integration kernel
__global__ void integrateMotionKernel(float3 *positions, float3 *velocities, float3 *accelerations, int particle_count,
                                      float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count)
        return;

    // Euler integration
    velocities[idx] =
        make_float3(velocities[idx].x + accelerations[idx].x * dt, velocities[idx].y + accelerations[idx].y * dt,
                    velocities[idx].z + accelerations[idx].z * dt);

    velocities[idx] =
        make_float3(velocities[idx].x * d_params.damping_factor, velocities[idx].y * d_params.damping_factor,
                    velocities[idx].z * d_params.damping_factor);

    positions[idx] = make_float3(positions[idx].x + velocities[idx].x * dt, positions[idx].y + velocities[idx].y * dt,
                                 positions[idx].z + velocities[idx].z * dt);

    // Reset acceleration
    accelerations[idx] = make_float3(0.0f, 0.0f, 0.0f);
}

// Boundary collision kernel
__global__ void handleBoundaryCollisionsKernel(float3 *positions, float3 *velocities, int particle_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count)
        return;

    float3 pos = positions[idx];
    float3 vel = velocities[idx];
    float restitution = d_params.restitution;

    // X boundaries
    if (pos.x < d_params.boundary_min.x)
    {
        pos.x = d_params.boundary_min.x;
        if (vel.x < 0)
            vel.x = -vel.x * restitution;
    }
    else if (pos.x > d_params.boundary_max.x)
    {
        pos.x = d_params.boundary_max.x;
        if (vel.x > 0)
            vel.x = -vel.x * restitution;
    }

    // Y boundaries
    if (pos.y < d_params.boundary_min.y)
    {
        pos.y = d_params.boundary_min.y;
        if (vel.y < 0)
            vel.y = -vel.y * restitution;
    }
    else if (pos.y > d_params.boundary_max.y)
    {
        pos.y = d_params.boundary_max.y;
        if (vel.y > 0)
            vel.y = -vel.y * restitution;
    }

    // Z boundaries
    if (pos.z < d_params.boundary_min.z)
    {
        pos.z = d_params.boundary_min.z;
        if (vel.z < 0)
            vel.z = -vel.z * restitution;
    }
    else if (pos.z > d_params.boundary_max.z)
    {
        pos.z = d_params.boundary_max.z;
        if (vel.z > 0)
            vel.z = -vel.z * restitution;
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}

// Host functions
bool initializeCudaPhysics(int max_particles)
{
    if (max_allocated_particles >= max_particles)
    {
        return true; // Already allocated enough memory
    }

    // Free existing memory
    freeParticleBuffers();

    // Allocate device memory
    cudaError_t error;
    // Neighbor list arrays
    int max_neighbors = 64;

    error = cudaMalloc(&d_positions, max_particles * sizeof(float3));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_velocities, max_particles * sizeof(float3));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_accelerations, max_particles * sizeof(float3));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_densities, max_particles * sizeof(float));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_pressures, max_particles * sizeof(float));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_masses, max_particles * sizeof(float));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_neighbor_indices, max_particles * max_neighbors * sizeof(int));
    if (error != cudaSuccess)
        goto error_cleanup;

    error = cudaMalloc(&d_neighbor_counts, max_particles * sizeof(int));
    if (error != cudaSuccess)
        goto error_cleanup;

    max_allocated_particles = max_particles;
    std::cout << "CUDA physics memory allocated for " << max_particles << " particles" << std::endl;
    return true;

error_cleanup:
    std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(error) << std::endl;
    freeParticleBuffers();
    return false;
}

void cleanupCudaPhysics()
{
    freeParticleBuffers();
}

void freeParticleBuffers()
{
    if (d_positions)
    {
        cudaFree(d_positions);
        d_positions = nullptr;
    }
    if (d_velocities)
    {
        cudaFree(d_velocities);
        d_velocities = nullptr;
    }
    if (d_accelerations)
    {
        cudaFree(d_accelerations);
        d_accelerations = nullptr;
    }
    if (d_densities)
    {
        cudaFree(d_densities);
        d_densities = nullptr;
    }
    if (d_pressures)
    {
        cudaFree(d_pressures);
        d_pressures = nullptr;
    }
    if (d_masses)
    {
        cudaFree(d_masses);
        d_masses = nullptr;
    }
    if (d_neighbor_indices)
    {
        cudaFree(d_neighbor_indices);
        d_neighbor_indices = nullptr;
    }
    if (d_neighbor_counts)
    {
        cudaFree(d_neighbor_counts);
        d_neighbor_counts = nullptr;
    }

    max_allocated_particles = 0;
}

void updatePhysicsParams(const PhysicsParams &params)
{
    cudaMemcpyToSymbol(d_params, &params, sizeof(PhysicsParams));
}

bool copyParticlesToDevice(const float *positions, const float *velocities, const float *masses, int count)
{
    if (!d_positions || count > max_allocated_particles)
        return false;

    // Copy position data (convert from flat array to float3)
    cudaError_t error = cudaMemcpy(d_positions, positions, count * sizeof(float3), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return false;

    // Copy velocity data
    error = cudaMemcpy(d_velocities, velocities, count * sizeof(float3), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return false;

    // Copy mass data
    error = cudaMemcpy(d_masses, masses, count * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return false;

    return true;
}

bool copyParticlesFromDevice(float *positions, float *velocities, int count)
{
    if (!d_positions || count > max_allocated_particles)
        return false;

    // Copy position data back
    cudaError_t error = cudaMemcpy(positions, d_positions, count * sizeof(float3), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        return false;

    // Copy velocity data back
    error = cudaMemcpy(velocities, d_velocities, count * sizeof(float3), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        return false;

    return true;
}

void runPhysicsSimulationCUDA(int particle_count, float dt)
{
    if (particle_count == 0 || !d_positions)
        return;

    int block_size = 256;
    int grid_size = (particle_count + block_size - 1) / block_size;

    // 1. Find neighbors
    findNeighborsKernel<<<grid_size, block_size>>>(d_positions, d_neighbor_indices, d_neighbor_counts, particle_count);

    // 2. Calculate densities
    calculateDensityKernel<<<grid_size, block_size>>>(d_positions, d_densities, d_masses, d_neighbor_indices,
                                                      d_neighbor_counts, particle_count);

    // 3. Calculate pressures
    calculatePressureKernel<<<grid_size, block_size>>>(d_densities, d_pressures, particle_count);

    // 4. Calculate forces
    calculateForcesKernel<<<grid_size, block_size>>>(d_positions, d_velocities, d_accelerations, d_densities,
                                                     d_pressures, d_masses, d_neighbor_indices, d_neighbor_counts,
                                                     particle_count);

    // 5. Integrate motion
    integrateMotionKernel<<<grid_size, block_size>>>(d_positions, d_velocities, d_accelerations, particle_count, dt);

    // 6. Handle boundary collisions
    handleBoundaryCollisionsKernel<<<grid_size, block_size>>>(d_positions, d_velocities, particle_count);

    // Synchronize to ensure all kernels complete
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << std::endl;
    }
}

} // namespace FluidSim
