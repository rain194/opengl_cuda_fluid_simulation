#pragma once
#include "../utils/config.hh"
#include "particle.hh"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace FluidSim
{

/**
 * Main fluid simulation class using SPH (Smoothed Particle Hydrodynamics)
 * Handles physics calculations and particle interactions
 */
class FluidSimulator
{
  private:
    std::unique_ptr<ParticleSystem> particle_system;

    // Timing
    float time_step;
    float current_time;
    int frame_count;

    // Physics parameters
    glm::vec3 gravity;
    float damping_factor;
    bool use_cuda;

    // SPH calculation buffers (for performance)
    std::vector<std::vector<size_t>> neighbor_lists;
    std::vector<float> densities;
    std::vector<glm::vec3> pressure_forces;
    std::vector<glm::vec3> viscosity_forces;

  public:
    FluidSimulator(bool enable_cuda = true);
    ~FluidSimulator();

    // Simulation control
    void initialize(const SimulationConfig &config);
    void update(float delta_time);
    void reset();
    void pause();
    void resume();

    // Particle system access
    ParticleSystem *getParticleSystem()
    {
        return particle_system.get();
    }
    const ParticleSystem *getParticleSystem() const
    {
        return particle_system.get();
    }

    // Simulation state
    bool isRunning() const
    {
        return is_running;
    }
    bool isCudaEnabled() const
    {
        return use_cuda;
    }
    float getCurrentTime() const
    {
        return current_time;
    }
    int getFrameCount() const
    {
        return frame_count;
    }

    // Physics parameters
    void setGravity(glm::vec3 g)
    {
        gravity = g;
    }
    void setTimeStep(float dt)
    {
        time_step = dt;
    }
    void setDamping(float damping)
    {
        damping_factor = damping;
    }

    // Statistics
    float getAverageFrameTime() const;
    size_t getActiveParticleCount() const;

  private:
    bool is_running;
    bool is_paused;

    // Core SPH physics methods
    void updatePhysics(float dt);
    void calculateDensities();
    void calculatePressures();
    void calculateForces();
    void integrateMotion(float dt);
    void handleBoundaryCollisions();

    // SPH helper methods
    void findNeighbors();
    float kernelPoly6(float r, float h);
    glm::vec3 kernelSpikyGradient(glm::vec3 r, float h);
    float kernelViscosityLaplacian(float r, float h);

    // CUDA methods (implemented in fluid_sim.cu)
    void updatePhysicsCUDA(float dt);
    void initializeCUDA();
    void cleanupCUDA();
};

} // namespace FluidSim

