#include "config.hh"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace FluidSim
{

// Static member definitions
SimulationConfig ConfigManager::current_config;
RuntimeConstants ConfigManager::runtime_constants;
bool ConfigManager::initialized = false;

// SimulationConfig implementation
bool SimulationConfig::loadFromFile(const std::string &filename)
{
    // For now, just return false - file loading not implemented yet
    std::cout << "File loading not implemented yet: " << filename << std::endl;
    return false;
}

bool SimulationConfig::saveToFile(const std::string &filename) const
{
    // For now, just return false - file saving not implemented yet
    std::cout << "File saving not implemented yet: " << filename << std::endl;
    return false;
}

void SimulationConfig::printConfiguration() const
{
    std::cout << "=== Configuration Summary ===" << std::endl;
    std::cout << "Max particles: " << max_particles << std::endl;
    std::cout << "Time step: " << time_step << std::endl;
    std::cout << "Use CUDA: " << (use_cuda ? "Yes" : "No") << std::endl;
    std::cout << "Window: " << window_width << "x" << window_height << std::endl;
}

// ConfigManager implementation
bool ConfigManager::initialize(const SimulationConfig &config)
{
    current_config = config;
    runtime_constants.updateFromConfig(config);
    initialized = true;
    return true;
}

bool ConfigManager::loadConfiguration(const std::string &filename)
{
    SimulationConfig config;
    if (config.loadFromFile(filename))
    {
        return initialize(config);
    }
    return false;
}

bool ConfigManager::saveConfiguration(const std::string &filename)
{
    return current_config.saveToFile(filename);
}

void ConfigManager::updateRuntimeConstants()
{
    runtime_constants.updateFromConfig(current_config);
}

void ConfigManager::resetRuntimeStats()
{
    runtime_constants.reset();
}

void ConfigManager::applyPreset(const std::string &preset_name)
{
    if (preset_name == "performance")
    {
        current_config = SimulationConfig::getHighPerformanceConfig();
    }
    else if (preset_name == "quality")
    {
        current_config = SimulationConfig::getHighQualityConfig();
    }
    else if (preset_name == "debug")
    {
        current_config = SimulationConfig::getDebugConfig();
    }
    else
    {
        current_config = SimulationConfig::getDefaultConfig();
    }
    updateRuntimeConstants();
}

void ConfigManager::createCustomPreset(const std::string &name, const SimulationConfig &config)
{
    // Not implemented yet
    std::cout << "Custom preset creation not implemented: " << name << std::endl;
}

bool ConfigManager::validateCurrentConfig()
{
    return current_config.validate();
}

void ConfigManager::printConfigurationSummary()
{
    current_config.printConfiguration();
}

void ConfigManager::printPerformanceStats()
{
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1)
              << (1000.0f / runtime_constants.average_frame_time) << std::endl;
    std::cout << "Physics time: " << runtime_constants.physics_time_ms << "ms" << std::endl;
    std::cout << "Rendering time: " << runtime_constants.rendering_time_ms << "ms" << std::endl;
    std::cout << "Total frames: " << runtime_constants.total_frames << std::endl;
    std::cout << "===============================" << std::endl;
}

void ConfigManager::updatePerformanceMetrics(float physics_time, float rendering_time, float total_time)
{
    runtime_constants.physics_time_ms = physics_time;
    runtime_constants.rendering_time_ms = rendering_time;
    runtime_constants.total_frame_time_ms = total_time;
}

void ConfigManager::updateFrameStats(float frame_time)
{
    runtime_constants.total_frames++;
    runtime_constants.accumulated_time += frame_time / 1000.0f;
    runtime_constants.average_frame_time = frame_time;
    runtime_constants.total_frame_time_ms = frame_time;
}

std::string ConfigManager::getConfigDirectory()
{
    return "configs/";
}

std::string ConfigManager::getDefaultConfigPath()
{
    return getConfigDirectory() + "default.conf";
}

} // namespace FluidSim
