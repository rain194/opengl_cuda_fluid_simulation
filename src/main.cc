#define GLM_ENABLE_EXPERIMENTAL
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

// OpenGL and window management
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Our simulation components
#include "core/fluid_sim.hh"
#include "physics/collision.hh"
#include "physics/forces.hh"
#include "rendering/renderer.hh"
#include "utils/config.hh"
#include "utils/cuda_helper.hh"
#include "utils/math_utils.hh"

// Namespace usage
using namespace FluidSim;

// === GLOBAL APPLICATION STATE ===
class FluidSimulationApp
{
  private:
    // Core systems
    GLFWwindow *window;
    std::unique_ptr<FluidSimulator> simulator;
    std::unique_ptr<Renderer> renderer;

    // Application state
    bool initialized;
    bool running;
    bool paused;
    bool show_debug_info;

    // Timing and performance
    MathUtils::Timer frame_timer;
    MathUtils::Timer physics_timer;
    MathUtils::Timer render_timer;
    MathUtils::MovingAverage<float> fps_average;

    // Statistics
    int frame_count;
    float total_runtime;
    float last_stats_print;

  public:
    FluidSimulationApp()
        : window(nullptr), initialized(false), running(false), paused(false), show_debug_info(true), fps_average(60),
          frame_count(0), total_runtime(0.0f), last_stats_print(0.0f)
    {
    }

    ~FluidSimulationApp()
    {
        cleanup();
    }

    // === INITIALIZATION ===
    bool initialize(const std::string &config_preset = "default")
    {
        printHeader();

        // Step 1: Load configuration
        if (!loadConfiguration(config_preset))
        {
            return false;
        }

        // Step 2: Initialize systems
        if (!initializeWindow())
            return false;
        if (!initializeGraphics())
            return false;
        if (!initializeCUDA())
            return false;
        if (!initializeSimulation())
            return false;
        if (!initializeRenderer())
            return false;

        // Step 3: Setup callbacks and initial state
        setupCallbacks();
        createInitialScene();

        initialized = true;
        running = true;

        printInitializationSummary();
        return true;
    }

    // === MAIN LOOP ===
    void run()
    {
        if (!initialized)
        {
            std::cerr << "❌ Application not initialized!" << std::endl;
            return;
        }

        std::cout << "\n🚀 Starting simulation loop...\n" << std::endl;

        while (running && !glfwWindowShouldClose(window))
        {
            std::cout << "=== MAIN LOOP ITERATION ===" << std::endl;

            frame_timer.start();

            // Process input
            glfwPollEvents();
            processInput();
            std::cout << "Input processed" << std::endl;

            // Update simulation
            updateSimulation();
            std::cout << "Simulation updated" << std::endl;

            // Render frame
            renderFrame();
            std::cout << "Frame rendered" << std::endl;

            // Update timing and statistics
            updateStatistics();

            // Print periodic status
            printPeriodicStatus();

            frame_count++;

            std::cout << "Frame " << frame_count << " completed" << std::endl;
        }

        printShutdownSummary();
    }

    // === CONFIGURATION MANAGEMENT ===
    bool loadConfiguration(const std::string &preset)
    {
        std::cout << "📋 Loading configuration preset: '" << preset << "'" << std::endl;

        SimulationConfig config;

        if (preset == "default")
        {
            config = SimulationConfig::getDefaultConfig();
        }
        else if (preset == "performance")
        {
            config = SimulationConfig::getHighPerformanceConfig();
        }
        else if (preset == "quality")
        {
            config = SimulationConfig::getHighQualityConfig();
        }
        else if (preset == "debug")
        {
            config = SimulationConfig::getDebugConfig();
        }
        else
        {
            // Try to load from file
            if (!config.loadFromFile("configs/" + preset + ".conf"))
            {
                std::cerr << "❌ Failed to load configuration: " << preset << std::endl;
                std::cout << "   Using default configuration instead." << std::endl;
                config = SimulationConfig::getDefaultConfig();
            }
        }

        if (!config.validate())
        {
            std::cerr << "❌ Configuration validation failed!" << std::endl;
            return false;
        }

        if (!ConfigManager::initialize(config))
        {
            std::cerr << "❌ Failed to initialize configuration manager!" << std::endl;
            return false;
        }

        std::cout << "✅ Configuration loaded successfully" << std::endl;
        return true;
    }

  private:
    // === SYSTEM INITIALIZATION ===
    bool initializeWindow()
    {
        std::cout << "🪟 Initializing window system..." << std::endl;

        if (!glfwInit())
        {
            std::cerr << "❌ Failed to initialize GLFW" << std::endl;
            return false;
        }

        const auto &config = GET_CONFIG();

        // Set OpenGL version and profile
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4); // 4x MSAA

        // Create window
        window = glfwCreateWindow(config.window_width, config.window_height, config.window_title.c_str(),
                                  config.fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

        if (!window)
        {
            std::cerr << "❌ Failed to create window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(config.vsync ? 1 : 0);

        std::cout << "✅ Window created: " << config.window_width << "x" << config.window_height << std::endl;
        return true;
    }

    bool initializeGraphics()
    {
        std::cout << "🎨 Initializing graphics system..." << std::endl;

        if (glewInit() != GLEW_OK)
        {
            std::cerr << "❌ Failed to initialize GLEW" << std::endl;
            return false;
        }

        // Print OpenGL information
        std::cout << "   OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
        std::cout << "   OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
        std::cout << "   OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "   GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

        std::cout << "✅ Graphics system initialized" << std::endl;
        return true;
    }

    bool initializeCUDA()
    {
        const auto &config = GET_CONFIG();

        if (!config.use_cuda)
        {
            std::cout << "⚠️  CUDA disabled in configuration, using CPU-only mode" << std::endl;
            return true;
        }

        std::cout << "🚀 Initializing CUDA system..." << std::endl;

        if (!CudaHelper::isCudaAvailable())
        {
            std::cout << "⚠️  CUDA not available, falling back to CPU-only mode" << std::endl;
            ConfigManager::getMutableConfig().use_cuda = false;
            return true;
        }

        if (!CudaHelper::initializeCuda())
        {
            std::cout << "⚠️  CUDA initialization failed, using CPU-only mode" << std::endl;
            ConfigManager::getMutableConfig().use_cuda = false;
            return true;
        }

        // Print CUDA information
        std::cout << "   CUDA Runtime Version: " << CudaHelper::getCudaRuntimeVersion() << std::endl;
        std::cout << "   CUDA Driver Version: " << CudaHelper::getCudaDriverVersion() << std::endl;
        std::cout << "   Available GPU Memory: " << (CudaHelper::getAvailableMemory() / (1024 * 1024)) << " MB"
                  << std::endl;

        std::cout << "✅ CUDA system initialized" << std::endl;
        return true;
    }

    bool initializeSimulation()
    {
        std::cout << "⚗️  Initializing physics simulation..." << std::endl;

        const auto &config = GET_CONFIG();

        // Create fluid simulator
        simulator = std::make_unique<FluidSimulator>(config.use_cuda);

        if (!simulator)
        {
            std::cerr << "❌ Failed to create fluid simulator" << std::endl;
            return false;
        }

        // Initialize with configuration
        simulator->initialize(config);

        std::cout << "   Physics Engine: " << (config.use_cuda ? "CUDA GPU" : "CPU") << std::endl;
        std::cout << "   Max Particles: " << config.max_particles << std::endl;
        std::cout << "   Time Step: " << config.time_step << " seconds" << std::endl;
        std::cout << "   SPH Kernel Radius: " << config.smoothing_radius << " meters" << std::endl;

        std::cout << "✅ Physics simulation initialized" << std::endl;
        return true;
    }

    bool initializeRenderer()
    {
        std::cout << "🎬 Initializing rendering system..." << std::endl;

        renderer = std::make_unique<Renderer>(window);

        if (!renderer || !renderer->initialize())
        {
            std::cerr << "❌ Failed to initialize renderer" << std::endl;
            return false;
        }

        const auto &config = GET_CONFIG();

        std::cout << "   Rendering Mode: " << (config.render_as_spheres ? "High Quality Spheres" : "Fast Point Sprites")
                  << std::endl;
        std::cout << "   Lighting: " << (config.enable_lighting ? "Enabled" : "Disabled") << std::endl;
        std::cout << "   Color Mode: ";
        switch (config.color_mode)
        {
        case SimulationConfig::SOLID_COLOR:
            std::cout << "Solid Color";
            break;
        case SimulationConfig::VELOCITY_BASED:
            std::cout << "Velocity-Based";
            break;
        case SimulationConfig::DENSITY_BASED:
            std::cout << "Density-Based";
            break;
        case SimulationConfig::PRESSURE_BASED:
            std::cout << "Pressure-Based";
            break;
        default:
            std::cout << "Unknown";
            break;
        }
        std::cout << std::endl;

        std::cout << "✅ Rendering system initialized" << std::endl;
        return true;
    }

    // === SCENE CREATION ===
    void createInitialScene()
    {
        std::cout << "🌊 Creating initial fluid scene..." << std::endl;

        auto *particle_system = simulator->getParticleSystem();
        if (!particle_system)
            return;

        // Create a small grid of particles at origin
        glm::vec3 start_pos = glm::vec3(-1.0f, -1.0f, -1.0f);
        glm::vec3 end_pos = glm::vec3(1.0f, 1.0f, 1.0f);
        float spacing = 0.2f;

        particle_system->createParticleGrid(start_pos, end_pos, spacing);

        std::cout << "   Created " << particle_system->getActiveCount() << " particles" << std::endl;
    }

    // === INPUT HANDLING ===
    void setupCallbacks()
    {
        glfwSetWindowUserPointer(window, this);

        // Key callback
        glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
            auto *app = static_cast<FluidSimulationApp *>(glfwGetWindowUserPointer(window));
            app->handleKeyInput(key, scancode, action, mods);
        });

        // Mouse callbacks
        glfwSetCursorPosCallback(window, [](GLFWwindow *window, double x, double y) {
            auto *app = static_cast<FluidSimulationApp *>(glfwGetWindowUserPointer(window));
            app->handleMouseMovement(x, y);
        });

        glfwSetScrollCallback(window, [](GLFWwindow *window, double x_offset, double y_offset) {
            auto *app = static_cast<FluidSimulationApp *>(glfwGetWindowUserPointer(window));
            app->handleMouseScroll(x_offset, y_offset);
        });

        // Window resize callback
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height) {
            auto *app = static_cast<FluidSimulationApp *>(glfwGetWindowUserPointer(window));
            app->handleWindowResize(width, height);
        });
    }

    void processInput()
    {
        if (renderer)
        {
            renderer->processInput();
        }
    }

    void handleKeyInput(int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            switch (key)
            {
            case GLFW_KEY_ESCAPE:
                running = false;
                break;

            case GLFW_KEY_SPACE:
                paused = !paused;
                std::cout << (paused ? "⏸️  Simulation PAUSED" : "▶️  Simulation RESUMED") << std::endl;
                if (paused)
                    simulator->pause();
                else
                    simulator->resume();
                break;

            case GLFW_KEY_R:
                std::cout << "🔄 Restarting simulation..." << std::endl;
                simulator->reset();
                createInitialScene();
                frame_count = 0;
                total_runtime = 0.0f;
                break;

            case GLFW_KEY_F1:
                show_debug_info = !show_debug_info;
                std::cout << (show_debug_info ? "📊 Debug info ENABLED" : "📊 Debug info DISABLED") << std::endl;
                break;

            case GLFW_KEY_F2:
                ConfigManager::printPerformanceStats();
                break;

            case GLFW_KEY_F3:
                if (GET_CONFIG().use_cuda)
                {
                    CudaHelper::printMemoryStats();
                }
                break;

            case GLFW_KEY_1:
            case GLFW_KEY_2:
            case GLFW_KEY_3:
            case GLFW_KEY_4:
                switchColorMode(key - GLFW_KEY_1);
                break;
            }
        }
    }

    void handleMouseMovement(double x, double y)
    {
        if (renderer)
        {
            renderer->handleMouseMovement(x, y);
        }
    }

    void handleMouseScroll(double x_offset, double y_offset)
    {
        if (renderer)
        {
            renderer->handleMouseScroll(x_offset, y_offset);
        }
    }

    void handleWindowResize(int width, int height)
    {
        if (renderer)
        {
            renderer->handleWindowResize(width, height);
        }
    }

    // === SIMULATION UPDATE ===
    void updateSimulation()
    {
        if (!paused && simulator)
        {
            physics_timer.start();
            simulator->update(GET_CONFIG().time_step);
            float physics_time = physics_timer.stop();

            // Update performance metrics
            ConfigManager::updatePerformanceMetrics(physics_time, 0.0f, 0.0f);
        }
    }

    void renderFrame()
    {
        if (!renderer || !simulator)
        {
            std::cout << "Renderer or simulator is null!" << std::endl;
            return;
        }

        std::cout << "Starting render frame..." << std::endl;

        render_timer.start();

        // Clear and render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render the scene
        renderer->renderFrame(simulator->getParticleSystem());

        // Swap buffers
        glfwSwapBuffers(window);

        float render_time = render_timer.stop();

        std::cout << "Render frame completed in " << render_time << "ms" << std::endl;

        // Update render performance
        auto &runtime = ConfigManager::getMutableRuntimeConstants();
        runtime.rendering_time_ms = render_time;
    }

    // === STATISTICS AND MONITORING ===
    void updateStatistics()
    {
        float frame_time = frame_timer.stop();
        fps_average.addValue(1000.0f / frame_time); // Convert to FPS
        total_runtime += frame_time / 1000.0f;      // Convert to seconds

        // Update runtime constants
        ConfigManager::updateFrameStats(frame_time);
    }

    void printPeriodicStatus()
    {
        if (!show_debug_info)
            return;

        // Print status every 2 seconds
        if (total_runtime - last_stats_print >= 2.0f)
        {
            printRuntimeStatus();
            last_stats_print = total_runtime;
        }
    }

    void printRuntimeStatus()
    {
        auto *particle_system = simulator->getParticleSystem();
        const auto &runtime = GET_RUNTIME();

        std::cout << "\n📊 === Runtime Status === (Frame " << frame_count << ")" << std::endl;
        std::cout << "   FPS: " << std::fixed << std::setprecision(1) << fps_average.getAverage()
                  << " | Frame Time: " << std::setprecision(2) << runtime.total_frame_time_ms << "ms" << std::endl;
        std::cout << "   Physics: " << runtime.physics_time_ms << "ms | Rendering: " << runtime.rendering_time_ms
                  << "ms" << std::endl;

        if (particle_system)
        {
            std::cout << "   Particles: " << particle_system->getActiveCount() << "/"
                      << particle_system->getMaxParticles() << std::endl;
        }

        if (GET_CONFIG().use_cuda)
        {
            std::cout << "   GPU Memory: " << std::setprecision(1) << CudaHelper::getMemoryUsagePercent() << "% used"
                      << std::endl;
        }

        std::cout << "   Runtime: " << std::setprecision(1) << total_runtime << "s" << std::endl;
    }

    // === UTILITY FUNCTIONS ===
    void switchColorMode(int mode)
    {
        auto &config = ConfigManager::getMutableConfig();

        switch (mode)
        {
        case 0:
            config.color_mode = SimulationConfig::SOLID_COLOR;
            std::cout << "🎨 Color Mode: Solid Color" << std::endl;
            break;
        case 1:
            config.color_mode = SimulationConfig::VELOCITY_BASED;
            std::cout << "🎨 Color Mode: Velocity-Based" << std::endl;
            break;
        case 2:
            config.color_mode = SimulationConfig::DENSITY_BASED;
            std::cout << "🎨 Color Mode: Density-Based" << std::endl;
            break;
        case 3:
            config.color_mode = SimulationConfig::PRESSURE_BASED;
            std::cout << "🎨 Color Mode: Pressure-Based" << std::endl;
            break;
        }
    }

    void printHeader()
    {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                   FLUID SIMULATION ENGINE                   ║" << std::endl;
        std::cout << "║               SPH + CUDA + OpenGL Renderer                  ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\n";
    }

    void printInitializationSummary()
    {
        const auto &config = GET_CONFIG();

        std::cout << "\n🎯 === Initialization Summary ===" << std::endl;
        std::cout << "   Window: " << config.window_width << "x" << config.window_height
                  << (config.fullscreen ? " (Fullscreen)" : "") << std::endl;
        std::cout << "   Physics: " << (config.use_cuda ? "CUDA GPU" : "CPU")
                  << " | Max Particles: " << config.max_particles << std::endl;
        std::cout << "   Rendering: " << (config.render_as_spheres ? "High Quality" : "High Performance") << std::endl;
        std::cout << "   Color Mode: " << getColorModeName(config.color_mode) << std::endl;

        std::cout << "\n🎮 === Controls ===" << std::endl;
        std::cout << "   WASD + Mouse: Camera movement" << std::endl;
        std::cout << "   SPACE: Pause/Resume simulation" << std::endl;
        std::cout << "   R: Restart simulation" << std::endl;
        std::cout << "   F1: Toggle debug info" << std::endl;
        std::cout << "   F2: Print performance stats" << std::endl;
        std::cout << "   F3: Print GPU memory stats" << std::endl;
        std::cout << "   1-4: Switch color modes" << std::endl;
        std::cout << "   ESC: Exit" << std::endl;
        std::cout << "\n✅ All systems ready!" << std::endl;
    }

    void printShutdownSummary()
    {
        std::cout << "\n🏁 === Simulation Summary ===" << std::endl;
        std::cout << "   Total Runtime: " << std::fixed << std::setprecision(1) << total_runtime << " seconds"
                  << std::endl;
        std::cout << "   Total Frames: " << frame_count << std::endl;
        std::cout << "   Average FPS: " << std::setprecision(1) << fps_average.getAverage() << std::endl;

        if (simulator && simulator->getParticleSystem())
        {
            std::cout << "   Final Particle Count: " << simulator->getParticleSystem()->getActiveCount() << std::endl;
        }

        std::cout << "\n👋 Thank you for using Fluid Simulation Engine!" << std::endl;
    }

    std::string getColorModeName(SimulationConfig::ColorMode mode)
    {
        switch (mode)
        {
        case SimulationConfig::SOLID_COLOR:
            return "Solid Color";
        case SimulationConfig::VELOCITY_BASED:
            return "Velocity-Based";
        case SimulationConfig::DENSITY_BASED:
            return "Density-Based";
        case SimulationConfig::PRESSURE_BASED:
            return "Pressure-Based";
        default:
            return "Unknown";
        }
    }

    // === CLEANUP ===
    void cleanup()
    {
        std::cout << "\n🧹 Cleaning up systems..." << std::endl;

        simulator.reset();
        renderer.reset();

        if (window)
        {
            glfwDestroyWindow(window);
            glfwTerminate();
        }

        if (GET_CONFIG().use_cuda)
        {
            CudaHelper::cleanupCuda();
        }

        std::cout << "✅ Cleanup complete" << std::endl;
    }
};

// === COMMAND LINE ARGUMENT PARSING ===
struct CommandLineArgs
{
    std::string config_preset = "default";
    bool help = false;
    bool list_presets = false;

    void parse(int argc, char *argv[])
    {
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h")
            {
                help = true;
            }
            else if (arg == "--preset" || arg == "-p")
            {
                if (i + 1 < argc)
                {
                    config_preset = argv[++i];
                }
            }
            else if (arg == "--list-presets")
            {
                list_presets = true;
            }
        }
    }

    void printHelp()
    {
        std::cout << "Usage: FluidSimulation [OPTIONS]\n" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --preset, -p <name>    Use configuration preset (default: default)" << std::endl;
        std::cout << "  --list-presets         List available configuration presets" << std::endl;
        std::cout << "  --help, -h             Show this help message" << std::endl;
        std::cout << "\nConfiguration Presets:" << std::endl;
        std::cout << "  default     - Balanced settings for general use" << std::endl;
        std::cout << "  performance - Optimized for maximum FPS" << std::endl;
        std::cout << "  quality     - Optimized for visual quality" << std::endl;
        std::cout << "  debug       - Reduced particle count with debug info" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  ./FluidSimulation                    # Use default settings" << std::endl;
        std::cout << "  ./FluidSimulation -p performance     # Use performance preset" << std::endl;
        std::cout << "  ./FluidSimulation -p quality         # Use quality preset" << std::endl;
    }

    void printPresets()
    {
        std::cout << "Available Configuration Presets:\n" << std::endl;

        std::cout << "📋 default" << std::endl;
        std::cout << "   - Balanced settings for general use" << std::endl;
        std::cout << "   - 1000 particles, CUDA enabled, velocity-based coloring\n" << std::endl;

        std::cout << "⚡ performance" << std::endl;
        std::cout << "   - Optimized for maximum FPS" << std::endl;
        std::cout << "   - 50000 particles, point sprites, minimal effects\n" << std::endl;

        std::cout << "💎 quality" << std::endl;
        std::cout << "   - Optimized for visual quality" << std::endl;
        std::cout << "   - 10000 particles, sphere rendering, lighting effects\n" << std::endl;

        std::cout << "🐛 debug" << std::endl;
        std::cout << "   - Reduced particle count with debug visualization" << std::endl;
        std::cout << "   - 1000 particles, debug overlays, performance monitoring" << std::endl;
    }
};

// === MAIN FUNCTION ===
int main(int argc, char *argv[])
{
    // Parse command line arguments
    CommandLineArgs args;
    args.parse(argc, argv);

    if (args.help)
    {
        args.printHelp();
        return 0;
    }

    if (args.list_presets)
    {
        args.printPresets();
        return 0;
    }

    try
    {
        // Create and initialize application
        FluidSimulationApp app;

        if (!app.initialize(args.config_preset))
        {
            std::cerr << "❌ Failed to initialize application!" << std::endl;
            return -1;
        }

        // Run main loop
        app.run();

        std::cout << "\n✅ Application exited successfully" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n💥 Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "\n💥 Unknown exception caught!" << std::endl;
        std::cin.get();
        return -1;
    }

    std::cin.get();
    return 0;
}
