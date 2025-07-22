#include "cuda_helper.hh"
#include <algorithm>
#include <iomanip>

namespace FluidSim
{
namespace CudaHelper
{

// Device management implementation
bool initializeCuda()
{
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess || device_count == 0)
    {
        std::cerr << "No CUDA devices found or CUDA not available" << std::endl;
        return false;
    }

    // Set the first device as default
    error = cudaSetDevice(0);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    std::cout << "CUDA initialized successfully with " << device_count << " device(s)" << std::endl;
    printDeviceInfo(0);

    return true;
}

void cleanupCuda()
{
    cudaError_t error = cudaDeviceReset();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA cleanup failed: " << cudaGetErrorString(error) << std::endl;
    }
    else
    {
        std::cout << "CUDA cleaned up successfully" << std::endl;
    }
}

std::vector<DeviceInfo> getDeviceInfo()
{
    std::vector<DeviceInfo> devices;

    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        size_t free_mem, total_mem;
        cudaSetDevice(i);
        cudaMemGetInfo(&free_mem, &total_mem);

        DeviceInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.total_memory = total_mem;
        info.free_memory = free_mem;
        info.major_compute_capability = prop.major;
        info.minor_compute_capability = prop.minor;
        info.multiprocessor_count = prop.multiProcessorCount;
        info.max_threads_per_block = prop.maxThreadsPerBlock;
        info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
        info.warp_size = prop.warpSize;
        info.unified_addressing = prop.unifiedAddressing;
        info.can_map_host_memory = prop.canMapHostMemory;

        devices.push_back(info);
    }

    return devices;
}

bool setDevice(int device_id)
{
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to set device " << device_id << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

int getCurrentDevice()
{
    int device;
    cudaGetDevice(&device);
    return device;
}

void printDeviceInfo(int device_id)
{
    auto devices = getDeviceInfo();

    if (device_id == -1)
    {
        device_id = getCurrentDevice();
    }

    if (device_id >= 0 && device_id < static_cast<int>(devices.size()))
    {
        const auto &info = devices[device_id];

        std::cout << "\n=== CUDA Device " << info.device_id << " Information ===" << std::endl;
        std::cout << "Name: " << info.name << std::endl;
        std::cout << "Compute Capability: " << info.major_compute_capability << "." << info.minor_compute_capability
                  << std::endl;
        std::cout << "Total Memory: " << (info.total_memory / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "Free Memory: " << (info.free_memory / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "Multiprocessors: " << info.multiprocessor_count << std::endl;
        std::cout << "Max Threads per Block: " << info.max_threads_per_block << std::endl;
        std::cout << "Max Threads per MP: " << info.max_threads_per_multiprocessor << std::endl;
        std::cout << "Warp Size: " << info.warp_size << std::endl;
        std::cout << "Unified Addressing: " << (info.unified_addressing ? "Yes" : "No") << std::endl;
        std::cout << "Can Map Host Memory: " << (info.can_map_host_memory ? "Yes" : "No") << std::endl;
        std::cout << "======================================\n" << std::endl;
    }
}

size_t getAvailableMemory()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t getTotalMemory()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

float getMemoryUsagePercent()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (total_mem == 0)
        return 0.0f;

    size_t used_mem = total_mem - free_mem;
    return (static_cast<float>(used_mem) / total_mem) * 100.0f;
}

// Kernel launch helpers
int getOptimalBlockSize(const void *kernel_func, size_t dynamic_shared_mem)
{
    int min_grid_size;
    int block_size;

    // Use CUDA occupancy calculator
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_func, dynamic_shared_mem, 0);

    return block_size;
}

bool synchronizeDevice()
{
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        std::cerr << "Device synchronization failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

bool isCudaAvailable()
{
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

int getCudaRuntimeVersion()
{
    int version;
    cudaRuntimeGetVersion(&version);
    return version;
}

int getCudaDriverVersion()
{
    int version;
    cudaDriverGetVersion(&version);
    return version;
}

// Error handling
std::string cudaErrorToString(cudaError_t error)
{
    return std::string(cudaGetErrorString(error));
}

void checkMemoryLeaks()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t used_mem = total_mem - free_mem;

    if (used_mem > 0)
    {
        std::cout << "Warning: " << (used_mem / (1024 * 1024)) << " MB of GPU memory still allocated" << std::endl;
    }
    else
    {
        std::cout << "No GPU memory leaks detected" << std::endl;
    }
}

void printMemoryStats()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t used_mem = total_mem - free_mem;
    float usage_percent = (static_cast<float>(used_mem) / total_mem) * 100.0f;

    std::cout << "\n=== GPU Memory Statistics ===" << std::endl;
    std::cout << "Total Memory: " << (total_mem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Used Memory: " << (used_mem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Free Memory: " << (free_mem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Usage: " << std::fixed << std::setprecision(1) << usage_percent << "%" << std::endl;
    std::cout << "============================\n" << std::endl;
}

} // namespace CudaHelper
} // namespace FluidSim
