#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <vector>

namespace FluidSim
{
namespace CudaHelper
{

// === ERROR CHECKING MACROS ===

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error)         \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = cudaGetLastError();                                                                        \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << std::endl;                              \
        }                                                                                                              \
        CUDA_CHECK(cudaDeviceSynchronize());                                                                           \
    } while (0)

// === DEVICE INFORMATION ===

struct DeviceInfo
{
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int major_compute_capability;
    int minor_compute_capability;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    bool unified_addressing;
    bool can_map_host_memory;
};

// Device management
bool initializeCuda();
void cleanupCuda();
std::vector<DeviceInfo> getDeviceInfo();
bool setDevice(int device_id);
int getCurrentDevice();
void printDeviceInfo(int device_id = -1);

// Memory information
size_t getAvailableMemory();
size_t getTotalMemory();
float getMemoryUsagePercent();

// === MEMORY MANAGEMENT ===

template <typename T> class DeviceBuffer
{
  private:
    T *d_ptr;
    size_t size;
    size_t capacity;

  public:
    DeviceBuffer() : d_ptr(nullptr), size(0), capacity(0)
    {
    }

    explicit DeviceBuffer(size_t count) : d_ptr(nullptr), size(0), capacity(0)
    {
        allocate(count);
    }

    ~DeviceBuffer()
    {
        free();
    }

    // Disable copy constructor and assignment
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    // Move constructor and assignment
    DeviceBuffer(DeviceBuffer &&other) noexcept : d_ptr(other.d_ptr), size(other.size), capacity(other.capacity)
    {
        other.d_ptr = nullptr;
        other.size = 0;
        other.capacity = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            free();
            d_ptr = other.d_ptr;
            size = other.size;
            capacity = other.capacity;
            other.d_ptr = nullptr;
            other.size = 0;
            other.capacity = 0;
        }
        return *this;
    }

    // Memory operations
    bool allocate(size_t count)
    {
        if (count <= capacity)
        {
            size = count;
            return true;
        }

        free();

        cudaError_t error = cudaMalloc(&d_ptr, count * sizeof(T));
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        size = count;
        capacity = count;
        return true;
    }

    void free()
    {
        if (d_ptr)
        {
            cudaFree(d_ptr);
            d_ptr = nullptr;
        }
        size = 0;
        capacity = 0;
    }

    bool resize(size_t new_size)
    {
        if (new_size <= capacity)
        {
            size = new_size;
            return true;
        }
        return allocate(new_size);
    }

    // Data transfer
    bool copyFromHost(const T *host_ptr, size_t count = 0)
    {
        if (count == 0)
            count = size;
        if (!d_ptr || count > capacity)
            return false;

        cudaError_t error = cudaMemcpy(d_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
        return error == cudaSuccess;
    }

    bool copyToHost(T *host_ptr, size_t count = 0) const
    {
        if (count == 0)
            count = size;
        if (!d_ptr || count > size)
            return false;

        cudaError_t error = cudaMemcpy(host_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
        return error == cudaSuccess;
    }

    bool copyFromDevice(const DeviceBuffer<T> &other, size_t count = 0)
    {
        if (count == 0)
            count = std::min(size, other.size);
        if (!d_ptr || !other.d_ptr || count > size || count > other.size)
            return false;

        cudaError_t error = cudaMemcpy(d_ptr, other.d_ptr, count * sizeof(T), cudaMemcpyDeviceToDevice);
        return error == cudaSuccess;
    }

    void clear()
    {
        if (d_ptr && size > 0)
        {
            cudaMemset(d_ptr, 0, size * sizeof(T));
        }
    }

    // Accessors
    T *data()
    {
        return d_ptr;
    }
    const T *data() const
    {
        return d_ptr;
    }
    size_t getSize() const
    {
        return size;
    }
    size_t getCapacity() const
    {
        return capacity;
    }
    bool empty() const
    {
        return size == 0;
    }
    bool valid() const
    {
        return d_ptr != nullptr;
    }

    // Memory usage in bytes
    size_t sizeInBytes() const
    {
        return size * sizeof(T);
    }
    size_t capacityInBytes() const
    {
        return capacity * sizeof(T);
    }
};

// === KERNEL LAUNCH HELPERS ===

// Calculate optimal grid and block dimensions
struct LaunchConfig
{
    dim3 grid_size;
    dim3 block_size;
    size_t shared_mem_size;

    LaunchConfig() : shared_mem_size(0)
    {
    }

    LaunchConfig(int total_threads, int preferred_block_size = 256) : shared_mem_size(0)
    {
        calculateOptimal(total_threads, preferred_block_size);
    }

    void calculateOptimal(int total_threads, int preferred_block_size = 256)
    {
        block_size = dim3(preferred_block_size, 1, 1);
        grid_size = dim3((total_threads + preferred_block_size - 1) / preferred_block_size, 1, 1);
    }

    void calculate2D(int width, int height, int block_x = 16, int block_y = 16)
    {
        block_size = dim3(block_x, block_y, 1);
        grid_size = dim3((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
    }

    void calculate3D(int width, int height, int depth, int block_x = 8, int block_y = 8, int block_z = 8)
    {
        block_size = dim3(block_x, block_y, block_z);
        grid_size =
            dim3((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, (depth + block_z - 1) / block_z);
    }
};

// Get optimal block size for a kernel
int getOptimalBlockSize(const void *kernel_func, size_t dynamic_shared_mem = 0);

// === PERFORMANCE MONITORING ===

class CudaTimer
{
  private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool timing_active;

  public:
    CudaTimer() : timing_active(false)
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start()
    {
        cudaEventRecord(start_event);
        timing_active = true;
    }

    float stop()
    {
        if (!timing_active)
            return 0.0f;

        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        timing_active = false;

        return elapsed_time;
    }
};

// === UTILITY FUNCTIONS ===

// Safe kernel launch with error checking
template <typename KernelFunc, typename... Args>
bool launchKernel(KernelFunc kernel, const LaunchConfig &config, Args &&...args)
{
    kernel<<<config.grid_size, config.block_size, config.shared_mem_size>>>(std::forward<Args>(args)...);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Synchronize device
bool synchronizeDevice();

// Check if CUDA is available
bool isCudaAvailable();

// Get CUDA runtime version
int getCudaRuntimeVersion();
int getCudaDriverVersion();

// === MEMORY UTILITIES ===

// Pinned memory allocator
template <typename T> class PinnedBuffer
{
  private:
    T *h_ptr;
    size_t size;

  public:
    PinnedBuffer() : h_ptr(nullptr), size(0)
    {
    }

    explicit PinnedBuffer(size_t count) : h_ptr(nullptr), size(0)
    {
        allocate(count);
    }

    ~PinnedBuffer()
    {
        free();
    }

    bool allocate(size_t count)
    {
        free();

        cudaError_t error = cudaMallocHost(&h_ptr, count * sizeof(T));
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA pinned malloc failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        size = count;
        return true;
    }

    void free()
    {
        if (h_ptr)
        {
            cudaFreeHost(h_ptr);
            h_ptr = nullptr;
        }
        size = 0;
    }

    T *data()
    {
        return h_ptr;
    }
    const T *data() const
    {
        return h_ptr;
    }
    size_t getSize() const
    {
        return size;
    }
    bool valid() const
    {
        return h_ptr != nullptr;
    }
};

// Unified memory allocator
template <typename T> class UnifiedBuffer
{
  private:
    T *u_ptr;
    size_t size;

  public:
    UnifiedBuffer() : u_ptr(nullptr), size(0)
    {
    }

    explicit UnifiedBuffer(size_t count) : u_ptr(nullptr), size(0)
    {
        allocate(count);
    }

    ~UnifiedBuffer()
    {
        free();
    }

    bool allocate(size_t count)
    {
        free();

        cudaError_t error = cudaMallocManaged(&u_ptr, count * sizeof(T));
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA unified malloc failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        size = count;
        return true;
    }

    void free()
    {
        if (u_ptr)
        {
            cudaFree(u_ptr);
            u_ptr = nullptr;
        }
        size = 0;
    }

    void prefetchToDevice(int device = -1)
    {
        if (u_ptr && size > 0)
        {
            if (device == -1)
            {
                cudaGetDevice(&device);
            }
            cudaMemPrefetchAsync(u_ptr, size * sizeof(T), device);
        }
    }

    void prefetchToHost()
    {
        if (u_ptr && size > 0)
        {
            cudaMemPrefetchAsync(u_ptr, size * sizeof(T), cudaCpuDeviceId);
        }
    }

    T *data()
    {
        return u_ptr;
    }
    const T *data() const
    {
        return u_ptr;
    }
    size_t getSize() const
    {
        return size;
    }
    bool valid() const
    {
        return u_ptr != nullptr;
    }
};

// === PHYSICS-SPECIFIC STRUCTURES ===

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

// Neighbor list structure
struct NeighborList
{
    int *indices;       // Flattened neighbor indices
    int *counts;        // Number of neighbors per particle
    int max_neighbors;  // Maximum neighbors per particle
    int particle_count; // Total number of particles
};

// Spatial grid structure for neighbor finding
struct SpatialGrid
{
    int3 *cell_indices;   // Grid cell for each particle
    int *cell_starts;     // Start index for each cell
    int *cell_ends;       // End index for each cell
    int3 grid_dimensions; // Grid size
    float3 grid_origin;   // Grid origin
    float cell_size;      // Grid cell size
};

// === ERROR HANDLING ===

// CUDA error to string conversion
std::string cudaErrorToString(cudaError_t error);

// Check for memory leaks
void checkMemoryLeaks();

// Print memory usage statistics
void printMemoryStats();

} // namespace CudaHelper
} // namespace FluidSim
