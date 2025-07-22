#pragma once
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>
#include <random>
#include <vector>
#include <chrono>

namespace FluidSim
{
namespace MathUtils
{

// === CONSTANTS ===
constexpr float PI = 3.14159265359f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float HALF_PI = PI * 0.5f;
constexpr float INV_PI = 1.0f / PI;
constexpr float EPSILON = 1e-6f;
constexpr float LARGE_FLOAT = 1e30f;

// === UTILITY FUNCTIONS ===

// Clamp value between min and max
template <typename T> inline T clamp(T value, T min_val, T max_val)
{
    return std::max(min_val, std::min(value, max_val));
}

// Linear interpolation
template <typename T> inline T lerp(T a, T b, float t)
{
    return a + t * (b - a);
}

// Smooth step function
inline float smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Square function
template <typename T> inline T square(T x)
{
    return x * x;
}

// Fast inverse square root (Quake algorithm)
inline float fastInvSqrt(float x)
{
    float xhalf = 0.5f * x;
    int i = *(int *)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float *)&i;
    x = x * (1.5f - xhalf * x * x);
    return x;
}

// === VECTOR OPERATIONS ===

// Distance between two points
inline float distance(const glm::vec3 &a, const glm::vec3 &b)
{
    return glm::length(b - a);
}

// Squared distance (faster when you don't need the actual distance)
inline float distanceSquared(const glm::vec3 &a, const glm::vec3 &b)
{
    glm::vec3 diff = b - a;
    return glm::dot(diff, diff);
}

// Normalize vector with zero-check
inline glm::vec3 safeNormalize(const glm::vec3 &v, const glm::vec3 &fallback = glm::vec3(0.0f, 1.0f, 0.0f))
{
    float length = glm::length(v);
    return (length > EPSILON) ? v / length : fallback;
}

// Reflect vector around normal
inline glm::vec3 reflect(const glm::vec3 &incident, const glm::vec3 &normal)
{
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

// Project vector a onto vector b
inline glm::vec3 project(const glm::vec3 &a, const glm::vec3 &b)
{
    float dot_product = glm::dot(a, b);
    float b_length_squared = glm::dot(b, b);
    return (b_length_squared > EPSILON) ? (dot_product / b_length_squared) * b : glm::vec3(0.0f);
}

// === SPH KERNEL FUNCTIONS ===

// Poly6 kernel for density calculation
inline float kernelPoly6(float r, float h)
{
    if (r > h || r < 0)
        return 0.0f;

    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float diff = h2 - r * r;

    return (315.0f / (64.0f * PI * h9)) * diff * diff * diff;
}

// Spiky kernel gradient for pressure forces
inline glm::vec3 kernelSpikyGradient(const glm::vec3 &r_vec, float h)
{
    float r = glm::length(r_vec);
    if (r > h || r <= 0.0f)
        return glm::vec3(0.0f);

    float h6 = h * h * h * h * h * h;
    float diff = h - r;
    float coefficient = -45.0f / (PI * h6) * diff * diff;

    return coefficient * (r_vec / r);
}

// Viscosity kernel laplacian for viscosity forces
inline float kernelViscosityLaplacian(float r, float h)
{
    if (r > h || r < 0)
        return 0.0f;

    float h6 = h * h * h * h * h * h;
    return (45.0f / (PI * h6)) * (h - r);
}

// === RANDOM NUMBER GENERATION ===

class RandomGenerator
{
  private:
    static thread_local std::mt19937 generator;
    static thread_local std::uniform_real_distribution<float> uniform_dist;
    static thread_local std::normal_distribution<float> normal_dist;

  public:
    // Initialize with seed
    static void seed(unsigned int seed)
    {
        generator.seed(seed);
    }

    // Random float between 0 and 1
    static float random()
    {
        return uniform_dist(generator);
    }

    // Random float between min and max
    static float random(float min, float max)
    {
        return min + (max - min) * random();
    }

    // Random integer between min and max (inclusive)
    static int randomInt(int min, int max)
    {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator);
    }

    // Random point in unit sphere
    static glm::vec3 randomInSphere()
    {
        glm::vec3 point;
        do
        {
            point = glm::vec3(random(-1, 1), random(-1, 1), random(-1, 1));
        } while (glm::dot(point, point) > 1.0f);
        return point;
    }

    // Random point on unit sphere surface
    static glm::vec3 randomOnSphere()
    {
        return glm::normalize(randomInSphere());
    }

    // Random point in unit circle (2D)
    static glm::vec2 randomInCircle()
    {
        glm::vec2 point;
        do
        {
            point = glm::vec2(random(-1, 1), random(-1, 1));
        } while (glm::dot(point, point) > 1.0f);
        return point;
    }

    // Gaussian/normal distribution
    static float randomGaussian(float mean = 0.0f, float stddev = 1.0f)
    {
        return mean + stddev * normal_dist(generator);
    }

    // Random color
    static glm::vec3 randomColor()
    {
        return glm::vec3(random(), random(), random());
    }

    // Random velocity within bounds
    static glm::vec3 randomVelocity(float max_speed)
    {
        return randomOnSphere() * random(0.0f, max_speed);
    }
};

// === GEOMETRIC OPERATIONS ===

// Check if point is inside AABB
inline bool pointInAABB(const glm::vec3 &point, const glm::vec3 &min_bound, const glm::vec3 &max_bound)
{
    return point.x >= min_bound.x && point.x <= max_bound.x && point.y >= min_bound.y && point.y <= max_bound.y &&
           point.z >= min_bound.z && point.z <= max_bound.z;
}

// Check if sphere intersects AABB
inline bool sphereAABBIntersection(const glm::vec3 &sphere_center, float sphere_radius, const glm::vec3 &aabb_min,
                                   const glm::vec3 &aabb_max)
{
    glm::vec3 closest_point = glm::clamp(sphere_center, aabb_min, aabb_max);
    float distance_squared = glm::distance2(sphere_center, closest_point);
    return distance_squared <= sphere_radius * sphere_radius;
}

// Check if two spheres intersect
inline bool sphereSphereIntersection(const glm::vec3 &center1, float radius1, const glm::vec3 &center2, float radius2)
{
    float distance_squared = glm::distance2(center1, center2);
    float radius_sum = radius1 + radius2;
    return distance_squared <= radius_sum * radius_sum;
}

// Ray-sphere intersection
struct RayHit
{
    bool hit;
    float distance;
    glm::vec3 point;
    glm::vec3 normal;
};

inline RayHit raySphereIntersection(const glm::vec3 &ray_origin, const glm::vec3 &ray_direction,
                                    const glm::vec3 &sphere_center, float sphere_radius)
{
    RayHit result = {false, 0.0f, glm::vec3(0.0f), glm::vec3(0.0f)};

    glm::vec3 oc = ray_origin - sphere_center;
    float a = glm::dot(ray_direction, ray_direction);
    float b = 2.0f * glm::dot(oc, ray_direction);
    float c = glm::dot(oc, oc) - sphere_radius * sphere_radius;

    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0)
    {
        float sqrt_discriminant = sqrt(discriminant);
        float t1 = (-b - sqrt_discriminant) / (2.0f * a);
        float t2 = (-b + sqrt_discriminant) / (2.0f * a);

        float t = (t1 > 0) ? t1 : t2;

        if (t > 0)
        {
            result.hit = true;
            result.distance = t;
            result.point = ray_origin + t * ray_direction;
            result.normal = glm::normalize(result.point - sphere_center);
        }
    }

    return result;
}

// === TRANSFORMATION UTILITIES ===

// Create transformation matrix
inline glm::mat4 createTransform(const glm::vec3 &translation, const glm::vec3 &rotation, const glm::vec3 &scale)
{
    glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 r = glm::rotate(glm::mat4(1.0f), rotation.x, glm::vec3(1, 0, 0));
    r = glm::rotate(r, rotation.y, glm::vec3(0, 1, 0));
    r = glm::rotate(r, rotation.z, glm::vec3(0, 0, 1));
    glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
    return t * r * s;
}

// Extract translation from matrix
inline glm::vec3 extractTranslation(const glm::mat4 &matrix)
{
    return glm::vec3(matrix[3]);
}

// Extract scale from matrix
inline glm::vec3 extractScale(const glm::mat4 &matrix)
{
    return glm::vec3(glm::length(glm::vec3(matrix[0])), glm::length(glm::vec3(matrix[1])),
                     glm::length(glm::vec3(matrix[2])));
}

// === TIMING UTILITIES ===

class Timer
{
  private:
    std::chrono::high_resolution_clock::time_point start_time;
    bool running;

  public:
    Timer() : running(false)
    {
    }

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    float stop()
    {
        if (!running)
            return 0.0f;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        running = false;

        return duration.count() / 1000.0f; // Return milliseconds
    }

    float peek() const
    {
        if (!running)
            return 0.0f;

        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);

        return duration.count() / 1000.0f; // Return milliseconds
    }

    bool isRunning() const
    {
        return running;
    }
};

// === STATISTICS UTILITIES ===

template <typename T> class MovingAverage
{
  private:
    std::vector<T> values;
    size_t max_size;
    size_t current_index;
    bool filled;
    T sum;

  public:
    MovingAverage(size_t size = 60) : max_size(size), current_index(0), filled(false), sum(T{})
    {
        values.resize(max_size);
    }

    void addValue(T value)
    {
        if (filled)
        {
            sum -= values[current_index];
        }

        values[current_index] = value;
        sum += value;

        current_index = (current_index + 1) % max_size;
        if (current_index == 0)
        {
            filled = true;
        }
    }

    T getAverage() const
    {
        if (!filled && current_index == 0)
            return T{};

        size_t count = filled ? max_size : current_index;
        return sum / static_cast<T>(count);
    }

    void clear()
    {
        current_index = 0;
        filled = false;
        sum = T{};
    }
};

} // namespace MathUtils
} // namespace FluidSim
