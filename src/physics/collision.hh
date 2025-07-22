#pragma once
#include <glm/glm.hpp>
#include <vector>

namespace FluidSim
{

// Forward declarations
struct Particle;
class ParticleSystem;

/**
 * Collision detection and response for fluid particles
 * Handles boundaries, obstacles, and particle-particle collisions
 */
class CollisionHandler
{
  private:
    // Boundary parameters
    glm::vec3 boundary_min;
    glm::vec3 boundary_max;
    float restitution_coefficient; // Bounce factor (0.0 = no bounce, 1.0 = perfect bounce)
    float friction_coefficient;    // Surface friction

    // Obstacle list (simple spheres for now)
    struct SphereObstacle
    {
        glm::vec3 center;
        float radius;
        float restitution;
        float friction;
        bool is_static;

        SphereObstacle(glm::vec3 c, float r, float rest = 0.5f, float fric = 0.1f)
            : center(c), radius(r), restitution(rest), friction(fric), is_static(true)
        {
        }
    };

    std::vector<SphereObstacle> sphere_obstacles;

  public:
    CollisionHandler();
    ~CollisionHandler() = default;

    // Initialization
    void setBoundaries(glm::vec3 min_bound, glm::vec3 max_bound);
    void setRestitution(float restitution)
    {
        restitution_coefficient = restitution;
    }
    void setFriction(float friction)
    {
        friction_coefficient = friction;
    }

    // Obstacle management
    void addSphereObstacle(glm::vec3 center, float radius, float restitution = 0.5f, float friction = 0.1f);
    void clearObstacles();
    size_t getObstacleCount() const
    {
        return sphere_obstacles.size();
    }

    // Main collision detection and response
    void handleCollisions(ParticleSystem *system, float dt);

    // Individual collision types
    void handleBoundaryCollisions(ParticleSystem *system);
    void handleObstacleCollisions(ParticleSystem *system);
    void handleParticleCollisions(ParticleSystem *system);

    // Collision detection utilities
    bool checkSphereCollision(glm::vec3 point, glm::vec3 sphere_center, float point_radius, float sphere_radius);

    glm::vec3 resolveCollision(glm::vec3 position, glm::vec3 velocity, glm::vec3 normal, float restitution,
                               float friction);

    // Spatial optimization for particle-particle collisions
    void buildSpatialGrid(ParticleSystem *system);
    std::vector<std::vector<size_t>> spatial_grid;
    glm::ivec3 grid_dimensions;
    float grid_cell_size;
    glm::vec3 grid_origin;

  private:
    // Helper methods
    glm::ivec3 worldToGrid(glm::vec3 world_pos);
    size_t gridToIndex(glm::ivec3 grid_pos);
    void insertParticleIntoGrid(size_t particle_idx, glm::vec3 position);
};

/**
 * Advanced collision shapes (for future expansion)
 */
class CollisionShapes
{
  public:
    // Plane collision
    struct Plane
    {
        glm::vec3 point;
        glm::vec3 normal;
        float restitution;
        float friction;

        Plane(glm::vec3 p, glm::vec3 n, float rest = 0.5f, float fric = 0.1f)
            : point(p), normal(glm::normalize(n)), restitution(rest), friction(fric)
        {
        }
    };

    // Box collision
    struct Box
    {
        glm::vec3 center;
        glm::vec3 half_extents;
        glm::mat3 rotation; // For oriented boxes
        float restitution;
        float friction;

        Box(glm::vec3 c, glm::vec3 extents, float rest = 0.5f, float fric = 0.1f)
            : center(c), half_extents(extents), rotation(1.0f), restitution(rest), friction(fric)
        {
        }
    };

    // Collision detection methods
    static bool checkPlaneCollision(glm::vec3 point, const Plane &plane, float radius);
    static bool checkBoxCollision(glm::vec3 point, const Box &box, float radius);

    // Collision response methods
    static glm::vec3 resolvePlaneCollision(glm::vec3 position, glm::vec3 velocity, const Plane &plane, float radius);
    static glm::vec3 resolveBoxCollision(glm::vec3 position, glm::vec3 velocity, const Box &box, float radius);
};

} // namespace FluidSim
