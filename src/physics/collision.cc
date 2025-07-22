#include "collision.hh"
#include "../core/particle.hh"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace FluidSim
{

CollisionHandler::CollisionHandler()
    : boundary_min(-5.0f, -5.0f, -5.0f), boundary_max(5.0f, 5.0f, 5.0f), restitution_coefficient(0.5f),
      friction_coefficient(0.1f), grid_cell_size(0.2f), grid_origin(-10.0f, -10.0f, -10.0f),
      grid_dimensions(100, 100, 100)
{

    // Initialize spatial grid
    spatial_grid.resize(grid_dimensions.x * grid_dimensions.y * grid_dimensions.z);
}

void CollisionHandler::setBoundaries(glm::vec3 min_bound, glm::vec3 max_bound)
{
    boundary_min = min_bound;
    boundary_max = max_bound;

    std::cout << "Collision boundaries set: (" << min_bound.x << "," << min_bound.y << "," << min_bound.z << ") to ("
              << max_bound.x << "," << max_bound.y << "," << max_bound.z << ")\n";
}

void CollisionHandler::addSphereObstacle(glm::vec3 center, float radius, float restitution, float friction)
{
    sphere_obstacles.emplace_back(center, radius, restitution, friction);

    std::cout << "Added sphere obstacle: center(" << center.x << "," << center.y << "," << center.z
              << ") radius=" << radius << "\n";
}

void CollisionHandler::clearObstacles()
{
    sphere_obstacles.clear();
}

void CollisionHandler::handleCollisions(ParticleSystem *system, float dt)
{
    if (!system || system->getActiveCount() == 0)
        return;

    // Handle different collision types
    handleBoundaryCollisions(system);
    handleObstacleCollisions(system);

    // Particle-particle collisions are expensive, only do if needed
    // handleParticleCollisions(system);
}

void CollisionHandler::handleBoundaryCollisions(ParticleSystem *system)
{
    auto &particles = system->getParticles();

    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        Particle &p = particles[i];
        bool collision_occurred = false;

        // Check X boundaries
        if (p.position.x < boundary_min.x)
        {
            p.position.x = boundary_min.x;
            if (p.velocity.x < 0)
            {
                p.velocity.x = -p.velocity.x * restitution_coefficient;
                collision_occurred = true;
            }
        }
        else if (p.position.x > boundary_max.x)
        {
            p.position.x = boundary_max.x;
            if (p.velocity.x > 0)
            {
                p.velocity.x = -p.velocity.x * restitution_coefficient;
                collision_occurred = true;
            }
        }

        // Check Y boundaries
        if (p.position.y < boundary_min.y)
        {
            p.position.y = boundary_min.y;
            if (p.velocity.y < 0)
            {
                p.velocity.y = -p.velocity.y * restitution_coefficient;
                collision_occurred = true;
            }
        }
        else if (p.position.y > boundary_max.y)
        {
            p.position.y = boundary_max.y;
            if (p.velocity.y > 0)
            {
                p.velocity.y = -p.velocity.y * restitution_coefficient;
                collision_occurred = true;
            }
        }

        // Check Z boundaries
        if (p.position.z < boundary_min.z)
        {
            p.position.z = boundary_min.z;
            if (p.velocity.z < 0)
            {
                p.velocity.z = -p.velocity.z * restitution_coefficient;
                collision_occurred = true;
            }
        }
        else if (p.position.z > boundary_max.z)
        {
            p.position.z = boundary_max.z;
            if (p.velocity.z > 0)
            {
                p.velocity.z = -p.velocity.z * restitution_coefficient;
                collision_occurred = true;
            }
        }

        // Apply friction if collision occurred
        if (collision_occurred && friction_coefficient > 0.0f)
        {
            p.velocity *= (1.0f - friction_coefficient);
        }
    }
}

void CollisionHandler::handleObstacleCollisions(ParticleSystem *system)
{
    auto &particles = system->getParticles();
    float particle_radius = 0.02f; // Assume small particle radius

    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        Particle &p = particles[i];

        // Check collision with each sphere obstacle
        for (const auto &obstacle : sphere_obstacles)
        {
            glm::vec3 diff = p.position - obstacle.center;
            float distance = glm::length(diff);
            float min_distance = particle_radius + obstacle.radius;

            if (distance < min_distance && distance > 0.0f)
            {
                // Collision detected - move particle out of obstacle
                glm::vec3 normal = diff / distance; // normalize
                p.position = obstacle.center + normal * min_distance;

                // Reflect velocity along normal
                float vel_along_normal = glm::dot(p.velocity, normal);
                if (vel_along_normal < 0)
                { // Moving towards obstacle
                    glm::vec3 reflection = p.velocity - 2.0f * vel_along_normal * normal;
                    p.velocity = reflection * obstacle.restitution;

                    // Apply friction (tangential velocity reduction)
                    if (obstacle.friction > 0.0f)
                    {
                        glm::vec3 tangent_velocity = p.velocity - glm::dot(p.velocity, normal) * normal;
                        p.velocity -= tangent_velocity * obstacle.friction;
                    }
                }
            }
        }
    }
}

void CollisionHandler::handleParticleCollisions(ParticleSystem *system)
{
    // This is computationally expensive O(n²) without spatial optimization
    // For now, we'll skip this and focus on boundary/obstacle collisions

    auto &particles = system->getParticles();
    float particle_radius = 0.02f;
    float min_distance = 2.0f * particle_radius;

    size_t collision_count = 0;

    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        for (size_t j = i + 1; j < system->getActiveCount(); j++)
        {
            glm::vec3 diff = particles[i].position - particles[j].position;
            float distance = glm::length(diff);

            if (distance < min_distance && distance > 0.0f)
            {
                collision_count++;

                // Separate particles
                glm::vec3 normal = diff / distance;
                float overlap = min_distance - distance;
                glm::vec3 separation = normal * (overlap * 0.5f);

                particles[i].position += separation;
                particles[j].position -= separation;

                // Simple elastic collision response
                float restitution = 0.8f;
                glm::vec3 relative_velocity = particles[i].velocity - particles[j].velocity;
                float vel_along_normal = glm::dot(relative_velocity, normal);

                if (vel_along_normal > 0)
                    continue; // Particles separating

                float impulse = 2.0f * vel_along_normal / (particles[i].mass + particles[j].mass);

                particles[i].velocity -= impulse * particles[j].mass * normal * restitution;
                particles[j].velocity += impulse * particles[i].mass * normal * restitution;
            }
        }
    }

    if (collision_count > 0)
    {
        std::cout << "Particle collisions: " << collision_count << std::endl;
    }
}

bool CollisionHandler::checkSphereCollision(glm::vec3 point, glm::vec3 sphere_center, float point_radius,
                                            float sphere_radius)
{
    float distance = glm::length(point - sphere_center);
    return distance < (point_radius + sphere_radius);
}

glm::vec3 CollisionHandler::resolveCollision(glm::vec3 position, glm::vec3 velocity, glm::vec3 normal,
                                             float restitution, float friction)
{
    // Reflect velocity along normal
    float vel_along_normal = glm::dot(velocity, normal);
    glm::vec3 reflected_velocity = velocity - (1.0f + restitution) * vel_along_normal * normal;

    // Apply friction to tangential component
    if (friction > 0.0f)
    {
        glm::vec3 tangent_velocity = reflected_velocity - glm::dot(reflected_velocity, normal) * normal;
        reflected_velocity -= tangent_velocity * friction;
    }

    return reflected_velocity;
}

// Spatial grid methods (for optimization)
void CollisionHandler::buildSpatialGrid(ParticleSystem *system)
{
    // Clear grid
    for (auto &cell : spatial_grid)
    {
        cell.clear();
    }

    // Insert particles into grid
    auto &particles = system->getParticles();
    for (size_t i = 0; i < system->getActiveCount(); i++)
    {
        insertParticleIntoGrid(i, particles[i].position);
    }
}

glm::ivec3 CollisionHandler::worldToGrid(glm::vec3 world_pos)
{
    glm::vec3 relative_pos = world_pos - grid_origin;
    glm::ivec3 grid_pos =
        glm::ivec3(static_cast<int>(relative_pos.x / grid_cell_size), static_cast<int>(relative_pos.y / grid_cell_size),
                   static_cast<int>(relative_pos.z / grid_cell_size));

    // Clamp to grid bounds
    grid_pos.x = std::max(0, std::min(grid_pos.x, grid_dimensions.x - 1));
    grid_pos.y = std::max(0, std::min(grid_pos.y, grid_dimensions.y - 1));
    grid_pos.z = std::max(0, std::min(grid_pos.z, grid_dimensions.z - 1));

    return grid_pos;
}

size_t CollisionHandler::gridToIndex(glm::ivec3 grid_pos)
{
    return grid_pos.x + grid_pos.y * grid_dimensions.x + grid_pos.z * grid_dimensions.x * grid_dimensions.y;
}

void CollisionHandler::insertParticleIntoGrid(size_t particle_idx, glm::vec3 position)
{
    glm::ivec3 grid_pos = worldToGrid(position);
    size_t grid_index = gridToIndex(grid_pos);

    if (grid_index < spatial_grid.size())
    {
        spatial_grid[grid_index].push_back(particle_idx);
    }
}

// CollisionShapes implementation
bool CollisionShapes::checkPlaneCollision(glm::vec3 point, const Plane &plane, float radius)
{
    float distance = glm::dot(point - plane.point, plane.normal);
    return distance < radius;
}

bool CollisionShapes::checkBoxCollision(glm::vec3 point, const Box &box, float radius)
{
    // Transform point to box local space
    glm::vec3 local_point = glm::transpose(box.rotation) * (point - box.center);

    // Find closest point on box to the sphere center
    glm::vec3 closest_point = glm::clamp(local_point, -box.half_extents, box.half_extents);

    // Check if distance is less than radius
    float distance = glm::length(local_point - closest_point);
    return distance < radius;
}

glm::vec3 CollisionShapes::resolvePlaneCollision(glm::vec3 position, glm::vec3 velocity, const Plane &plane,
                                                 float radius)
{
    float distance = glm::dot(position - plane.point, plane.normal);

    if (distance < radius)
    {
        // Move point outside plane
        position = position + plane.normal * (radius - distance);

        // Reflect velocity
        float vel_along_normal = glm::dot(velocity, plane.normal);
        if (vel_along_normal < 0)
        {
            velocity = velocity - (1.0f + plane.restitution) * vel_along_normal * plane.normal;

            // Apply friction
            if (plane.friction > 0.0f)
            {
                glm::vec3 tangent_velocity = velocity - glm::dot(velocity, plane.normal) * plane.normal;
                velocity -= tangent_velocity * plane.friction;
            }
        }
    }

    return velocity;
}

glm::vec3 CollisionShapes::resolveBoxCollision(glm::vec3 position, glm::vec3 velocity, const Box &box, float radius)
{
    // Transform to box local space
    glm::vec3 local_pos = glm::transpose(box.rotation) * (position - box.center);
    glm::vec3 local_vel = glm::transpose(box.rotation) * velocity;

    // Find closest point on box
    glm::vec3 closest_point = glm::clamp(local_pos, -box.half_extents, box.half_extents);
    glm::vec3 diff = local_pos - closest_point;
    float distance = glm::length(diff);

    if (distance < radius && distance > 0.0f)
    {
        // Calculate collision normal
        glm::vec3 normal = diff / distance;

        // Move position outside box
        local_pos = closest_point + normal * radius;

        // Reflect velocity
        float vel_along_normal = glm::dot(local_vel, normal);
        if (vel_along_normal < 0)
        {
            local_vel = local_vel - (1.0f + box.restitution) * vel_along_normal * normal;

            // Apply friction
            if (box.friction > 0.0f)
            {
                glm::vec3 tangent_velocity = local_vel - glm::dot(local_vel, normal) * normal;
                local_vel -= tangent_velocity * box.friction;
            }
        }

        // Transform back to world space
        velocity = box.rotation * local_vel;
    }

    return velocity;
}

} // namespace FluidSim
