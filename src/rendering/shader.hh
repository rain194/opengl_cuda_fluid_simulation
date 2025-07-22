#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <unordered_map>
#include <memory>

namespace FluidSim
{

/**
 * OpenGL shader program wrapper
 * Handles loading, compilation, and uniform management
 */
class Shader
{
  private:
    GLuint program_id;
    bool is_compiled;
    mutable std::unordered_map<std::string, GLint> uniform_cache;

  public:
    Shader();
    ~Shader();

    // Disable copy constructor and assignment operator
    Shader(const Shader &) = delete;
    Shader &operator=(const Shader &) = delete;

    // Move constructor and assignment operator
    Shader(Shader &&other) noexcept;
    Shader &operator=(Shader &&other) noexcept;

    // Shader loading methods
    bool loadFromFiles(const std::string &vertex_path, const std::string &fragment_path);
    bool loadFromFiles(const std::string &vertex_path, const std::string &fragment_path,
                       const std::string &geometry_path);
    bool loadFromSource(const std::string &vertex_source, const std::string &fragment_source);
    bool loadFromSource(const std::string &vertex_source, const std::string &fragment_source,
                        const std::string &geometry_source);

    // Shader usage
    void use() const;
    void unbind() const;

    // Uniform setters
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec2(const std::string &name, const glm::vec2 &value) const;
    void setVec2(const std::string &name, float x, float y) const;
    void setVec3(const std::string &name, const glm::vec3 &value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setVec4(const std::string &name, float x, float y, float z, float w) const;
    void setMat2(const std::string &name, const glm::mat2 &value) const;
    void setMat3(const std::string &name, const glm::mat3 &value) const;
    void setMat4(const std::string &name, const glm::mat4 &value) const;

    // Array uniform setters
    void setFloatArray(const std::string &name, const float *values, int count) const;
    void setVec3Array(const std::string &name, const glm::vec3 *values, int count) const;
    void setMat4Array(const std::string &name, const glm::mat4 *values, int count) const;

    // Utility
    GLuint getID() const
    {
        return program_id;
    }
    bool isCompiled() const
    {
        return is_compiled;
    }
    void cleanup();

    // Attribute and uniform locations
    GLint getAttributeLocation(const std::string &name) const;
    GLint getUniformLocation(const std::string &name) const;

    // Shader introspection
    void printActiveUniforms() const;
    void printActiveAttributes() const;

    // Static utility methods
    static std::string loadShaderSource(const std::string &file_path);
    static bool checkCompileErrors(GLuint shader, const std::string &type);
    static bool checkLinkErrors(GLuint program);

  private:
    // Compilation helpers
    GLuint compileShader(const std::string &source, GLenum shader_type);
    bool linkProgram(GLuint vertex_shader, GLuint fragment_shader, GLuint geometry_shader = 0);

    // Uniform location caching
    GLint getUniformLocationCached(const std::string &name) const;
};

/**
 * Shader library for managing commonly used shaders
 */
class ShaderLibrary
{
  private:
    static std::unordered_map<std::string, std::unique_ptr<Shader>> shaders;

  public:
    // Load predefined shaders
    static bool loadDefaultShaders();

    // Shader management
    static Shader *getShader(const std::string &name);
    static bool addShader(const std::string &name, std::unique_ptr<Shader> shader);
    static void removeShader(const std::string &name);
    static void cleanup();

    // Predefined shader creation
    static std::unique_ptr<Shader> createParticleShader();
    static std::unique_ptr<Shader> createSphereShader();
    static std::unique_ptr<Shader> createGridShader();
    static std::unique_ptr<Shader> createBoundaryShader();
    static std::unique_ptr<Shader> createFluidSurfaceShader();
};

} // namespace FluidSim
