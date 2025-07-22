#include "shader.hh"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

namespace FluidSim
{

// Static member initialization
std::unordered_map<std::string, std::unique_ptr<Shader>> ShaderLibrary::shaders;

// Shader Implementation
Shader::Shader() : program_id(0), is_compiled(false)
{
}

Shader::~Shader()
{
    cleanup();
}

Shader::Shader(Shader &&other) noexcept
    : program_id(other.program_id), is_compiled(other.is_compiled), uniform_cache(std::move(other.uniform_cache))
{
    other.program_id = 0;
    other.is_compiled = false;
}

Shader &Shader::operator=(Shader &&other) noexcept
{
    if (this != &other)
    {
        cleanup();
        program_id = other.program_id;
        is_compiled = other.is_compiled;
        uniform_cache = std::move(other.uniform_cache);

        other.program_id = 0;
        other.is_compiled = false;
    }
    return *this;
}

bool Shader::loadFromFiles(const std::string &vertex_path, const std::string &fragment_path)
{
    std::string vertex_source = loadShaderSource(vertex_path);
    std::string fragment_source = loadShaderSource(fragment_path);

    if (vertex_source.empty() || fragment_source.empty())
    {
        std::cerr << "Failed to load shader files: " << vertex_path << ", " << fragment_path << std::endl;
        return false;
    }

    return loadFromSource(vertex_source, fragment_source);
}

bool Shader::loadFromFiles(const std::string &vertex_path, const std::string &fragment_path,
                           const std::string &geometry_path)
{
    std::string vertex_source = loadShaderSource(vertex_path);
    std::string fragment_source = loadShaderSource(fragment_path);
    std::string geometry_source = loadShaderSource(geometry_path);

    if (vertex_source.empty() || fragment_source.empty() || geometry_source.empty())
    {
        std::cerr << "Failed to load shader files" << std::endl;
        return false;
    }

    return loadFromSource(vertex_source, fragment_source, geometry_source);
}

bool Shader::loadFromSource(const std::string &vertex_source, const std::string &fragment_source)
{
    // Compile vertex shader
    GLuint vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER);
    if (vertex_shader == 0)
    {
        std::cerr << "Failed to compile vertex shader" << std::endl;
        return false;
    }

    // Compile fragment shader
    GLuint fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER);
    if (fragment_shader == 0)
    {
        std::cerr << "Failed to compile fragment shader" << std::endl;
        glDeleteShader(vertex_shader);
        return false;
    }

    // Link program
    bool success = linkProgram(vertex_shader, fragment_shader);

    // Clean up individual shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    is_compiled = success;
    return success;
}

bool Shader::loadFromSource(const std::string &vertex_source, const std::string &fragment_source,
                            const std::string &geometry_source)
{
    // Compile shaders
    GLuint vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER);
    GLuint fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER);
    GLuint geometry_shader = compileShader(geometry_source, GL_GEOMETRY_SHADER);

    if (vertex_shader == 0 || fragment_shader == 0 || geometry_shader == 0)
    {
        if (vertex_shader)
            glDeleteShader(vertex_shader);
        if (fragment_shader)
            glDeleteShader(fragment_shader);
        if (geometry_shader)
            glDeleteShader(geometry_shader);
        return false;
    }

    // Link program
    bool success = linkProgram(vertex_shader, fragment_shader, geometry_shader);

    // Clean up individual shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    glDeleteShader(geometry_shader);

    is_compiled = success;
    return success;
}

GLuint Shader::compileShader(const std::string &source, GLenum shader_type)
{
    GLuint shader = glCreateShader(shader_type);
    const char *source_cstr = source.c_str();
    glShaderSource(shader, 1, &source_cstr, NULL);
    glCompileShader(shader);

    std::string type_name;
    switch (shader_type)
    {
    case GL_VERTEX_SHADER:
        type_name = "VERTEX";
        break;
    case GL_FRAGMENT_SHADER:
        type_name = "FRAGMENT";
        break;
    case GL_GEOMETRY_SHADER:
        type_name = "GEOMETRY";
        break;
    default:
        type_name = "UNKNOWN";
        break;
    }

    if (!checkCompileErrors(shader, type_name))
    {
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

bool Shader::linkProgram(GLuint vertex_shader, GLuint fragment_shader, GLuint geometry_shader)
{
    // Clean up previous program
    cleanup();

    // Create program
    program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader);
    glAttachShader(program_id, fragment_shader);
    if (geometry_shader != 0)
    {
        glAttachShader(program_id, geometry_shader);
    }

    glLinkProgram(program_id);

    if (!checkLinkErrors(program_id))
    {
        glDeleteProgram(program_id);
        program_id = 0;
        return false;
    }

    return true;
}

void Shader::use() const
{
    if (is_compiled && program_id != 0)
    {
        glUseProgram(program_id);
    }
}

void Shader::unbind() const
{
    glUseProgram(0);
}

void Shader::cleanup()
{
    if (program_id != 0)
    {
        glDeleteProgram(program_id);
        program_id = 0;
    }
    is_compiled = false;
    uniform_cache.clear();
}

// Uniform setters
void Shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(getUniformLocationCached(name), (int)value);
}

void Shader::setInt(const std::string &name, int value) const
{
    glUniform1i(getUniformLocationCached(name), value);
}

void Shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(getUniformLocationCached(name), value);
}

void Shader::setVec2(const std::string &name, const glm::vec2 &value) const
{
    glUniform2fv(getUniformLocationCached(name), 1, &value[0]);
}

void Shader::setVec2(const std::string &name, float x, float y) const
{
    glUniform2f(getUniformLocationCached(name), x, y);
}

void Shader::setVec3(const std::string &name, const glm::vec3 &value) const
{
    glUniform3fv(getUniformLocationCached(name), 1, &value[0]);
}

void Shader::setVec3(const std::string &name, float x, float y, float z) const
{
    glUniform3f(getUniformLocationCached(name), x, y, z);
}

void Shader::setVec4(const std::string &name, const glm::vec4 &value) const
{
    glUniform4fv(getUniformLocationCached(name), 1, &value[0]);
}

void Shader::setVec4(const std::string &name, float x, float y, float z, float w) const
{
    glUniform4f(getUniformLocationCached(name), x, y, z, w);
}

void Shader::setMat2(const std::string &name, const glm::mat2 &value) const
{
    glUniformMatrix2fv(getUniformLocationCached(name), 1, GL_FALSE, &value[0][0]);
}

void Shader::setMat3(const std::string &name, const glm::mat3 &value) const
{
    glUniformMatrix3fv(getUniformLocationCached(name), 1, GL_FALSE, &value[0][0]);
}

void Shader::setMat4(const std::string &name, const glm::mat4 &value) const
{
    glUniformMatrix4fv(getUniformLocationCached(name), 1, GL_FALSE, &value[0][0]);
}

void Shader::setFloatArray(const std::string &name, const float *values, int count) const
{
    glUniform1fv(getUniformLocationCached(name), count, values);
}

void Shader::setVec3Array(const std::string &name, const glm::vec3 *values, int count) const
{
    glUniform3fv(getUniformLocationCached(name), count, glm::value_ptr(values[0]));
}

void Shader::setMat4Array(const std::string &name, const glm::mat4 *values, int count) const
{
    glUniformMatrix4fv(getUniformLocationCached(name), count, GL_FALSE, glm::value_ptr(values[0]));
}

GLint Shader::getUniformLocationCached(const std::string &name) const
{
    auto it = uniform_cache.find(name);
    if (it != uniform_cache.end())
    {
        return it->second;
    }

    GLint location = glGetUniformLocation(program_id, name.c_str());
    uniform_cache[name] = location;

    if (location == -1)
    {
        std::cerr << "Warning: Uniform '" << name << "' not found in shader" << std::endl;
    }

    return location;
}

GLint Shader::getAttributeLocation(const std::string &name) const
{
    return glGetAttribLocation(program_id, name.c_str());
}

GLint Shader::getUniformLocation(const std::string &name) const
{
    return glGetUniformLocation(program_id, name.c_str());
}

// Static utility methods
std::string Shader::loadShaderSource(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open shader file: " << file_path << std::endl;
        return "";
    }

    std::stringstream stream;
    stream << file.rdbuf();
    file.close();

    return stream.str();
}

bool Shader::checkCompileErrors(GLuint shader, const std::string &type)
{
    GLint success;
    GLchar info_log[1024];

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 1024, NULL, info_log);
        std::cerr << "Shader compilation error (" << type << "):\n" << info_log << std::endl;
        return false;
    }

    return true;
}

bool Shader::checkLinkErrors(GLuint program)
{
    GLint success;
    GLchar info_log[1024];

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 1024, NULL, info_log);
        std::cerr << "Shader linking error:\n" << info_log << std::endl;
        return false;
    }

    return true;
}

void Shader::printActiveUniforms() const
{
    if (!is_compiled)
        return;

    GLint count;
    glGetProgramiv(program_id, GL_ACTIVE_UNIFORMS, &count);

    std::cout << "Active uniforms (" << count << "):" << std::endl;

    for (GLint i = 0; i < count; i++)
    {
        char name[256];
        GLsizei length;
        GLint size;
        GLenum type;

        glGetActiveUniform(program_id, i, sizeof(name), &length, &size, &type, name);
        GLint location = glGetUniformLocation(program_id, name);

        std::cout << "  " << i << ": " << name << " (location=" << location << ")" << std::endl;
    }
}

// ShaderLibrary Implementation
bool ShaderLibrary::loadDefaultShaders()
{
    bool success = true;

    // Create particle shader
    auto particle_shader = createParticleShader();
    if (particle_shader && particle_shader->isCompiled())
    {
        addShader("particle", std::move(particle_shader));
        std::cout << "Loaded particle shader" << std::endl;
    }
    else
    {
        std::cerr << "Failed to load particle shader" << std::endl;
        success = false;
    }

    // Create sphere shader
    auto sphere_shader = createSphereShader();
    if (sphere_shader && sphere_shader->isCompiled())
    {
        addShader("sphere", std::move(sphere_shader));
        std::cout << "Loaded sphere shader" << std::endl;
    }
    else
    {
        std::cout << "Sphere shader not available, using particle shader fallback" << std::endl;
    }

    // Create grid shader
    auto grid_shader = createGridShader();
    if (grid_shader && grid_shader->isCompiled())
    {
        addShader("grid", std::move(grid_shader));
        std::cout << "Loaded grid shader" << std::endl;
    }

    return success;
}

Shader *ShaderLibrary::getShader(const std::string &name)
{
    auto it = shaders.find(name);
    return (it != shaders.end()) ? it->second.get() : nullptr;
}

bool ShaderLibrary::addShader(const std::string &name, std::unique_ptr<Shader> shader)
{
    if (!shader)
        return false;

    shaders[name] = std::move(shader);
    return true;
}

void ShaderLibrary::removeShader(const std::string &name)
{
    shaders.erase(name);
}

void ShaderLibrary::cleanup()
{
    shaders.clear();
}

std::unique_ptr<Shader> ShaderLibrary::createParticleShader()
{
    auto shader = std::make_unique<Shader>();

    std::string vertex_source = R"(
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 color;
        
        uniform mat4 view;
        uniform mat4 projection;
        uniform float particle_size;
        
        out vec3 particle_color;
        
        void main() {
            gl_Position = projection * view * vec4(position, 1.0);
            gl_PointSize = particle_size;
            particle_color = color;
        }
    )";

    std::string fragment_source = R"(
        #version 330 core
        in vec3 particle_color;
        out vec4 FragColor;
        
        uniform vec3 base_color;
        
        void main() {
            FragColor = vec4(particle_color * base_color, 1.0);
        }
    )";

    if (shader->loadFromSource(vertex_source, fragment_source))
    {
        return shader;
    }

    return nullptr;
}

std::unique_ptr<Shader> ShaderLibrary::createGridShader()
{
    auto shader = std::make_unique<Shader>();

    std::string vertex_source = R"(
        #version 330 core
        layout (location = 0) in vec3 position;
        
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * view * vec4(position, 1.0);
        }
    )";

    std::string fragment_source = R"(
        #version 330 core
        out vec4 FragColor;
        
        uniform vec3 color;
        
        void main() {
            FragColor = vec4(color, 0.3);
        }
    )";

    if (shader->loadFromSource(vertex_source, fragment_source))
    {
        return shader;
    }

    return nullptr;
}

std::unique_ptr<Shader> ShaderLibrary::createSphereShader()
{
    // Advanced sphere shader with lighting - optional for now
    return nullptr;
}

} // namespace FluidSim
