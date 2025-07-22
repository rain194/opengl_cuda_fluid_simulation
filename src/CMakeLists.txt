# src/CMakeLists.txt

# Add all module subdirectories (order matters for dependencies)
add_subdirectory(utils)      # No dependencies
add_subdirectory(shaders)    # No dependencies
add_subdirectory(physics)    # Depends on utils
add_subdirectory(core)       # Depends on utils, physics
add_subdirectory(rendering)  # Depends on utils

# Create main executable
add_executable(${PROJECT_NAME} main.cc)

# Link all static libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    core_lib
    physics_lib
    rendering_lib
    utils_lib
    OpenGL::GL
    glfw
    GLEW::GLEW
    glm::glm
)
# Set CUDA architecture
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 75)
