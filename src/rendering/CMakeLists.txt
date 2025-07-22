# src/rendering/CMakeLists.txt

add_library(rendering_lib STATIC
    renderer.cc
    shader.cc
    buffer_manager.cc
)

target_include_directories(rendering_lib PUBLIC .)

# Link dependencies
target_link_libraries(rendering_lib
    PUBLIC
    utils_lib
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm::glm
)

# CUDA specific settings for OpenGL interop
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(rendering_lib PUBLIC CUDA::cudart)
endif()
