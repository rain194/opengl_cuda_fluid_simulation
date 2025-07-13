cmake_minimum_required(VERSION 3.18)

set(CMAKE_TOOLCHAIN_FILE "C:/Libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

project(FluidSimulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(${PROJECT_NAME} 
    src/main.cu
    # Add CUDA files here later: src/fluid.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 75)

target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL 
    glfw 
    GLEW::GLEW 
    glm::glm
)
