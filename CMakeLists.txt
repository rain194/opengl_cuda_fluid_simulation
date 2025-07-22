# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
set(CMAKE_TOOLCHAIN_FILE "C:/Libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

project(FluidSimulation LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)

# Add source directory
add_subdirectory(src)
