# src/physics/CMakeLists.txt

add_library(physics_lib STATIC
    forces.cc
    collision.cc
    cuda_kernels.cu
)

target_include_directories(physics_lib PUBLIC .)

# Link dependencies
target_link_libraries(physics_lib
    PUBLIC
    utils_lib
    glm::glm
)

# CUDA specific settings
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(physics_lib PUBLIC CUDA::cudart)
    set_target_properties(physics_lib PROPERTIES CUDA_ARCHITECTURES 75)
endif()
