# src/core/CMakeLists.txt

add_library(core_lib STATIC
    fluid_sim.cc
    fluid_sim.cu
    # particle.hh is header-only
)

target_include_directories(core_lib PUBLIC .)

# Link dependencies
target_link_libraries(core_lib
    PUBLIC
    utils_lib
    physics_lib
    glm::glm
)

# CUDA specific settings
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(core_lib PUBLIC CUDA::cudart)
    set_target_properties(core_lib PROPERTIES CUDA_ARCHITECTURES 75)
endif()
