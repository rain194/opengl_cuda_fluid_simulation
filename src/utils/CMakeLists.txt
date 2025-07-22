# src/utils/CMakeLists.txt

add_library(utils_lib STATIC
    cuda_helper.cc
    config.cc
    # math_utils.hh is header-only
    # config.hh is header-only
)

target_include_directories(utils_lib PUBLIC .)

# Link CUDA if available
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(utils_lib PUBLIC CUDA::cudart)
    set_target_properties(utils_lib PROPERTIES CUDA_ARCHITECTURES 75)
endif()

target_link_libraries(utils_lib PUBLIC glm::glm)
