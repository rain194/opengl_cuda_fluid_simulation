# src/shaders/CMakeLists.txt

# Copy shader files to build directory
set(SHADER_FILES
    particle.vert
    particle.frag
    fluid.vert
    fluid.frag
    grid.vert
    grid.frag
    boundary.vert
    boundary.frag
    sphere.vert
    sphere.frag
)

# Create shaders directory in build
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

# Copy each shader file to the correct location
foreach(SHADER ${SHADER_FILES})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        ${CMAKE_BINARY_DIR}/shaders/${SHADER}
        COPYONLY
    )
endforeach()

# Also copy to Debug/Release directories for Visual Studio
foreach(SHADER ${SHADER_FILES})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        ${CMAKE_BINARY_DIR}/src/Debug/shaders/${SHADER}
        COPYONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        ${CMAKE_BINARY_DIR}/src/Release/shaders/${SHADER}
        COPYONLY
    )
endforeach()

# Create empty library (just for organization)
add_library(shaders_lib INTERFACE)
target_include_directories(shaders_lib INTERFACE .)