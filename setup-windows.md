# Windows CUDA + OpenGL Development Setup Guide

## Prerequisites
- Windows 11
- NVIDIA GPU with drivers installed
- Administrator access
- 8GB free disk space

## Installation Steps

### Step 1: Install CUDA Toolkit
Check if already installed:
```cmd
nvcc --version
```
If not present, download **CUDA Toolkit 12.6** from NVIDIA (~3-4GB).

### Step 2: Install Build Tools for Visual Studio
1. Download **"Build Tools for Visual Studio 2022"** from Microsoft
2. Select **"C++ build tools"** workload only (~3-5GB)
3. Skip full Visual Studio IDE if not needed.

### Step 3: Fix CUDA Integration (Critical Step)
Copy CUDA Visual Studio integration files:
Might have to do it manually if this command didn't work.
```cmd
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions\*" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\"
```

### Step 4: Configure Environment Variables
1. Open **Developer Command Prompt for VS 2022**
2. Run `where cl.exe` to find compiler path
3. Add to Windows PATH via System Properties → Environment Variables:
   - Compiler directory (e.g., `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\[version]\bin\Hostx64\x64`)
   - MSBuild path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin`
4. Set CUDA environment variables:
```cmd
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
```

### Step 5: Install Package Manager and Libraries
```cmd
git clone https://github.com/Microsoft/vcpkg.git C:\Libs\vcpkg
cd C:\Libs\vcpkg
setx VCPKG_DISABLE_METRICS 1
.\bootstrap-vcpkg.bat
.\vcpkg install glfw3:x64-windows glew:x64-windows glm:x64-windows
```

### Step 6: Verification
Test installation:
```cmd
cl.exe
nvcc --version
```

## CMake Project Setup
CMakeLists.txt template:
```cmake
cmake_minimum_required(VERSION 3.18)
set(CMAKE_TOOLCHAIN_FILE "C:/Libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
project(ProjectName LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/main.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 75)
target_link_libraries(${PROJECT_NAME} OpenGL::GL glfw GLEW::GLEW glm::glm)
```

**Build commands:**
```cmd
mkdir build && cd build
cmake ..
cmake --build .
```

**Total Installation Size: ~8GB**
- CUDA Toolkit: 3-4GB
- Build Tools: 3-5GB  
- vcpkg + Libraries: 300MB

**Note:** The CUDA integration file copy (Step 3) is essential - without it, CMake will fail with "No CUDA toolset found" error.
