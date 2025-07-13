# Windows CUDA + OpenGL Development Setup Guide

## Prerequisites
- Windows 11
- NVIDIA GPU with drivers installed
- Administrator access
- ~8GB free disk space

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
3. Skip full Visual Studio IDE

### Step 3: Configure Environment Variables
1. Open **Developer Command Prompt for VS 2022**
2. Run `where cl.exe` to find compiler path
3. Add to Windows PATH via System Properties → Environment Variables:
   - Compiler directory (e.g., `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\[version]\bin\Hostx64\x64`)
   - MSBuild path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin`

### Step 4: Install Package Manager and Libraries
```cmd
git clone https://github.com/Microsoft/vcpkg.git C:\Libs\vcpkg
cd C:\Libs\vcpkg
setx VCPKG_DISABLE_METRICS 1
.\bootstrap-vcpkg.bat
.\vcpkg install glfw3:x64-windows glew:x64-windows glm:x64-windows
```

### Step 5: Verification
Test installation:
```cmd
cl.exe
nvcc --version
cmake --version
```

## CMake Integration
Add to CMakeLists.txt:
```cmake
set(CMAKE_TOOLCHAIN_FILE "C:/Libs/vcpkg/scripts/buildsystems/vcpkg.cmake")
```
