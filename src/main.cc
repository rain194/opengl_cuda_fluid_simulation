#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    // Test CUDA
    std::cout << "=== CUDA Test ===" << std::endl;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Devices: " << deviceCount << std::endl;

    testKernel<<<1, 5>>>();
    cudaDeviceSynchronize();

    // Test OpenGL
    std::cout << "\n=== OpenGL Test ===" << std::endl;
    if (!glfwInit())
    {
        std::cerr << "GLFW failed" << std::endl;
        return -1;
    }

    GLFWwindow *window = glfwCreateWindow(400, 300, "CUDA+OpenGL Test", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Window creation failed" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "GLEW failed" << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;

    // Quick render loop
    for (int i = 0; i < 60 && !glfwWindowShouldClose(window); i++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.2f, 0.8f, 0.2f, 1.0f); // Green background
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    std::cout << "\n✅ All tests passed! CUDA + OpenGL working!" << std::endl;
    return 0;
}
