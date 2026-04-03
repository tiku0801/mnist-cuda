cmake_minimum_required(VERSION 3.20)

project(MNIST_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86)

option(ENABLE_TEST ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS 
    "src/*.cpp" 
    "src/*.cu"
)

add_library(core_lib STATIC ${SRC_FILES})

target_include_directories(core_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(core_lib PUBLIC CUDA::cublas CUDA::cublasLt)


if(ENABLE_TEST)
    add_subdirectory(tests)
endif()