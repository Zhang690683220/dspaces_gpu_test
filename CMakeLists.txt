cmake_minimum_required(VERSION 3.16)
project(GPUTest
        LANGUAGES CXX 
        VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
find_package(MPI REQUIRED)
# find_package(OpenACC REQUIRED)
find_package(DataSpaces REQUIRED)
find_package(CUDAToolkit REQUIRED)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
