cmake_minimum_required(VERSION 3.14)
project(GPUTest
        LANGUAGES CXX 
        VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
find_package(MPI REQUIRED)
find_package(OpenACC REQUIRED)
find_package(DataSpaces REQUIRED)

add_executable(gpu_put gpu_put.cpp)
target_link_libraries(gpu_put PUBLIC MPI::MPI_CXX)
target_compile_options(gpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(gpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(gpu_put PUBLIC DataSpaces::DataSpaces)

add_executable(gpu_cpu_put gpu_cpu_put.cpp)
target_link_libraries(gpu_cpu_put PUBLIC MPI::MPI_CXX)
target_compile_options(gpu_cpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(gpu_cpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(gpu_cpu_put PUBLIC DataSpaces::DataSpaces)

add_executable(gpu_cpu_put_local gpu_cpu_put_local.cpp)
target_link_libraries(gpu_cpu_put_local PUBLIC MPI::MPI_CXX)
target_compile_options(gpu_cpu_put_local PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(gpu_cpu_put_local PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(gpu_cpu_put_local PUBLIC DataSpaces::DataSpaces)

add_executable(cpu_put cpu_put.cpp)
target_link_libraries(cpu_put PUBLIC MPI::MPI_CXX)
target_compile_options(cpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(cpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
target_link_libraries(cpu_put PUBLIC DataSpaces::DataSpaces)

add_executable(cpu_get cpu_get.cpp)
target_link_libraries(cpu_get PUBLIC MPI::MPI_CXX)
target_link_libraries(cpu_get PUBLIC DataSpaces::DataSpaces)