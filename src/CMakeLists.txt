# add_executable(cpu_put cpu_put.cpp)
# target_include_directories(cpu_put PRIVATE ${INCLUDE_PATH})
# target_link_libraries(cpu_put PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces)


# add_executable(cpu_get cpu_get.cpp)
# target_include_directories(cpu_get PRIVATE ${INCLUDE_PATH})
# target_link_libraries(cpu_get PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces)

# add_executable(stdio_put stdio_put.cpp)
# target_include_directories(stdio_put PRIVATE ${INCLUDE_PATH})
# target_link_libraries(stdio_put PUBLIC MPI::MPI_CXX)

# add_executable(stdio_get stdio_get.cpp)
# target_include_directories(stdio_get PRIVATE ${INCLUDE_PATH})
# target_link_libraries(stdio_get PUBLIC MPI::MPI_CXX)

# add_executable(gpu_cpu_put gpu_cpu_put.cpp)
# target_include_directories(gpu_cpu_put PRIVATE ${INCLUDE_PATH})
# target_link_libraries(gpu_cpu_put PUBLIC MPI::MPI_CXX)
# target_compile_options(gpu_cpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
# target_link_libraries(gpu_cpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
# target_link_libraries(gpu_cpu_put PUBLIC DataSpaces::DataSpaces)

# add_executable(gpu_cpu_put_local gpu_cpu_put_local.cpp)
# target_include_directories(gpu_cpu_put_local PRIVATE ${INCLUDE_PATH})
# target_link_libraries(gpu_cpu_put_local PUBLIC MPI::MPI_CXX)
# target_compile_options(gpu_cpu_put_local PUBLIC ${OpenACC_CXX_OPTIONS})
# target_link_libraries(gpu_cpu_put_local PUBLIC ${OpenACC_CXX_OPTIONS})
# target_link_libraries(gpu_cpu_put_local PUBLIC DataSpaces::DataSpaces)

# add_executable(gpu_put gpu_put.cpp ACCH.cpp)
# target_link_libraries(gpu_put PUBLIC MPI::MPI_CXX)
# target_compile_options(gpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
# target_link_libraries(gpu_put PUBLIC ${OpenACC_CXX_OPTIONS})
# target_include_directories(gpu_put PRIVATE ${INCLUDE_PATH})
# target_link_libraries(gpu_put PUBLIC DataSpaces::DataSpaces)

option(CASPER OFF)
if(CASPER)
    add_definitions(-DCASPER)
endif()

add_executable(cuda_put cuda_put.cpp)
set_source_files_properties(cuda_put.cuh PROPERTIES LANGUAGE "CUDA")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_include_directories(cuda_put PRIVATE ${INCLUDE_PATH} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(cuda_put PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces
                            CUDA::cudart CUDA::cuda_driver)
else()
    target_include_directories(cuda_put PRIVATE ${INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cuda_put PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces
                            ${CUDA_LIBRARIES})
endif()

add_executable(cuda_get cuda_get.cpp)
set_source_files_properties(cuda_get.cuh PROPERTIES LANGUAGE "CUDA")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_include_directories(cuda_get PRIVATE ${INCLUDE_PATH} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(cuda_get PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces
                            CUDA::cudart CUDA::cuda_driver)
else()
    target_include_directories(cuda_get PRIVATE ${INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cuda_get PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces
                            ${CUDA_LIBRARIES})
endif()