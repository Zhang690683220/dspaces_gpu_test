add_executable(cpu_get cpu_get.cpp)
target_include_directories(cpu_get PRIVATE ${INCLUDE_PATH})
target_link_libraries(cpu_get PUBLIC MPI::MPI_CXX)
target_link_libraries(cpu_get PUBLIC DataSpaces::DataSpaces)