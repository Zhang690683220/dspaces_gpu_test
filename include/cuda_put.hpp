#ifndef CUDA_PUT_HPP
#define CUDA_PUT_HPP

#include <cstring>
#include <string>
#include <vector>
#include "unistd.h"
#include "mpi.h"


template <typename Data_t>
struct Run {
    static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate);
};


#endif // CUDA_PUT_HPP

