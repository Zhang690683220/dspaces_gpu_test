#ifndef CUDA_PUT_HPP
#define CUDA_PUT_HPP

#include <cstring>
#include <string>
#include <vector>
#include "unistd.h"
#include "mpi.h"


/*
int timer_cb(dspaces_client_t client, struct dspaces_req* req, void* timer) {
    Timer* timer_ptr = (Timer*) timer;
    double put_time_async = timer_ptr->stop();
    std::cout<< "DSPACES_CPU_PUT() Version = "<< req->ver << " TIME(Sync) = " << put_time_async << "(ms)" << std::endl;
    return 0;
}
*/

template <typename Data_t>
struct Run {
    static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate);
};


template <>
struct Run <double> {
static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name, bool terminate);
};


#endif // CUDA_PUT_HPP

