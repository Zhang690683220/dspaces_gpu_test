#ifndef CUDA_PUT_HPP
#define CUDA_PUT_HPP

#include <iostream>
#include <cstring>
#include "unistd.h"
#include "mpi.h"
#include "dspaces.h"
#include "timer.hpp"
#include "cuda_runtime.h"

/*
int timer_cb(dspaces_client_t client, struct dspaces_req* req, void* timer) {
    Timer* timer_ptr = (Timer*) timer;
    double put_time_async = timer_ptr->stop();
    std::cout<< "DSPACES_CPU_PUT() Version = "<< req->ver << " TIME(Sync) = " << put_time_async << "(ms)" << std::endl;
    return 0;
}
*/

template <typename Data_t>
struct CUDA {
    __global__ void assign(Data_t *ptr, int size, int var_idx);
};

template <>
struct CUDA <double> {
__global__ void assign(double *ptr, int size, int var_idx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        ptr[idx] = idx + 0.01*var_idx;
    }
}
};


template <typename Data_t>
struct Run {
    static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate);
};


struct Run <double> {
static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    char* listen_addr_str = NULL;
    if(!listen_addr.empty()) {
        listen_addr_str = (char*) malloc(sizeof(char)*128);
        strcpy(listen_addr_str, listen_addr.c_str());
    }
    dspaces_init(rank, &ndcl);

    uint64_t grid_size = 1;
    for(int i=0; i<dims; i++) {
        grid_size *= sp[i];
    }

    uint64_t* off = (uint64_t*) malloc(dims*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(dims*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(dims*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<dims; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    int dev_num, dev_rank;
    cudaError_t cuda_status;
    cudaDeviceProp dev_prop;
    cuda_status = cudaGetDeviceCount(&dev_num);
    dev_rank = rank%dev_num;
    cuda_status = cudaSetDevice(dev_rank);
    cuda_status = cudaGetDeviceProperties(&dev_prop,dev_rank);

    int threadsPerBlock = dev_prop.maxThreadsPerBlock;
    int numBlocks = (grid_size + threadsPerBlock) / threadsPerBlock;

    // same init data for each var at host
    //double *data_tab_h = (double *) malloc(sizeof(double) * grid_size);

    // n vars at device
    double **data_tab_d = (double **) malloc(sizeof(double*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        cuda_status = cudaMalloc((void**)&data_tab_d[i],sizeof(double) * grid_size);
        var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
        sprintf(var_name_tab[i], "test_var_%d", i);
    }

    
    std::ofstream log;
    double* avg_put = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_put = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, put_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // emulate computing time
        sleep(delay);

        // output every $interval timesteps from timestep 1
        if((ts-1)%interval==0) {

            for(int i=0; i<var_num; i++) {
                CUDA<double>::assign<<numBlocks, threadIdx>>(data_tab_d[i], grid_size, i);
            }

            // wait device to finish
            cuda_status = cudaThreadSynchronize();

            Timer timer_put;
            timer_put.start();
            for(int i=0; i<var_num; i++) {
                dspaces_cuda_put(ndcl, var_name_tab[i], ts, sizeof(double), dims, lb, ub, data_tab_d[i]);
            }
            double time_put = timer_put.stop();

            double *avg_time_put = nullptr;

            if(rank == 0) {
                avg_time_put = (double*) malloc(sizeof(double)*nprocs);
            }

            MPI_Gather(&time_put, 1, MPI_DOUBLE, avg_time_put, 1, MPI_DOUBLE, 0, gcomm);

            if(rank == 0) {
                for(int i=0; i<nprocs; i++) {
                    avg_put[ts-1] += avg_time_put[i];
                }
                avg_put[ts-1] /= nprocs;
                log << ts << ", " << avg_put[ts-1] << std::endl;
                total_avg += avg_put[ts-1];
                free(avg_time_put);
            }
        }
    }

    for(int i=0; i<var_num; i++) {
        cuda_status = cudaFree(data_tab[i]);
        free(var_name_tab[i]);
    }
    free(data_tab_d);
    free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);
    free(listen_addr_str);

    if(rank == 0) {
        total_avg /= (timesteps/interval);
        log << "Total" << ", " << total_avg << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            dspaces_kill(ndcl);
        }
    }

    dspaces_fini(ndcl);

    return 0;

}
};



#endif // CUDA_PUT_HPP

