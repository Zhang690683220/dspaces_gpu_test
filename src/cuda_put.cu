#include <cstring>
#include <string>
#include <vector>
#include "unistd.h"
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "dspaces.h"
#include "timer.hpp"
#include "cuda_runtime.h"

#include "CLI11.hpp"


template <typename Data_t>
__global__ void assign(Data_t *ptr, int size, int var_idx);


template <>
__global__ void assign<double>(double *ptr, int size, int var_idx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        ptr[idx] = idx + 0.01*var_idx;
    }
}

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
#ifdef CASPER
    dspaces_init(rank, &ndcl, listen_addr_str);
#else
    dspaces_init(rank, &ndcl);
#endif
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
                assign<double><<<numBlocks, threadsPerBlock>>>(data_tab_d[i], grid_size, i);
            }

            // wait device to finish
            cuda_status = cudaDeviceSynchronize();

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
        cuda_status = cudaFree(data_tab_d[i]);
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

    return cuda_status;

}
};

void print_usage()
{
    std::cerr<<"Usage: cuda_put --dims <dims> --np <np[0] .. np[dims-1]> --sp <sp[0] ... sp[dims-1]> "
               "--ts <timesteps> [-s <elem_size>] [-c <var_count>] [--log <log_file>] [--delay <delay_second>] "
               "[--interval <output_freq>] [-t]"<<std::endl
             <<"--dims                      - number of data dimensions. Must be [1-8]"<<std::endl
             <<"--np                        - the number of processes in the ith dimension. "
               "The product of np[0],...,np[dim-1] must be the number of MPI ranks"<<std::endl
             <<"--sp                        - the per-process data size in the ith dimension"<<std::endl
             <<"--ts                        - the number of timestep iterations written"<<std::endl
             <<"-l, --listen_addr (optional)- listen address of the mercury network. Default to be "
               "the same as server's address"<<std::endl
             <<"-t, --type (optional)       - type of each element [float|double]. Defaults to double"<<std::endl
             <<"-c, --var_count (optional)  - the number of variables written in each iteration. "
               "Defaults to 1"<<std::endl
             <<"--log (optional)            - output log file name. Default to cpu_put.log"<<std::endl
             <<"--delay (optional)          - sleep(delay) seconds in each timestep. Default to 0"<<std::endl
             <<"--interval (optional)       - Output timestep interval. Default to 1"<<std::endl
             <<"-k (optional)               - send server kill signal after writing is complete"<<std::endl;
}



int main(int argc, char* argv[]) {

    CLI::App app{"CUDA PUT Emulator for DataSpaces"};
    int dims;              // number of dimensions
    std::vector<int> np;
    std::vector<uint64_t> sp;
    int timestep;
    std::string listen_addr;
    int elem_type = 1;
    int num_vars = 1;
    std::string log_name = "cuda_put.log";
    int delay = 0;
    int interval = 1;
    bool terminate = false;
    app.add_option("--dims", dims, "number of data dimensions. Must be [1-8]")->required();
    app.add_option("--np", np, "the number of processes in the ith dimension. The product of np[0],"
                    "...,np[dim-1] must be the number of MPI ranks")->expected(1, 8);
    app.add_option("--sp", sp, "the per-process data size in the ith dimension")->expected(1, 8);
    app.add_option("--ts", timestep, "the number of timestep iterations")->required();
    app.add_option("-l, --listen_addr", listen_addr, "listen address of the mercury network");
    app.add_option("-t, --type", elem_type, "type of each element [float|double]. Defaults to double",
                    true)->transform(CLI::CheckedTransformer(std::map<std::string, int>({{"double", 1},
                    {"float", 2}})));
    app.add_option("-c, --var_count", num_vars, "the number of variables written in each iteration."
                    "Defaults to 1", true);
    app.add_option("--log", log_name, "output log file name. Default to cpu_put.log", true);
    app.add_option("--delay", delay, "sleep(delay) seconds in each timestep. Default to 0", true);
    app.add_option("--interval", interval, "Output timestep interval. Default to 1", true);
    app.add_flag("-k", terminate, "send server kill signal after writing is complete");

    CLI11_PARSE(app, argc, argv);

    int npapp = 1;             // number of application processes
    for(int i = 0; i < dims; i++) {
        npapp *= np[i];
    }

    int nprocs, rank;
    MPI_Comm gcomm;
    // Using SPMD style programming
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    gcomm = MPI_COMM_WORLD;

    int color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &gcomm);

    if(npapp != nprocs) {
        std::cerr<<"Product of np[i] args must equal number of MPI processes!"<<std::endl;
        print_usage();
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
    switch (elem_type)
    {
    case 1:
        Run<double>::put(gcomm, listen_addr, dims, np, sp, timestep, num_vars, delay, interval,
                        log_name, terminate);
        break;

    case 2:
        Run<float>::put(gcomm, listen_addr, dims, np, sp, timestep, num_vars, delay, interval,
                        log_name, terminate);
        break;
    
    default:
        std::cerr<<"Element type is not supported !!!"<<std::endl;
        print_usage();
        MPI_Abort(MPI_COMM_WORLD, 1);
        break;
    }
    */

   Run<double>::put(gcomm, listen_addr, dims, np, sp, timestep, num_vars, delay, interval,
                        log_name, terminate);

    

    
    MPI_Finalize();

    return 0;
    

}