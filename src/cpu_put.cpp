#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio>
#include <iostream>
#include <unistd.h>

#include "dspaces.h"
#include "CLI11.hpp"
#include "timer.hpp"

#include "cpu_put.hpp"


constexpr int DEFAULT_DIM = 1024;
constexpr double DEFAULT_VALUE = 1.l;
constexpr int DEFAULT_TIMESTEP = 10;

void print_usage()
{
    std::cerr<<"Usage: cpu_put --dims <dims> --np <np[0] .. np[dims-1]> --sp <sp[0] ... sp[dims-1]> "
               "--ts <timesteps> [-s <elem_size>] [-c <var_count>] [--log <log_file>] [--delay <delay_second>] "
               "[--interval <output_freq>] [-t]"<<std::endl
             <<"--dims                      - number of data dimensions. Must be [1-8]"<<std::endl
             <<"--np                        - the number of processes in the ith dimension. "
               "The product of np[0],...,np[dim-1] must be the number of MPI ranks"<<std::endl
             <<"--sp                        - the per-process data size in the ith dimension"<<std::endl
             <<"--ts                        - the number of timestep iterations written"<<std::endl
             <<"-s, --elem_size (optional)  - the number of bytes in each element. Defaults to 8"<<std::endl
             <<"-c, --var_count (optional)  - the number of variables written in each iteration. "
               "Defaults to 1"<<std::endl
             <<"--log (optional)            - output log file name. Default to cpu_put.log"<<std::endl
             <<"--delay (optional)          - sleep(delay) seconds in each timestep. Default to 0"<<std::endl
             <<"--interval (optional)       - Output timestep interval. Default to 1"<<std::endl
             <<"-t (optional)               - send server termination after writing is complete"<<std::endl;
}

int timer_cb(dspaces_client_t client, struct dspaces_req* req, void* timer) {
    Timer* timer_ptr = (Timer*) timer;
    double put_time_async = timer_ptr->stop();
    std::cout<< "DSPACES_CPU_PUT() Version = "<< req->ver << " TIME(Sync) = " << put_time_async << "(ms)" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {

    CLI::App app{"CPU PUT Emulator for DataSpaces"};
    int dims;              // number of dimensions
    std::vector<int> np;
    std::vector<uint64_t> sp;
    int timestep;
    std::string listen_addr;
    size_t elem_size = 8;
    int num_vars = 1;
    std::string log_name = "cpu_put.log";
    int delay = 0;
    int interval = 1;
    bool terminate = false;
    app.add_option("--dims", dims, "number of data dimensions. Must be [1-8]")->required();
    app.add_option("--np", np, "the number of processes in the ith dimension. The product of np[0],"
                    "...,np[dim-1] must be the number of MPI ranks")->expected(1, 8);
    app.add_option("--sp", sp, "the per-process data size in the ith dimension")->expected(1, 8);
    app.add_option("--ts", timestep, "the number of timestep iterations")->required();
    app.add_option()
    app.add_option("-s, --elem_size", elem_size, "the number of bytes in each element. Defaults to 8",
                    true);
    app.add_option("-c, --var_count", num_vars, "the number of variables written in each iteration."
                    "Defaults to 1", true);
    app.add_option("--log", log_name, "output log file name. Default to cpu_put.log", true);
    app.add_option("--delay", delay, "sleep(delay) seconds in each timestep. Default to 0", true);
    app.add_option("--interval", interval, "Output timestep interval. Default to 1", true);
    app.add_flag("-t", terminate, "send server termination after writing is complete", true);

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
        fprintf(stderr,
                "Product of np[i] args must equal number of MPI processes!\n");
        print_usage();
        MPI_Abort(MPI_COMM_WORLD, 1)
    }

    Run<double>::put(gcomm, listen_addr, dims, np, sp, timestep, num_vars, delay, interval,
                     log_name, terminate);
    /*
    char* listen_addr_str = NULL;
    if(argc == 2) {
        listen_addr_str = argv[1];
    }
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    Timer timer_sync;
    Timer timer_async;
    double put_time_async;

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    dspaces_init(rank, &ndcl, listen_addr_str);

    char var_name[128];
    sprintf(var_name, "test_gpu_data");

    double* cpu_data;
    int dim0, dim1, dim2;
    int ndims = 3;
    dim0 = 64;
    dim1 = DEFAULT_DIM;
    dim2 = DEFAULT_DIM;
    int size = dim0*dim1*dim2;

    cpu_data = (double*) malloc(size*sizeof(double));

    for(int ts=1; ts<=DEFAULT_TIMESTEP; ts++) {

    for(int i=0; i<dim0; i++) {
        for(int j=0; j<dim1; j++) {
            for(int k=0; k<dim2; k++) {
                cpu_data[i*dim1*dim2+j*dim2+k] = DEFAULT_VALUE;
            }
        }
    }

    uint64_t lb[3] = {0}, ub[3] = {0};

    ub[0] = 63;
    ub[1] = 1023;
    ub[2] = 1023;

    dspaces_sub(ndcl, var_name, ts, sizeof(double), ndims, lb, ub, timer_cb, &timer_sync);
    sleep(3);

    timer_sync.start();
    timer_async.start();
    dspaces_put(ndcl, var_name, ts, sizeof(double), ndims, lb, ub, cpu_data);
    put_time_async = timer_async.stop();

    std::cout<< "DSPACES_CPU_PUT() Version = "<< ts << " TIME(ASync) = " << put_time_async << "(ms)" << std::endl;

    }

    dspaces_fini(ndcl);

    MPI_Barrier(MPI_COMM_WORLD);

    //std::cout<< "DSPACES_CPU_PUT() TIME = " << put_time << "(ms)" << std::endl;
    */
   
    MPI_Finalize();

    return 0;
    

}
