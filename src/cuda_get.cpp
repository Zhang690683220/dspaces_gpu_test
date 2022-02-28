#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "CLI11.hpp"
#include "mpi.h"
#include "cuda_get.cuh"

void print_usage()
{
    std::cerr<<"Usage: cpu_get --dims <dims> --np <np[0] .. np[dims-1]> --sp <sp[0] ... sp[dims-1]> "
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
             <<"--log (optional)            - output log file name. Default to cpu_get.log"<<std::endl
             <<"--delay (optional)          - sleep(delay) seconds in each timestep. Default to 0"<<std::endl
             <<"--interval (optional)       - Output timestep interval. Default to 1"<<std::endl
             <<"-k (optional)               - send server kill signal after reading is complete"<<std::endl;
}

int main(int argc, char* argv[]) {
    
    CLI::App app{"CUDA GET Emulator for DataSpaces"};
    int dims;              // number of dimensions
    std::vector<int> np;
    std::vector<uint64_t> sp;
    int timestep;
    std::string listen_addr;
    int elem_type = 1;
    int num_vars = 1;
    std::string log_name = "cuda_get.log";
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
    app.add_flag("-k", terminate, "send server kill signal after reading is complete");

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

    switch (elem_type)
    {
    case 1:
        Run<double>::get(gcomm, listen_addr, dims, np, sp, timestep, num_vars, delay, interval,
                        log_name, terminate);
        break;

    case 2:
        Run<float>::get(gcomm, listen_addr, dims, np, sp, timestep, num_vars, delay, interval,
                        log_name, terminate);
        break;
    
    default:
        std::cerr<<"Element type is not supported !!!"<<std::endl;
        print_usage();
        MPI_Abort(MPI_COMM_WORLD, 1);
        break;
    }

    

    
    MPI_Finalize();

    return 0;



}