#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio>
#include <iostream>

#include <openacc.h>
#include "dspaces.h"
#include "timer.hpp"

constexpr int DEFAULT_DIM = 1024;
constexpr int DEFAULT_VALUE = 1.l;
constexpr int DEFAULT_TIMESTEP = 10;

bool validate(double *data) {
    double epsilon = 1e-6;
    for(int i=0; i<DEFAULT_DIM*DEFAULT_DIM; i++) {
        if(abs(data[i]-DEFAULT_VALUE) > epsilon)
            return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    char* listen_addr_str = NULL;
    if(argc == 2) {
        listen_addr_str = argv[1];
    }
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    Timer timer;
    double get_time;

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    dspaces_init(rank, &ndcl, listen_addr_str);

    char var_name[128];
    sprintf(var_name, "test_gpu_data");

    double* cpu_data;
    int dim0, dim1, dim2;
    int ndims = 3;
    dim0 = DEFAULT_DIM;
    dim1 = DEFAULT_DIM;
    dim2 = DEFAULT_DIM;
    int size = dim0*dim1*dim2;

    cpu_data = (double*) malloc(size*sizeof(double));


    uint64_t lb[3] = {0}, ub[3] = {0};

    ub[0] = 63;
    ub[1] = 1023;
    ub[2] = 1023;

    for(int ts=1; ts<=DEFAULT_TIMESTEP; ts++) {

    
    timer.start();
    dspaces_get(ndcl, var_name, ts, sizeof(double), ndims, lb, ub, cpu_data, -1);
    get_time = timer.stop();

    std::cout<< "DSPACES_CPU_GET() Version = "<< ts << " TIME = " << get_time << "(ms)" << std::endl;

    if(validate(cpu_data)) {
        std::cout << "Successful Validation !" << std::endl;
    } else {
        std::cout << "Validation Failed !" << std::endl;
    }

    }

    dspaces_fini(ndcl);


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;



}
