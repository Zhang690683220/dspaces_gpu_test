#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio>
#include <iostream>
#include <unistd.h>

#include <openacc.h>
#include "dspaces.h"
#include "timer.hpp"

constexpr int DEFAULT_DIM = 1024;
constexpr double DEFAULT_VALUE = 1.l;
constexpr int DEFAULT_TIMESTEP = 10;

int timer_cb(dspaces_client_t client, struct dspaces_req* req, void* timer) {
    Timer* timer_ptr = (Timer*) timer;
    double put_time_async = timer_ptr->stop();
    std::cout<< "DSPACES_GPU_PUT() Version = "<< req->ver << " TIME(Sync) = " << put_time_async << "(ms)" << std::endl;
    return 0;
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

    Timer timer_sync;
    Timer timer_async;
    double put_time_async;

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    dspaces_init(rank, &ndcl, listen_addr_str);

    char var_name[128];
    sprintf(var_name, "test_gpu_data");

    double* gpu_data;
    int dim0, dim1, dim2;
    int ndims = 3;
    dim0 = 64;
    dim1 = DEFAULT_DIM;
    dim2 = DEFAULT_DIM;
    int size = dim0*dim1*dim2;

    gpu_data = (double*) malloc(size*sizeof(double));

    for(int ts=1; ts<=DEFAULT_TIMESTEP; ts++) {

#pragma acc enter data create(gpu_data[0:size])

    #pragma acc parallel loop collapse(3)
    for(int i=0; i<dim0; i++) {
        for(int j=0; j<dim1; j++) {
            for(int k=0; k<dim2; k++) {
                gpu_data[i*dim1*dim2+j*dim2+k] = DEFAULT_VALUE;
            }
        }
    }

    uint64_t lb[3] = {0}, ub[3] = {0};

    ub[0] = 63;
    ub[1] = 1023;
    ub[2] = 1023;

    dspaces_sub(ndcl, var_name, ts, sizeof(double), ndims, lb, ub, timer_cb, &timer_sync);
    sleep(3);

    #pragma acc host_data use_device(gpu_data)
    {
        timer_sync.start();
        timer_async.start();
        dspaces_put(ndcl, var_name, ts, sizeof(double), ndims, lb, ub, gpu_data);
        put_time_async = timer_async.stop();
    }

#pragma acc exit data delete(gpu_data[0:size])

    std::cout<< "DSPACES_GPU_PUT() Version = "<< ts << " TIME(ASync) = " << put_time_async << "(ms)" << std::endl;

    }

    dspaces_fini(ndcl);

    MPI_Barrier(MPI_COMM_WORLD);

    //std::cout<< "DSPACES_GPU_PUT() TIME = " << put_time << "(ms)" << std::endl;

    MPI_Finalize();

    return 0;

}
