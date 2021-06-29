#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio>

#include <openacc.h>
#include "dspaces.h"

constexpr int DEFAULT_DIM = 1024;
constexpr double DEFAULT_VALUE = 1.l;

int main(int argc, char* argv[]) {
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    dspaces_init(rank, &ndcl);

    char var_name[128];
    sprintf(var_name, "test_gpu_data");

    double* gpu_data;
    int dim0, dim1;
    int ndims = 2;
    dim0 = DEFAULT_DIM;
    dim1 = DEFAULT_DIM;
    int size = dim0*dim1;

    gpu_data = (double*) malloc(size*sizeof(double));

#pragma acc enter data create(gpu_data[0:size])

    #pragma acc parallel loop collapse(2)
    for(int i=0; i<dim0; i++) {
        for(int j=0; j<dim1; j++) {
            gpu_data[i*dim1+j] = DEFAULT_VALUE;
        }
    }

    uint64_t lb[2] = {0}, ub[2] = {0};

    ub[0] = 1023;
    ub[1] = 1023;

    #pragma acc host_data use_device(gpu_data)
    {
        dspaces_put(ndcl, var_name, 0, sizeof(double), ndims, lb, ub, gpu_data);
    }

#pragma acc exit data delete(gpu_data[0:size])

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;

}