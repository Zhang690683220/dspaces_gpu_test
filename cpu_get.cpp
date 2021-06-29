#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio>
#include <iostream>

#include <openacc.h>
#include "dspaces.h"

constexpr int DEFAULT_DIM = 1024;
constexpr int DEFAULT_VALUE = 1.l;

bool validate(double *data) {
    double epsilon = 1e-6;
    for(int i=0; i<DEFAULT_DIM*DEFAULT_DIM; i++) {
        if(abs(data[i]-DEFAULT_VALUE) > epsilon)
            return false;
    }

    return true;
}

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

    double* cpu_data;
    int dim0, dim1;
    int ndims = 2;
    dim0 = DEFAULT_DIM;
    dim1 = DEFAULT_DIM;
    int size = dim0*dim1;

    cpu_data = (double*) malloc(size*sizeof(double));


    uint64_t lb[2] = {0}, ub[2] = {0};

    ub[0] = 1023;
    ub[1] = 1023;


    dspaces_get(ndcl, var_name, 0, sizeof(double), ndims, lb, ub, cpu_data, -1);

    if(validate(cpu_data)) {
        std::cout << "Successful Validation !" << std::endl;
    } else {
        std::cout << "Validation Failed !" << std::endl;
    }




    MPI_Barrier(MPI_COMM_WORLD);

    return 0;



}
