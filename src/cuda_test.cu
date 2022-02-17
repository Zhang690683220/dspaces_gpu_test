#include <iostream>
#include <vector>
#include <cstring>
#include "unistd.h"
#include "mpi.h"
#include "cuda_runtime.h"
#include "dspaces.h"
#include "CLI11.hpp"

__global__ void device_assign(double *ptr, int size)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < size)
        ptr[idx] = idx;
}

int main(int argc, char* argv[]) {

    CLI::App app{"CUDA TEST"};
    int dims;              // number of dimensions
    std::vector<int> np;
    std::vector<uint64_t> sp;
    std::string listen_addr;
    std::string log_name = "cuda_put.log";
    app.add_option("--dims", dims, "number of data dimensions. Must be [1-8]")->required();
    app.add_option("--np", np, "the number of processes in the ith dimension. The product of np[0],"
                    "...,np[dim-1] must be the number of MPI ranks")->expected(1, 8);
    app.add_option("--sp", sp, "the per-process data size in the ith dimension")->expected(1, 8);
    app.add_option("-l, --listen_addr", listen_addr, "listen address of the mercury network");

    CLI11_PARSE(app, argc, argv);

    int npapp = 1;             // number of application processes
    for(int i = 0; i < dims; i++) {
        npapp *= np[i];
    }

    int nprocs, rank;
    // Using SPMD style programming
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if(npapp != nprocs) {
        std::cerr<<"Product of np[i] args must equal number of MPI processes!"<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    char* listen_addr_str = NULL;
    if(!listen_addr.empty()) {
        listen_addr_str = (char*) malloc(sizeof(char)*128);
        strcpy(listen_addr_str, listen_addr.c_str());
    }
    dspaces_init(rank, &ndcl, listen_addr_str);

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

    std::cout<<"threadsPerBlock = "<< threadsPerBlock <<", numBlocks = " << numBlocks << std::endl;

    // double * data_h = (double*) malloc(sizeof(double) * grid_size);

    double *data_d;
    cuda_status = cudaMalloc((void**)&data_d, sizeof(double) * grid_size);

    device_assign<<<numBlocks, threadsPerBlock>>>(data_d, grid_size);

    cuda_status = cudaThreadSynchronize();

    // cudaMemcpy(data_h, data_d, sizeof(double) * grid_size, cudaMemcpyDeviceToHost);

    dspaces_put(ndcl, "CUDA_TEST", 1, sizeof(double), dims, lb, ub, data_d);

    cudaFree(data_d);

    // std::cout<<"Data :"<< std::endl;

    // for(int i=0; i<sp[0]; i++) {
    //     for(int j=0; j<sp[1]; j++) {
    //         std::cout<< data_h[j+i*sp[1]] << " ";
    //     }

    //     std::cout<< std::endl;
    // }

    // free(data_h);

    dspaces_fini(ndcl);

    MPI_Finalize();


    return 0;
}
