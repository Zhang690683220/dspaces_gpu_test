#ifndef CUDA_GET_HPP
#define CUDA_GET_HPP

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include "unistd.h"
#include "mpi.h"
#include "dspaces.h"
#include "timer.hpp"
#include "cuda_runtime.h"

template <typename Data_t>
struct Run {
    static int get(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate, bool interference_cpu, bool interference_gpu);
};

template <>
struct Run <double> {
static int get(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name, bool terminate, bool interference_cpu, bool interference_gpu)
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
    dspaces_init_mpi(gcomm, &ndcl);
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

    // n vars at device
    double **data_tab_d = (double **) malloc(sizeof(double*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        cuda_status = cudaMalloc((void**)&data_tab_d[i],sizeof(double) * grid_size);
        var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
        sprintf(var_name_tab[i], "test_var_%d", i);
    }

    size_t elem_num = 1 << 27; // 1GB allgather

    // malloc CUDA memory for MPI_Iallgather()
    bool interference = interference_cpu || interference_gpu;
    double *mpi_send_buf, *mpi_recv_buf, *mpi_host_buf;

    if(interference_cpu) {
        if(rank < (nprocs/2)) {
            mpi_send_buf = (double*) malloc(elem_num*sizeof(double));
            for(int i=0; i<elem_num; i++) {
                mpi_send_buf[i] = 1.0;
            }
        } else {
            mpi_recv_buf = (double*) malloc(elem_num*sizeof(double));
        }
    }

    if(interference_gpu) {
        if(rank < (nprocs/2)) {
            mpi_host_buf = (double*) malloc(elem_num*sizeof(double));
            for(int i=0; i<elem_num; i++) {
                mpi_host_buf[i] = 1.0;
            }
            cudaMalloc((void**)&mpi_send_buf, elem_num*sizeof(double));
            cudaMemcpy(mpi_send_buf, mpi_host_buf, elem_num*sizeof(double), cudaMemcpyHostToDevice);
            free(mpi_host_buf);
        } else {
            cudaMalloc((void**)&mpi_recv_buf, elem_num*nprocs*sizeof(double));
        }
    }


    // if(interference) {
    //     if(rank < (nprocs/2)) {
    //         mpi_host_buf = (double*) malloc(elem_num*sizeof(double));
    //         for(int i=0; i<elem_num; i++) {
    //             mpi_host_buf[i] = 1.0;
    //         }
    //         cudaMalloc((void**)&mpi_send_buf, elem_num*sizeof(double));
    //         cudaMemcpy(mpi_send_buf, mpi_host_buf, elem_num*sizeof(double), cudaMemcpyHostToDevice);
    //         free(mpi_host_buf);
    //     } else {
    //         cudaMalloc((void**)&mpi_recv_buf, elem_num*nprocs*sizeof(double));
    //     }
    // }
    MPI_Request mpi_req;

    std::ofstream log;
    double* avg_get = nullptr;
    double* avg_copy = nullptr;
    double* avg_transfer = nullptr;
    double total_avg = 0;
    double total_avg_copy = 0;
    double total_avg_transfer = 0;

    if(rank == 0) {
        avg_get = (double*) malloc(sizeof(double)*timesteps);
        avg_copy = (double*) malloc(sizeof(double)*timesteps);
        avg_transfer = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, get_ms, copy_ms, transfer_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // output every $interval timesteps from timestep 1
        if((ts-1)%interval==0) {
            if(interference) {
                if(ts != 1) {
                    MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
                }
                if(rank < (nprocs/2)) {
                    MPI_Isend(mpi_send_buf, elem_num, MPI_DOUBLE, nprocs-1-rank, 0, gcomm, &mpi_req);
                } else {
                    MPI_Irecv(mpi_recv_buf, elem_num, MPI_DOUBLE, nprocs-1-rank, 0, gcomm, &mpi_req);
                }
            }
            double time_copy, time_transfer;
            Timer timer_get;
            timer_get.start();
            for(int i=0; i<var_num; i++) {
                dspaces_cuda_get(ndcl, var_name_tab[i], ts, sizeof(double), dims, lb, ub, data_tab_d[i], -1, &time_transfer, &time_copy);
            }
            double time_get = timer_get.stop();

            double *avg_time_get = nullptr;
            double *avg_time_copy = nullptr;
            double *avg_time_transfer = nullptr;

            if(rank == 0) {
                avg_time_get = (double*) malloc(sizeof(double)*nprocs);
                avg_time_copy = (double*) malloc(sizeof(double)*nprocs);
                avg_time_transfer = (double*) malloc(sizeof(double)*nprocs);
            }

            MPI_Gather(&time_get, 1, MPI_DOUBLE, avg_time_get, 1, MPI_DOUBLE, 0, gcomm);
            MPI_Gather(&time_copy, 1, MPI_DOUBLE, avg_time_copy, 1, MPI_DOUBLE, 0, gcomm);
            MPI_Gather(&time_transfer, 1, MPI_DOUBLE, avg_time_transfer, 1, MPI_DOUBLE, 0, gcomm);

            if(rank == 0) {
                avg_get[ts-1] = 0;
                avg_copy[ts-1] = 0;
                avg_transfer[ts-1] = 0;
                for(int i=0; i<nprocs; i++) {
                    avg_get[ts-1] += avg_time_get[i];
                    avg_copy[ts-1] += avg_time_copy[i];
                    avg_transfer[ts-1] += avg_time_transfer[i];
                }
                avg_get[ts-1] /= nprocs;
                avg_copy[ts-1] /= nprocs;
                avg_transfer[ts-1] /= nprocs;
                log << ts << ", " << avg_get[ts-1] << ", " << avg_copy[ts-1] << ", " << avg_transfer[ts-1] << std::endl;
                total_avg += avg_get[ts-1];
                total_avg_copy += avg_copy[ts-1];
                total_avg_transfer += avg_transfer[ts-1];
                free(avg_time_get);
            }
        }

        // emulate computing time
        sleep(delay);
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
    free(avg_get);
    free(listen_addr_str);

    if(interference_cpu) {
        MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
        if(rank < (nprocs/2)) {
            free(mpi_send_buf);
        } else {
            free(mpi_recv_buf);
        }
    }

    if(interference_gpu) {
        MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
        if(rank < (nprocs/2)) {
            cudaFree(mpi_send_buf);
        } else {
            cudaFree(mpi_recv_buf);
        }
    }

    if(rank == 0) {
        total_avg /= (timesteps/interval);
        total_avg_copy /= (timesteps/interval);
        total_avg_transfer /= (timesteps/interval);
        log << "Average" << ", " << total_avg << ", " << total_avg_copy << ", " << total_avg_transfer << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            dspaces_kill(ndcl);
        }
    }

    dspaces_fini(ndcl);

    return cuda_status;
}
};

template <>
struct Run <float> {
static int get(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name, bool terminate, bool interference_cpu, bool interference_gpu)
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
    dspaces_init_mpi(gcomm, &ndcl);
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

    // n vars at device
    float **data_tab_d = (float **) malloc(sizeof(float*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        cuda_status = cudaMalloc((void**)&data_tab_d[i],sizeof(float) * grid_size);
        var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
        sprintf(var_name_tab[i], "test_var_%d", i);
    }

    size_t elem_num = 1 << 27; // 1GB allgather
    // malloc CUDA memory for MPI_Iallgather()
    bool interference = interference_cpu || interference_gpu;
    double *mpi_send_buf, *mpi_recv_buf, *mpi_host_buf;

    if(interference_cpu) {
        if(rank < (nprocs/2)) {
            mpi_send_buf = (double*) malloc(elem_num*sizeof(double));
            for(int i=0; i<elem_num; i++) {
                mpi_send_buf[i] = 1.0;
            }
        } else {
            mpi_recv_buf = (double*) malloc(elem_num*sizeof(double));
        }
    }

    if(interference_gpu) {
        if(rank < (nprocs/2)) {
            mpi_host_buf = (double*) malloc(elem_num*sizeof(double));
            for(int i=0; i<elem_num; i++) {
                mpi_host_buf[i] = 1.0;
            }
            cudaMalloc((void**)&mpi_send_buf, elem_num*sizeof(double));
            cudaMemcpy(mpi_send_buf, mpi_host_buf, elem_num*sizeof(double), cudaMemcpyHostToDevice);
            free(mpi_host_buf);
        } else {
            cudaMalloc((void**)&mpi_recv_buf, elem_num*nprocs*sizeof(double));
        }
    }
    MPI_Request mpi_req;

    std::ofstream log;
    double* avg_get = nullptr;
    double* avg_copy = nullptr;
    double* avg_transfer = nullptr;
    double total_avg = 0;
    double total_avg_copy = 0;
    double total_avg_transfer = 0;

    if(rank == 0) {
        avg_get = (double*) malloc(sizeof(double)*timesteps);
        avg_copy = (double*) malloc(sizeof(double)*timesteps);
        avg_transfer = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, get_ms, copy_ms, transfer_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // output every $interval timesteps from timestep 1
        if((ts-1)%interval==0) {
            if(interference) {
                if(ts != 1) {
                    MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
                }
                if(rank < (nprocs/2)) {
                    MPI_Isend(mpi_send_buf, elem_num, MPI_DOUBLE, nprocs-1-rank, 0, gcomm, &mpi_req);
                } else {
                    MPI_Irecv(mpi_recv_buf, elem_num, MPI_DOUBLE, nprocs-1-rank, 0, gcomm, &mpi_req);
                }
            }
            double time_copy, time_transfer;
            Timer timer_get;
            timer_get.start();
            for(int i=0; i<var_num; i++) {
                dspaces_cuda_get(ndcl, var_name_tab[i], ts, sizeof(float), dims, lb, ub, data_tab_d[i], -1, &time_transfer, &time_copy);
            }
            double time_get = timer_get.stop();

            double *avg_time_get = nullptr;
            double *avg_time_copy = nullptr;
            double *avg_time_transfer = nullptr;

            if(rank == 0) {
                avg_time_get = (double*) malloc(sizeof(double)*nprocs);
                avg_time_copy = (double*) malloc(sizeof(double)*nprocs);
                avg_time_transfer = (double*) malloc(sizeof(double)*nprocs);
            }

            MPI_Gather(&time_get, 1, MPI_DOUBLE, avg_time_get, 1, MPI_DOUBLE, 0, gcomm);
            MPI_Gather(&time_copy, 1, MPI_DOUBLE, avg_time_copy, 1, MPI_DOUBLE, 0, gcomm);
            MPI_Gather(&time_transfer, 1, MPI_DOUBLE, avg_time_transfer, 1, MPI_DOUBLE, 0, gcomm);

            if(rank == 0) {
                avg_get[ts-1] = 0;
                avg_copy[ts-1] = 0;
                avg_transfer[ts-1] = 0;
                for(int i=0; i<nprocs; i++) {
                    avg_get[ts-1] += avg_time_get[i];
                    avg_copy[ts-1] += avg_time_copy[i];
                    avg_transfer[ts-1] += avg_time_transfer[i];
                }
                avg_get[ts-1] /= nprocs;
                avg_copy[ts-1] /= nprocs;
                avg_transfer[ts-1] /= nprocs;
                log << ts << ", " << avg_get[ts-1] << ", " << avg_copy[ts-1] << ", " << avg_transfer[ts-1] << std::endl;
                total_avg += avg_get[ts-1];
                total_avg_copy += avg_copy[ts-1];
                total_avg_transfer += avg_transfer[ts-1];
                free(avg_time_get);
            }
        }

        // emulate computing time
        sleep(delay);
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
    free(avg_get);
    free(listen_addr_str);

    if(interference_cpu) {
        MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
        if(rank < (nprocs/2)) {
            free(mpi_send_buf);
        } else {
            free(mpi_recv_buf);
        }
    }

    if(interference_gpu) {
        MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
        if(rank < (nprocs/2)) {
            cudaFree(mpi_send_buf);
        } else {
            cudaFree(mpi_recv_buf);
        }
    }

    if(rank == 0) {
        total_avg /= (timesteps/interval);
        total_avg_copy /= (timesteps/interval);
        total_avg_transfer /= (timesteps/interval);
        log << "Average" << ", " << total_avg << ", " << total_avg_copy << ", " << total_avg_transfer << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            dspaces_kill(ndcl);
        }
    }

    dspaces_fini(ndcl);

    return cuda_status;
}
};

#endif // CUDA_GET_HPP