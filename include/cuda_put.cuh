#ifndef CUDA_PUT_HPP
#define CUDA_PUT_HPP

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

cudaError_t cuda_assign_double(int dev_rank, double *ptr, int size, int var_idx);
cudaError_t cuda_assign_float(int dev_rank, float *ptr, int size, int var_idx);

template <typename Data_t>
struct Run {
    static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate, bool interference_cpu, bool interference_gpu);
};

template <>
struct Run <double> {
static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
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
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaGetDeviceCount() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }
    dev_rank = rank%dev_num;
    cuda_status = cudaSetDevice(dev_rank);
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaSetDevice() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }

    // same init data for each var at host
    //double *data_tab_h = (double *) malloc(sizeof(double) * grid_size);

    // n vars at device
    double **data_tab_d = (double **) malloc(sizeof(double*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        cuda_status = cudaMalloc((void**)&data_tab_d[i],sizeof(double) * grid_size);
        if(cuda_status != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMalloc() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
            return -1;
        }
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
    double* avg_put = nullptr;
    double* avg_itime = nullptr;
    double total_avg = 0;
    double total_itime = 0;

    if(rank == 0) {
        avg_put = (double*) malloc(sizeof(double)*timesteps);
        avg_itime = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, put_ms, internal_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // emulate computing time
        sleep(delay);

        // output every $interval timesteps from timestep 1
        if((ts-1)%interval==0) {

            for(int i=0; i<var_num; i++) {
                cuda_status = cuda_assign_double(dev_rank, data_tab_d[i], grid_size, i);
                if(cuda_status != cudaSuccess) {
                    fprintf(stderr, "ERROR: (%s): cuda_assign_double() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
                    return -1;
                }
            }

            // wait device to finish
            cuda_status = cudaDeviceSynchronize();
            if(cuda_status != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaDeviceSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
                return -1;
            }

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
            double time_itime = 0;
            Timer timer_put;
            timer_put.start();
            for(int i=0; i<var_num; i++) {
                dspaces_cuda_put(ndcl, var_name_tab[i], ts, sizeof(double), dims, lb, ub, data_tab_d[i], &time_itime);
            }
            double time_put = timer_put.stop();

            double *avg_time_put = nullptr;
            double *avg_time_itime = nullptr;

            if(rank == 0) {
                avg_time_put = (double*) malloc(sizeof(double)*nprocs);
                avg_time_itime = (double*) malloc(sizeof(double)*nprocs);
            }

            MPI_Gather(&time_put, 1, MPI_DOUBLE, avg_time_put, 1, MPI_DOUBLE, 0, gcomm);
            MPI_Gather(&time_itime, 1, MPI_DOUBLE, avg_time_itime, 1, MPI_DOUBLE, 0, gcomm);

            if(rank == 0) {
                avg_put[ts-1] = 0;
                avg_itime[ts-1] = 0;
                for(int i=0; i<nprocs; i++) {
                    avg_put[ts-1] += avg_time_put[i];
                    avg_itime[ts-1] += avg_time_itime[i];
                }
                avg_put[ts-1] /= nprocs;
                avg_itime[ts-1] /= nprocs;
                log << ts << ", " << avg_put[ts-1] << ", " << avg_itime[ts-1] << std::endl;
                total_avg += avg_put[ts-1];
                total_itime +=avg_itime[ts-1];
                free(avg_time_put);
            }
        }
    }

    for(int i=0; i<var_num; i++) {
        cuda_status = cudaFree(data_tab_d[i]);
        if(cuda_status != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaFree() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
            return -1;
        }
        free(var_name_tab[i]);
    }
    free(data_tab_d);
    free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);
    free(avg_itime);
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
        total_itime /= (timesteps/interval);
        log << "Average" << ", " << total_avg << ", " << total_itime << std::endl;
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

template <>
struct Run <float> {
static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
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
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaGetDeviceCount() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }
    dev_rank = rank%dev_num;
    cuda_status = cudaSetDevice(dev_rank);
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaSetDevice() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }

    // same init data for each var at host
    //float *data_tab_h = (float *) malloc(sizeof(float) * grid_size);

    // n vars at device
    float **data_tab_d = (float **) malloc(sizeof(float*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        cuda_status = cudaMalloc((void**)&data_tab_d[i],sizeof(float) * grid_size);
        if(cuda_status != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMalloc() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
            return -1;
        }
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
    double* avg_put = nullptr;
    double* avg_itime = nullptr;
    double total_avg = 0;
    double total_itime = 0;


    if(rank == 0) {
        avg_put = (double*) malloc(sizeof(double)*timesteps);
        avg_itime = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, put_ms, internal_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // emulate computing time
        sleep(delay);

        // output every $interval timesteps from timestep 1
        if((ts-1)%interval==0) {

            for(int i=0; i<var_num; i++) {
                cuda_status = cuda_assign_float(dev_rank, data_tab_d[i], grid_size, i);
                if(cuda_status != cudaSuccess) {
                    fprintf(stderr, "ERROR: (%s): cuda_assign_double() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
                    return -1;
                }
            }

            // wait device to finish
            cuda_status = cudaDeviceSynchronize();
            if(cuda_status != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaDeviceSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
                return -1;
            }

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
            double time_itime = 0;
            Timer timer_put;
            timer_put.start();
            for(int i=0; i<var_num; i++) {
                dspaces_cuda_put(ndcl, var_name_tab[i], ts, sizeof(float), dims, lb, ub, data_tab_d[i], &time_itime);
            }
            double time_put = timer_put.stop();

            double *avg_time_put = nullptr;
            double *avg_time_itime = nullptr;

            if(rank == 0) {
                avg_time_put = (double*) malloc(sizeof(double)*nprocs);
                avg_time_itime = (double*) malloc(sizeof(double)*nprocs);
            }

            MPI_Gather(&time_put, 1, MPI_DOUBLE, avg_time_put, 1, MPI_DOUBLE, 0, gcomm);
            MPI_Gather(&time_itime, 1, MPI_DOUBLE, avg_time_itime, 1, MPI_DOUBLE, 0, gcomm);

            if(rank == 0) {
                avg_put[ts-1] = 0;
                avg_itime[ts-1] = 0;
                for(int i=0; i<nprocs; i++) {
                    avg_put[ts-1] += avg_time_put[i];
                    avg_itime[ts-1] += avg_time_itime[i];
                }
                avg_put[ts-1] /= nprocs;
                avg_itime[ts-1] /= nprocs;
                log << ts << ", " << avg_put[ts-1] << ", " << avg_itime[ts-1] << std::endl;
                total_avg += avg_put[ts-1];
                total_itime +=avg_itime[ts-1];
                free(avg_time_put);
            }
        }
    }

    for(int i=0; i<var_num; i++) {
        cuda_status = cudaFree(data_tab_d[i]);
        if(cuda_status != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaFree() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
            return -1;
        }
        free(var_name_tab[i]);
    }
    free(data_tab_d);
    free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);
    free(avg_itime);
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
        total_itime /= (timesteps/interval);
        log << "Average" << ", " << total_avg << ", " << total_itime << std::endl;
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