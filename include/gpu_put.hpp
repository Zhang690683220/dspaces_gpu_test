#ifndef GPU_PUT_HPP
#define GPU_PUT_HPP

#include <iostream>
#include <cstring>
#include <openacc.h>
#include "ACCH.hpp"
#include "unistd.h"
#include "mpi.h"
#include "dspaces.h"
#include "timer.hpp"

template <typename Data_t>
struct Set {
static int set_value(Data_t* ptr, uint64_t grid_size, int var_index);
};


struct Set <double> {
static int set_value(double* ptr, uint64_t grid_size, int var_index)
{
    #pragma acc parallel loop present(ptr[0:grid_size])
    for(int i=0; i<grid_size; i++) {
        ptr[i] = (double) i+0.01*var_index;
    }

    return 0;
}
};

struct Set <float> {
static int set_value(float* ptr, uint64_t grid_size, int var_index)
{
    #pragma acc parallel loop present(ptr[0:grid_size])
    for(int i=0; i<grid_size; i++) {
        ptr[i] = (float) i+0.01*var_index;
    }

    return 0;
}
};

template <typename Data_t>
struct Run {
    static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate);
};


struct Run <double> {
static int put(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name, bool terminate)
{
    constexpr int DEFAULT_DIM = 1024;
    constexpr double DEFAULT_VALUE = 1.l;
    constexpr int DEFAULT_TIMESTEP = 10;
    
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);
    //MPI_Barrier(MPI_COMM_WORLD);

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

    int size = grid_size;
    std::cout<<"grid_size:"<<grid_size<<std::endl;
    double *gpu_data = (double*) malloc(sizeof(double) * grid_size);

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
    std::cout<<"lb={";
    for(int i=0; i<dims; i++) {
        std::cout<<lb[i]<<", ";
    }
    std::cout<<"}"<<std::endl;

    std::cout<<"ub={";
    for(int i=0; i<dims; i++) {
        std::cout<<ub[i]<<", ";
    }
    std::cout<<"}"<<std::endl;

    Timer timer_sync;
    Timer timer_async;
    double put_time_async;

    char var_name[128];
    sprintf(var_name, "test_gpu_data");

    //double* gpu_data;
    int dim0, dim1, dim2;
    int ndims = 3;
    dim0 = 64;
    dim1 = DEFAULT_DIM;
    dim2 = DEFAULT_DIM;
    //int size = dim0*dim1*dim2;

    //gpu_data = (double*) malloc(size*sizeof(double));

    for(int ts=1; ts<=DEFAULT_TIMESTEP; ts++) {

#pragma acc enter data create(gpu_data[0:size])

    #pragma acc parallel loop collapse(3)
    for(int i=0; i<sp[0]; i++) {
        for(int j=0; j<sp[1]; j++) {
            for(int k=0; k<sp[2]; k++) {
                gpu_data[i*sp[1]*sp[2]+j*sp[2]+k] = DEFAULT_VALUE;
            }
        }
    }
    /*
    uint64_t lb[3] = {0}, ub[3] = {0};

    ub[0] = 63;
    ub[1] = 1023;
    ub[2] = 1023;
    */
    //dspaces_sub(ndcl, var_name, ts, sizeof(double), ndims, lb, ub, timer_cb, &timer_sync);
    sleep(3);

    #pragma acc host_data use_device(gpu_data)
    {
        //timer_sync.start();
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
    /*
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

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

    double *data = (double*) malloc(sizeof(double) * grid_size);

    double **data_tab = (double **) malloc(sizeof(double*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        data_tab[i] = (double*) malloc(sizeof(double) * grid_size);
        var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
        sprintf(var_name_tab[i], "test_var_%d", i);
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

    std::ofstream log;
    double* avg_put = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_put = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, put_ms" << std::endl;
    }

    #pragma acc enter data create(data[0:grid_size])

    for(int ts=1; ts<=timesteps; ts++) {
        // emulate computing time
        sleep(delay);
        //for(int i=0; i<var_num; i++) {
            
            #pragma acc parallel loop
            for(int j=0; j<grid_size; j++) {
                data[j] = (double) j;
            }
            //Set<double>::set_value(data_tab[i], grid_size, i);
        //}

        Timer timer_put;
        timer_put.start();
        for(int i=0; i<var_num; i++) {
            #pragma acc host_data use_device(data)
            {
            dspaces_put(ndcl, var_name_tab[i], ts, sizeof(double), dims, lb, ub,
                        data);
            }
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
    #pragma acc exit data delete(data[0:grid_size])

    for(int i=0; i<var_num; i++) {
        //ACCH::Free(data_tab[i], sizeof(double) * grid_size);
        
        free(data_tab[i]);
        free(var_name_tab[i]);
    }
    free(data_tab);
    free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);
    free(listen_addr_str);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << ", " << total_avg << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            dspaces_kill(ndcl);
        }
    }

    dspaces_fini(ndcl);

    return 0;
    */

}
};

struct Run <float> {
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
    dspaces_init(rank, &ndcl, listen_addr_str);

    uint64_t grid_size = 1;
    for(int i=0; i<dims; i++) {
        grid_size *= sp[i];
    }

    float **data_tab = (float **) malloc(sizeof(float*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        data_tab[i] = (float*) ACCH::Malloc(sizeof(float) * grid_size);
        var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
        sprintf(var_name_tab[i], "test_var_%d", i);
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
        for(int i=0; i<var_num; i++) {
            Set<float>::set_value(data_tab[i], grid_size, i);
        }

        Timer timer_put;
        timer_put.start();
        for(int i=0; i<var_num; i++) {
            dspaces_put(ndcl, var_name_tab[i], ts, sizeof(float), dims, lb, ub,
                        ACCH::GetDevicePtr(data_tab[i]));
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

    for(int i=0; i<var_num; i++) {
        ACCH::Free(data_tab[i], sizeof(double) * grid_size);
        free(var_name_tab[i]);
    }
    free(data_tab);
    free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);
    free(listen_addr_str);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << ", " << total_avg << std::endl;
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

#endif // GPU_PUT_HPP

