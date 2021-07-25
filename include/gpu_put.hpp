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

    static int put_fixed(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name, bool terminate);
};


struct Run <double> {
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

    char *var_name = (char *) malloc(sizeof(char) * 128);
    double *data = (double*) malloc(sizeof(double) * grid_size);
    
    double **data_tab = (double **) malloc(sizeof(double*) * var_num);
    char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    for(int i=0; i<var_num; i++) {
        data_tab[i] = (double*) ACCH::Malloc(sizeof(double) * grid_size);
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
        sleep(delay);
        if((ts-1)%interval==0) {
            //#pragma acc enter data create(data_tab[0:var_num][0:grid_size])
            for(int i=0; i<var_num; i++) {
                #pragma acc parallel loop present(data_tab[i][0:grid_size])
                for(int j=0; j<grid_size; j++) {
                    data_tab[i][j] = (double) j+0.01*i;
                }
            }
            
            Timer timer_put;
            timer_put.start();
            for(int i=0; i<var_num; i++) {
                //#pragma acc host_data use_device(data_tab[i])
                {
                    dspaces_put(ndcl, var_name, ts, sizeof(double), dims, lb, ub, ACCH::GetDevicePtr(data_tab[i]));
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
            //#pragma acc exit data delete(data_tab[0:var_num][0:grid_size])
            
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
        total_avg /= (timesteps/interval);
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

// gpu_put() 16x 3D vars, only for presentation
static int put_fixed(MPI_Comm gcomm, std::string listen_addr, int dims, std::vector<int>& np,
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

    char *var_name = (char *) malloc(sizeof(char) * 128);
    double *data = (double*) malloc(sizeof(double) * grid_size);
    
    //double **data_tab = (double **) malloc(sizeof(double*) * var_num);
    double *data0 = (double*) malloc(sizeof(double) * grid_size);
    double *data1 = (double*) malloc(sizeof(double) * grid_size);
    double *data2 = (double*) malloc(sizeof(double) * grid_size);
    double *data3 = (double*) malloc(sizeof(double) * grid_size);
    double *data4 = (double*) malloc(sizeof(double) * grid_size);
    double *data5 = (double*) malloc(sizeof(double) * grid_size);
    double *data6 = (double*) malloc(sizeof(double) * grid_size);
    double *data7 = (double*) malloc(sizeof(double) * grid_size);
    double *data8 = (double*) malloc(sizeof(double) * grid_size);
    double *data9 = (double*) malloc(sizeof(double) * grid_size);
    double *data10 = (double*) malloc(sizeof(double) * grid_size);
    double *data11 = (double*) malloc(sizeof(double) * grid_size);
    double *data12 = (double*) malloc(sizeof(double) * grid_size);
    double *data13 = (double*) malloc(sizeof(double) * grid_size);
    double *data14 = (double*) malloc(sizeof(double) * grid_size);
    double *data15 = (double*) malloc(sizeof(double) * grid_size);

    char var_name_tab[16][128];
    for(int i=0; i<var_num; i++) {
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
        sleep(delay);
        if((ts-1)%interval==0) {
            #pragma acc enter data create(data0[0:grid_size])
    #pragma acc enter data create(data1[0:grid_size])
    #pragma acc enter data create(data2[0:grid_size])
    #pragma acc enter data create(data3[0:grid_size])
    #pragma acc enter data create(data4[0:grid_size])
    #pragma acc enter data create(data5[0:grid_size])
    #pragma acc enter data create(data6[0:grid_size])
    #pragma acc enter data create(data7[0:grid_size])
    #pragma acc enter data create(data8[0:grid_size])
    #pragma acc enter data create(data9[0:grid_size])
    #pragma acc enter data create(data10[0:grid_size])
    #pragma acc enter data create(data11[0:grid_size])
    #pragma acc enter data create(data12[0:grid_size])
    #pragma acc enter data create(data13[0:grid_size])
    #pragma acc enter data create(data14[0:grid_size])
    #pragma acc enter data create(data15[0:grid_size])
            
            #pragma acc parallel loop
            for(int j=0; j<grid_size; j++) {
                data0[j] = (double) j+0.01*0;
                data1[j] = (double) j+0.01*1;
                data2[j] = (double) j+0.01*2;
                data3[j] = (double) j+0.01*3;
                data4[j] = (double) j+0.01*4;
                data5[j] = (double) j+0.01*5;
                data6[j] = (double) j+0.01*6;
                data7[j] = (double) j+0.01*7;
                data8[j] = (double) j+0.01*8;
                data9[j] = (double) j+0.01*9;
                data10[j] = (double) j+0.01*10;
                data11[j] = (double) j+0.01*11;
                data12[j] = (double) j+0.01*12;
                data13[j] = (double) j+0.01*13;
                data14[j] = (double) j+0.01*14;
                data15[j] = (double) j+0.01*15;
            }
            
            
            Timer timer_put;
            timer_put.start();
            
            #pragma acc host_data use_device(data0)
            {
                dspaces_put(ndcl, var_name_tab[0], ts, sizeof(double), dims, lb, ub, data0);
            }

            #pragma acc host_data use_device(data1)
            {
                dspaces_put(ndcl, var_name_tab[1], ts, sizeof(double), dims, lb, ub, data1);
            }

            #pragma acc host_data use_device(data2)
            {
                dspaces_put(ndcl, var_name_tab[2], ts, sizeof(double), dims, lb, ub, data2);
            }

            #pragma acc host_data use_device(data3)
            {
                dspaces_put(ndcl, var_name_tab[3], ts, sizeof(double), dims, lb, ub, data3);
            }

            #pragma acc host_data use_device(data4)
            {
                dspaces_put(ndcl, var_name_tab[4], ts, sizeof(double), dims, lb, ub, data4);
            }

            #pragma acc host_data use_device(data5)
            {
                dspaces_put(ndcl, var_name_tab[5], ts, sizeof(double), dims, lb, ub, data5);
            }

            #pragma acc host_data use_device(data6)
            {
                dspaces_put(ndcl, var_name_tab[6], ts, sizeof(double), dims, lb, ub, data6);
            }

            #pragma acc host_data use_device(data7)
            {
                dspaces_put(ndcl, var_name_tab[7], ts, sizeof(double), dims, lb, ub, data7);
            }

            #pragma acc host_data use_device(data8)
            {
                dspaces_put(ndcl, var_name_tab[8], ts, sizeof(double), dims, lb, ub, data8);
            }

            #pragma acc host_data use_device(data9)
            {
                dspaces_put(ndcl, var_name_tab[9], ts, sizeof(double), dims, lb, ub, data9);
            }

            #pragma acc host_data use_device(data10)
            {
                dspaces_put(ndcl, var_name_tab[10], ts, sizeof(double), dims, lb, ub, data10);
            }

            #pragma acc host_data use_device(data11)
            {
                dspaces_put(ndcl, var_name_tab[11], ts, sizeof(double), dims, lb, ub, data11);
            }

            #pragma acc host_data use_device(data12)
            {
                dspaces_put(ndcl, var_name_tab[12], ts, sizeof(double), dims, lb, ub, data12);
            }

            #pragma acc host_data use_device(data13)
            {
                dspaces_put(ndcl, var_name_tab[13], ts, sizeof(double), dims, lb, ub, data13);
            }

            #pragma acc host_data use_device(data14)
            {
                dspaces_put(ndcl, var_name_tab[14], ts, sizeof(double), dims, lb, ub, data14);
            }

            #pragma acc host_data use_device(data15)
            {
                dspaces_put(ndcl, var_name_tab[15], ts, sizeof(double), dims, lb, ub, data15);
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

        #pragma acc exit data delete(data0[0:grid_size])
    #pragma acc exit data delete(data1[0:grid_size])
    #pragma acc exit data delete(data2[0:grid_size])
    #pragma acc exit data delete(data3[0:grid_size])
    #pragma acc exit data delete(data4[0:grid_size])
    #pragma acc exit data delete(data5[0:grid_size])
    #pragma acc exit data delete(data6[0:grid_size])
    #pragma acc exit data delete(data7[0:grid_size])
    #pragma acc exit data delete(data8[0:grid_size])
    #pragma acc exit data delete(data9[0:grid_size])
    #pragma acc exit data delete(data10[0:grid_size])
    #pragma acc exit data delete(data11[0:grid_size])
    #pragma acc exit data delete(data12[0:grid_size])
    #pragma acc exit data delete(data13[0:grid_size])
    #pragma acc exit data delete(data14[0:grid_size])
    #pragma acc exit data delete(data15[0:grid_size])    
            
        }
    }

    

    free(data0);
    free(data1);
    free(data2);
    free(data3);
    free(data4);
    free(data5);
    free(data6);
    free(data7);
    free(data8);
    free(data9);
    free(data10);
    free(data11);
    free(data12);
    free(data13);
    free(data14);
    free(data15);
    /*
    for(int i=0; i<var_num; i++) {
        free(var_name_tab[i]);
    }
    */
    //free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);
    free(listen_addr_str);

    if(rank == 0) {
        total_avg /= (timesteps/interval);
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

