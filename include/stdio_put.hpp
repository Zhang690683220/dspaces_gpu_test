#ifndef CPU_PUT_HPP
#define CPU_PUT_HPP

#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include "unistd.h"
#include "mpi.h"
#include "timer.hpp"

/*
int timer_cb(dspaces_client_t client, struct dspaces_req* req, void* timer) {
    Timer* timer_ptr = (Timer*) timer;
    double put_time_async = timer_ptr->stop();
    std::cout<< "DSPACES_CPU_PUT() Version = "<< req->ver << " TIME(Sync) = " << put_time_async << "(ms)" << std::endl;
    return 0;
}
*/

template <typename Data_t>
struct Run {
    static int put(MPI_Comm gcomm, int dims, std::vector<int>& np,
                    std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                    std::string log_name);
};

template <>
struct Run <double> {
static int put(MPI_Comm gcomm, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    uint64_t grid_size = 1;
    for(int i=0; i<dims; i++) {
        grid_size *= sp[i];
    }

    double **data_tab = (double **) malloc(sizeof(double*) * var_num);
    std::vector<std::string> filename;
    filename.resize(var_num);
    // char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    // for(int i=0; i<var_num; i++) {
    //     data_tab[i] = (double*) malloc(sizeof(double) * grid_size);
    //     var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
    //     sprintf(var_name_tab[i], "test_var_%d", i);
    // }

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

        // output every $interval timesteps from timestep 1
        if((ts-1)%interval==0) {
            for(int i=0; i<var_num; i++) {
                filename[i] = "test_var_" + std::to_string(i);
                for(int j=0; j<dims; j++) {
                    filename[i] += "_" + std::to_string(lb[j]);
                }
                for(int j=0; j<dims; j++) {
                    filename[i] += "_" + std::to_string(ub[j]);
                }
                filename[i] += "_t" + std::to_string(ts) + ".bin";
                for(int j=0; j<grid_size; j++) {
                    data_tab[i][j] = (double) j+0.01*i;
                }
            }

            Timer timer_put;
            timer_put.start();
            for(int i=0; i<var_num; i++) {
                std::ofstream ofs;
                ofs.(filename[i], std::ios::out | std::ios::trunc | std::ios::binary);
                ofs.write(data_tab[i], grid_size*sizeof(double));
                if(ofs.fail()) {
                    ofs.close();
                    MPI_Abort(gcomm, -1);
                }
                ofs.close();
                // dspaces_put(ndcl, var_name_tab[i], ts, sizeof(double), dims, lb, ub, data_tab[i]);
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
    }

    for(int i=0; i<var_num; i++) {
        free(data_tab[i]);
        free(var_name_tab[i]);
    }
    free(data_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);

    if(rank == 0) {
        total_avg /= (timesteps/interval);
        log << "Total" << ", " << total_avg << std::endl;
        log.close();
    }


    return 0;

}
};

template <>
struct Run <float> {
static int put(MPI_Comm gcomm, int dims, std::vector<int>& np,
                std::vector<uint64_t>& sp, int timesteps, int var_num, int delay, int interval, 
                std::string log_name)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    uint64_t grid_size = 1;
    for(int i=0; i<dims; i++) {
        grid_size *= sp[i];
    }

    float **data_tab = (float **) malloc(sizeof(float*) * var_num);
    std::vector<std::string> filename;
    filename.resize(var_num);
    // char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    // for(int i=0; i<var_num; i++) {
    //     data_tab[i] = (float*) malloc(sizeof(float) * grid_size);
    //     var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
    //     sprintf(var_name_tab[i], "test_var_%d", i);
    // }

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
        if((ts-1)%interval==0) {
            for(int i=0; i<var_num; i++) {
                filename[i] = "test_var_" + std::to_string(i);
                for(int j=0; j<dims; j++) {
                    filename[i] += "_" + std::to_string(lb[j]) + "_" + std::to_string(ub[j]);
                }
                filename[i] += "_t" + std::to_string(ts) + ".bin";
                for(int j=0; j<grid_size; j++) {
                    data_tab[i][j] = (float) j+0.01*i;
                }
            }

            Timer timer_put;
            timer_put.start();
            for(int i=0; i<var_num; i++) {
                std::ofstream ofs;
                ofs.(filename[i], std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                ofs.write(data_tab[i], grid_size*sizeof(double));
                if(ofs.fail()) {
                    ofs.close();
                    MPI_Abort(gcomm, -1);
                }
                ofs.close();
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
    }

    for(int i=0; i<var_num; i++) {
        free(data_tab[i]);
        free(var_name_tab[i]);
    }
    free(data_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);

    if(rank == 0) {
        total_avg /= (timesteps/interval);
        log << "Total" << ", " << total_avg << std::endl;
        log.close();
    }


    return 0;

}
};


#endif // CPU_PUT_HPP

