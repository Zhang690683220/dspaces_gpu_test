#ifndef CPU_PUT_HPP
#define CPU_PUT_HPP

#include <cstring>
#include "unistd.h"
#include "mpi.h"
#include "dspaces.h"

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
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    dspaces_client_t ndcl = dspaces_CLIENT_NULL;
    char listen_addr_str[128];
    if(listen_addr.empty()) {
        listen_addr_str = NULL;
    } else {
        strcpy(listen_addr_str, listen_addr.c_str());
    }
    dspaces_init(rank, &ndcl, listen_addr_str);

    uint64_t grid_size = 1;
    for(int i=0; i<dims; i++) {
        grid_size *= sp[i];
    }

    double **data_tab = (double **)malloc(sizeof(double*) * var_num);
    std::vector<std::string> var_name;
    for(int i=0; i<var_num; i++) {
        data_tab[i] = (double*) malloc(sizeof(double) * grid_size);
        for(int j=0; j<grid_size; j++) {
            data_tab[i][j] = (double) j+0.01*i;
        }
        var_name[i] = "test_var_" + std::to_string(i);
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

        Timer timer_put;
        timer_put.start();
        for(int i=0; i<var_num; i++) {
            dspaces_put(ndcl, var_name[i].c_str(), ts, sizeof(double), dims, lb, ub, data_tab[i]);
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
        free(data_tab[i]);
    }
    free(data_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_put);

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


#endif // CPU_PUT_HPP

