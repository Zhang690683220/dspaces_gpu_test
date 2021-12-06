#ifndef CPU_GET_HPP
#define CPU_GET_HPP

#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include "unistd.h"
#include "mpi.h"
#include "dspaces.h"
#include "timer.hpp"

#define bb_max(a, b) (a) > (b) ? (a) : (b)
#define bb_min(a, b) (a) < (b) ? (a) : (b)

#define BBOX_MAX_NDIM 8

struct coord {
    uint64_t c[BBOX_MAX_NDIM];
};

struct bbox {
    int num_dims;
    struct coord lb, ub;
};

struct meta_file {
    struct bbox bb;
    std::string filename;
};


/*
  Test if bounding boxes b0 and b1 intersect along dimension dim.
*/
static int bbox_intersect_ondim(const struct bbox *b0, const struct bbox *b1,
                                int dim)
{
    if((b0->lb.c[dim] <= b1->lb.c[dim] && b1->lb.c[dim] <= b0->ub.c[dim]) ||
       (b1->lb.c[dim] <= b0->lb.c[dim] && b0->lb.c[dim] <= b1->ub.c[dim]))
        return 1;
    else
        return 0;
}

/*
  Test if bounding boxes b0 and b1 intersect (on all dimensions).
*/
int bbox_does_intersect(const struct bbox *b0, const struct bbox *b1)
{
    int i;

    for(i = 0; i < b0->num_dims; i++) {
        if(!bbox_intersect_ondim(b0, b1, i))
            return 0;
    }

    return 1;
}

/*
  Compute the intersection of bounding boxes b0 and b1, and store it on
  b2. Implicit assumption: b0 and b1 intersect.
*/
void bbox_intersect(const struct bbox *b0, const struct bbox *b1,
                    struct bbox *b2)
{
    int i;

    b2->num_dims = b0->num_dims;
    for(i = 0; i < b0->num_dims; i++) {
        b2->lb.c[i] = bb_max(b0->lb.c[i], b1->lb.c[i]);
        b2->ub.c[i] = bb_min(b0->ub.c[i], b1->ub.c[i]);
    }
}

template <typename Data_t, unsigned int Dims>
struct Run {
    static int get(MPI_Comm gcomm, int dims, std::vector<int>& np, std::vector<int>& src_np,
                    std::vector<uint64_t>& sp, std::vector<uint64_t>& src_sp, int timesteps,
                    int var_num, int delay, int interval, std::string log_name);
};

template <>
struct Run <double, 3> {
static int get(MPI_Comm gcomm, int dims, std::vector<int>& np, std::vector<int>& src_np,
                    std::vector<uint64_t>& sp, std::vector<uint64_t>& src_sp, int timesteps,
                    int var_num, int delay, int interval, std::string log_name);
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);


    uint64_t grid_size = 1;
    for(int i=0; i<3; i++) {
        grid_size *= sp[i];
    }

    double **data_tab = (double **) malloc(sizeof(double*) * var_num);
    // char **var_name_tab = (char **) malloc(sizeof(char*) * var_num);
    // for(int i=0; i<var_num; i++) {
    //     data_tab[i] = (double*) malloc(sizeof(double) * grid_size);
    //     var_name_tab[i] = (char*) malloc(sizeof(char) * 128);
    //     sprintf(var_name_tab[i], "test_var_%d", i);
    // }

    uint64_t* off = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(3*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<3; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    struct bbox local_bb;

    local_bb.num_dims = 3;
    memcpy(local_bb.lb.c, lb, 3*sizeof(uint64_t));
    memcpy(local_bb.ub.c, ub, 3*sizeof(uint64_t));

    struct bbox* src_bbox_tab = (struct bbox*) malloc(src_np[0]*src_np[1]*src_np[2]*sizeof(struct bbox));
    int iter[3];
    for(iter[0]=0; iter[0]<src_np[0]; iter[0]++) {
        for(iter[1]=0; iter[1]<src_np[1]; iter[1]++) {
            for(iter[2]=0; iter[2]<src_np[2]; iter[2]++){
                src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].num_dims = 3;
                for(int d=0; d<3; d++) {
                    src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].lb.c[d] = iter[d]*src_sp[d];
                    src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].ub.c[d] = (iter[d]+1)*src_sp[d]-1;
                }
            }
        }
    }

    int open_num = 0;
    for(int i=0; i<src_np[0]*src_np[1]*src_np[2]; i++) {
        if(bbox_does_intersect(&local_bb, &src_bbox_tab[i])) {
            open_num++;
        }
    }

    std::ofstream log;
    double* avg_get = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_get = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, get_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // emulate computing time
        sleep(delay);

        Timer timer_get;
        timer_get.start();
        for(int i=0; i<var_num; i++) {
            std::vector<struct meta_file> open_tab;
            open_tab.resize(open_num);
            int index_entry = 0;
            for(int j=0; j<src_np[0]*src_np[1]*src_np[2]; j++) {
                if(bbox_does_intersect(&local_bb, &src_bbox_tab[j])) {
                std::string tmp = "test_var_" + std::to_string(i) + "_"
                                + std::to_string(src_bbox_tab[j].lb.c[0]) + "_" + std::to_string(src_bbox_tab[j].lb.c[1]) + "_"
                                + std::to_string(src_bbox_tab[j].lb.c[2]) + "_"
                                + std::to_string(src_bbox_tab[j].ub.c[0]) + "_" + std::to_string(src_bbox_tab[j].ub.c[1]) + "_"
                                + std::to_string(src_bbox_tab[j].ub.c[2])
                                + "_t" + std::to_string(ts) + ".bin";
                open_tab[index_entry].filename = tmp;
                memcpy(&open_tab[index_entry].bb, &src_bbox_tab[i], sizeof(struct bbox));
                index_entry++;
                }
            }
            for(int j=0; j<open_tab.size(); j++) {
                double* tmp_buf = (double*) malloc(src_sp[0]*src_sp[1]*src_sp[2]*sizeof(double));
                std::ifstream ifs;
                ifs.open(open_tab[j].filename, std::ios::in | std::ios::binary);
                ifs.read(tmp_buf, src_sp[0]*src_sp[1]*src_sp[2]*sizeof(double));
                if(ofs.fail()) {
                    ofs.close();
                    MPI_Abort(gcomm, -1);
                }
                ofs.close();

                struct bbox tmp_bbox;

                bbox_intersect(&local_bb, &open_tab[j].bb, &tmp_bbox);

                int dst_offset, src_offset;
                for(int i0=0; i0<src_sp[0]; i0++) {
                    for(int i1=0; i1<src_sp[1]; i1++) {
                        for(int i2=0; i2<src_sp[2]; i2++) {
                            dst_offset = ((i0+tmp_bbox.lb.c[0]-local_bb.lb.c[0])*src_sp[1]
                                         + i1+tmp_bbox.lb.c[1]-local_bb.lb.c[1])*src_sp[2]
                                         + i2+tmp_bbox.lb.c[2]-local_bb.lb.c[2];
                            src_offset = ((i0+tmp_bbox.lb.c[0]-open_tab[i].bb.lb.c[0])*src_sp[1]
                                         + i1+tmp_bbox.lb.c[1]-open_tab[i].bb.lb.c[1]*src_sp[2])
                                         + i2+tmp_bbox.lb.c[2]-open_tab[i].bb.lb.c[2];

                            data_tab[i][dst_offest] = tmp_buf[src_offset];
                        }
                    }
                }
            }
        }

        double time_get = timer_get.stop();

        double *avg_time_get = nullptr;

        if(rank == 0) {
            avg_time_get = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_get, 1, MPI_DOUBLE, avg_time_get, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_get[ts-1] += avg_time_get[i];
            }
            avg_get[ts-1] /= nprocs;
            log << ts << ", " << avg_get[ts-1] << std::endl;
            total_avg += avg_get[ts-1];
            free(avg_time_get);
        }
    }

    for(int i=0; i<var_num; i++) {
        free(data_tab[i]);
        // free(var_name_tab[i]);
    }
    free(data_tab);
    // free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_get);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << ", " << total_avg << std::endl;
        log.close();
    }

    return 0;
}
};

template <>
struct Run <float> {
static int get(MPI_Comm gcomm, int dims, std::vector<int>& np, std::vector<int>& src_np,
                    std::vector<uint64_t>& sp, std::vector<uint64_t>& src_sp, int timesteps,
                    int var_num, int delay, int interval, std::string log_name);
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
        data_tab[i] = (float*) malloc(sizeof(float) * grid_size);
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
    double* avg_get = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_get = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step, get_ms" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {
        // emulate computing time
        sleep(delay);

        Timer timer_get;
        timer_get.start();
        for(int i=0; i<var_num; i++) {
            dspaces_get(ndcl, var_name_tab[i], ts, sizeof(float), dims, lb, ub, data_tab[i], -1);
        }
        double time_get = timer_get.stop();

        double *avg_time_get = nullptr;

        if(rank == 0) {
            avg_time_get = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_get, 1, MPI_DOUBLE, avg_time_get, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_get[ts-1] += avg_time_get[i];
            }
            avg_get[ts-1] /= nprocs;
            log << ts << ", " << avg_get[ts-1] << std::endl;
            total_avg += avg_get[ts-1];
            free(avg_time_get);
        }
    }

    for(int i=0; i<var_num; i++) {
        free(data_tab[i]);
        free(var_name_tab[i]);
    }
    free(data_tab);
    free(var_name_tab);

    free(off);
    free(lb);
    free(ub);
    free(avg_get);
    free(listen_addr_str);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << ", " << total_avg << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            dspaces_kill(ndcl);
        }
    }

    dspaces_fini(ndcl);

    return 0;
}
};

#endif // CPU_GET_HPP

