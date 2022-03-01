#include "cuda_put.hpp"

template <typename Data_t>
__global__ void assign(Data_t *ptr, int size, int var_idx);


template <>
__global__ void assign<double>(double *ptr, int size, int var_idx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        ptr[idx] = idx + 0.01*var_idx;
    }
}

template <>
__global__ void assign<float>(float *ptr, int size, int var_idx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        ptr[idx] = idx + 0.01*var_idx;
    }
}

cudaError_t cuda_assign_double(MPI_Comm gcomm, double *ptr, int size, int var_idx)
{
    MPI_Comm_rank(gcomm, &rank);
    int dev_num, dev_rank;
    cudaError_t cuda_status;
    cudaDeviceProp dev_prop;
    cuda_status = cudaGetDeviceCount(&dev_num);
    dev_rank = rank%dev_num;
    cuda_status = cudaSetDevice(dev_rank);
    cuda_status = cudaGetDeviceProperties(&dev_prop,dev_rank);

    int threadsPerBlock = dev_prop.maxThreadsPerBlock;
    int numBlocks = (grid_size + threadsPerBlock) / threadsPerBlock;

    assign<double><<<numBlocks, threadsPerBlock>>>(ptr, size, var_idx);

    return cuda_status
}

cudaError_t cuda_assign_float(MPI_Comm gcomm, float *ptr, int size, int var_idx)
{
    MPI_Comm_rank(gcomm, &rank);
    int dev_num, dev_rank;
    cudaError_t cuda_status;
    cudaDeviceProp dev_prop;
    cuda_status = cudaGetDeviceCount(&dev_num);
    dev_rank = rank%dev_num;
    cuda_status = cudaSetDevice(dev_rank);
    cuda_status = cudaGetDeviceProperties(&dev_prop,dev_rank);

    int threadsPerBlock = dev_prop.maxThreadsPerBlock;
    int numBlocks = (grid_size + threadsPerBlock) / threadsPerBlock;

    assign<float><<<numBlocks, threadsPerBlock>>>(ptr, size, var_idx);

    return cuda_status
}