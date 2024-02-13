/*
CUDA.
*/

#include <cmath>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define cuda_check(call)                                       \
{                                                              \
    cudaError_t code = call;                                   \
    if (code != cudaSuccess)                                   \
    {   fprintf(stderr, "CUDA error: %s\nfile: %s line: %d\n", \
            cudaGetErrorString(code), __FILE__, __LINE__);     \
        exit(code);                                            \
    }                                                          \
}

__global__
void task(
    const double *__restrict__ x,
    const double *__restrict__ y,
    double *__restrict__ z,
    size_t n, size_t N)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
        for (size_t j = 0; j < N; j++)
            z[i] += sqrt(x[i]+y[i]);
}

int main()
{
    namespace chrono = std::chrono;
    constexpr size_t n = 1e9, N = 1e5;
    double *x, *y, *z, *d_x, *d_y, *d_z;
    x = (double *) malloc(sizeof(double)*3*n);
    y = x+n; z = y+n;
    for (size_t i = 0; i < n; i++)
    {   x[i] = 1./n/N;
        y[i] = 2./n/N;
        z[i] = 0;
    }
    cuda_check(cudaMalloc(&d_x, sizeof(double)*n));
    cuda_check(cudaMalloc(&d_y, sizeof(double)*n));
    cuda_check(cudaMalloc(&d_z, sizeof(double)*n));
    auto t_1 = chrono::high_resolution_clock::now();
    cuda_check(cudaMemcpy(d_x, x, sizeof(double)*n, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_y, y, sizeof(double)*n, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_z, z, sizeof(double)*n, cudaMemcpyHostToDevice));
    constexpr int block_size = 128; // need not be constexpr
    constexpr int grid_size = (n+block_size-1)/block_size; // need not be constexpr
    task<<<grid_size, block_size>>>(d_x, d_y, d_z, n, N);
    cuda_check(cudaPeekAtLastError()); // for `task` kernel
    cuda_check(cudaMemcpy(z, d_z, sizeof(double)*n, cudaMemcpyDeviceToHost));
    auto t_2 = chrono::high_resolution_clock::now();
    auto delta_t = chrono::duration_cast<chrono::milliseconds>(t_2-t_1).count();
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += z[i];
    std::cout << "sum " << sum << "\ntime " << delta_t << " ms\n";
    cuda_check(cudaFree(d_x));
    cuda_check(cudaFree(d_y));
    cuda_check(cudaFree(d_z));
    free(x);
}
