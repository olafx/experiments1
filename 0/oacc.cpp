/*
OpenACC on GPU.
*/

#include <cmath>
#include <iostream>
#include <chrono>

void task(
    const double *__restrict__ x,
    const double *__restrict__ y,
    double *__restrict__ z,
    size_t n, size_t N)
{
    #pragma acc kernels loop copyin(x[0:n], y[0:n]) copyout(z[0:n])
    for (size_t j = 0; j < N; j++)
        for (size_t i = 0; i < n; i++)
            z[i] += sqrt(x[i]+y[i]);
}

int main()
{
    namespace chrono = std::chrono;
    constexpr size_t n = 1e5, N = 1e5;
    auto *x = new double[3*n], *y = x+n, *z = y+n;
    for (size_t i = 0; i < n; i++)
    {   x[i] = 1./n/N;
        y[i] = 2./n/N;
        z[i] = 0;
    }        
    auto t_1 = chrono::high_resolution_clock::now();
    task(x, y, z, n, N);
    auto t_2 = chrono::high_resolution_clock::now();
    auto delta_t = chrono::duration_cast<chrono::milliseconds>(t_2-t_1).count();
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += z[i];
    std::cout << "sum " << sum << "\ntime " << delta_t << " ms\n";
}
