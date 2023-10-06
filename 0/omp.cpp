/*
OpenMP on CPU.
*/

#include <cmath>
#include <iostream>
#include <chrono>

int main()
{
    namespace chrono = std::chrono;
    constexpr size_t n = 1e5;
    constexpr size_t N = 1e5;
    auto *x = new double[3*n], *y = x+n, *z = y+n;
    for (size_t i = 0; i < n; i++)
    {   x[i] = 1./n/N;
        y[i] = 2./n/N;
        z[i] = 0;
    }
    auto t_1 = chrono::high_resolution_clock::now();
    for (size_t j = 0; j < N; j++)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            z[i] += sqrt(x[i]+y[i]);
    }
    auto t_2 = chrono::high_resolution_clock::now();
    auto delta_t = chrono::duration_cast<chrono::milliseconds>(t_2-t_1).count();
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += z[i];
    std::cout << "sum " << sum << "\ntime " << delta_t << " ms\n";
}
