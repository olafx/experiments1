import cupy as cp
from time import monotonic_ns

n = 10**5
N = 10**5
x = cp.full((n), 1/n/N, dtype=cp.float64)
y = cp.full((n), 2/n/N, dtype=cp.float64)
z = cp.full((n), 0, dtype=cp.float64)
t1 = monotonic_ns()
for i in range(N):
  z += cp.sqrt(x+y)
t2 = monotonic_ns()
sum = cp.sum(z)
print(f'sum {int(sum)}')
print(f'{int((t2-t1)//1e6)} ms')

task = cp.RawKernel(r'''
extern "C" __global__
void task(
    const double *__restrict__ x,
    const double *__restrict__ y,
    double *__restrict__ z,
    size_t n, size_t N)
{
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < n)
        for (size_t j = 0; j < N; j++)
            z[i] += sqrt(x[i]+y[i]);
}''', 'task')
z = cp.full((n), 0, dtype=cp.float64)
block_size = 128
grid_size = (n+block_size-1)//block_size
t1 = monotonic_ns()
task((grid_size,), (block_size,), (x, y, z, n, N))
cp.cuda.stream.get_current_stream().synchronize()
t2 = monotonic_ns()
sum = cp.sum(z)
print(f'sum {int(sum)}')
print(f'{int((t2-t1)//1e6)} ms')
