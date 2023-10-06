nvcc -std=c++20 -O3 -o cu cu.cu --gpu-architecture=native
./cu
