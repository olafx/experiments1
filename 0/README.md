# 0

Learning how to use CUDA, OpenMP, OpenACC, MPI, etc., very basic.

The example program here is a sort of vector addition. It's compute bottlenecked, but there's a decent amount of memory bandwidth as well. We get a 30x speedup from single threaded CPU to OpenACC GPU, which is about the expected amount, so that's fine. The CPU parallelized speedup is sublinear (~4x on a CPU with 16 threads) because the number of iterations is quite low. It's approximately square root scaling, which is also about the expected amount, so that's fine.

With the nvc++ compiler, the OpenMP default device is the CPU, and the OpenACC default is the GPU. Makes sense, highlights the usual application. But OpenMP can also do GPU, and OpenACC can also do CPU.

OpenACC has some quirks around pointers. It seems to really insist on a pointer being restrict, and it almost never understands from context that a pointer acts as a restrict pointer, with the exception of C-style arrays, i.e. a useless exception. Practically this means it's best to put OpenACC compiler commands inside of functions that take as input restrict pointers. This is particularly bizarre because OpenACC is C++ compatible, and we're using a C++ compiler, and restrict pointers are not a thing in C++. It doesn't seem to care, restrict works, but the most sensible thing to do here is probably to use the `__restrict__` compiler extension, which is standard in clang, and NVIDIA supports it, even tho nvc++ and cuda are no longer clang based. So when writing OpenACC code in C++, it insists on absolute 100% correctness, and `__restrict__` in C++ makes sense. In C, e.g. the nvc compiler, but who uses that, can also use `restrict` because it's part of the language specification. CUDA on the other hand doesn't insist on them, but CUDA isn't that much more clever apparently, and sometimes it does result in a performance increase.

The lesson is that it's good to be as explicit as possible with all these compilers, so marking pointers are `const` and `__restrict__` where applicable.

The way I wrote the OpenACC code, I made it explicit how memory is copied between CPU and GPU.

In the CPU code, the parallelization happened at the level of the inner loop. This makes sense because it needs to finish the vector sum before it can go on to the next iteration. But with a GPU, the complexity of moving memory from host to device is added, and obviously this should only be done once, not during every iteration. This copying is slow, ms level latency and dozens of GB/s bandwidth (similar to RAM), as opposed to hundreds of GB/s GPU memory. So it makes more sense to put the OpenACC command over the entire loop, so it only copies once. Or you could use separate commands to move from host to device and device to host at the end, it doesn't really matter.

The NVIDIA Nsight Compute CLI profiler is very useful. But when the HPC SDK in the usual way, following NVIDIA's instructions, the superuser does not have the compilers and tools in path. The profiler requires the superuser. Many solutions, mine is specifying the full path through `sudo $(which ncu)`.

The profiler confirms the program is compute bottlenecked (good), and that it runs at around 90% of theoretical maximum device performance, so that's good.

Can use top and nvtop to see what the CPU and GPU are doing roughly.

Can check cuda version from the cuda compiler, `nvcc --version`. My pycuda is installed for version 12.x.

The NVIDIA HPC SDK works only with the driver that was used when the SDK was installed. Don't change driver. Don't upgrade anything, unless upgrading everything.

By default, NVIDIA compilers compile GPU code for your GPU. Same as compilers compiling for your specific CPU. But in the same way programs can be compiled to work for everyone on some platform with your CPU family (e.g. ARM, x86), NVIDIA compilers can compile for a large set of GPUs. This is of course a huge simplification. But practically this results in many kernels optimized and taking advantage of features of specific GPUs.

CUDA works fine, but need to be careful about the architecture. My CUDA driver is of a sligthly older CUDA version than my compiler, so the compiler by default writer for a version that my driver doesn't understand. It seems good practice to specify the architecture as native, that solves that.

The example in CUDA is very explicit and verbose (that's the purpose of CUDA) but not more difficult than OpenACC if you understand how the GPUs work, and you need to understand that anyway to write good code, even with OpenACC. The CUDA implementation is 20% faster or so than the OpenACC solution and is much more consistent in terms of performance. Who understands this stuff, not many people; let's not deep dive here.

For device memory, large alignment is important. So it's not a good idea to use a single `cudaMalloc` call to allocate multiple arrays. The same goes for matrices, the NVIDIA standard libraries often work with padded memory from `cudaMallocPitch`. Will deep dive here later probably when using NVIDIA's standard libraries.

cupy does not work out of the box with NVIDIA HPC SDK installed, because it can't find the libraries. It wants NVIDIA libraries in `LD_LIBRARY_PATH`. So I added these for SDK version 23.7.
```
export LD_LIBRARY_PATH=$NVCOMPILERS/$NVARCH/23.7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVCOMPILERS/$NVARCH/23.7/math_libs/lib64:$LD_LIBRARY_PATH
```
Can add more libraries too, like cuDNN if necessary, cupy can use that too.

cupy for this example is slow, 5x slower than our manual CUDA. There's definitely no copying between host and device between steps going on, that would be far worse still. Not sure why it's slower, who cares, let's not deep dive. cupy mimicks numpy, and numpy is not lazily evaluated. There is no other syntax, so no point deep diving. Writing the kernel manually in cupy, we get back to manual CUDA performance; that is of course always an option.

Many things can go wrong in CUDA, so error checking is really necessary.
