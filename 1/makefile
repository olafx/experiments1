CC = nvc
NVCC = nvcc
GPU_ARCH = native

# nvCOMP is not yet part of the NVIDIA package manager. It's manual.
NVCOMP_INCLUDE_DIR = /home/taylor/Documents/nvcomp_3.0.4_x86_64_12.x/include
NVCOMP_LIB_DIR = /home/taylor/Documents/nvcomp_3.0.4_x86_64_12.x/lib

CCFLAGS = -std=c17 -O3
NVCCFLAGS = -std=c++20 -O3 --gpu-architecture=$(GPU_ARCH)

.PHONY: all clean

all: png_to_tiff tiff_to_png list_codecs mp4_to_tiff
clean:
	rm -f png_to_tiff tiff_to_png list_codecs mp4_to_tiff

png_to_tiff: png_to_tiff.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lnvtiff -lz -lpng
tiff_to_png: tiff_to_png.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -I$(NVCOMP_INCLUDE_DIR) -L$(NVCOMP_LIB_DIR) -lnvtiff -lnvjpeg -lnvcomp -lnvcomp_gdeflate -lz -lpng
list_codecs: list_codecs.c
	$(CC) $(CCFLAGS) -o $@ $< -lavcodec
mp4_to_tiff: mp4_to_tiff.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lavcodec -lavformat -lavutil -lswscale -lnvcuvid -lnvtiff
