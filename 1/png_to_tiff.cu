/*
Read a folder with PNGs using libpng of the form "06%d.png", and write them
using nvTIFF as "06%d.tif" in another folder.
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <png.h>
#include <nvtiff.h>

#define check_cuda(call)                                       \
{                                                              \
    cudaError_t code = call;                                   \
    if (code != cudaSuccess)                                   \
    {   fprintf(stderr, "CUDA error: %s\nfile: %s line: %d\n", \
            cudaGetErrorString(code), __FILE__, __LINE__);     \
        exit(code);                                            \
    }                                                          \
}

void read_png(const char *path, png_byte *&image, unsigned int &ncol, unsigned int &nrow)
{
    png_struct *reader = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (reader == NULL)
    {   fprintf(stderr, "libpng: couldn't create read struct\n");
        exit(-1);
    }
    png_info *info = png_create_info_struct(reader);
    if (info == NULL)
    {   fprintf(stderr, "libpng: couldn't create info struct\n");
        exit(-1);
    }
    FILE *fp = fopen(path, "rb");
    if (fp == NULL)
    {   fprintf(stderr, "libpng: couldn't open png\n");
        exit(-1);
    }
    // for simplicity, we don't check if it's a valid png file

    png_init_io(reader, fp);
    png_read_info(reader, info);
    ncol = png_get_image_width(reader, info);
    nrow = png_get_image_height(reader, info);
    png_byte color_type = png_get_color_type(reader, info);
    png_byte bit_depth = png_get_bit_depth(reader, info);
    if (color_type != PNG_COLOR_TYPE_RGB)
    {   fprintf(stderr, "libpng: color type must be RGB\n");
        exit(-1);
    }
    if (bit_depth != 8)
    {   fprintf(stderr, "libpng: bit depth must be 8\n");
        exit(-1);
    }

    image = new png_byte[nrow*ncol*3];
    auto **row_buffer = new png_byte*[nrow];
    for (size_t i = 0; i < nrow; i++)
        row_buffer[i] = image+ncol*3*i;
    png_read_image(reader, row_buffer);

    png_destroy_read_struct(&reader, &info, NULL);
    fclose(fp);
    delete[] row_buffer;
}

void write_tiff(const png_byte *image_h, unsigned int ncol, unsigned int nrow, const char *path, const cudaStream_t stream)
{
    namespace chrono = std::chrono;
    using clock = chrono::high_resolution_clock;
    auto t_1 = clock::now();

    unsigned int n_subfiles = 1; // 1 tiff file may contain multiple images, we don't do that
    unsigned int photometric_int = 2; // RGB
    
    unsigned int enc_rows_per_strip = 4; // for image partitioning
    unsigned int planar_config = 1; // contiguous pixels, image is array of pixel structs
    unsigned short sample_fmt = 1; // unsigned integer style
    unsigned short samples_per_pixel = 3; // 3 components per pixel
    unsigned short bits_per_sample[samples_per_pixel] {8, 8, 8}; // each component is 1 byte
    unsigned int pixel_size = 3; // 3 bytes per pixel
    unsigned int n_strips_out = (enc_rows_per_strip-1+nrow)/enc_rows_per_strip; // strips per image
    unsigned int n_tot_strips = n_subfiles*n_strips_out;
    unsigned long long enc_strip_size = enc_rows_per_strip*ncol*pixel_size;

    auto **images_d = new unsigned char *[n_subfiles];
    unsigned long long *strip_sizes_d, *strip_offsets_d;
    unsigned char *strip_data_d;

    check_cuda(cudaMalloc(&images_d[0], nrow*ncol*pixel_size));
    check_cuda(cudaMemcpy(images_d[0], image_h, nrow*ncol*pixel_size, cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&strip_sizes_d, sizeof(*strip_sizes_d)*n_tot_strips));
	check_cuda(cudaMalloc(&strip_offsets_d, sizeof(*strip_offsets_d)*n_tot_strips));
	check_cuda(cudaMalloc(&strip_data_d, n_tot_strips*enc_strip_size));

    auto t_2 = clock::now();
    int dev_id = 0; // device
    nvTiffEncodeCtx_t *ctx = nvTiffEncodeCtxCreate(dev_id, n_subfiles, n_strips_out);
    for (size_t attempt = 1;;)
    {   int ret_code = nvTiffEncode(ctx, nrow, ncol, pixel_size, enc_rows_per_strip, n_subfiles,
            images_d, enc_strip_size, strip_sizes_d, strip_offsets_d, strip_data_d, stream);
        if (ret_code != NVTIFF_ENCODE_SUCCESS)
        {   fprintf(stderr, "nvTIFF: encoding error\n");
            exit(-1);
        }
        ret_code = nvTiffEncodeFinalize(ctx, stream);
        if (ret_code == NVTIFF_ENCODE_SUCCESS) break;
        if (ret_code != NVTIFF_ENCODE_COMP_OVERFLOW)
        {   fprintf(stderr, "nvTIFF: error finalizing encoded images\n");
            exit(-1);
        }
        else
        {   if (attempt == 2)
            {   fprintf(stderr, "nvTIFF: encoder buffer overflowed twice, unknown error\n");
                exit(-1);
            }
            fprintf(stderr, "nvTIFF: (warning) encoder buffer overflow\n");
            enc_strip_size = ctx->stripSizeMax;
            nvTiffEncodeCtxDestroy(ctx);
            check_cuda(cudaFree(strip_data_d));
            check_cuda(cudaMalloc(&strip_data_d, n_tot_strips*enc_strip_size));
            ctx = nvTiffEncodeCtxCreate(dev_id, n_subfiles, n_strips_out);
            attempt++;
        }
    }
    check_cuda(cudaStreamSynchronize(stream)); // must sync after encoding
    nvTiffEncodeCtxDestroy(ctx);
    auto t_3 = clock::now();

    auto *strip_sizes_h = new unsigned long long[n_tot_strips];
    auto *strip_offsets_h = new unsigned long long[n_tot_strips];
    auto *strip_data_h = new unsigned char[ctx->stripSizeTot];
    check_cuda(cudaMemcpy(strip_sizes_h, strip_sizes_d, sizeof(*strip_sizes_h)*n_tot_strips, cudaMemcpyDeviceToHost));
	check_cuda(cudaMemcpy(strip_offsets_h, strip_offsets_d, sizeof(*strip_offsets_h)*n_tot_strips, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(strip_data_h, strip_data_d, ctx->stripSizeTot, cudaMemcpyDeviceToHost));

    auto t_4 = clock::now();
    int ret_code = nvTiffWriteFile(path, VER_REG_TIFF, n_subfiles, nrow, ncol,
        enc_rows_per_strip, samples_per_pixel, bits_per_sample, photometric_int, planar_config,
        strip_sizes_h, strip_offsets_h, strip_data_h, sample_fmt);
    if (ret_code != NVTIFF_WRITE_SUCCESS)
    {   fprintf(stderr, "nvTIFF: write error\n");
        exit(-1);
    }
    auto t_5 = clock::now();

    delete[] strip_sizes_h;
    delete[] strip_offsets_h;
    delete[] strip_data_h;
    check_cuda(cudaFree(strip_sizes_d));
    check_cuda(cudaFree(strip_offsets_d));
    check_cuda(cudaFree(strip_data_d));
    for (unsigned int i = 0; i < n_subfiles; i++)
        check_cuda(cudaFree(images_d[i]));
    delete[] images_d;

    auto t_6 = clock::now();
    printf("encoding %s: total %.0f ms encode %.0f ms write %.0f ms\n", path,
        chrono::duration<float>(t_6-t_1).count()*1e3,
        chrono::duration<float>(t_3-t_2).count()*1e3,
        chrono::duration<float>(t_5-t_4).count()*1e3);
}

int main()
{
    const char *dir_in = "/media/taylor/0A/frames_in", *dir_out = "/media/taylor/0A/frames_out";
    size_t n = 10, first = 1000; // first, first+1, ..., first+(n-1)

    char path[256];
    png_byte *image;
    unsigned int ncol, nrow;
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    for (size_t i = 0; i < n; i++)
    {   sprintf(path, "%s/%06zu.png", dir_in, first+i);
        read_png(path, image, ncol, nrow);
        sprintf(path, "%s/%06zu.tif", dir_out, first+i);
        write_tiff(image, ncol, nrow, path, stream);
        delete[] image;
    }

    check_cuda(cudaStreamDestroy(stream));
    return 0;
}
