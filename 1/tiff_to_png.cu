/*
Read a folder with TIFFs using nvTIFF of the form "06%d.tif", and write them
using libpng as "06%d.png" in another folder.
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

void write_png(
    const char *path, png_byte *image,
    const unsigned int ncol, const unsigned int nrow)
{
    png_struct *writer = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (writer == NULL)
    {   fprintf(stderr, "libpng: couldn't create write struct\n");
        exit(-1);
    }
    png_info *info = png_create_info_struct(writer);
    if (info == NULL)
    {   fprintf(stderr, "libpng: couldn't create info struct\n");
        exit(-1);
    }
    FILE *fp = fopen(path, "wb");
    if (fp == NULL)
    {   fprintf(stderr, "couldn't open png\n");
        exit(-1);
    }
    // for simplicity, we don't check if it's a valid png file

    png_init_io(writer, fp);
    png_set_IHDR(writer, info, ncol, nrow, 8 /* bits per pixel component */,
        PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(writer, info);

    auto **row_buffer = new png_byte*[nrow];
    for (size_t i = 0; i < nrow; i++)
        row_buffer[i] = image+ncol*3*i;
    png_write_image(writer, row_buffer);
    png_write_end(writer, NULL);

    png_destroy_write_struct(&writer, &info);
    fclose(fp);
    delete[] row_buffer;
}

void read_tiff(const char *path, unsigned char *&image_h, unsigned int &ncol, unsigned int &nrow, const cudaStream_t stream)
{
    namespace chrono = std::chrono;
    using clock = chrono::high_resolution_clock;
    auto t_1 = clock::now();

    nvtiffStream_t tiff_stream;
    nvtiffStatus_t status = nvtiffStreamCreate(&tiff_stream);
    if (status != NVTIFF_STATUS_SUCCESS)
    {   fprintf(stderr, "nvTIFF: couldn't create stream (code %d)\n", status);
        exit(-1);
    }
    nvtiffDecoder_t decoder;
    status = nvtiffDecoderCreate(&decoder, nullptr, nullptr, stream);
    if (status != NVTIFF_STATUS_SUCCESS)
    {   fprintf(stderr, "nvTIFF: couldn't create decoder (code %d)\n", status);
        exit(-1);
    }
    auto t_2 = clock::now();
    status = nvtiffStreamParseFromFile(path, tiff_stream);
    auto t_3 = clock::now();
    if (status != NVTIFF_STATUS_SUCCESS)
    {   fprintf(stderr, "nvTIFF: couldn't read file (code %d)\n", status);
        printf("%d\n", status);
        exit(-1);
    }

    nvtiffFileInfo_t file_info;
    status = nvtiffStreamGetFileInfo(tiff_stream, &file_info);
    if (status != NVTIFF_STATUS_SUCCESS)
    {   fprintf(stderr, "nvTIFF: couldn't read file info (code %d)\n", status);
        exit(-1);
    }
    if (file_info.num_images != 1)
    {   fprintf(stderr, "nvTIFF: support for subfiles is not implemented\n");
        exit(-1);
    }
    if (file_info.photometric_int != 2)
    {   fprintf(stderr, "nvTIFF: only support for RGB is implemented\n");
        exit(-1);
    }
    if (file_info.planar_config != 1)
    {   fprintf(stderr, "nvTIFF: only support for contiguous pixels is implemented\n");
        exit(-1);
    }
    if (file_info.sample_format[0] != 1 ||
        file_info.sample_format[0] != 1 ||
        file_info.sample_format[0] != 1)
    {   fprintf(stderr, "nvTIFF: only support for unsigned integer style is implemented\n");
        exit(-1);
    }
    if (file_info.samples_per_pixel != 3)
    {   fprintf(stderr, "nvTIFF: only support for 3 samples per pixel is implemented\n");
        exit(-1);
    }
    if (file_info.bits_per_sample[0] != 8 ||
        file_info.bits_per_sample[1] != 8 ||
        file_info.bits_per_sample[2] != 8 ||
        file_info.bits_per_sample[3] != 0)
    {   fprintf(stderr, "nvTIFF: only support for 1 byte pixel samples is implemented\n");
        exit(-1);
    }
    if (file_info.bits_per_pixel != 24)
    {   fprintf(stderr, "nvTIFF: only support for 3 byte pixels (1 byte per sample) is implemented\n");
        exit(-1);
    }

    ncol = file_info.image_width;
    nrow = file_info.image_height;
    unsigned char *image_d;
    check_cuda(cudaMalloc(&image_d, ncol*nrow*3));
    auto t_4 = clock::now();
    status = nvtiffDecode(tiff_stream, decoder, &image_d, stream);
    if (status != NVTIFF_STATUS_SUCCESS)
    {   fprintf(stderr, "nvTIFF: decoding error (code %d)", status);
        exit(-1);
    }
    check_cuda(cudaStreamSynchronize(stream));
    nvtiffDecoderDestroy(decoder, stream);
    auto t_5 = clock::now();

    image_h = new unsigned char[ncol*nrow*3];
    check_cuda(cudaMemcpy(image_h, image_d, ncol*nrow*3, cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(image_d));
    nvtiffStreamDestroy(tiff_stream);

    auto t_6 = clock::now();
    printf("decoding %s: total %0.f ms read %0.f ms decode %0.f ms\n", path,
        chrono::duration<float>(t_6-t_1).count()*1e3,
        chrono::duration<float>(t_3-t_2).count()*1e3,
        chrono::duration<float>(t_5-t_4).count()*1e3);
}

int main()
{
    const char *dir_in = "/media/taylor/0A/frames_out", *dir_out = "/media/taylor/0A/frames_out";
    size_t n = 10, first = 1000; // first, first+1, ..., first+(n-1)

    char path[256];
    unsigned char *image;
    unsigned int ncol, nrow;
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    for (size_t i = 0; i < n; i++)
    {   sprintf(path, "%s/%06zu.tif", dir_in, first+i);
        read_tiff(path, image, ncol, nrow, stream);
        sprintf(path, "%s/%06zu.png", dir_out, first+i);
        write_png(path, image, ncol, nrow);
        delete[] image;
    }

    check_cuda(cudaStreamDestroy(stream));
    return 0;
}
