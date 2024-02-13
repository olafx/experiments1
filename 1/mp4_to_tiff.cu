/*
Read a video using nvdec/nvcuvid through libavcodec, and write every 5th second
as a TIFF using nvTIFF.
*/

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <nvtiff.h>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#define check_cuda(call)                                       \
{                                                              \
    cudaError_t code = call;                                   \
    if (code != cudaSuccess)                                   \
    {   fprintf(stderr, "CUDA error: %s\nfile: %s line: %d\n", \
            cudaGetErrorString(code), __FILE__, __LINE__);     \
        exit(code);                                            \
    }                                                          \
}

void write_tiff(const unsigned char *image_h, unsigned int ncol, unsigned int nrow, const char *path, const cudaStream_t stream)
{
    unsigned int n_subfiles = 1; // 1 tiff file may contain multiple images, we don't do that
    unsigned int photometric_int = 2; // RGB
    
    unsigned int enc_rows_per_strip = 4; // for image partitioning
    unsigned int planar_config = 1; // contiguous pixels, image is array of pixel structs
    unsigned short sample_fmt = 1; // unsigned integer style
    unsigned short samples_per_pixel = 3; // 3 components per pixel
    unsigned short bits_per_sample[samples_per_pixel] {8, 8, 8}; // each component is 1 byte
    unsigned int pixel_size = 3; // 3 bytes per pixel
    unsigned int n_strip_out = (enc_rows_per_strip-1+nrow)/enc_rows_per_strip; // strips per image
    unsigned int n_tot_strips = n_subfiles*n_strip_out;
    unsigned long long enc_strip_size = enc_rows_per_strip*ncol*pixel_size;

    auto **images_d = new unsigned char *[n_subfiles];
    unsigned long long *strip_sizes_d, *strip_offsets_d;
    unsigned char *strip_data_d;

    check_cuda(cudaMalloc(&images_d[0], nrow*ncol*pixel_size));
    check_cuda(cudaMemcpy(images_d[0], image_h, nrow*ncol*pixel_size, cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&strip_sizes_d, sizeof(*strip_sizes_d)*n_tot_strips));
	check_cuda(cudaMalloc(&strip_offsets_d, sizeof(*strip_offsets_d)*n_tot_strips));
	check_cuda(cudaMalloc(&strip_data_d, n_tot_strips*enc_strip_size));

    int dev_id = 0; // device
    nvTiffEncodeCtx_t *ctx = nvTiffEncodeCtxCreate(dev_id, n_subfiles, n_strip_out);
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
            ctx = nvTiffEncodeCtxCreate(dev_id, n_subfiles, n_strip_out);
            attempt++;
        }
    }
    check_cuda(cudaStreamSynchronize(stream)); // must sync after encoding
    nvTiffEncodeCtxDestroy(ctx);

    auto *strip_sizes_h = new unsigned long long[n_tot_strips];
    auto *strip_offsets_h = new unsigned long long[n_tot_strips];
    auto *strip_data_h = new unsigned char[ctx->stripSizeTot];
    check_cuda(cudaMemcpy(strip_sizes_h, strip_sizes_d, sizeof(*strip_sizes_h)*n_tot_strips, cudaMemcpyDeviceToHost));
	check_cuda(cudaMemcpy(strip_offsets_h, strip_offsets_d, sizeof(*strip_offsets_h)*n_tot_strips, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(strip_data_h, strip_data_d, ctx->stripSizeTot, cudaMemcpyDeviceToHost));

    int ret_code = nvTiffWriteFile(path, VER_REG_TIFF, n_subfiles, nrow, ncol,
        enc_rows_per_strip, samples_per_pixel, bits_per_sample, photometric_int, planar_config,
        strip_sizes_h, strip_offsets_h, strip_data_h, sample_fmt);
    if (ret_code != NVTIFF_WRITE_SUCCESS)
    {   fprintf(stderr, "nvTIFF: write error\n");
        exit(-1);
    }

    delete[] strip_sizes_h;
    delete[] strip_offsets_h;
    delete[] strip_data_h;
    check_cuda(cudaFree(strip_sizes_d));
    check_cuda(cudaFree(strip_offsets_d));
    check_cuda(cudaFree(strip_data_d));
    for (unsigned int i = 0; i < n_subfiles; i++)
        check_cuda(cudaFree(images_d[i]));
    delete[] images_d;
}

int main()
{
    const char* path = "/media/taylor/0A/vid_in/a.mp4", *dir_out = "/media/taylor/0A/frames_out";
    size_t n = SIZE_MAX; // maximum number of output images
    int seconds_between_images = 60;

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    AVFormatContext* format_context = avformat_alloc_context();

    if (avformat_open_input(&format_context, path, NULL, NULL) != 0)
    {   fprintf(stderr, "ffmpeg: couldn't read file headers\n");
        return -1;
    }
    if (avformat_find_stream_info(format_context, NULL) < 0)
    {   fprintf(stderr, "ffmpeg: couldn't get stream info\n");
        return -1;
    }

    size_t i_video_stream = 0;
    for (; i_video_stream < format_context->nb_streams; i_video_stream++)
    {   if (format_context->streams[i_video_stream]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            break;
        if (i_video_stream == format_context->nb_streams-1)
        {   fprintf(stderr, "ffmpeg: couldn't find a video stream\n");
            return -1;
        }
    }
    AVStream *video_stream = format_context->streams[i_video_stream];

    AVCodecParameters* codec_parameters = video_stream->codecpar;
    const char *codec_name = NULL;
    switch (codec_parameters->codec_id)
    {
    case AV_CODEC_ID_H264:
        codec_name = "h264_cuvid";
        break;
    case AV_CODEC_ID_HEVC:
        codec_name = "hevc_cuvid";
        break;
    default:
        fprintf(stderr, "ffmpeg: video codec is not H.264 or HEVC, which are the showcase of this program\n");
        return -1;
    }
    AVCodec* codec = avcodec_find_decoder_by_name(codec_name);
    if (codec == NULL)
    {   fprintf(stderr, "ffmpeg: %s codec is unavailable\n", codec_name);
        return -1;
    }

    AVCodecContext* codec_context = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_context, codec_parameters);
    if (avcodec_open2(codec_context, codec, NULL) < 0)
    {   fprintf(stderr, "ffmpeg: couldn't open %s codec\n", codec_name);
        return -1;
    }

    int i_frame = 0;
    int i_image = 0;
    char image_path[256];
    AVRational fps = video_stream->r_frame_rate;
    printf("fps %d/%d\n", fps.num, fps.den);
    AVFrame* frame = av_frame_alloc();
    AVPacket packet;
    while (av_read_frame(format_context, &packet) >= 0)
    {   if (packet.stream_index == i_video_stream)
        {   int ret_code = avcodec_send_packet(codec_context, &packet);
            if (ret_code < 0)
            {   fprintf(stderr, "ffmpeg: error decoding packet\n");
                return -1;
            }
            while (ret_code >= 0)
            {   ret_code = avcodec_receive_frame(codec_context, frame);
                if (ret_code == AVERROR(EAGAIN) || ret_code == AVERROR_EOF)
                    break;
                else if (ret_code < 0)
                {   fprintf(stderr, "ffmpeg: error decoding frame\n");
                    return -1;
                }
                if (seconds_between_images*i_image*fps.num < i_frame*fps.den)
                {   i_image++;
                    sprintf(image_path, "%s/%06d.tif", dir_out, i_image);
                    // convert to RGB24
                    AVFrame *frame_dest = av_frame_alloc();
                    frame_dest->width = frame->width;
                    frame_dest->height = frame->height;
                    frame_dest->format = AV_PIX_FMT_RGB24;
                    av_frame_get_buffer(frame_dest, 1); // no padding
                    SwsContext *sws_ctx = sws_getContext( // need some casts here due to C++ != C
                        frame->width, frame->height, (AVPixelFormat) frame->format,
                        frame_dest->width, frame_dest->height, (AVPixelFormat) frame_dest->format,
                        SWS_BILINEAR, NULL, NULL, NULL);
                    sws_scale(
                        sws_ctx, frame->data, frame->linesize, 0, frame->height,
                        frame_dest->data, frame_dest->linesize);
                    write_tiff(*frame_dest->data, frame_dest->width, frame_dest->height, image_path, stream);
                    printf("frame %08d: scaled %dx%d %s to %dx%d %s: wrote %s\n", i_frame,
                        frame->width, frame->height, av_get_pix_fmt_name((AVPixelFormat) frame->format),
                        frame_dest->width, frame_dest->height, av_get_pix_fmt_name((AVPixelFormat) frame_dest->format),
                        image_path);
                    av_frame_free(&frame_dest);
                    sws_freeContext(sws_ctx);
                    if (i_image == n)
                        goto exit;
                }
                i_frame++;
            }
        }
        av_packet_unref(&packet);
    }

    exit:
    check_cuda(cudaStreamDestroy(stream));
    avformat_close_input(&format_context);
    avformat_free_context(format_context);
    avcodec_free_context(&codec_context);
    av_frame_free(&frame);

    return 0;
}
