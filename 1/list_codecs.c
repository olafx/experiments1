#include <stdio.h>
#include <libavcodec/avcodec.h>

int main()
{
    void *opaque = NULL;
    for (;;)
    {   const AVCodec *codec = av_codec_iterate(&opaque);
        if (codec == NULL)
            break;
        printf("%s: %s\n", codec->name, codec->long_name);
    }
}
