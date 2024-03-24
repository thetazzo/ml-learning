#ifndef NFVT_H_
#define NFVT_H

#include "raylib.h"

#include "nf.h"

#ifndef NFVT_ASSERT
#define NFVT_ASSERT NNF_ASSERT
#endif // NFVT_ASSERT

#define NFVT_BACKGOUND CLITERAL(Color) {0x16, 0x16, 0x16, 0xFF}

typedef struct {
    float *data;
    size_t count;
    size_t capacity;
} NFVT_Cost_Plot;

void nfvt_nn_render(NF_NN nn, float rx, float ry, float rw, float rh);
void nfvt_plot_cost(NFVT_Cost_Plot plot, int rx, int ry, int rw, int rh);
void nfvt_render_single_frame(NF_NN nn, float img_index);
int nfvt_render_upscaled_screenshot(NF_NN nn, float img_index, const char *out_file_path);
int nfvt_render_upscaled_video(NF_NN nn, float duration, const char *out_file_path);

#endif // NFVY_H_

#ifdef NFVT_IMPLEMENTATION

void nfvt_nn_render(NF_NN nn, float rx, float ry, float rw, float rh) {
    Color low_color  = { 0xFF, 0x00, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

    float neuron_rad = rh*0.03;
    float layer_border_hpad = 50;
    float layer_border_vpad = 50;

    size_t arch_count = nn.count + 1;

    float nn_width   = rw  - 2*layer_border_hpad;
    float nn_height  = rh - 2*layer_border_vpad;
    float nn_x = rx + rw/2 - nn_width/2;
    float nn_y = ry + rh/2 - nn_height/2;

    int layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; ++l) {
        int layer_vpad1 = nn_height/nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            float cx1 = nn_x + l*layer_hpad + layer_hpad/2; 
            float cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if (l+1 < arch_count) {
                float layer_vpad2 = nn_height/nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    float cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2; 
                    float cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    float value = nf_sigmoidf(NF_MAT_AT(nn.ws[l], j, i));
                    high_color.a = floorf(255.f*value);
                    float thicc = rh*0.004f;

                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};

                    DrawLineEx(
                        start,
                        end,
                        thicc,
                        ColorAlphaBlend(low_color, high_color, WHITE)
                    );
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f*nf_sigmoidf(NF_MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_rad, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, neuron_rad, GRAY);
            }
        }
    }
}

void nfvt_plot_cost(NFVT_Cost_Plot plot, int rx, int ry, int rw, int rh) 
{
    float min = FLT_MAX;
    float max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (max < plot.data[i]) { max = plot.data[i]; }
        if (min > plot.data[i]) { min = plot.data[i]; }
    }
    if (min > 0) min = 0;

    size_t n = plot.count;

    if (n < 100) n = 100;

    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = rx + (float)rw/n * i; 
        float y1 = ry + (1-(plot.data[i] - min)/(max-min))*rh;
        float x2 = rx + (float)rw/n * (i+1); 
        float y2 = ry + (1-(plot.data[i+1] - min)/(max-min))*rh;

        DrawLineEx((Vector2){x1,y1}, (Vector2){x2,y2}, rh*0.0035f, YELLOW);
        DrawLine(0, ry+rh, rx+rw+60, ry+rh, RAYWHITE);
        DrawLine(rx, ry+rh+50, rx, 50, RAYWHITE);
        DrawCircle(rx, ry+rh, rh*0.008f, RAYWHITE);
        DrawText("0", rx-rh*0.03f, ry+rh+2, rh*0.03f, RAYWHITE);
    }
}

// TODO: remove
#define STR2(x) #x
#define STR(x) STR2(x)
 
#define out_width 256
#define out_height 256
#define FPS 30
uint32_t out_pixles[out_width*out_height];

#define READ_END 0
#define WRITE_END 1

void nfvt_render_single_frame(NF_NN nn, float img_index)
{
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);;
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 2) = img_index;
            nf_nn_forward(nn);
            float activation = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0);
            if (activation < 0) activation = 0;
            if (activation > 1) activation = 1;
            uint32_t bright = activation*255.f;
            uint32_t pixel = 0xFF000000|bright|(bright<<8)|(bright<<16);
            out_pixles[y*out_width + x] = pixel;
        }
    }
}

int nfvt_render_upscaled_screenshot(NF_NN nn, float img_index, const char *out_file_path)
{
    assert(out_pixles != NULL);
    nfvt_render_single_frame(nn, img_index);
    if (!stbi_write_png(out_file_path, out_width, out_height, 4, out_pixles, out_width*sizeof(*out_pixles))) {
        fprintf(stderr, "ERROR: could not write image %s\n", out_file_path);
        return 1;
    }
    printf("Generated %s\n", out_file_path);
    return 0;
}

int nfvt_render_upscaled_video(NF_NN nn, float duration, const char *out_file_path)
{
    // connecting two processes with a pipe ~ unidirectional pipe
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        fprintf(stderr, "ERROR: could not create a pipe: %s\n", strerror(errno));
        return 1;
    }
    // Fork the current process
    pid_t child = fork();
    // if child pid is negative it means that the child process was not created
    if (child < 0) {
        fprintf(stderr, "ERROR: could not fork a child: %s\n", strerror(errno));
        return 1;
    }
    // if you are the child process pid is equal to 0
    if (child == 0) {
        // replace the stdinput with the read end of the pipe
        if (dup2(pipefd[READ_END], STDIN_FILENO) < 0) {
            fprintf(stderr, "ERROR: could not reopen read end of the pipe as stdin: %s\n", strerror(errno));
            return 1;
        }
        close(pipefd[WRITE_END]);

        int ret = execlp("ffmpeg",
                         "ffmpeg",
                         "-loglevel", "verbose",
                         "-y",
                         "-f", "rawvideo",
                         "-pix_fmt", "rgb32",
                         "-s", STR(out_width) "x" STR(out_height),
                         "-r", STR(FPS),
                         "-an",
                         "-i", "-", 
                         "-c:v", "libx264",
                         out_file_path,
                         NULL
                         );
        if (ret < 0) {
            fprintf(stderr, "ERROR: could not run ffmpeg as a child process: %s\n", strerror(errno));
            return 1;
        }
        assert(0 && "unreachable");
    }

    close(pipefd[READ_END]);

    typedef struct {
        float start;
        float end;
    } Segment;
    
    Segment segments[] = {
        {0, 0},
        {0, 1},
        {1, 1},
        {1, 0},
    };

    size_t segments_count = NF_ARRAY_LEN(segments);
    float segment_length = 1.f/segments_count;

    // render video
    size_t frame_count = FPS*duration;
    for (size_t i = 0; i < frame_count; ++i) {
        float img_index = (float)i/frame_count;
        
        // easing animation
        size_t segment_index = floorf(img_index/segment_length);
        float segment_porgress = img_index/segment_length - segment_index;
        if (segment_index > segments_count) segment_index = segments_count - 1;
        Segment segment = segments[segment_index];
        float a = segment.start + (segment.end - segment.start)*sqrtf(segment_porgress);
        nfvt_render_single_frame(nn, a);
        write(pipefd[WRITE_END], out_pixles, sizeof(*out_pixles)*out_width*out_height); 
    }

    close(pipefd[WRITE_END]);

    // wait for the child to finish executing
    wait(NULL);

    printf("Generated %s\n", out_file_path);
    return 0;
}

#endif // NFVT_IMPLEMENTATION
