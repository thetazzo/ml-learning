// nnf is a GUI app that trains and visualizes your neural network

#include <assert.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"
#define NF_NN_IMPLEMENTATION
#include "nf.h"
#define SV_IMPLEMENTATION
#include "sv.h"

#define SC_FACTOR 120
#define SCREEN_WIDTH (16*SC_FACTOR)
#define SCREEN_HEIGHT (9*SC_FACTOR)

typedef struct {
   size_t *data;
   size_t count;
   size_t capacity;
} NNF_Arch;

typedef struct {
    float *data;
    size_t count;
    size_t capacity;
} NNF_Cost_Plot;

#define NNF_DA_INIT_CAP 256
#define nnf_da_append(da, item)                                                          \
    do {                                                                                 \
        if ((da)->count >= (da)->capacity) {                                             \
            (da)->capacity = (da)->capacity == 0 ? NNF_DA_INIT_CAP : (da)->capacity*2;   \
            (da)->data = realloc((da)->data, (da)->capacity*sizeof(*(da)->data));        \
            assert((da)->data != NULL && "Buy more RAM");                                \
        }                                                                                \
        (da)->data[(da)->count++] = (item);                                              \
    } while (0)                                                                           

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

void nf_nn_render_raylib(NF_NN nn, int rx, int ry, int rw, int rh) {
    Color low_color  = { 0xFF, 0x00, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };


    int neuron_rad = rh*0.03;
    int layer_border_hpad = 50;
    int layer_border_vpad = 50;

    size_t arch_count = nn.count + 1;

    int nn_width   = rw  - 2*layer_border_hpad;
    int nn_height  = rh - 2*layer_border_vpad;
    int nn_x = rx + rw/2 - nn_width/2;
    int nn_y = ry + rh/2 - nn_height/2;

    int layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; ++l) {
        int layer_vpad1 = nn_height/nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2; 
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if (l+1 < arch_count) {
                int layer_vpad2 = nn_height/nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2; 
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    float value = sigmoidf(NF_MAT_AT(nn.ws[l], j, i));
                    high_color.a = floorf(255.f*value);
                    float thicc = rh*0.004f;
                    Vector2 p1 = {cx1, cy1};
                    Vector2 p2 = {cx2, cy2};
                    DrawLineEx(p1, p2, thicc, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f*sigmoidf(NF_MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_rad, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, neuron_rad, GRAY);
            }
        }
    }
}

void nnf_cost_plot_minmax(NNF_Cost_Plot plot, float *min, float *max)
{
    *min = FLT_MAX;
    *max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (*max < plot.data[i]) { *max = plot.data[i]; }
        if (*min > plot.data[i]) { *min = plot.data[i]; }
    }
}

void nnf_plot_cost(NNF_Cost_Plot plot, int rx, int ry, int rw, int rh) 
{
    float min, max;
    nnf_cost_plot_minmax(plot, &min, &max);
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
        DrawText("0", 35, ry+rh+2, rh*0.02f, RAYWHITE);
    }
}

int main(int argc, char **argv)
{
    srand(time(0));
    const char *program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: no architecture file provided\n");
        return 1;
    }

    const char *arch_file_path = args_shift(&argc, &argv);
    if (argc <= 0) {
        fprintf(stderr, "ERROR: no matrix data file provided\n"); return 1; 
    }
    const char *data_file_path = args_shift(&argc, &argv);

    int buffer_len = 0;
    unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);
    if (buffer == NULL) {
        return 1;
    }

    String_View content = sv_from_parts((const char*)buffer, buffer_len);

    NNF_Arch arch = {0};

    content = sv_trim_left(content);
    while (content.count > 0 && isdigit(content.data[0])) {
        size_t x = sv_chop_u64(&content);
        nnf_da_append(&arch, x);
        content = sv_trim_left(content);
    }

    FILE *in = fopen(data_file_path, "rb");
    if (in == NULL) {
        fprintf(stderr, "ERROR: could not read file %s\n", data_file_path);
        return 1;
    }
    NF_Mat td = nf_mat_load(in);
    fclose(in);
    
    NF_ASSERT(arch.count > 1);
    size_t ins_size  = arch.data[0];
    size_t outs_size = arch.data[arch.count-1];
    NF_ASSERT(td.cols == ins_size + outs_size);

    NF_Mat ti = {
        .rows = td.rows,
        .cols = ins_size,
        .stride = td.stride,
        .es = &NF_MAT_AT(td, 0, 0),
    };

    NF_Mat to = {
        .rows = td.rows,
        .cols = outs_size,
        .stride = td.stride,
        .es = &NF_MAT_AT(td, 0, ins_size),
    };

    NF_NN nn = nf_nn_alloc(arch.data, arch.count);
    NF_NN gn = nf_nn_alloc(arch.data, arch.count);
    nf_nn_rand(nn, 0, 1);

    float rate = 0.5;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "NNF");
    SetTargetFPS(60);

    Color bg_color = { 0x18, 0x18, 0x18, 0xFF };

    NNF_Cost_Plot plot = {0};

    size_t max_epoch = 5*1000;
    size_t epoch = 0;
    while (!WindowShouldClose()) {
        for (size_t i = 0; i<10&&epoch < max_epoch;++i) {
#if 1
            nf_nn_backprop(nn, gn, ti, to); 
#else
            float eps = 1e-3;
            nf_nn_finite_diff(nn, gn, eps, ti, to);
#endif
            nf_nn_learn(nn, gn, rate);
            epoch += 1;
            float cost = nf_nn_cost(nn, ti, to);
            nnf_da_append(&plot, cost);
        }  
        BeginDrawing();
        ClearBackground(bg_color);
        {
            int rx,ry,rw,rh;

            int w = GetRenderWidth();
            int h = GetRenderHeight();

            rh = h*2/3;
            rw = w/2;
            rx = 50;
            ry = h/2 - rh/2;
            nnf_plot_cost(plot, rx, ry, rw, rh);

            char buffer[256]; 

            float cost = nf_nn_cost(nn, ti, to);
            sprintf(buffer, "Cost: %g", cost);
            DrawText(buffer, w/4 - 150, 50, h*0.04f, RAYWHITE);

            sprintf(buffer, "Epochs: %zu/%zu, Rate: %f", epoch, max_epoch, rate);
            DrawText(buffer, w/2-550, h-150, h*0.04f, RAYWHITE);

            rw = w/2;
            rh = h*2/3;
            rx = w - rw;
            ry = h/2 - rh/2;
            nf_nn_render_raylib(nn, rx, ry, rw, rh);

        }
        EndDrawing();
    }
    CloseWindow();

    return 0;
}
