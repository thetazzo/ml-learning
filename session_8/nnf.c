// nnf is a GUI app that trains and visualizes your neural network

#include <assert.h>
#include <ctype.h>
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
    Color bg_color   = { 0x18, 0x18, 0x18, 0xFF };
    Color low_color  = { 0xFF, 0x00, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

    ClearBackground(bg_color);

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

int main(int argc, char **argv)
{
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

    float rate = 1;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "NNF");
    SetTargetFPS(60);

    Color bg_color = { 0x18, 0x18, 0x18, 0xFF };
    size_t epochs = 1000*5;

    size_t i = 0;
    while (!WindowShouldClose()) {
        if (i < epochs) {
            nf_nn_backprop(nn, gn, ti, to); 
            nf_nn_learn(nn, gn, rate);
            i += 1;
            printf("cost: %f\n", nf_nn_cost(nn, ti,to));
        }
        BeginDrawing();
        {
            int rw = SCREEN_WIDTH/2;
            int rh = SCREEN_HEIGHT*2/3;
            int rx = SCREEN_WIDTH - rw;
            int ry = SCREEN_HEIGHT/2 - rh/2;
            nf_nn_render_raylib(nn, rx, ry, rw, rh);
        }
        EndDrawing();
    }

    return 0;
}
