#include <time.h>
#include "raylib.h"

#define NF_IMPLEMENTATION
#include "./nf.h"

#define FACTOR 128 
#define IMG_WIDTH (16*FACTOR) 
#define IMG_HEIGHT (9*FACTOR)

void nf_nn_render_raylib(NF_NN nn) {
    Color bg_color   = { 0x18, 0x18, 0x18, 0xFF };
    Color low_color  = { 0xFF, 0x00, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

    ClearBackground(bg_color);

    int neuron_rad = 25;
    int layer_border_hpad = 50;
    int layer_border_vpad = 50;

    size_t arch_count = nn.count + 1;

    int nn_width   = IMG_WIDTH  - 2*layer_border_hpad;
    int nn_height  = IMG_HEIGHT - 2*layer_border_vpad;
    int nn_y = IMG_HEIGHT/2 - nn_height/2;
    int nn_x = IMG_WIDTH/2 - nn_width/2;

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
                    high_color.a = floorf(255.f*sigmoidf(NF_MAT_AT(nn.ws[l], j, i)));
                    DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_color, high_color, WHITE));
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

#define BITS 2 

void adder_validate(NF_NN nn, size_t n, size_t *fails) 
{
    for (size_t x = 0; x < n; ++x) {
        for (size_t y = 0; y < n; ++y) {
            size_t z = x + y;
            for (size_t j = 0; j < BITS; ++j) {
                NF_MAT_AT(NF_NN_INPUT(nn), 0, j)        = (x>>j)&1;
                NF_MAT_AT(NF_NN_INPUT(nn), 0, j + BITS) = (y>>j)&1;
            }
            nf_nn_forward(nn);
            if (NF_MAT_AT(NF_NN_OUTPUT(nn), 0, BITS) > 0.5f) {
                if (z < n) {
                    printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, z);
                    *fails += 1;
                }
            } else {
                size_t a = 0;
                for (size_t j = 0; j < BITS; ++j) {
                    size_t bit = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, j) > 0.5f;
                    a |= bit<<j;
                }
                if (z != a) {
                    printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
                    *fails += 1;
                }
            }
        }
    }
}

int main()
{


    srand(time(0));

    size_t n = 1<<BITS; // 0001 << 2 -> 0010 aka 2^BITS 
    size_t rows = n*n;
    
    NF_Mat ti = nf_mat_alloc(rows, 2*BITS);
    NF_Mat to = nf_mat_alloc(rows, BITS + 1);  // + 1 ~ carry bit

    for (size_t i = 0; i < ti.rows; ++i) {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;

        for (size_t j = 0; j < BITS; ++j) {
            NF_MAT_AT(ti, i, j)        = (x>>j)&1;
            NF_MAT_AT(ti, i, j + BITS) = (y>>j)&1;
            NF_MAT_AT(to, i, j)    = (z>>j)&1;
        }

        NF_MAT_AT(to, i, BITS) = z >= n;
    }

    size_t arch[] = { 2*BITS, 4*BITS,  2*BITS, BITS+1 };
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    NF_NN gn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, 0, 1);

    float rate = 1;
    size_t epochs = 1000*5;

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Adder");
    SetTargetFPS(60);

    size_t i = 0;
    bool running = false;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            running = !running;
        }   
        if (i < epochs && running) {
#if 1
            nf_nn_backprop(nn, gn, ti, to); 
#else
            nf_nn_finite_diff(nn, gn, 1e-3, ti, to);
#endif
            nf_nn_learn(nn, gn, rate);
            i += 1;
        }

        BeginDrawing();
        {
            nf_nn_render_raylib(nn);
            if (running) {
                DrawText("Running", IMG_WIDTH/2 - 50, 25, 24, RAYWHITE);
            } else {
                DrawText("Stopped", IMG_WIDTH/2 - 50, 25, 24, RAYWHITE);
            }
        }
        EndDrawing();
    }

    printf("----------------------------\n");

    size_t fails = 0;

    adder_validate(nn, n, &fails);

    if (fails == 0) {
        printf("You are OK, you are OK Annie\n");
    } else {
        printf("fails: %zu\n", fails);
    }

    return 0;
}
