#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "raylib.h"
#include "stb_image.h"
#include "stb_image_write.h"
#define NF_NN_IMPLEMENTATION
#include "nf.h"

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
                    DrawLineEx(
                        (Vector2){cx1, cy1},
                        (Vector2){cx2, cy2},
                        thicc,
                        ColorAlphaBlend(low_color, high_color, WHITE)
                    );
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
        DrawCircle(rx, ry+rh, rh*0.008f, RAYWHITE);
        DrawText("0", rx-rh*0.03f, ry+rh+2, rh*0.04f, RAYWHITE);
    }
}
char *p2m_shift_args(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1; 
    (*argv) += 1;
    return result;
}

int main(int argc, char **argv)
{
    char *program = p2m_shift_args(&argc, &argv);
    if (argc <= 0) {
        fprintf(stderr, "ERROR: missing image path\n");
        fprintf(stderr, "Usage: %s <img_file_path>\n", program);
        return 1;
    }

    // read image
    char *img_file_path = p2m_shift_args(&argc, &argv);
    int img_width, img_height, img_comp;
    uint8_t *data = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);
    if (data == NULL) {
        fprintf(stderr, "ERROR: could not load image %s\n", img_file_path);
        return 1;
    }
    if (img_comp != 1) {
        fprintf(stderr, "ERROR:  image %s is %d bits image, Only 8 bit grayscale images are supported", img_file_path, img_comp*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);


    NF_Mat td = nf_mat_alloc(img_width*img_height, 2+1); // 2 inputs 1 output

    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            size_t i = y*img_width + x;
            NF_MAT_AT(td, i, 0) = (float)x/(img_width - 1);;
            NF_MAT_AT(td, i, 1) = (float)y/(img_height - 1);
            NF_MAT_AT(td, i, 2) = data[i]/255.f;;
        }
    }

    size_t arch[] = {2, 7, 4, 1};
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    NF_NN gn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, -1, 1);
    
    NF_Mat ti = {
        .rows = td.rows,
        .cols = 2,
        .stride = td.stride,
        .es = &NF_MAT_AT(td, 0, 0),
    };

    NF_Mat to = {
        .rows = td.rows,
        .cols = 1,
        .stride = td.stride,
        .es = &NF_MAT_AT(td, 0, ti.cols),
    };

    float rate = 0.75f;

    size_t SC_FACTOR=120;
    size_t SCREEN_WIDTH=(16*SC_FACTOR);
    size_t SCREEN_HEIGHT=(9*SC_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "NNF");
    SetTargetFPS(60);

    Color bg_color = { 0x18, 0x18, 0x18, 0xFF };

    NNF_Cost_Plot plot = {0};
    Image preview_image = GenImageColor(img_width, img_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);
    Image original_image = GenImageColor(img_width, img_height, BLACK);
    Texture2D original_texture = LoadTextureFromImage(preview_image);

    size_t max_epoch = 50*1000;
    size_t epoch = 0;
    bool isRunning = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            isRunning = !isRunning;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            NNF_Cost_Plot ncp = {0};
            plot = ncp;
            nf_nn_rand(nn, -1, 1);
        }
        for (size_t i = 0; i<60&&epoch<max_epoch&&isRunning;++i) {
            nf_nn_backprop(nn, gn, ti, to);
            nf_nn_learn(nn, gn, rate);
            epoch += 1;
            nnf_da_append(&plot, nf_nn_cost(nn, ti, to));
        }  
        BeginDrawing();
        ClearBackground(bg_color);
        {
            int rx,ry,rw,rh;

            int w = GetRenderWidth();
            int h = GetRenderHeight();

            rw = w/3;
            rh = h*2/3;
            rx = 50;
            ry = h/2 - rh/2;

            nnf_plot_cost(plot, rx, ry, rw, rh);

            char buffer[256]; 
            float cost = nf_nn_cost(nn, ti, to);
            sprintf(buffer, "Cost: %g", cost);
            DrawText(buffer, rx+rw/2, 50, h*0.04f, RAYWHITE);

            sprintf(buffer, "Epochs: %zu/%zu, Rate: %f", epoch, max_epoch, rate);
            DrawText(buffer, rx+rw/2, h-150, h*0.04f, RAYWHITE);

            rx += rw;
            nf_nn_render_raylib(nn, rx, ry, rw, rh);
            for (int y = 0; y < img_height; ++y) {
                for (int x = 0; x < img_width; ++x) {
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(img_width - 1);;
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(img_height - 1);
                    nf_nn_forward(nn);
                    uint8_t pixel = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(
                        &preview_image,
                        x,
                        y,
                        CLITERAL(Color){pixel, pixel, pixel, 255}
                    );
                }
            }

            rx += rw;

            UpdateTexture(preview_texture, preview_image.data);
            float scale = 20;
            DrawTextureEx(preview_texture,CLITERAL(Vector2){rx, ry + img_height*scale + 50}, 0, scale, WHITE);

            for (int y = 0; y < img_height; ++y) {
                for (int x = 0; x < img_width; ++x) {
                    uint8_t pixel = data[y*img_width + x]; 
                    ImageDrawPixel(
                        &original_image,
                        x,
                        y,
                        CLITERAL(Color){pixel, pixel, pixel, 255}
                    );
                }
            }
            UpdateTexture(original_texture, original_image.data);
            DrawTextureEx(original_texture,CLITERAL(Vector2){rx, ry}, 0, scale, WHITE);
        }
        EndDrawing();
    }
    CloseWindow();

    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(img_width - 1);;
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(img_height - 1);
            nf_nn_forward(nn);
            uint8_t pixel = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)*255.f;

            if (pixel == 0) {
                printf("    ");
            } else {
                printf("%3u ", pixel);
            }
        }
        printf("\n");
    }

    int out_width = 512;
    int out_height = 512;
    uint8_t *out_pixles = malloc(sizeof(*out_pixles)*out_width*out_height);
    assert(out_pixles != NULL);

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);;
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            nf_nn_forward(nn);
            uint8_t pixel = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)*255.f;
            out_pixles[y*out_width + x] = pixel;
        }
    }

    const char *out_file_path = "image-render.png";
    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixles, out_width*sizeof(*out_pixles))) {
        fprintf(stderr, "ERROR: could not write image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s from %s\n", out_file_path, img_file_path);

    return 0;
}
