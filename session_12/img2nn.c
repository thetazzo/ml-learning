#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>

// posix specific headers
// allows forking childs in linux
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
        DrawText("0", rx-rh*0.03f, ry+rh+2, rh*0.03f, RAYWHITE);
    }
}

#define STR2(x) #x
#define STR(x) STR2(x)
 
#define out_width 256
#define out_height 256
#define FPS 30
uint32_t out_pixles[out_width*out_height];

void nnf_render_single_frame(NF_NN nn, float img_index)
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

int nnf_render_upscaled_screenshot(NF_NN nn, float img_index, const char *out_file_path)
{
    assert(out_pixles != NULL);
    nnf_render_single_frame(nn, img_index);
    if (!stbi_write_png(out_file_path, out_width, out_height, 4, out_pixles, out_width*sizeof(*out_pixles))) {
        fprintf(stderr, "ERROR: could not write image %s\n", out_file_path);
        return 1;
    }
    printf("Generated %s\n", out_file_path);
    return 0;
}

#define READ_END 0
#define WRITE_END 1

int nnf_render_upscaled_video(NF_NN nn, float duration, const char *out_file_path)
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
        nnf_render_single_frame(nn, a);
        write(pipefd[WRITE_END], out_pixles, sizeof(*out_pixles)*out_width*out_height); 
    }

    close(pipefd[WRITE_END]);

    // wait for the child to finish executing
    wait(NULL);

    printf("Generated %s\n", out_file_path);
    return 0;
}

char *p2m_shift_args(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1; 
    (*argv) += 1;
    return result;
}

// neural network architecture
size_t arch[] = {3, 28, 14, 7, 1};

int main(int argc, char **argv)
{
    char *program = p2m_shift_args(&argc, &argv);
    if (argc <= 0) {
        fprintf(stderr, "ERROR: no image 1 provided\n");
        fprintf(stderr, "Usage: %s <img1_file_path> <img2_file_path>\n", program);
        return 1;
    }


    // read image
    char *img1_file_path = p2m_shift_args(&argc, &argv);
    if (argc <= 0) {
        fprintf(stderr, "ERROR: missing second image\n");
        return 1;
    }

    char *img2_file_path = p2m_shift_args(&argc, &argv);

    int img1_width, img1_height, img1_comp;
    uint8_t *img1_data = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
    if (img1_data == NULL) {
        fprintf(stderr, "ERROR: could not load image %s\n", img1_file_path);
        return 1;
    }
    if (img1_comp != 1) {
        fprintf(stderr, "ERROR:  image %s is %d bits image, Only 8 bit grayscale images are supported", img1_file_path, img1_comp*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp*8);

    int img2_width, img2_height, img2_comp;
    uint8_t *img2_data = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
    if (img2_data == NULL) {
        fprintf(stderr, "ERROR: could not load image %s\n", img2_file_path);
        return 1;
    }
    if (img2_comp != 1) {
        fprintf(stderr, "ERROR:  image %s is %d bits image, Only 8 bit grayscale images are supported", img2_file_path, img2_comp*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);

    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    NF_NN gn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));

    NF_Mat td = nf_mat_alloc(
        img1_width*img1_height + img2_width*img2_height,
        NF_NN_INPUT(nn).cols + NF_NN_OUTPUT(nn).cols
    );

    // add image 1 to training data
    for (int y = 0; y < img1_height; ++y) {
        for (int x = 0; x < img1_width; ++x) {
            size_t i = y*img1_width + x;
            NF_MAT_AT(td, i, 0) = (float)x/(img1_width - 1);;
            NF_MAT_AT(td, i, 1) = (float)y/(img1_height - 1);
            NF_MAT_AT(td, i, 2) = 0.f; // 0 is the index of the first image
            NF_MAT_AT(td, i, 3) = img1_data[i]/255.f;;
        }
    }

    // add image 2 to training data
    for (int y = 0; y < img2_height; ++y) {
        for (int x = 0; x < img2_width; ++x) {
            size_t i = img1_width*img1_height + y*img2_width + x;
            NF_MAT_AT(td, i, 0) = (float)x/(img2_width - 1);;
            NF_MAT_AT(td, i, 1) = (float)y/(img2_height - 1);
            NF_MAT_AT(td, i, 2) = 1.f; // 1 is the index of the second image
            NF_MAT_AT(td, i, 3) = img2_data[y*img2_width + x]/255.f;;
        }
    }

    nf_nn_rand(nn, -1, 1);

    size_t SC_FACTOR=120;
    size_t SCREEN_WIDTH=(16*SC_FACTOR);
    size_t SCREEN_HEIGHT=(9*SC_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "img2nn");
    SetTargetFPS(60);

    Color bg_color = { 0x18, 0x18, 0x18, 0xFF };

    NNF_Cost_Plot plot = {0};

    size_t preview_width = 28;
    size_t preview_height  = 28;

    Image preview_image1 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

    Image preview_image2 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

    Image preview_image3 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture3 = LoadTextureFromImage(preview_image3);

    // Draw original image 1
    Image original_image1 = GenImageColor(img1_width, img1_height, BLACK);
    for (int y = 0; y < img1_height; ++y) {
        for (int x = 0; x < img1_width; ++x) {
            uint8_t pixel = img1_data[y*img1_width + x]; 
            ImageDrawPixel(
                &original_image1,
                x,
                y,
                CLITERAL(Color){pixel, pixel, pixel, 255}
            );
        }
    }
    Texture2D original_texture1 = LoadTextureFromImage(original_image1);

    // Draw original image 2
    Image original_image2 = GenImageColor(img2_width, img2_height, BLACK);
    for (int y = 0; y < img2_height; ++y) {
        for (int x = 0; x < img2_width; ++x) {
            uint8_t pixel = img2_data[y*img2_width + x]; 
            ImageDrawPixel(
                &original_image2,
                x,
                y,
                CLITERAL(Color){pixel, pixel, pixel, 255}
            );
        }
    }
    Texture2D original_texture2 = LoadTextureFromImage(original_image2);

    size_t max_epoch = 50*1000;
    size_t epoch = 0;
    size_t bpf = 200;  // batches per frame
    size_t batch_size = 28;
    size_t batch_count = (td.rows + batch_size-1)/batch_size;
    size_t batch_begin = 0;
    float cost = 0.f;
    float rate = 1.f;
    bool isRunning = false;
    float preview_scroll = 0.5f;
    bool preview_scroll_dragging = false;
    bool lrate_scroll_dragging = false;

    while (!WindowShouldClose()) {
        // Start/Stop learning 
        if (IsKeyPressed(KEY_SPACE)) {
            isRunning = !isRunning;
        }

        // Reset neural network
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            NNF_Cost_Plot ncp = {0};
            plot = ncp;
            nf_nn_rand(nn, -1, 1);
        }

        // screenshot neural network based on preview scroll 
        if (IsKeyPressed(KEY_S)) {
            nnf_render_upscaled_screenshot(nn, preview_scroll, "number-upscaled.png");
        }

        // render upscaled video
        if (IsKeyPressed(KEY_X)) {
            nnf_render_upscaled_video(nn, 5, "number-upscaled.mp4");
        }

        for (size_t i = 0; i < bpf && epoch < max_epoch && isRunning; ++i) {
            size_t size = batch_size;

            if (batch_begin + batch_size >= td.rows) {
                // handle last batch that is not the full batch size
                size = td.rows - batch_begin;
            }

            NF_Mat batch_ti = {
                .rows = size,
                .cols = 3,
                .stride = td.stride,
                .es = &NF_MAT_AT(td, batch_begin, 0),
            };

            NF_Mat batch_to = {
                .rows = size, 
                .cols = 1,
                .stride = td.stride,
                .es = &NF_MAT_AT(td, batch_begin, batch_ti.cols),
            };

            nf_nn_backprop(nn, gn, batch_ti, batch_to);
            nf_nn_learn(nn, gn, rate);
            cost += nf_nn_cost(nn, batch_ti, batch_to);
            batch_begin += batch_size;

            if (batch_begin >= td.rows) {
                epoch += 1;
                nnf_da_append(&plot, cost/batch_count);
                cost = 0.0f;
                batch_begin = 0;
                nf_mat_shuffle_rows(td);
            }
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
            float scale = rh*0.009f;

            nnf_plot_cost(plot, rx, ry, rw, rh);

            char buffer[256]; 
            sprintf(buffer, "Cost: %g", plot.count > 0 ? plot.data[plot.count - 1] : 0);
            DrawText(buffer, rx+rw/2, 50, rh*0.04f, RAYWHITE);

            sprintf(buffer, "Epochs: %zu/%zu, Rate: %f", epoch, max_epoch, rate);
            DrawText(buffer, rx+rw/8+20, ry+rh*1.025, rh*0.04f, RAYWHITE);

            {
                float pad = rh*0.05f;
                Vector2 position = { rx+rw*0.2f, ry + rh*1.05 + pad, };
                Vector2 size = { preview_width*scale*2, rh*0.008f, };
                float knob_radious = rh*0.02f;
                DrawRectangleV(position, size, RAYWHITE);
                Vector2 knob_position = {position.x + size.x*rate, position.y + size.y*0.5f}; 
                DrawCircleV(knob_position, knob_radious, RED);

                if (lrate_scroll_dragging) {
                    float x = GetMousePosition().x;
                    if (x < position.x) { x = position.x; }
                    if (x > position.x + size.x) { x = position.x + size.x; }
                    rate = (x - position.x)/size.x;
                }

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    Vector2 mouse_position = GetMousePosition();
                    if (CheckCollisionPointCircle(mouse_position, knob_position, knob_radious)) {
                        lrate_scroll_dragging = true;
                    }
                }

                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                    lrate_scroll_dragging = false;
                }

            }

            rx += rw;
            nf_nn_render_raylib(nn, rx, ry, rw, rh);

            rx += rw;

            // Draw original image 1 
            DrawTextureEx(original_texture1,CLITERAL(Vector2){rx, ry-50}, 0, scale, WHITE);
            // Draw original image 2 
            DrawTextureEx(original_texture2,CLITERAL(Vector2){rx+img1_width*scale, ry-50}, 0, scale, WHITE);

            // Draw preview image 1
            for (size_t y = 0; y < preview_height; ++y) {
                for (size_t x = 0; x < preview_width; ++x) {
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);;
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 2) = 0.f;
                    nf_nn_forward(nn);
                    uint8_t pixel = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(
                        &preview_image1,
                        x,
                        y,
                        CLITERAL(Color){pixel, pixel, pixel, 255}
                    );
                }
            }
            UpdateTexture(preview_texture1, preview_image1.data);
            DrawTextureEx(
                preview_texture1,
                CLITERAL(Vector2){rx, ry + preview_height*scale},
                0,
                scale,
                WHITE
            );

            // Draw preview image 2
            for (size_t y = 0; y < preview_height; ++y) {
                for (size_t x = 0; x < preview_width; ++x) {
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);;
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 2) = 1.f;
                    nf_nn_forward(nn);
                    uint8_t pixel = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(
                        &preview_image2,
                        x,
                        y,
                        CLITERAL(Color){pixel, pixel, pixel, 255}
                    );
                }
            }
            UpdateTexture(preview_texture2, preview_image2.data);
            DrawTextureEx(
                preview_texture2,
                CLITERAL(Vector2){rx + preview_width*scale, ry + preview_height*scale},
                0,
                scale,
                WHITE
            );

            // Draw preview image 3
            for (size_t y = 0; y < preview_height; ++y) {
                for (size_t x = 0; x < preview_width; ++x) {
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);;
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    NF_MAT_AT(NF_NN_INPUT(nn), 0, 2) = preview_scroll;
                    nf_nn_forward(nn);
                    uint8_t pixel = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(
                        &preview_image3,
                        x,
                        y,
                        CLITERAL(Color){pixel, pixel, pixel, 255}
                    );
                }
            }
            UpdateTexture(preview_texture3, preview_image3.data);
            DrawTextureEx(
                preview_texture3,
                CLITERAL(Vector2){rx, ry + preview_height*scale*2},
                0,
                scale*2,
                WHITE
            );

            {
                float pad = rh*0.04f;
                Vector2 position = { rx, ry + preview_height*scale*4 + pad, };
                Vector2 size = { preview_width*scale*2, rh*0.008f, };
                float knob_radious = rh*0.02f;
                DrawRectangleV(position, size, RAYWHITE);
                Vector2 knob_position = {position.x + size.x*preview_scroll, position.y + size.y*0.5f}; 
                DrawCircleV(knob_position, knob_radious, RED);
                sprintf(buffer, "%f", preview_scroll);
                DrawText(buffer, position.x + size.x*3/8, position.y + 50, rh*0.04f, RAYWHITE);

                if (preview_scroll_dragging) {
                    float x = GetMousePosition().x;
                    if (x < position.x) { x = position.x; }
                    if (x > position.x + size.x) { x = position.x + size.x; }
                    preview_scroll = (x - position.x)/size.x;
                }

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    Vector2 mouse_position = GetMousePosition();
                    if (CheckCollisionPointCircle(mouse_position, knob_position, knob_radious)) {
                        preview_scroll_dragging = true;
                    }
                }

                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                    preview_scroll_dragging = false;
                }
            }
        }
        EndDrawing();
    }
    CloseWindow();

    return 0;
}
