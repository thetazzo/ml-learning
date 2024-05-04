#define NF_VISUALISATION
#define NF_IMPLEMENTATION
#include "nf.h"

NF_Mat prepare_training_data()
{
    NF_Mat td = nf_mat_alloc(NULL, 4, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            size_t row = 2*i + j; 
            NF_MAT_AT(td, row, 0) = i;
            NF_MAT_AT(td, row, 1) = j;
            NF_MAT_AT(td, row, 2) = j^i;
        }
    }
    return td;
}

#define F 120
#define WIDTH 16*F
#define HEIGHT 9*F

size_t arch[] = {2, 2, 1};
float rate = 0.5f;
size_t max_epoch = 10000;
size_t batch_pre_frame = 100;
bool running = false;

int main(void)
{
    Region tmp_mem = region_alloc_alloc(256*1024*1024);

    NF_Mat td = prepare_training_data();

    NF_NN nn = nf_nn_alloc(NULL, arch, NF_ARRAY_LEN(arch));
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
        .es = &NF_MAT_AT(td, 0, 2),
    };


    InitWindow(WIDTH, HEIGHT, "xor");
    SetTargetFPS(60);

    Font font = LoadFontEx("./fonts/iosevka-term-ss02-regular.ttf", 72, NULL, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);

    Batch batch = {0};
    size_t batch_size = 20;
    size_t epoch = 0;

    NF_V_Plot plot = {0};

    while (!WindowShouldClose()) {

        if (IsKeyPressed(KEY_SPACE)) {
            running = !running;
        }

        if (IsKeyPressed(KEY_R)) {
            NF_V_Plot tmp = {0};
            plot = tmp;
            epoch = 0;
            nf_nn_rand(nn, -1, 1);
        }

        int w = GetScreenWidth();
        int h = GetScreenHeight();
        for (size_t i = 0; i < batch_pre_frame && epoch < max_epoch && running; ++i) {
            nf_batch_process(&tmp_mem, &batch, batch_size, nn, td, rate);
            if (batch.done) {
                epoch += 1;
                da_append(&plot, batch.cost);
                nf_mat_shuffle_rows(td);
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        NF_V_Rect root = {
            0, 0, w, h,
        };
        root.x += 50;
        root.w -= 50;
        root.y += 100;
        root.h -= 200;
        nf_v_layout_begin(VLO_HORZ, root, 3, 0);
            char buffer[512];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f, Temporary Memory: %zu\n", epoch, max_epoch, rate, nf_nn_cost(nn, ti, to), region_occupied_bytes(&tmp_mem));
            DrawTextEx(font, buffer, CLITERAL(Vector2){root.x, root.y-100}, h*0.04, 0, WHITE); 
            nf_v_plot(plot, nf_v_layout_slot());
            nf_v_render_nn(nn, nf_v_layout_slot());
            NF_V_Rect out_slot = nf_v_layout_slot();
            // --------------------------------------------------
            NF_MAT_AT(NF_NN_INPUT(nn),0,0) = 0;
            NF_MAT_AT(NF_NN_INPUT(nn),0,1) = 0;
            nf_nn_forward(nn);
            snprintf(buffer, sizeof(buffer), "0 ^ 0 = %f",NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)); 
            DrawTextEx(font, buffer, CLITERAL(Vector2){out_slot.x, out_slot.y + 0*100}, h*0.04, 0, WHITE); 
            // --------------------------------------------------
            NF_MAT_AT(NF_NN_INPUT(nn),0,0) = 0;
            NF_MAT_AT(NF_NN_INPUT(nn),0,1) = 1;
            nf_nn_forward(nn);
            snprintf(buffer, sizeof(buffer), "0 ^ 1 = %f",NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)); 
            DrawTextEx(font, buffer, CLITERAL(Vector2){out_slot.x, out_slot.y + 1*100}, h*0.04, 0, WHITE); 
            // --------------------------------------------------
            NF_MAT_AT(NF_NN_INPUT(nn),0,0) = 1;
            NF_MAT_AT(NF_NN_INPUT(nn),0,1) = 0;
            nf_nn_forward(nn);
            snprintf(buffer, sizeof(buffer), "1 ^ 0 = %f",NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)); 
            DrawTextEx(font, buffer, CLITERAL(Vector2){out_slot.x, out_slot.y + 2*100}, h*0.04, 0, WHITE); 
            // --------------------------------------------------
            NF_MAT_AT(NF_NN_INPUT(nn),0,0) = 1;
            NF_MAT_AT(NF_NN_INPUT(nn),0,1) = 1;
            nf_nn_forward(nn);
            snprintf(buffer, sizeof(buffer), "1 ^ 1 = %f",NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0)); 
            DrawTextEx(font, buffer, CLITERAL(Vector2){out_slot.x, out_slot.y + 3*100}, h*0.04, 0, WHITE); 
        nf_v_layout_end();

        EndDrawing();
        region_reset(&tmp_mem);
    }

    CloseWindow();
    return 0;
}

