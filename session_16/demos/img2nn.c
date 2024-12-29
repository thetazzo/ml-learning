#define NF_NN_ACT NF_ACT_SIG

#define NF_VISUALISATION
#define NF_IMAGE_GENERATION
#define NF_IMPLEMENTATION
#include "../nf.h"

char *p2m_shift_args(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1; 
    (*argv) += 1;
    return result;
}

void nf_v_preview_image(NF_NN nn, NF_V_Rect r, float scale, float pimg_index)
{
    Image preview_image = GenImageColor(r.w, r.h, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    // Draw preview image 1
    for (size_t y = 0; y < r.h; ++y) {
        for (size_t x = 0; x < r.w; ++x) {
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(r.w - 1);;
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(r.h - 1);
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 2) = pimg_index;
            nf_nn_forward(nn);
            float act = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0);
            if (act < 0) act = 0;
            if (act > 1) act = 1;
            uint8_t pixel = act*255.f;
            ImageDrawPixel(
                &preview_image,
                x,
                y,
                CLITERAL(Color){pixel, pixel, pixel, 255}
            );
        }
    }
    UpdateTexture(preview_texture, preview_image.data);
    DrawTextureEx(
        preview_texture,
        CLITERAL(Vector2){r.x, r.y + r.h*scale},
        0,
        scale,
        WHITE
    );
}

// neural network architecture
//size_t arch[] = {3, 28, 16, 15, 14, 13, 12, 10, 6, 12, 7, 4, 3, 1};
size_t arch[] = {3, 11, 18, 9, 4, 1};
bool isRunning = false;

int main(int argc, char **argv)
{
    Region temp = region_alloc_alloc(256*1024*1024);

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

    NF_NN nn = nf_nn_alloc(NULL, arch, NF_ARRAY_LEN(arch));

    NF_Mat td = nf_mat_alloc(
        NULL,
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

    //SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "img2nn");
    SetTargetFPS(60);

    Color bg_color = { 0x12, 0x12, 0x12, 0xFF };

    NF_V_Plot plot = {0};

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
    Batch batch = {0};
    size_t batch_size = 28;
    float rate = 1.f;
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
            NF_V_Plot ncp = {0};
            plot = ncp;
            nf_nn_rand(nn, -1, 1);
        }

        // screenshot neural network based on preview scroll 
        if (IsKeyPressed(KEY_S)) {
            nf_v_render_upscaled_screenshot(nn, preview_scroll, "number-upscaled.png");
        }

        // render upscaled video
        if (IsKeyPressed(KEY_X)) {
            nf_v_render_upscaled_video(nn, 5, "number-upscaled.mp4");
        }

        for (size_t i = 0; i < bpf && epoch < max_epoch && isRunning; ++i) {
            nf_batch_process(&temp, &batch, batch_size, nn, td, rate);
            if (batch.done) {
                epoch += 1;
                da_append(&plot, batch.cost);
                nf_mat_shuffle_rows(td);
            }
        }  
        BeginDrawing();
        ClearBackground(bg_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();
            float scale = h*0.009f;
            float frame = h*0.1;

            nf_v_layout_begin(VLO_HORZ, (CLITERAL(NF_V_Rect){frame,frame,w-2*frame,h-2*frame}), 3, 0);
            NF_V_Rect fsr = nf_v_layout_slot();
            nf_v_plot(plot, fsr);

            char buffer[256]; 
            sprintf(buffer, "Cost: %g", plot.count > 0 ? plot.items[plot.count - 1] : 0);
            DrawText(buffer, fsr.x+fsr.w + 100, fsr.y+fsr.h-40, fsr.h*0.05f, RAYWHITE);

            sprintf(buffer, "Epochs: %zu/%zu, Rate: %f, Mem Usage: %lu\n", epoch, max_epoch, rate, region_occupied_bytes(&temp));
            DrawText(buffer, fsr.x+fsr.w + 50, 20, fsr.h*0.03f, RAYWHITE);

            nf_v_slider(&rate, &lrate_scroll_dragging, fsr.x + fsr.w*2 + 50, fsr.y - 80, fsr.w, 20);

            //nf_v_nn_render(nn, nf_v_layout_slot());
            NF_V_Rect nnr = nf_v_layout_slot();
            nnr.h -= 70;
            nnr.y += 30;
            nnr.w -= 70;
            nnr.x += 30;
            //nf_v_render_nn_activations_heatmap(nn, nnr);
            nf_v_render_nn(nn, nnr);

            NF_V_Rect isr = nf_v_layout_slot();
            isr.x += isr.w/12;
            //isr.y += isr.h/12;
            // Draw original image 1 
            DrawTextureEx(original_texture1,CLITERAL(Vector2){isr.x, isr.y}, 0, scale, WHITE);
            // Draw original image 2 
            DrawTextureEx(original_texture2,CLITERAL(Vector2){isr.x+img1_width*scale, isr.y}, 0, scale, WHITE);
            
            // Draw preview image 1
            NF_V_Rect pi1r = { isr.x, isr.y, preview_width, preview_height };
            nf_v_preview_image(nn, pi1r, scale, 0.f);
            // Draw preview image 2
            NF_V_Rect pi2r = { isr.x + preview_width*scale, isr.y, preview_width, preview_height };
            nf_v_preview_image(nn, pi2r, scale, 1.f);
            // Draw preview image 3
            NF_V_Rect pi3r = { isr.x + preview_width/2*scale, isr.y+preview_height*scale, preview_width, preview_height };
            nf_v_preview_image(nn, pi3r, scale, preview_scroll);
            // Preview image 3 slider
            nf_v_slider(&preview_scroll, &preview_scroll_dragging, isr.x - isr.w/12, isr.y+ 3*(preview_height*scale) + 75, isr.w, 20);
            nf_v_layout_end();
        }
        EndDrawing();
        region_reset(&temp);
    }
    CloseWindow();

    return 0;
}
