#define NF_IMPLEMENTATION
#include "nf.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_img_write.h"

#include<time.h>

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_WIDTH*IMG_HEIGHT];

void nf_nn_render(Olivec_Canvas img, NF_NN nn) {
    uint32_t bg_color = 0xFF181818;
    uint32_t low_color  = 0x00FF00FF;
    uint32_t high_color = 0x0000FF00;
    uint32_t neuron_rad = 25;
    int layer_border_hpad = 50;
    int layer_border_vpad = 50;

    size_t arch_count = nn.count + 1;

    int nn_width   = img.width  - 2*layer_border_hpad;
    int nn_height  = img.height - 2*layer_border_vpad;
    int nn_y = img.height/2 - nn_height/2;
    int nn_x = img.width/2 - nn_width/2;

    int layer_hpad = nn_width / arch_count;

    olivec_fill(img, bg_color);
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
                    uint32_t alpha = floorf(255.f*sigmoidf(NF_MAT_AT(nn.ws[l], j, i)));
                    uint32_t connection_color = 0xFF000000|low_color;
                    olivec_blend_color(&connection_color, (alpha<<(8*3))|high_color);
                    olivec_line(img, cx1, cy1, cx2,cy2, connection_color);
                }
            }
            if (l > 0) {
                uint32_t alpha = floorf(255.f*sigmoidf(NF_MAT_AT(nn.bs[l-1], 0, i)));
                uint32_t neuron_color = 0xFF000000|low_color;
                olivec_blend_color(&neuron_color, (alpha<<(8*3))|high_color);
                olivec_circle(img, cx1, cy1, neuron_rad, neuron_color);
            } else {
                olivec_circle(img, cx1, cy1, neuron_rad, 0xFF505050);
            }
        }
    }
}

int main(void)
{
    srand(time(0));
    size_t arch[] ={4, 4, 2, 1};
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, -1, 1);
    NF_NN_PRINT(nn);

    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    nf_nn_render(img, nn);

    uint32_t frame_thicc = 10;
    uint32_t frame_color = 0xFFAAAAAA;
    olivec_frame(img, 0, 0, img.width-1, img.height-1, frame_thicc, frame_color);

    const char *img_file_path = "nf-render.png";
    if (!stbi_write_png(
        img_file_path,
        img.width,
        img.height,
        4,
        img.pixels,
        img.stride*sizeof(uint32_t))
    ) {
        printf("ERROR: could not write file '%s'", img_file_path);
        return 1;
    }

    printf("Saved neural network render to '%s'", img_file_path);

    return 0;
}
