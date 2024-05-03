#include <float.h>

#define NF_IMPLEMENTATION
#define NF_VISUALISATION
#include "nf.h"

int main() {
    size_t arch[] = {2, 3, 4, 10, 12, 18, 10, 10, 2};
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, -1, 1);
    NF_NN_PRINT(nn);

    float factor = 120;
    int w = 16*factor;
    int h = 9*factor;
    InitWindow(w, h, "Simple");

    while(!WindowShouldClose()) {
        BeginDrawing();
            ClearBackground(BLACK);
            nf_v_layout_begin(VLO_HORZ, (CLITERAL(NF_V_Rect){0, 0, w, h}), 2, 0);
                nf_v_render_nn(nn, nf_v_layout_slot());
                nf_v_render_nn_as_cake(nn, nf_v_layout_slot());
            nf_v_layout_end();
        EndDrawing();
    }

    return 0;
}
