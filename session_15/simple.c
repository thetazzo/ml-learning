#include <float.h>

#define NF_IMPLEMENTATION
#define NF_VISUALISATION
#include "nf.h"

int main() {
    size_t arch[] = {2, 3, 4, 2};
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
            nf_v_nn_render(nn, CLITERAL(NF_V_Rect){0, 0, w, h});
        EndDrawing();
    }

    return 0;
}
