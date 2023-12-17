
#include <time.h>

#define NF_IMPLEMENTATION
#include "./nf.h"

#define BITS 4

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

    size_t arch[] = { 2*BITS, 4*BITS, BITS+1 };
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    NF_NN gn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, 0, 1);

    float rate = 1;
    for (size_t i = 0; i < 1000*20; ++i) {
#if 1
        nf_nn_backprop(nn, gn, ti, to); 
#else
        nf_nn_finite_diff(nn, gn, 1e-1, ti, to);
#endif
        nf_nn_learn(nn, gn, rate);
        printf("%zu: cost: %f\n", i, nf_nn_cost(nn, ti, to));
    }

    printf("----------------------------\n");

    size_t fails = 0;
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
                    fails += 1;
                }
            } else {
                size_t a = 0;
                for (size_t j = 0; j < BITS; ++j) {
                    size_t bit = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, j) > 0.5f;
                    a |= bit<<j;
                }
                if (z != a) {
                    printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
                    fails += 1;
                }
            }
        }
    }
    if (fails == 0) {
        printf("You are OK, you are OK Annie\n");
    } else {
        printf("fails: %zu\n", fails);
    }

    return 0;
}
