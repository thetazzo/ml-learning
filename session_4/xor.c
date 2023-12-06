#include <stdio.h>
#include <time.h>

#define NF_IMPLEMENTATION
#include "nf.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0, 
};

int main()
{
    // srand(time(0));
    srand(69);

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    NF_Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    NF_Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    size_t arch[] = {2, 2, 1};
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    NF_NN gn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, 0, 1);

    float rate = 1;

    for (size_t i = 0; i < 1000;++i) {
#if 0
        float eps = 1e-1;
        nf_nn_finite_diff(nn, gn, eps, ti, to);
#else
        nf_nn_backpropaga(nn, gn, ti, to);
#endif
        nf_nn_learn(nn, gn, rate);
        // printf("cost: %f\n", nf_nn_cost(nn, ti, to));
    }
    printf("cost: %f\n", nf_nn_cost(nn, ti, to));
    // NF_NN_PRINT(gn);

    printf("-----------------------------------\n");

    for (size_t i = 0; i <= 1; ++i) {
        for (size_t j = 0; j <= 1; ++j) {
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = i;
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = j;
            nf_nn_forward(nn);
            float y = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0);
            printf("%zu ^ %zu = %d\n", i, j, y > 0.5 ? 1 : 0);
        }
    }

    return 0;
}
