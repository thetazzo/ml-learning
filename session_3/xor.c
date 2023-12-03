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
    srand(time(0));

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
    NF_NN nn = nf_nn_alloc(arch, ARRAY_LEN(arch));
    NF_NN ng = nf_nn_alloc(arch, ARRAY_LEN(arch));
    nf_nn_rand(nn, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost: %f\n", nf_nn_cost(nn, ti, to));
    for (size_t i = 0; i < 15*1000;++i) {
        nf_nn_finite_diff(nn, ng, eps, ti, to);
        nf_nn_learn(nn, ng, rate);
        printf("cost: %f\n", nf_nn_cost(nn, ti, to));
    }

    printf("-----------------------------------\n");

    for (size_t i = 0; i <= 1; ++i) {
        for (size_t j = 0; j <= 1; ++j) {
            MAT_AT(NF_NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NF_NN_INPUT(nn), 0, 1) = j;
            nf_nn_forward(nn);
            float y = MAT_AT(NF_NN_OUTPUT(nn), 0, 0);
            printf("%zu ^ %zu = %d\n", i, j, y > 0.5 ? 1 : 0);
        }
    }

    return 0;
}
