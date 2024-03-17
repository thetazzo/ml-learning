#include "./nf.h"
#include <stdio.h>

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main() {
    size_t rc = 4; // rows count
    size_t stride = 3;

    NF_Mat ti = {
        .rows = rc,
        .cols = 2,
        .stride = stride,
        .es = td,
    };
    NF_Mat to = {
        .rows = rc,
        .cols = 1,
        .stride = stride,
        .es = td,
    };

    size_t arch[] = {2,2,1};
    NF_NN nn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    NF_NN gn = nf_nn_alloc(arch, NF_ARRAY_LEN(arch));
    nf_nn_rand(nn, 0, 1);

    float rate = 1;

    printf("cost: %f\n", nf_nn_cost(nn, ti, to));

    size_t max_epoch = 10*1000;
    for (size_t i = 0; i < max_epoch; ++i) { 
        nf_nn_backprop(nn, gn, ti, to);
        nf_nn_learn(nn, gn, rate);
    }

    printf("cost: %f\n", nf_nn_cost(nn, ti, to));

    NF_NN_PRINT(nn);

    return 0;
}
