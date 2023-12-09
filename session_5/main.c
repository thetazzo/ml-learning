#include <stdio.h>
#include <time.h>

#define NF_IMPLEMENTATION
#include "./nf.h"

float td[] = {
    1, 2,
    2, 3,
    3, 4,
    4, 5,
};

int main()
{
    srand(time(0));
    // srand(69);

    NF_Mat ti = nf_mat_alloc(4, 1);
    ti.es = td;
    NF_Mat to = nf_mat_alloc(4, 1);
    to.es = td+1;

    size_t arch[] = {1, 2, 3, 1};
    size_t arch_count = NF_ARRAY_LEN(arch);
    NF_NN nn = nf_nn_alloc(arch,  arch_count);
    NF_NN gn = nf_nn_alloc(arch,  arch_count);
    nf_nn_rand(nn, 0, 1);

    float rate = 1e-2;

    for (size_t i = 0; i < 18*20; ++i) {
        nf_nn_backpropaga(nn, gn, ti, to);
        nf_nn_learn(nn, gn, rate);
        printf("cost: %f\n", nf_nn_cost(nn, ti, to));
    }
    printf("---------------------------------\n");

    size_t i = 2;
    NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = i;
    nf_nn_forward(nn);
    float y = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0);
    printf("cost: %f, %zu*w+b: %f\n", nf_nn_cost(nn, ti, to), i, y);

    return 0;
}
