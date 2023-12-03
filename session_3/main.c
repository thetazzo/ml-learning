
#include <stdio.h>
#include <time.h>

#define NF_IMPLEMENTATION
#include "nf.h"

int main(void)
{
    srand(time(0));

    NF_Mat a = nf_mat_alloc(1, 2);
    nf_mat_rand(a, 0, 1);

    float id_mat_data[] = {
        1, 0,
        0, 1,
    };

    NF_Mat b = {
        .rows = 2,
        .cols = 2,
        .es = id_mat_data,
    };

    NF_Mat c = nf_mat_alloc(1, 2);

    printf("-------------------------\n");
    nf_mat_print(a);
    printf("-------------------------\n");
    nf_mat_print(b);
    printf("-------------------------\n");
    nf_mat_dot(c, a, b);
    nf_mat_print(c);

    return 0;
}
