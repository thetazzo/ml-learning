
#include <stdio.h>
#include <time.h>

#define NL_IMPLEMENTATION
#include "nl.h"

int main(void)
{
    srand(time(0));

    NL_Mat a = nl_mat_alloc(1, 2);
    nl_mat_rand(a, 0, 1);

    float id_mat_data[] = {
        1, 0,
        0, 1,
    };

    NL_Mat b = {
        .rows = 2,
        .cols = 2,
        .es = id_mat_data,
    };

    NL_Mat c = nl_mat_alloc(1, 2);

    printf("-------------------------\n");
    nl_mat_print(a);
    printf("-------------------------\n");
    nl_mat_print(b);
    printf("-------------------------\n");
    nl_mat_dot(c, a, b);
    nl_mat_print(c);

    return 0;
}
