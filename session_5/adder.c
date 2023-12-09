
#define NF_IMPLEMENTATION
#include "./nf.h"

#define BITS 2

int main()
{
    size_t n = 1<<BITS;
    size_t rows = n*n;
    NF_Mat ti = nf_mat_alloc(rows, 2*BITS);

    for (size_t i = 0; i < ti.rows; ++i) {
        size_t x = i/n;
        size_t y = i%n;
        for (size_t j = 0; j < BITS; ++j) {
            NF_MAT_AT(ti, i, j) =rows, 2*BITS (x>>j)&1;
            NF_MAT_AT(ti, i, j*BITS) = (i>>j)&1;
        }
    }

    return 0;
}
