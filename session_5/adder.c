
#include <stdio.h>
#define NF_IMPLEMENTATION
#include "./nf.h"

#define BITS 2

int main()
{
    size_t n = 1<<BITS; // 0001 << 2 -> 0010 aka 2^BITS 
    size_t rows = n*n;
    
    NF_Mat ti = nf_mat_alloc(rows, 2*BITS);
    NF_Mat to = nf_mat_alloc(rows, BITS + 1);  // + 1 ~ carry bit

    for (size_t i = 0; i < ti.rows; ++i) {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        size_t overflow = z >= n;
        for (size_t j = 0; j < BITS; ++j) {
            NF_MAT_AT(ti, i, j)      = (x>>j)&1;
            NF_MAT_AT(ti, i, j+BITS) = (y>>j)&1;
            if (overflow) {
                NF_MAT_AT(to, i, j)      = 0;
            } else {
                NF_MAT_AT(to, i, j)      = (z>>j)&1;
           }
        }
        NF_MAT_AT(to, i, BITS) = overflow;
    }

    NF_MAT_PRINT(ti);
    NF_MAT_PRINT(to);

    return 0;
}
