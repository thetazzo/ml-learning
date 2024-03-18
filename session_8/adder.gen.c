#define NF_NN_IMPLEMENTATION
#include "nf.h"

#define BITS 4 

int main()
{
    size_t n = (1<<BITS); // 0001 << 2 -> 0010 aka 2^BITS 
    size_t rows = n*n;
    
    NF_Mat td = nf_mat_alloc(rows, 2*BITS + BITS+1);
    NF_Mat ti = {
        .es = &NF_MAT_AT(td, 0, 0),
        .rows = td.rows,
        .cols = 2*BITS,
        .stride = td.stride,
    };
    NF_Mat to = {
        .es = &NF_MAT_AT(td, 0, 2*BITS),
        .rows = td.rows,
        .cols = BITS+1,
        .stride = td.stride,
    };
    
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

    const char *out_file_path = "adder.mat";
    FILE *out = fopen(out_file_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
        return 1;
    }
    nf_mat_save(out, td);
    fclose(out);
    printf("Generated martrix data for Adder: '%s'\n", out_file_path);
    return 0;
}
