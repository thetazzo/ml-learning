#include <stdio.h>
#define NF_NN_IMPLEMENTATION
#include "nf.h"

int main()
{
    NF_Mat td = nf_mat_alloc(4, 3);
    for (size_t i=0;i<2;++i) {
         for (size_t j=0;j<2;++j) {
            size_t row = i*2+j;
            NF_MAT_AT(td, row, 0) = i;
            NF_MAT_AT(td, row, 1) = j;
            NF_MAT_AT(td, row, 2) = i^j;
        }
    }
    
    NF_MAT_PRINT(td);

    const char *out_file_path = "xor.mat";
    FILE *out = fopen(out_file_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
        return 1;
    }
    nf_mat_save(out, td);
    printf("Generated %s\n", out_file_path);
    return 0;
}
