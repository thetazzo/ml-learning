#ifndef NF_H_
#define NF_H_ 

#include <stddef.h>
#include <stdio.h>

#ifndef NF_MALLOC
#include <stdlib.h>
#define NF_MALLOC malloc
#endif // NF_MALLOC

#ifndef NF_ASSERT
#include <assert.h>
#define NF_ASSERT assert
#endif // NF_ASSERT

float rand_float(void);

// ---------------------------------------
//   Declatrations For Matrix Operations
// ---------------------------------------

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float  *es;
} NF_Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

/**
 * Allocate memory for a matrix
 */
NF_Mat nl_mat_alloc(size_t rows, size_t cols);

void nl_mat_rand(NF_Mat m, float low, float high);
void nl_mat_fill(NF_Mat m, float a);

void nl_mat_dot(NF_Mat dst, NF_Mat a, NF_Mat b);
void nl_mat_sum(NF_Mat dst, NF_Mat a);
void nl_mat_print(NF_Mat m);

#endif // NF_H_

#ifdef NF_IMPLEMENTATION

/**
 * -------------------------------------
 *   Utility Functions Implementations
 * -------------------------------------
 */
float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

/**
 * -------------------------------------
 *        Matrix Implementations 
 * -------------------------------------
 */

NF_Mat nl_mat_alloc(size_t rows, size_t cols)
{
    NF_Mat m = {0}; 
    m.rows = rows;
    m.cols = cols;
    m.es   = NF_MALLOC(sizeof(*m.es) * rows * cols);
    NF_ASSERT(m.es != NULL);
    return m;
}

void nl_mat_rand(NF_Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

void nl_mat_fill(NF_Mat m, float a)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = a;
        }
    }
}

void nl_mat_dot(NF_Mat dst, NF_Mat a, NF_Mat b)
{
    // validate rules of matrix multiplication
    NF_ASSERT(a.cols == b.rows);
    NF_ASSERT(dst.rows == a.rows);
    NF_ASSERT(dst.cols == b.cols);

    for (size_t k = 0; k < b.rows; ++k) {
        for (size_t i = 0; i < dst.rows; ++i) {
            for (size_t j = 0; j < dst.cols; ++j) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j); 
            }
        }
    }
}

void nl_mat_sum(NF_Mat dst, NF_Mat a)
{
    // validate that matrices are of the same order
    NF_ASSERT(dst.rows == a.rows);
    NF_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.rows; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void nl_mat_print(NF_Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}

#endif // NF_IMPLEMENTATION
