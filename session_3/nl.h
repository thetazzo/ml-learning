#ifndef NL_H_
#define NL_H_ 

#include <stddef.h>
#include <stdio.h>

#ifndef NL_MALLOC
#include <stdlib.h>
#define NL_MALLOC malloc
#endif // NL_MALLOC

#ifndef NL_ASSERT
#include <assert.h>
#define NL_ASSERT assert
#endif // NL_ASSERT

float rand_float(void);

// ---------------------------------------
//   Declatrations For Matrix Operations
// ---------------------------------------

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float  *es;
} NL_Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

/**
 * Allocate memory for a matrix
 */
NL_Mat nl_mat_alloc(size_t rows, size_t cols);

void nl_mat_rand(NL_Mat m, float low, float high);
void nl_mat_fill(NL_Mat m, float a);

void nl_mat_dot(NL_Mat dst, NL_Mat a, NL_Mat b);
void nl_mat_sum(NL_Mat dst, NL_Mat a);
void nl_mat_print(NL_Mat m);

#endif // NL_H_

#ifdef NL_IMPLEMENTATION

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

NL_Mat nl_mat_alloc(size_t rows, size_t cols)
{
    NL_Mat m = {0}; 
    m.rows = rows;
    m.cols = cols;
    m.es   = NL_MALLOC(sizeof(*m.es) * rows * cols);
    NL_ASSERT(m.es != NULL);
    return m;
}

void nl_mat_rand(NL_Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

void nl_mat_fill(NL_Mat m, float a)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = a;
        }
    }
}

void nl_mat_dot(NL_Mat dst, NL_Mat a, NL_Mat b)
{
    // validate rules of matrix multiplication
    NL_ASSERT(a.cols == b.rows);
    NL_ASSERT(dst.rows == a.rows);
    NL_ASSERT(dst.cols == b.cols);

    for (size_t k = 0; k < b.rows; ++k) {
        for (size_t i = 0; i < dst.rows; ++i) {
            for (size_t j = 0; j < dst.cols; ++j) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j); 
            }
        }
    }
}

void nl_mat_sum(NL_Mat dst, NL_Mat a)
{
    // validate that matrices are of the same order
    NL_ASSERT(dst.rows == a.rows);
    NL_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.rows; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void nl_mat_print(NL_Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}

#endif // NL_IMPLEMENTATION
