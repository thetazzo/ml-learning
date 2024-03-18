#ifndef NF_H_
#define NF_H_ 

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NF_MALLOC
#include <stdlib.h>
#define NF_MALLOC malloc
#endif // NF_MALLOC

#ifndef NF_ASSERT
#include <assert.h>
#define NF_ASSERT assert
#endif // NF_ASSERT

#define NF_ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

// ------------------------------------------
//            Activation Functions
// ------------------------------------------


// ---------------------------------------
//   Declatrations For Matrix Operations
// ---------------------------------------

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float  *es;
} NF_Mat;

#define NF_MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

// Allocate memory for a matrix
NF_Mat nf_mat_alloc(size_t rows, size_t cols);

void nf_mat_rand(NF_Mat m, float low, float high);
void nf_mat_fill(NF_Mat m, float a);

NF_Mat nf_mat_row(NF_Mat m, size_t row);

void nf_mat_copy(NF_Mat dst, NF_Mat src);

void nf_mat_dot(NF_Mat dst, NF_Mat a, NF_Mat b);
void nf_mat_sum(NF_Mat dst, NF_Mat a);
void nf_mat_print(NF_Mat m, const char *name, size_t padding);
#define NF_MAT_PRINT(m)  nf_mat_print((m), #m, 0)

void nf_mat_sig(NF_Mat m);

// ------------------------------------------------
//              Neural Network Declarations 
// ------------------------------------------------

typedef struct {
    size_t count;

    NF_Mat *ws;
    NF_Mat *bs;
    NF_Mat *as; // amount of total activations is count+1 because we also have a0 ~ input
} NF_NN;

#define NF_NN_INPUT(nn) (nn).as[0]
#define NF_NN_OUTPUT(nn) (nn).as[(nn).count]

NF_NN nf_nn_alloc(size_t *arch, size_t arch_count);

void nf_nn_fill(NF_NN nn, float a);

void nf_nn_print(NF_NN nn, const char *name);
#define NF_NN_PRINT(nn) nf_nn_print((nn), #nn) 

void nf_nn_rand(NF_NN nn, float low, float high);

void nf_nn_forward(NF_NN nn);

float nf_nn_cost(NF_NN nn, NF_Mat ti, NF_Mat to);
void nf_nn_finite_diff(NF_NN nn, NF_NN gn, float eps, NF_Mat ti, NF_Mat to);
void nf_nn_backprop(NF_NN nn, NF_NN gn, NF_Mat ti, NF_Mat to);
void nf_nn_learn(NF_NN nn, NF_NN gn, float rate);

#endif // NF_H_

#ifndef NF_IMPLEMENTATION
#define NF_IMPLEMENTATION 
// #error "remove NF_IMPLEMENTATION from nf.h"
#endif

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

// ------------------------------------------
//            Activation Functions
// ------------------------------------------
float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

/**
 * -------------------------------------
 *        Matrix Implementations 
 * -------------------------------------
 */

NF_Mat nf_mat_alloc(size_t rows, size_t cols)
{
    NF_Mat m = {0}; 
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (float*) NF_MALLOC(sizeof(*m.es) * rows * cols);
    NF_ASSERT(m.es != NULL);
    return m;
}

void nf_mat_rand(NF_Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            NF_MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

void nf_mat_fill(NF_Mat m, float a)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            NF_MAT_AT(m, i, j) = a;
        }
    }
}

NF_Mat nf_mat_row(NF_Mat m, size_t row)
{
    return (NF_Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride, 
        .es = &NF_MAT_AT(m, row, 0),
    };
}

void nf_mat_copy(NF_Mat dst, NF_Mat src)
{
    NF_ASSERT(dst.rows == src.rows);
    NF_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            NF_MAT_AT(dst, i, j) = NF_MAT_AT(src, i, j);
        }
    }
}

void nf_mat_dot(NF_Mat dst, NF_Mat a, NF_Mat b)
{
    // validate rules of matrix multiplication
    NF_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NF_ASSERT(dst.rows == a.rows);
    NF_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            NF_MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; ++k) {
                NF_MAT_AT(dst, i, j) += NF_MAT_AT(a, i, k) * NF_MAT_AT(b, k, j);
            }
        }
    }
}

void nf_mat_sum(NF_Mat dst, NF_Mat a)
{
    // validate that matrices are of the same order
    NF_ASSERT(dst.rows == a.rows);
    NF_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            NF_MAT_AT(dst, i, j) += NF_MAT_AT(a, i, j);
        }
    }
}

void nf_mat_print(NF_Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ",  NF_MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}

void nf_mat_sig(NF_Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            NF_MAT_AT(m, i, j) = sigmoidf(NF_MAT_AT(m, i, j));
        }
    }
}

void nf_mat_relu(NF_Mat m)
{
    for (size_t k = 0; k < m.rows; ++k) {
        for (size_t p = 0; p < m.cols; ++p) {
            float x = NF_MAT_AT(m, k, p);
            if (x > 0) {
                NF_MAT_AT(m, k, p) = x;
            } else {
                NF_MAT_AT(m, k, p) = 0;
            }
        }
    }
}

void nf_mat_lrelu(NF_Mat m)
{
    for (size_t k = 0; k < m.rows; ++k) {
        for (size_t p = 0; p < m.cols; ++p) {
            float x = NF_MAT_AT(m, k, p);
            if (x > 0) {
                NF_MAT_AT(m, k, p) = x;
            } else {
                NF_MAT_AT(m, k, p) = 0.01*x;
            }
        }
    }
}

// ------------------------------------------------
//              Neural Network Implementations 
// ------------------------------------------------

NF_NN nf_nn_alloc(size_t *arch, size_t arch_count)
{
    NF_ASSERT(arch_count > 0);

    NF_NN nn;
    nn.count = arch_count - 1;

    nn.ws = (NF_Mat*) NF_MALLOC(sizeof(*nn.ws)*nn.count);
    NF_ASSERT(nn.ws != NULL);
    nn.bs = (NF_Mat*) NF_MALLOC(sizeof(*nn.bs)*nn.count);
    NF_ASSERT(nn.bs != NULL);
    nn.as = (NF_Mat*) NF_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NF_ASSERT(nn.as != NULL);

    nn.as[0] = nf_mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count;++i) {
        nn.ws[i - 1] = nf_mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i - 1] = nf_mat_alloc(1, arch[i]);
        nn.as[i]     = nf_mat_alloc(1, arch[i]);
    }

    return nn;
}

void nf_nn_fill(NF_NN nn, float a)
{
    for (size_t l = 0; l < nn.count; ++l) {
        nf_mat_fill(nn.ws[l], a);
        nf_mat_fill(nn.bs[l], a);
        nf_mat_fill(nn.as[l], a);
    }
    nf_mat_fill(nn.as[nn.count], a);
}

void nf_nn_print(NF_NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);

    NF_Mat *ws = nn.ws;
    NF_Mat *bs = nn.bs;

    for (size_t i = 0; i < nn.count; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        nf_mat_print(ws[i], buf, 3);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        nf_mat_print(bs[i], buf, 3);
    }
    
    printf("]\n");
}

void nf_nn_rand(NF_NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; ++i) {
        nf_mat_rand(nn.ws[i], low, high);
        nf_mat_rand(nn.bs[i], low, high);
    }
}

void nf_nn_forward(NF_NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
        nf_mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        nf_mat_sum(nn.as[i+1], nn.bs[i]);
        nf_mat_sig(nn.as[i+1]);
        // nf_mat_lrelu(nn.as[i+1]);
        // nf_mat_relu(nn.as[i+1]);
    }
}

float nf_nn_cost(NF_NN nn, NF_Mat ti, NF_Mat to)
{
    NF_ASSERT(ti.rows == to.rows);
    NF_ASSERT(to.cols == NF_NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0; i < n; ++i) {
        NF_Mat x = nf_mat_row(ti, i);
        NF_Mat y = nf_mat_row(to, i);

        nf_mat_copy(NF_NN_INPUT(nn), x);
        nf_nn_forward(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, j) - NF_MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}

void nf_nn_finite_diff(NF_NN nn, NF_NN gn, float eps, NF_Mat ti, NF_Mat to)
{
    float saved;
    float c = nf_nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                saved = NF_MAT_AT(nn.ws[i], j, k);
                NF_MAT_AT(nn.ws[i], j, k) += eps;
                NF_MAT_AT(gn.ws[i], j, k) = (nf_nn_cost(nn, ti, to) - c)/eps;
                NF_MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                saved = NF_MAT_AT(nn.bs[i], j, k);
                NF_MAT_AT(nn.bs[i], j, k) += eps;
                NF_MAT_AT(gn.bs[i], j, k) = (nf_nn_cost(nn, ti, to) - c)/eps;
                NF_MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nf_nn_backprop(NF_NN nn, NF_NN gn, NF_Mat ti, NF_Mat to)
{
    NF_ASSERT(ti.rows == to.rows);
    NF_ASSERT(NF_NN_OUTPUT(nn).cols == to.cols);
    size_t n = ti.rows;
    nf_nn_fill(gn, 0);                                                          // clear the gradient network
    
    // Feed-Forward With Back-Propagation
    // sample - i
    for (size_t i = 0; i < n; ++i) {
        // --------------------------------------------------------------------------------------------------------------------------------------
        //  Feed-Forward
        //  forward the current sample(i-th row of ti) into the neual network
        // --------------------------------------------------------------------------------------------------------------------------------------
        nf_mat_copy(NF_NN_INPUT(nn), nf_mat_row(ti, i));
        nf_nn_forward(nn);

        // clean up activations of the gradient network
        for (size_t l = 0; l <= gn.count; ++l) {
            nf_mat_fill(gn.as[l], 0);
        }
        
        // Compute the differances of the next layer and store it as the output activaion (last layer activation) of the gradient neural network
        for (size_t j = 0; j < to.cols; ++j) {
            NF_MAT_AT(NF_NN_OUTPUT(gn), 0, j) = 2*(NF_MAT_AT(NF_NN_OUTPUT(nn), 0, j) - NF_MAT_AT(to, i, j)); 
        }
        
        // --------------------------------------------------------------------------------------------------------------------------------------
        //  Back-Propagation
        //  layer - l
        //  Note: 
        //   - in fact we have count-1 layers bacuase the 0th layer is the input layer -> the reason why I compute ws, bs, as of the (l-1) layer 
        //   - a0 wb0 a1 wb1 a2 wb2 ... a(n-1) wb(n-1) an 
        //   - l points to the layer after the current one
        //   - the last layer is the output layer 
        // --------------------------------------------------------------------------------------------------------------------------------------
        for (size_t l = nn.count; l > 0; --l) {
            // current activation - j
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a  = NF_MAT_AT(nn.as[l], 0, j);                           // j-th activation of the l-th layer
                float da = NF_MAT_AT(gn.as[l], 0, j);                           // j-th derivitive of the l-th activation
                float q  = a*(1-a);

                NF_MAT_AT(gn.bs[l-1], 0, j) += 2*da*q;                  // compute the partial derivitive for the j-th bias of the previoius(l-1) layer
                                                                                //
                // previoius activation - k
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                                                                                // Note: from the way we compute activations we can deduce
                                                                                //  - weight matrix col - j ~ j represents the column inside the matrix of current   activations
                                                                                //  - weight matrix row - k ~ k represents the row    inside the matrix of previoius activations

                    float pa = NF_MAT_AT(nn.as[l-1], 0, k);                     // k-th     activation of the previoius layer (l-1) 
                    float pw = NF_MAT_AT(nn.ws[l-1], k, j);                     // k,j-th   weight of the previoius layer (l-1)
 
                    NF_MAT_AT(gn.ws[l-1], k, j) += 2*da*q*pa;           // compute the partial derivitive for the k,j-th weight of the previoius layer (l-1)

                    NF_MAT_AT(gn.as[l-1], 0, k) += 2*da*q*pw;           // compute the partial derivitive for the 
                }
            }
        }
    }
    
    // normalizse the gradint aka do the 1/n part
    for (size_t l = 0; l < gn.count; ++l) {
        for (size_t i = 0; i < gn.ws[l].rows; ++i) {
            for (size_t j = 0; j < gn.ws[l].cols; ++j) {
                NF_MAT_AT(gn.ws[l], i, j) /= n;
            }
        }
        for (size_t i = 0; i < gn.bs[l].rows; ++i) {
            for (size_t j = 0; j < gn.bs[l].cols; ++j) {
                NF_MAT_AT(gn.bs[l], i, j) /= n;
            }
        }
    }
}

void nf_nn_learn(NF_NN nn, NF_NN gn, float rate)
{
    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                NF_MAT_AT(nn.ws[i], j, k) -= rate*NF_MAT_AT(gn.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                NF_MAT_AT(nn.bs[i], j, k) -= rate*NF_MAT_AT(gn.bs[i], j, k);
            }
        }
    }
}

#endif // NF_IMPLEMENTATION
