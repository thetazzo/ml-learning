#ifndef NF_H_
#define NF_H_ 

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NF_BACKPROP_TRADITIONAL
// TODO: reafacor this to be part of the NF_NN typedef
#define NF_NN_ACT NF_ACT_TANH

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

// ------------------------------------------
//            Activation Functions
// ------------------------------------------
typedef enum {
    NF_ACT_SIG,
    NF_ACT_RELU,
    NF_ACT_LRELU,
    NF_ACT_TANH,
    NF_ACT_SIN,
} NF_Act;

float nf_sigmoidf(float x);
float nf_reluf(float x);
float nf_lreluf(float x);
float nf_tanhf(float x);
float nf_gelu(float x);

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
void nf_mat_save(FILE *out, NF_Mat m);
NF_Mat nf_mat_load(FILE *in);
void nf_mat_rand(NF_Mat m, float low, float high);
void nf_mat_fill(NF_Mat m, float a);
NF_Mat nf_mat_row(NF_Mat m, size_t row);
void nf_mat_copy(NF_Mat dst, NF_Mat src);
void nf_mat_dot(NF_Mat dst, NF_Mat a, NF_Mat b);
void nf_mat_sum(NF_Mat dst, NF_Mat a);
void nf_mat_shuffle_rows(NF_Mat m);
void nf_mat_print(NF_Mat m, const char *name, size_t padding);
#define NF_MAT_PRINT(m)  nf_mat_print((m), #m, 0)

// Handling the activation of the matrix
void nf_mat_act(NF_Mat m);

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
float nf_sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float nf_reluf(float x)
{
    if (x > 0) {
        return x;
    }
    return 0;
}

float nf_lreluf(float x)
{
    if (x < 0) {
        return 0.01*x;
    }
    return x;
}

float nf_tanhf(float x)
{
    return (expf(2*x) - 1)/(expf(2*x) + 1);
}

float nf_gelu(float x)
{
    (void)x;
    NF_ASSERT(0 && "TODO: implement!");
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

void nf_mat_save(FILE *out, NF_Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for (size_t i = 0; i < m.rows; ++i) {
        size_t n = fwrite(&NF_MAT_AT(m, i, 0), sizeof(*m.es), m.rows*m.cols, out);
        while (n < m.cols && !ferror(out)) {
            size_t k = fwrite(m.es + n, sizeof(*m.es), m.cols - n, out);
            n += k;
        }
    }
}

NF_Mat nf_mat_load(FILE *in)
{
    uint64_t magic;
    fread(&magic, sizeof(magic), 1, in);
    NF_ASSERT(magic == 0x74616d2e682e6e6e);
    size_t rows, cols;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);
    NF_Mat m = nf_mat_alloc(rows, cols);
    
    size_t n = fread(m.es, sizeof(*m.es), rows*cols, in);
    while (n < rows*cols && !ferror(in)) {
        size_t k = fread(m.es, sizeof(*m.es) + n, rows*cols - n, in);
        n += k;
    }
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

void nf_mat_shuffle_rows(NF_Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        size_t j = i+rand()%(m.rows - i);
        if (i != j) {
            for (size_t k = 0; k < m.cols; ++k) {
                float t = NF_MAT_AT(m, i, k);
                NF_MAT_AT(m, i, k) = NF_MAT_AT(m, j, k);
                NF_MAT_AT(m, j, k) = t;
            }
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

void nf_mat_act(NF_Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            switch (NF_NN_ACT) {
                case NF_ACT_SIG:
                    NF_MAT_AT(m, i, j) = nf_sigmoidf(NF_MAT_AT(m, i, j));
                    break;
                case NF_ACT_RELU: 
                    NF_MAT_AT(m, i, j) = nf_reluf(NF_MAT_AT(m, i, j));
                    break;
                case NF_ACT_LRELU:
                    NF_MAT_AT(m, i, j) = nf_lreluf(NF_MAT_AT(m, i, j));
                    break;
                case NF_ACT_TANH:
                    NF_MAT_AT(m, i, j) = nf_tanhf(NF_MAT_AT(m, i, j));
                    break;
                case NF_ACT_SIN:
                    NF_MAT_AT(m, i, j) = sinf(NF_MAT_AT(m, i, j));
                    break;
                default:
                    NF_ASSERT(0 && "Unreachable");
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
        nf_mat_act(nn.as[i+1]);
        //nf_mat_lrelu(nn.as[i+1]);
        //nf_mat_relu(nn.as[i+1]);
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
        #ifdef NF_BACKPROP_TRADITIONAL
            NF_MAT_AT(NF_NN_OUTPUT(gn), 0, j) = 2*(NF_MAT_AT(NF_NN_OUTPUT(nn), 0, j) - NF_MAT_AT(to, i, j)); 
        #else 
            NF_MAT_AT(NF_NN_OUTPUT(gn), 0, j) = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, j) - NF_MAT_AT(to, i, j); 
        #endif //NF_BACKPROP_TRADITIONAL
        }

        #ifdef NF_BACKPROP_TRADITIONAL
                float s = 1.f;
        #else
                float s = 2.f;
        #endif //NF_BACKPROP_TRADITIONAL
        
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
                // j-th activation of the l-th layer
                float a  = NF_MAT_AT(nn.as[l], 0, j);
                // j-th derivitive of the l-th activation                        
                float da = NF_MAT_AT(gn.as[l], 0, j);                           
                // comopute the derivitive of the activation function
                float act_deriv;
                switch (NF_NN_ACT) {
                    case NF_ACT_SIG:
                        act_deriv = a*(1-a);
                        break;
                    case NF_ACT_RELU:
                        act_deriv = a >= 0 ? 1 : 0;
                        break;
                    case NF_ACT_LRELU:
                        act_deriv = a >= 0 ? 1 : 0.01f;
                        break;
                    case NF_ACT_TANH:
                        act_deriv = 1 - a*a;
                        break;
                    case NF_ACT_SIN:
                        NF_ASSERT(0 && "Unsupported");
                        break;
                    default:
                        NF_ASSERT(0 && "Unreachable");
                }
                // compute the partial derivitive for the j-th bias of the previoius(l-1) layer
                NF_MAT_AT(gn.bs[l-1], 0, j) += s*da*act_deriv;
                // previoius activation - k
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    // k-th activation of the previoius layer (l-1)
                    float pa = NF_MAT_AT(nn.as[l-1], 0, k);                      
                    // k-th,j-th weight of the previoius layer (l-1)
                    float pw = NF_MAT_AT(nn.ws[l-1], k, j);                     
                    // compute the partial derivitive for the k,j-th weight of the previoius layer (l-1)
                    NF_MAT_AT(gn.ws[l-1], k, j) += s*da*act_deriv*pa;           
                    // compute the partial derivitive for the 
                    NF_MAT_AT(gn.as[l-1], 0, k) += s*da*act_deriv*pw;           
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
