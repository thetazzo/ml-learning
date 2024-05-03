#ifndef NF_H_
#define NF_H_ 

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
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

#ifdef NF_VISUALISATION
#include <math.h>
#include <errno.h>
// posix specific headers
// allows forking childs in linux
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <float.h>

#include "raylib.h"
#include "raymath.h"

typedef struct {
    float x;
    float y;
    float w;
    float h;
} NF_V_Rect;

typedef enum {
    VLO_HORZ,
    VLO_VERT,
} NF_V_Layout_Orient;

typedef struct {
    NF_V_Rect rect;
    NF_V_Layout_Orient orient;
    size_t count;
    size_t i;
    float gap;
} NF_V_Layout;

NF_V_Rect nf_v_layout_slot_loc(NF_V_Layout *l, const char *file_path, int line);

typedef struct {
    NF_V_Layout *items;
    size_t count;
    size_t capacity;
} NF_V_Layout_Stack;

void nf_v_layout_stack_push(NF_V_Layout_Stack *ls, NF_V_Layout_Orient orient, NF_V_Rect rect, size_t count, float gap);
#define nf_v_layout_stack_slot(ls) (assert((ls)->count > 0), nf_v_layout_slot_loc(&(ls)->items[(ls)->count - 1], __FILE__, __LINE__))
#define nf_v_layout_stack_pop(ls) do { assert((ls)->count > 0); (ls)->count -= 1; } while (0)

static NF_V_Layout_Stack default_nf_v_layout_stack = {0};

#define nf_v_layout_begin(orient, rect, count, gap) nf_v_layout_stack_push(&default_nf_v_layout_stack, (orient), (rect), (count), (gap))
#define nf_v_layout_end() nf_v_layout_stack_pop(&default_nf_v_layout_stack)
#define nf_v_layout_slot() nf_v_layout_stack_slot(&default_nf_v_layout_stack)

typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} NF_V_Plot;

#define DA_INIT_CAP 256
#define da_append(da, item)                                                                 \
    do {                                                                                    \
        if ((da)->count >= (da)->capacity) {                                                \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;          \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items));        \
            assert((da)->items != NULL && "Buy more RAM");                                  \
        }                                                                                   \
        (da)->items[(da)->count++] = (item);                                                \
    } while (0)                                                                           

void nf_v_widget(NF_V_Rect r);

void nf_v_render_nn(NF_NN nn, NF_V_Rect r);
void nf_v_plot_cost(NF_V_Plot plot, NF_V_Rect);
void nf_v_slider(float *value, bool *is_dragging, float rx, float ry, float rw, float rh);

void nf_v_render_mat_as_cake(NF_V_Rect r, NF_Mat m);
void nf_v_render_nn_as_cake(NF_NN nn, NF_V_Rect r);

#ifdef NF_IMAGE_GENERATION
#include "stb_image.h"
#include "stb_image_write.h"

void nf_v_render_single_frame(NF_NN nn, float img_index);
int  nf_v_render_upscaled_screenshot(NF_NN nn, float img_index, const char *out_file_path);
int  nf_v_render_upscaled_video(NF_NN nn, float duration, const char *out_file_path);
#endif //NF_IMAGE_GENERATION

#endif // NF_VISUALISATION

#endif // NF_H_

#ifdef NF_IMPLEMENTATION

//-------------------------------------
//  Utility Functions Implementations
//-------------------------------------
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
    assert(0 && "this is broken");
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

#ifdef NF_VISUALISATION

NF_V_Rect nf_v_layout_slot_loc(NF_V_Layout *l, const char *file_path, int line)
{
    if (l->i >= l->count) {
        fprintf(stderr, "%s:%d: ERROR: Layout overflow\n", file_path, line);
        exit(1);
    }

    NF_V_Rect r = {0};

    switch (l->orient) {
    case VLO_HORZ:
        r.w = l->rect.w/l->count;
        r.h = l->rect.h;
        r.x = l->rect.x + l->i*r.w;
        r.y = l->rect.y;

        if (l->i == 0) { // First
            r.w -= l->gap/2;
        } else if (l->i >= l->count - 1) { // Last
            r.x += l->gap/2;
            r.w -= l->gap/2;
        } else { // Middle
            r.x += l->gap/2;
            r.w -= l->gap;
        }

        break;

    case VLO_VERT:
        r.w = l->rect.w;
        r.h = l->rect.h/l->count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i*r.h;

        if (l->i == 0) { // First
            r.h -= l->gap/2;
        } else if (l->i >= l->count - 1) { // Last
            r.y += l->gap/2;
            r.h -= l->gap/2;
        } else { // Middle
            r.y += l->gap/2;
            r.h -= l->gap;
        }

        break;

    default:
        assert(0 && "Unreachable");
    }

    l->i += 1;

    return r;
}

void nf_v_layout_stack_push(NF_V_Layout_Stack *ls, NF_V_Layout_Orient orient, NF_V_Rect rect, size_t count, float gap)
{
    NF_V_Layout l = {0};
    l.orient = orient;
    l.rect = rect;
    l.count = count;
    l.gap = gap;
    da_append(ls, l);
}

void nf_v_render_nn(NF_NN nn, NF_V_Rect r) {
    Color low_color  = { 0xFF, 0x00, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

    float neuron_rad = r.h*0.03;
    float layer_border_hpad = 50;
    float layer_border_vpad = 50;

    size_t arch_count = nn.count + 1;

    float nn_width   = r.w - 2*layer_border_hpad;
    float nn_height  = r.h - 2*layer_border_vpad;
    float nn_x       = r.x + r.w/2 - nn_width/2;
    float nn_y       = r.y + r.h/2 - nn_height/2;

    int layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; ++l) {
        int layer_vpad1 = nn_height/nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            float cx1 = nn_x + l*layer_hpad + layer_hpad/2; 
            float cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if (l+1 < arch_count) {
                float layer_vpad2 = nn_height/nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    float cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2; 
                    float cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    float value = nf_sigmoidf(NF_MAT_AT(nn.ws[l], j, i));
                    high_color.a = floorf(255.f*value);
                    float thicc = r.h*0.004f;

                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};

                    DrawLineEx(
                        start,
                        end,
                        thicc,
                        ColorAlphaBlend(low_color, high_color, WHITE)
                    );
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f*nf_sigmoidf(NF_MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_rad, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, neuron_rad, GRAY);
            }
        }
    }
}

void nf_v_plot_cost(NF_V_Plot plot, NF_V_Rect r) 
{
    float min = FLT_MAX;
    float max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (max < plot.items[i]) { max = plot.items[i]; }
        if (min > plot.items[i]) { min = plot.items[i]; }
    }
    if (min > 0) min = 0;

    size_t n = plot.count;

    if (n < 100) n = 100;

    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = r.x + (float)r.w/n * i; 
        float y1 = r.y + (1-(plot.items[i] - min)/(max-min))*r.h;
        float x2 = r.x + (float)r.w/n * (i+1); 
        float y2 = r.y + (1-(plot.items[i+1] - min)/(max-min))*r.h;

        DrawLineEx((Vector2){x1,y1}, (Vector2){x2,y2}, r.h*0.0035f, YELLOW);
        DrawLine(0, r.y+r.h, r.x+r.w+60, r.y+r.h, RAYWHITE);
        DrawLine(r.x, r.y+r.h+50, r.x, 50, RAYWHITE);
        DrawCircle(r.x, r.y+r.h, r.h*0.008f, RAYWHITE);
        DrawText("0", r.x-r.h*0.03f, r.y+r.h+2, r.h*0.03f, RAYWHITE);
    }
}

void nf_v_slider(float *value, bool *is_dragging, float rx, float ry, float rw, float rh)
{
    float knob_radius = rh;
    Vector2 bar_size = {
        .x = rw - 2*knob_radius,
        .y = rh*0.25,
    };
    Vector2 bar_position = {
        .x = rx + knob_radius,
        .y = ry + rh/2 - bar_size.y/2
    };
    DrawRectangleV(bar_position, bar_size, WHITE);

    Vector2 knob_position = {
        .x = bar_position.x + bar_size.x*(*value),
        .y = ry + rh/2
    };
    DrawCircleV(knob_position, knob_radius, RED);

    if (*is_dragging) {
        float x = GetMousePosition().x;
        if (x < bar_position.x) x = bar_position.x;
        if (x > bar_position.x + bar_size.x) x = bar_position.x + bar_size.x;
        *value = (x - bar_position.x)/bar_size.x;
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_position = GetMousePosition();
        if (Vector2Distance(mouse_position, knob_position) <= knob_radius) {
            *is_dragging = true;
        }
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *is_dragging = false;
    }
}

void nf_v_render_mat_as_cake(NF_V_Rect r, NF_Mat m)
{
    Color low_color  = { 0xFF, 0x00, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

    float cell_width  = r.w/m.cols;
    float cell_height = r.h/m.rows;

    float gap = 5;

    nf_v_layout_begin(VLO_VERT, r, m.rows, gap);
    for (size_t y = 0; y < m.rows; ++y) {
        nf_v_layout_begin(VLO_HORZ, nf_v_layout_slot(), m.cols, gap);
        for (size_t x = 0; x < m.cols; ++x) {
            float alpha = nf_sigmoidf(NF_MAT_AT(m, y, x));
            high_color.a = floorf(255.f*alpha);
            Color clr = ColorAlphaBlend(low_color, high_color, WHITE);
            NF_V_Rect slot = nf_v_layout_slot();
            DrawRectangle(slot.x, slot.y, slot.w, slot.h, clr);
        }
        nf_v_layout_end();
    }
    nf_v_layout_end();
}

void nf_v_render_nn_as_cake(NF_NN nn, NF_V_Rect r)
{
    nf_v_layout_begin(VLO_VERT, r, nn.count, 10);
    for (size_t i = 0; i < nn.count; ++i) {
        nf_v_render_mat_as_cake(nf_v_layout_slot(), nn.ws[i]);
    }
    nf_v_layout_end();
}

#ifdef NF_IMAGE_GENERATION
// TODO: remove
#define STR2(x) #x
#define STR(x) STR2(x)
 
#define out_width 256
#define out_height 256
#define FPS 30
uint32_t out_pixles[out_width*out_height];

#define READ_END 0
#define WRITE_END 1

void nf_v_render_single_frame(NF_NN nn, float img_index)
{
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);;
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            NF_MAT_AT(NF_NN_INPUT(nn), 0, 2) = img_index;
            nf_nn_forward(nn);
            float activation = NF_MAT_AT(NF_NN_OUTPUT(nn), 0, 0);
            if (activation < 0) activation = 0;
            if (activation > 1) activation = 1;
            uint32_t bright = activation*255.f;
            uint32_t pixel = 0xFF000000|bright|(bright<<8)|(bright<<16);
            out_pixles[y*out_width + x] = pixel;
        }
    }
}

int nf_v_render_upscaled_screenshot(NF_NN nn, float img_index, const char *out_file_path)
{
    assert(out_pixles != NULL);
    nf_v_render_single_frame(nn, img_index);
    if (!stbi_write_png(out_file_path, out_width, out_height, 4, out_pixles, out_width*sizeof(*out_pixles))) {
        fprintf(stderr, "ERROR: could not write image %s\n", out_file_path);
        return 1;
    }
    printf("Generated %s\n", out_file_path);
    return 0;
}

int nf_v_render_upscaled_video(NF_NN nn, float duration, const char *out_file_path)
{
    // connecting two processes with a pipe ~ unidirectional pipe
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        fprintf(stderr, "ERROR: could not create a pipe: %s\n", strerror(errno));
        return 1;
    }
    // Fork the current process
    pid_t child = fork();
    // if child pid is negative it means that the child process was not created
    if (child < 0) {
        fprintf(stderr, "ERROR: could not fork a child: %s\n", strerror(errno));
        return 1;
    }
    // if you are the child process pid is equal to 0
    if (child == 0) {
        // replace the stdinput with the read end of the pipe
        if (dup2(pipefd[READ_END], STDIN_FILENO) < 0) {
            fprintf(stderr, "ERROR: could not reopen read end of the pipe as stdin: %s\n", strerror(errno));
            return 1;
        }
        close(pipefd[WRITE_END]);

        int ret = execlp("ffmpeg",
                         "ffmpeg",
                         "-loglevel", "verbose",
                         "-y",
                         "-f", "rawvideo",
                         "-pix_fmt", "rgb32",
                         "-s", STR(out_width) "x" STR(out_height),
                         "-r", STR(FPS),
                         "-an",
                         "-i", "-", 
                         "-c:v", "libx264",
                         out_file_path,
                         NULL
                         );
        if (ret < 0) {
            fprintf(stderr, "ERROR: could not run ffmpeg as a child process: %s\n", strerror(errno));
            return 1;
        }
        assert(0 && "unreachable");
    }

    close(pipefd[READ_END]);

    typedef struct {
        float start;
        float end;
    } Segment;
    
    Segment segments[] = {
        {0, 0},
        {0, 1},
        {1, 1},
        {1, 0},
    };

    size_t segments_count = NF_ARRAY_LEN(segments);
    float segment_length = 1.f/segments_count;

    // render video
    size_t frame_count = FPS*duration;
    for (size_t i = 0; i < frame_count; ++i) {
        float img_index = (float)i/frame_count;
        
        // easing animation
        size_t segment_index = floorf(img_index/segment_length);
        float segment_porgress = img_index/segment_length - segment_index;
        if (segment_index > segments_count) segment_index = segments_count - 1;
        Segment segment = segments[segment_index];
        float a = segment.start + (segment.end - segment.start)*sqrtf(segment_porgress);
        nf_v_render_single_frame(nn, a);
        write(pipefd[WRITE_END], out_pixles, sizeof(*out_pixles)*out_width*out_height); 
    }

    close(pipefd[WRITE_END]);

    // wait for the child to finish executing
    wait(NULL);

    printf("Generated %s\n", out_file_path);
    return 0;
}
#endif // NF_IMAGE_GENERATION
#endif // NF_VISUALISATION

#endif // NF_IMPLEMENTATION
