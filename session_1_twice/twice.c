
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * 1 2
 * 2 4
 * 3 6
 * 4 8
 *
 * x*w = y
 */

#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

float train_data[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

float cost(float w, float b)
{
    float train_count = ARRAY_LEN(train_data);
    float train_error = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x    = train_data[i][0];
        float y    = x*w + b;
        float dist = y - train_data[i][1]; 
        train_error += dist*dist;
    }
    train_error /= train_count;
    return train_error;
}

void train(float *w, float *b, size_t titer, float eps, float rate)
{
    for (size_t j=0; j < titer; ++j) {
        float dw = (cost(*w + eps, *b) - cost(*w, *b))/eps;
        float db = (cost(*w, *b+eps) - cost(*w, *b))/eps;
        *w -= rate*dw;
        *b -= rate*db;

        printf("ost: %f, w: %f, b: %f\n", cost(*w, *b), *w, *b);
    }
}

int main() 
{
    srand(time(0));

    float w = rand_float()*10.0f;
    float b = 0;

    float eps = 1e-3;
    float rate = 1e-3;

    train(&w, &b, 900, eps, rate);
    printf("--------------------------------\n");

    printf("w: %f, b: %f\n", w, b);
    printf("--------------------------------\n");

    printf("f(x) = x * w:\n");
    for (size_t i = 0; i < 5; ++i) {
        int y = i*w+b;
        printf("%zu -> %d\n", i, y);
    }

    return 0;
}
