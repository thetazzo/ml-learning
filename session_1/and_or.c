
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float or_td[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float and_td[][3] = {
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1},
};

#define train_data and_td

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

float cost(float w1, float w2, float b)
{
    float train_count = ARRAY_LEN(train_data);
    float train_error = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x1    = train_data[i][0];
        float x2    = train_data[i][1];
        float y     = sigmoidf(x1*w1 + x2*w2 + b);
        float dist  = y - train_data[i][2]; 
        train_error += dist*dist;
    }
    train_error /= train_count;
    return train_error;
}

void train(float *w1, float *w2, float *b, size_t titer, float eps, float rate)
{
    for (size_t j=0; j < titer; ++j) {
        float dw1 = (cost(*w1 + eps, *w2, *b) - cost(*w1, *w2, *b))/eps;
        float dw2 = (cost(*w1, *w2 + eps, *b) - cost(*w1, *w2, *b))/eps;
        float db  = (cost(*w1, *w2, *b + eps) - cost(*w1, *w2, *b))/eps;
        *w1 -= rate*dw1;
        *w2 -= rate*dw2;
        *b  -= rate*db;
        // printf("cost: %f, w1: %f, w2: %f, b: %f\n", cost(*w1, *w2, *b), *w1, *w2, *b);
    }
}

int main() 
{
    srand(time(0));

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float eps = 1e-1;
    float rate = 1e-1;

    train(&w1, &w2, &b, 1000*1000, eps, rate);
    printf("--------------------------------\n");

    printf("c: %f, w1: %f, w2: %f, b: %f\n", cost(w1, w2, b), w1, w2, b);

    for (size_t i = 0; i <= 1; ++i) {
        for (size_t j = 0; j <= 1; ++j) {
            float y = w1*i + w2*j + b;
            if (y > 0.5) {
                y = 1;
            } else {
                y = 0;
            }
            printf("%zu | %zu: %f\n", i, j, y);
        }
    }

    return 0;
}

