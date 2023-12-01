
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

float train_data[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

float cost(float w1, float w2)
{
    float train_count = ARRAY_LEN(train_data);
    float train_error = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x1    = train_data[i][0];
        float x2    = train_data[i][1];
        float y     = x1*w1 + x2*w2;
        float dist  = y - train_data[i][2]; 
        train_error += dist*dist;
    }
    train_error /= train_count;
    return train_error;
}

void train(float *w1, float *w2, size_t titer, float eps, float rate)
{
    for (size_t j=0; j < titer; ++j) {
        float dcost = (cost(*w1 + eps, *w2 + eps) - cost(*w1, *w2))/eps;
        *w1 -= rate*dcost;
        *w2 -= rate*dcost;
        printf("cost: %f, w1: %f, w2: %f\n", cost(*w1, *w2), *w1, *w2);
    }
}

int main() 
{
    srand(time(0));
    float w1 = rand_float()*10.0f;
    float w2 = rand_float()*10.0f;
    float eps = 1e-3;
    float rate = 1e-3;

    train(&w1, &w2, 8*2048, eps, rate);
    printf("--------------------------------\n");
    printf("w1: %f, w2: %f\n", w1, w2);

    for (size_t i = 0; i <= 1; ++i) {
        for (size_t j = 0; j <= 1; ++j) {
            printf("%zu & %zu: %f\n", i, j, w1*i + w2*j);
        }
    }

    return 0;
}

