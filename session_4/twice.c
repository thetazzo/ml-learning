#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    size_t train_count = ARRAY_LEN(train_data);
    float cost = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x    = train_data[i][0];
        float y    = x*w + b;
        float d = y - train_data[i][1]; 
        cost += d*d;
    }
    cost /= train_count;
    return cost;
}

float gd_cost(float w)
{
    size_t n = ARRAY_LEN(train_data);

    float costd = 0.f;

    for (size_t i = 0; i < n; ++i) {
        float xi = train_data[i][0];  
        float yi = train_data[i][1];  
        costd += 2 * (xi * w - yi) * xi;
    }
    costd /= n;
    return costd; 
}

void train(float *w, size_t titer,  float rate)
{
    for (size_t j=0; j < titer; ++j) {
        float dw = gd_cost(*w);// (cost(*w + eps, *b) - cost(*w, *b))/eps;
        float db = gd_cost(*w);//(cost(*w, *b+eps) - cost(*w, *b))/eps;
        *w -= rate*dw;

        printf("cost: %f, w: %f\n", gd_cost(*w), *w);
    }
}

int main() 
{
    // srand(time(0));
    srand(69);

    float w = rand_float()*10.0f;

    // float eps = 1e-3;
    float rate = 1e-1;

    train(&w, 30, rate);
    printf("--------------------------------\n");

    printf("w: %f\n", w);
    printf("--------------------------------\n");

    printf("f(x) = x * w:\n");
    for (size_t i = 0; i < 5; ++i) {
        int y = i*w;
        printf("%zu -> %d\n", i, y);
    }

    return 0;
}



