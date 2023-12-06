
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

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

float nand_td[][3] = {
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};

#define train_data nand_td

float cost(float w1, float w2, float b)
{
    size_t train_count = ARRAY_LEN(train_data);
    float cost = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x1    = train_data[i][0];
        float x2    = train_data[i][1];
        float y     = sigmoidf(x1*w1 + x2*w2 + b);
        float dist  = y - train_data[i][2]; 
        cost += dist*dist;
    }
    cost /= train_count;
    return cost;
}

void fd_cost(float eps, float w1, float w2, float b, float *dw1, float *dw2, float *db)
{
       *dw1 = (cost(w1 + eps, w2, b) - cost(w1, w2, b))/eps;
       *dw2 = (cost(w1, w2 + eps, b) - cost(w1, w2, b))/eps;
       *db  = (cost(w1, w2, b + eps) - cost(w1, w2, b))/eps;
}

void gd_cost(float w1, float w2, float b, float *dw1, float *dw2, float *db)
{
    size_t train_count = ARRAY_LEN(train_data);
    *dw1 = 0;
    *dw2 = 0;
    *db  = 0;
    for (size_t i = 0; i < train_count; ++i) {
        float xi    = train_data[i][0];
        float yi    = train_data[i][1];
        float zi     = train_data[i][2];
        float ai     = sigmoidf(xi*w1 + yi*w2 + b);
        float di = 2 * (ai - zi) * ai * (1 - ai); 
        *dw1  += di * xi; 
        *dw2  += di * yi; 
        *db   += di; 
    }
    *dw1 /= train_count;
    *dw2 /= train_count;
    *db  /= train_count;
}

void train(float *w1, float *w2, float *b, size_t titer, float rate)
{
    for (size_t j=0; j < titer; ++j) {
        float dw1;
        float dw2;
        float db;
#if 0   // Finite Difference ~ Derivitive Approcimation
        float eps = 1e-1;
        fd_cost(eps, *w1, *w2,*b, &dw1, &dw2, &db);
#else   //Gradient Descent
        gd_cost(*w1, *w2,*b, &dw1, &dw2, &db);
#endif
        *w1 -= rate*dw1;
        *w2 -= rate*dw2;
        *b  -= rate*db;
        // printf("w1: %f, w2: %f, b: %f\n", *w1, *w2, *b);
    }
}

/**
 *
 * Clock() is used to mesure CPU clock and consequntly the perofrmance of the alghoritm
 */
int main() 
{
    srand(time(0));

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float rate = 1e-1;
    size_t epoch = 1000*8000;

    clock_t ct = clock();
    train(&w1, &w2, &b, epoch, rate);
    clock_t nt = clock();
    float t = (float)(nt - ct)/CLOCKS_PER_SEC;
    printf("epoch: %zu => computation time: %f sec \n", epoch, t);
    printf("--------------------------------\n");

    printf("cost: %f, w1: %f, w2: %f, b: %f\n", cost(w1, w2, b), w1, w2, b);

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

