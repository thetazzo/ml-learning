#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
    // neuron 1
    float or_w1;
    float or_w2;
    float or_b;
    // neuron 2
    float nand_w1;
    float nand_w2;
    float nand_b;
    // neuron 3
    float and_w1;
    float and_w2;
    float and_b;
} Xor;

typedef float sample[3];
sample xor_td[] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,0},
};

sample *train_data = xor_td;
size_t train_count = 4;

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x1, float x2)
{
    float n1 = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    float n2 = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
    return sigmoidf(m.and_w1*n1 + m.and_w2*n2 + m.and_b);
}

float cost(Xor m)
{
    float train_error = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x1    = train_data[i][0];
        float x2    = train_data[i][1];
        float y     = forward(m, x1, x2);
        float dist  = y - train_data[i][2]; 
        train_error += dist*dist;
    }
    train_error /= train_count;
    return train_error;
}

void rand_xor(Xor *m)
{
    m->or_w1 = rand_float();
    m->or_w2 = rand_float();
    m->or_b = rand_float();

    m->nand_w1 = rand_float();
    m->nand_w2 = rand_float();
    m->nand_b = rand_float();

    m->and_w1 = rand_float();
    m->and_w2 = rand_float();
    m->and_b = rand_float();
}

void xor_print(Xor m)
{
    printf("or_w1 = %f\n", m.or_w1);
    printf("or_w2 = %f\n", m.or_w2);
    printf("or_b = %f\n", m.or_b);

    printf("nand_w1 = %f\n", m.nand_w1);
    printf("nand_w2 = %f\n", m.nand_w2);
    printf("nand_b = %f\n", m.nand_b);

    printf("and_w1 = %f\n", m.and_w1);
    printf("and_w2 = %f\n", m.and_w2);
    printf("and_b = %f\n", m.and_b);
}

Xor finite_diff(Xor m, float eps)
{
    Xor g = {0};
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
	m.or_w2 += eps;
	g.or_w2 = (cost(m) - c)/eps;
	m.or_w2 = saved;

    saved = m.or_b;
	m.or_b += eps;
	g.or_b = (cost(m) - c)/eps;
	m.or_b = saved;

    saved = m.nand_w1;
	m.nand_w1 += eps;
	g.nand_w1 = (cost(m) - c)/eps;
	m.nand_w1 = saved;

    saved = m.nand_w2;
	m.nand_w2 += eps;
	g.nand_w2 = (cost(m) - c)/eps;
	m.nand_w2 = saved;

    saved = m.nand_b;
	m.nand_b += eps;
	g.nand_b = (cost(m) - c)/eps;
	m.nand_b = saved;
   
    saved = m.and_w1;
	m.and_w1 += eps;
	g.and_w1 = (cost(m) - c)/eps;
	m.and_w1 = saved;

    saved = m.and_w2;
	m.and_w2 += eps;
	g.and_w2 = (cost(m) - c)/eps;
	m.and_w2 = saved;

    saved = m.and_b;
	m.and_b += eps;
	g.and_b = (cost(m) - c)/eps;
	m.and_b = saved;

    return g;
}

void xor_train(Xor *m, Xor g, float rate)
{
    m->or_w1 -= rate*g.or_w1;
    m->or_w2 -= rate*g.or_w2;
    m->or_b -= rate*g.or_b;
    m->nand_w1 -= rate*g.nand_w1;
    m->nand_w2 -= rate*g.nand_w2;
    m->nand_b -= rate*g.nand_b;
    m->and_w1 -= rate*g.and_w1;
    m->and_w2 -= rate*g.and_w2;
    m->and_b -= rate*g.and_b;
}

int main()
{
    srand(time(0));

    Xor m = {0};
    rand_xor(&m);

    float eps = 1e-1;
    float rate = 1e-1;

    for (size_t i = 0; i < 100*70;++i) {
        Xor g = finite_diff(m,eps);
        xor_train(&m, g, rate);
    }
    printf("cost: %f\n", cost(m));

    printf("---------------------------------\n");
    printf("XOR:\n");

    for (size_t i = 0; i <= 1; ++i) {
        for (size_t j = 0; j <= 1; ++j) {
            float f = forward(m, i, j);
            int y;
            if (f > 0.5) {
                y = 1;
            } else {
                y = 0;
            }
            printf("%zu ^ %zu: %d\n", i, j, y); 
        }
    }

    return 0;
}
