#include <stdio.h>
#include <time.h>
#include <math.h>
#include "ffnn.h"

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    uint16_t dim[] = {1, 100, 30, 1};
    ffnn* clf = ffnn_constructor(4, dim, rectifierF, 0.01f);
    float **x, **y;
    x = (float **) malloc(200 * sizeof(float *));
    y = (float **) malloc(200 * sizeof(float *));
    for (size_t i = 0; i < 200; i++) {
        x[i] = (float *) malloc(sizeof(float));
        y[i] = (float *) malloc(sizeof(float));
    }
    for(size_t i = 0; i < 200; i++) {
        x[i][0] = 0.01f * i - 1;
        y[i][0] = sinf(x[i][0] * 1000.0f) / 2.0f;
    }
    clock_t start= clock();
    for(size_t i = 0; i < 5; i++) {
        printf("Err: %.2f\n", (double) error(clf, 200, x, y));
        start = clock();
        trainNEpochs(clf, 100, x, y, 200);
        printf("Time passed while training 100 epochs: %lfs\n",((double)(clock()-start))/CLOCKS_PER_SEC);
    }

    return 0;
}
