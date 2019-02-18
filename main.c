#include <stdio.h>
#include <time.h>
#include <math.h>
#include "ffnn.h"

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    uint16_t dim[] = {1, 1000, 10, 1};
    ffnn* clf = ffnn_constructor(4, dim, rectifierF, 0.01f);
    float **x, **y;
    x = (float **) malloc(250 * sizeof(float *));
    y = (float **) malloc(250 * sizeof(float *));
    for (size_t i = 0; i < 250; i++) {
        x[i] = (float *) malloc(sizeof(float));
        y[i] = (float *) malloc(sizeof(float));
    }
    for(size_t i = 0; i < 250; i++) {
        x[i][0] = 0.01f * i - 1;
        y[i][0] = sinf(x[i][0] * 1000.0f) / 2.0f;
    }
    clock_t start= clock();
    for(size_t i = 0; i < 5; i++) {
        printf("Err: %.2f\n", (double) error(clf, 50, x+200, y+200));
        start = clock();
        trainNEpochs(clf, 10, x, y, 200);
        printf("Time passed while training 10 epochs: %lfs\n",((double)(clock()-start))/CLOCKS_PER_SEC);
    }

    return 0;
}
