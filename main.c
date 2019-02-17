#include <stdio.h>
#include <time.h>
#include "ffnn.h"

int main(int argc, char *argv[]) {
    srand(time(NULL));
    uint16_t dim[] = {1, 10, 5, 1};
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
        y[i][0] = -x[i][0];
    }
    for(size_t i = 0; i < 50; i++) {
        printf("Err: %f\n", error(clf, 200, x, y));
        trainNEpochs(clf, 10, x, y, 200);
    }

    return 0;
}
