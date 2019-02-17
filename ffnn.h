#ifndef FFNN_H
#define FFNN_H

#include <stdlib.h>

enum activationFunctionType {rectifierF = 0};

typedef struct {
    uint8_t length;  // number of layers in nn (min 2)
    uint16_t *width; // width of each layer (minimum of 2 elements)
    float **net;
    float **out;
    float **error;
    float ***weights; // starting layer, starting neuron, ending neuron
    float **bias;     // layer - 1, neuron
    float learnRate;
    float (*activationf)(float);
    float (*dactivation)(float);
} ffnn;

ffnn* ffnn_constructor(uint8_t length, uint16_t* widths, enum activationFunctionType, float learnRate);
void ffnn_destructor(ffnn* self);
void ffnnRandomize(ffnn* self);
void predict(ffnn* self, float* inputs, float* output);
float error(ffnn* self, uint32_t nSamples, float** inputs, float** outputs);
void trainNEpochs(ffnn* self, uint16_t nEpochs, float** data, float** ideal, uint32_t nSamples);
float rectifier(float);
float drectifier(float);

#endif // FFNN_H
