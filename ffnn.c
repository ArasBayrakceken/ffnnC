#include "ffnn.h"
#include <math.h>
#include <stdlib.h>

ffnn* ffnn_constructor(uint8_t length, uint16_t* widths, enum activationFunctionType activF, float learnRate) {
    ffnn* nn = (ffnn*) malloc(sizeof(ffnn));
    nn->learnRate = learnRate;
    nn->length = length;
    nn->width = (uint16_t *) malloc(length * sizeof(uint16_t));
    nn->net = (float **) malloc(length * sizeof(float *));
    nn->out = (float **) malloc(length * sizeof(float *));
    nn->error = (float **) malloc(length * sizeof(float *));
    nn->weights = (float ***) malloc((length - 1) * sizeof(float **));
    nn->bias = (float **) malloc((length - 1) * sizeof(float *));

    for(uint8_t i = 0; i < length; i++) {
        nn->width[i] = widths[i];
        nn->net[i] = (float *) malloc(widths[i] * sizeof(float));
        nn->out[i] = (float *) malloc(widths[i] * sizeof(float));
        nn->error[i] = (float *) malloc(widths[i] * sizeof(float));
        if (i < length - 1) {
            nn->bias[i] = (float *) malloc(widths[i+1] * sizeof(float));
            nn->weights[i] = (float **) malloc(widths[i] * sizeof(float*));
            for (uint16_t j = 0; j < widths[i]; j++) {
                nn->weights[i][j] = (float *) malloc(widths[i+1] * sizeof(float));
            }
        }
    }

    if (activF == rectifierF) {
        nn->activationf = rectifier;
        nn->dactivation = drectifier;
    }

    ffnnRandomize(nn);
    return nn;
}

void ffnn_destructor(ffnn* self) {
    for (uint8_t l = 0; l < self->length; l++) {
        free(self->net[l]);
        free(self->out[l]);
        free(self->error[l]);
        if (l < self->length - 1) {
            free(self->bias[l]);
            for (uint16_t i = 0; i < self->width[l]; i++) {
                free(self->weights[l][i]);
            }
            free(self->weights[l]);
        }
    }
    free(self->bias);
    free(self->weights);
    free(self->error);
    free(self->out);
    free(self->net);
    free(self->width);
    free(self);
}

void ffnnRandomize(ffnn* self) {
    for (uint8_t l = 0; l < self->length - 1; l++) {
        for (uint16_t p = 0; p < self->width[l+1]; p++) {
            self->bias[l][p] = rand()/(float)RAND_MAX/10.0f;
            for (uint16_t n = 0; n < self->width[l]; n++) {
                self->weights[l][n][p] = rand()/(float)RAND_MAX/10.0f;
            }
        }
    }
}

float rectifier(float x) {
    return logf(1 + powf((float)M_E, x));
}

float drectifier(float x) {
    return (1 /( 1 + powf((float)M_E, -x)));
}

void predict(ffnn* self, float* inputs, float* output) {
    for(uint8_t i = 0; i < self->width[0]; i++) {
        self->out[0][i] = inputs[i];
    }

    for(uint8_t layer = 1; layer < self->length; layer++) {
        for(uint16_t neuron = 0; neuron < self->width[layer]; neuron++) {
            self->net[layer][neuron] = self->bias[layer - 1][neuron];
            for(uint16_t preNeuron = 0; preNeuron < self->width[layer-1]; preNeuron++) {
                self->net[layer][neuron] += self->out[layer-1][preNeuron] * self->weights[layer-1][preNeuron][neuron];
            }
            self->out[layer][neuron] = self->activationf(self->net[layer][neuron]);
        }
    }

    for (uint16_t i = 0; i < self->width[self->length - 1]; i++) {
        output[i] = self->net[self->length - 1][i];
    }
}

float error(ffnn* self, uint32_t nSamples, float** inputs, float** outputs) {
    float err = 0;
    uint16_t outSize = self->width[self->length-1];
    float* actual = (float *) malloc(outSize * sizeof(float));
    for(uint32_t i = 0; i < nSamples; i++) {
        predict(self, inputs[i], actual);
        for(uint32_t j = 0; j < outSize; j++) {
            err += powf(outputs[i][j] - actual[j], 2);
        }
    }
    free(actual);
    return err;
}

void trainNEpochs(ffnn* self, uint16_t nEpochs, float** data, float** ideal, uint32_t nSamples) {
    for (uint16_t epoch = 0; epoch < nEpochs; epoch++) {
        for (uint32_t sample = 0; sample < nSamples; sample++) {
            for(uint8_t i = 0; i < self->width[0]; i++) {
                self->out[0][i] = data[sample][i];
            }

            //TODO: LAST LAYER, UNNECESSARY USE OF ACTIVATION FUNCTION
            for (uint8_t layer = 1; layer < self->length; layer++) {     //forward propagation
                for(uint16_t neuron = 0; neuron < self->width[layer]; neuron++) {
                    self->net[layer][neuron] = self->bias[layer - 1][neuron];
                    for(uint16_t preNeuron = 0; preNeuron < self->width[layer-1]; preNeuron++) {
                        self->net[layer][neuron] += self->out[layer-1][preNeuron]
                                                    * self->weights[layer-1][preNeuron][neuron];
                    }
                    self->out[layer][neuron] = self->activationf(self->net[layer][neuron]);
                }
            }

            for (uint8_t i = 0; i < self->width[self->length - 1]; i++) {    //error for output layer
                self->error[self->length - 1][i] = ideal[sample][i] - self->net[self->length - 1][i];
                self->bias[self->length - 2][i] += self->learnRate * self->error[self->length - 1][i];
                for (uint16_t anterior = 0; anterior < self->width[self->length - 2]; anterior++){
                    self->weights[self->length - 2][anterior][i] += self->learnRate * self->error[self->length - 1][i]
                                                          * self->out[self->length - 2][anterior];
                }
            }

            for (uint8_t layer = self->length - 2; layer > 0; layer--) { //error backpropagation
                for (uint16_t neuron = 0; neuron < self->width[layer]; neuron++){
                    self->error[layer][neuron] = 0;
                    for (uint16_t posterior = 0; posterior < self->width[layer+1]; posterior++) {
                        self->error[layer][neuron] += self->error[layer+1][posterior]
                                                    * self->weights[layer][neuron][posterior];
                    }
                    self->error[layer][neuron] *= self->dactivation(self->net[layer][neuron]);
                    self->bias[layer-1][neuron] += self->learnRate * self->error[layer][neuron];
                    for (uint16_t anterior = 0; anterior < self->width[layer-1]; anterior++) {
                        self->weights[layer-1][anterior][neuron] += self->learnRate
                                                    * self->error[layer][neuron] * self->out[layer-1][anterior];
                    }
                }
            }
        }
    }
}

