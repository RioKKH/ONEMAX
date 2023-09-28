#ifndef MISC_H
#define MISC_H

// #include <stdio.h>
// #include <stdlib.h>
#include "Parameters.h"
// #include "CUDAKernels.h"

int my_rand(void);

void initializePopulationOnCPU(int *population, Parameters *prms);
// void initializePopulationOnCPU(int *population, Parameters &prms);

void showPopulationOnCPU(int *population,
                         int *fitness,
                         int *parent1,
                         int *parent2,
                         Parameters *prms);

void showSummaryOnCPU(int gen, int *fitness, Parameters *prms);

void printDeviceInfo();

#endif // MISC_H
