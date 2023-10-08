#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "CUDAKernels.h"
#include "Parameters.hpp"
#include "Misc.h"


int my_rand(void)
{
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 1);

    return dist(rng);
}

void initializePopulationOnCPU(int *population, Parameters *prms)
{
    thrust::generate(population, population + prms->getN(), my_rand);
    // thrust::generate(population, population + N, my_rand);

#ifdef _DEBUG
    for (int i = 0; i < prms->getPopsizeActual(); ++i)
    // for (int i = 0; i < prms->getPopsize(); ++i)
    // for (int i = 0; i < POPSIZE; ++i)
	{
        printf("Individual %d:", i);
		for (int j = 0; j < prms->getChromosome(); ++j)
		// for (int j = 0; j < CHROMOSOME; ++j)
		{
			printf("%d", population[i * prms->getChromosome() + j]);
			// printf("%d", population[i * CHROMOSOME + j]);
		}
		printf("\n");
	}
	std::cout << "end of initialization" << std::endl;
#endif // _DEBUG
}

void showPopulationOnCPU(int *population, int *fitness,
                         int *parent1, int *parent2,
                         Parameters *prms)
// void showPopulationOnCPU(int *population, int *fitness, int *parent1, int *parent2)
{
	for (int i = 0; i < prms->getPopsizeActual(); ++i)
	// for (int i = 0; i < prms->getPopsize(); ++i)
	// for (int i = 0; i < POPSIZE; ++i)
	{
		printf("%d,%d,%d,%d,", i, fitness[i], parent1[i], parent2[i]);
		for (int j = 0; j < prms->getChromosome(); ++j)
		// for (int j = 0; j < CHROMOSOME; ++j)
		{
			printf("%d", population[i * prms->getChromosome() + j]);
			// printf("%d", population[i * CHROMOSOME + j]);
		}
		printf("\n");
	}
}

void showSummaryOnCPU(int gen, int *fitness, Parameters *prms)
{
    int fitnessMax = 0;
    int fitnessMin = prms->getChromosome();
    float fitnessAve = 0.0f;
    float fitnessVar = 0.0f;
    float fitnessStdev = 0.0f;

    for (int i = 0; i < prms->getPopsizeActual(); ++i)
    // for (int i = 0; i < prms->getPopsize(); ++i)
    {
        if (fitness[i] < fitnessMin) { fitnessMin = fitness[i]; }
        if (fitness[i] > fitnessMax) { fitnessMax = fitness[i]; }
        fitnessAve += fitness[i];
    }
    fitnessAve /= prms->getPopsizeActual();
    // fitnessAve /= prms->getPopsize();
    for (int i = 0; i < prms->getPopsize(); ++i)
    // for (int i = 0; i < prms->getPopsizeActual(); ++i)
    {
        fitnessVar += ((float)fitness[i] - fitnessAve) * ((float)fitness[i] - fitnessAve);
    }
    fitnessStdev = sqrt(fitnessVar / (prms->getPopsizeActual() - 1));
    // fitnessStdev = sqrt(fitnessVar / (prms->getPopsize() - 1));

    printf("%d,%f,%d,%d,%f\n", gen, fitnessAve, fitnessMin, fitnessMax, fitnessStdev);
}
