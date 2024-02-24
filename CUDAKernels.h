#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "Population.h"
#include "Parameters.h"

enum class PARENTS_e : short
{
    MALE   = 0,
    FEMALE = 1,
};

void copyToDevice(EvolutionParameters cpuEvoPrms);

/**
 * Check and report CUDA errors.
 * @param [in] sourceFileName    - Source file where the error happened.
 * @param [in] sourceLineNumber  - Line where the error happened.
 */
void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber);


__global__ void evaluation(PopulationData* populationData);
//__global__ void evaluation(int *population, int *fitness);

__global__ void pseudo_elitism(PopulationData* populationData);

__global__ void elitism(PopulationData* populationData);

__global__ void replaceWithElites(PopulationData *parentPopulation,
                                  PopulationData *offspringPopulation);

__global__ void swapPopulation(PopulationData* parentPopulation,
                               PopulationData* offspringPopulation);


__global__ void swapPopulation_pointer(PopulationData* parentPopulation,
                                       PopulationData* offspringPopulation);

inline __device__ int getBestIndividual(const PopulationData* populationData,
                                                 const int& idx1, const int& idx2,
                                                 const int& idx3, const int& idx4);

inline __device__ int tournamentSelection(const PopulationData* populationData,
                                          const int tournament_size,
                                          const std::uint32_t& random1,
                                          const std::uint32_t& random2,
                                          const std::uint32_t& random3,
                                          const std::uint32_t& random4);
                                          // const unsigned int& random1,
                                          // const unsigned int& random2,
                                          // const unsigned int& random3,
                                          // const unsigned int& random4);


inline __device__ void swap(unsigned int &point1,
                            unsigned int &point2);


inline __device__ void doublepointsCrossover(const PopulationData* parentPopulation,
                                             PopulationData* offspringPopulation,
                                             const unsigned int& offspringIdx,
                                             int& parent1Idx,
                                             int& parent2Idx,
                                             std::uint32_t& random1,
                                             std::uint32_t& random2);
                                             // unsigned int& rnadom3,
                                             // unsigned int& random4);


inline __device__ void bitFlipMutation(PopulationData* offspringPopulation,
                                       std::uint32_t& random1,
                                       std::uint32_t& random2,
                                       std::uint32_t& random3,
                                       std::uint32_t& random4);


__global__ void dev_show(int *population, int *fitness, int *sortedfitness,
                         int *parent1, int *parent2);

__global__ void dev_prms_show(void);

__global__ void cudaCallRandomNumber(unsigned int randomSeed);

//__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationDataEven,
//                                                  PopulationData* populationDataOdd,
__global__ void cudaKernelGenerateFirstPopulation(PopulationData* populationData,
                                                  unsigned int    randomSeed);

/**
 * Genetic manipulation (Selection, Crossover, Mutation)
 * @param [in]  populationDataEven    - Even-numbered generations of population.
 * @param [in]  populationDataOdd     - Odd-numbered generations of population.
 * @param [in]  randomSeed            - Random seed.
 */
__global__ void cudaGeneticManipulationKernel(PopulationData* populationDataEven,
                                              PopulationData* populationDataOdd,
                                              unsigned int    randomSeed);

__global__ void cudaKernelSelection(
        PopulationData* populationData,
        uint32_t* selectedParents1,
        uint32_t* selectedParents2,
        unsigned int randomSeed);

__global__ void cudaKernelCrossover(
        PopulationData* parent,
        PopulationData* offspring,
        uint32_t* selectedParents1,
        uint32_t* selectedParents2,
        unsigned int randomSeed);

__global__ void cudaKernelMutation(
        PopulationData* offspring,
        unsigned int randomSeed);

#endif // CUDA_KERNELS_H

