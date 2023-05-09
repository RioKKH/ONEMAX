#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

#include "Common/helper_cuda.h"
#include "Population.h"
#include "CUDAKernels.h"


//----- GPUPopulation  ------
//----- Public methods ------
/**
 * Constructor of the class
 */
GPUPopulation::GPUPopulation(const int populationSize,
                             const int chromosomeSize,
                             const int elitesSize)
{
    mHostPopulationHandler.chromosomeSize = chromosomeSize;
    mHostPopulationHandler.populationSize = populationSize;
    mHostPopulationHandler.elitesSize     = elitesSize;

    allocateMemory();
} // end of GPUPopulation


/**
 *  Destructor of the class
 */
GPUPopulation::~GPUPopulation()
{
    freeMemory();
} // end of GPUPopulation


/**
 * Copy data from CPU population structure to GPU
 * Both population must have the same size (sizes not being copied)!!
 * 
 * @param HostSource - Source of population data on the host side
 */
void GPUPopulation::copyToDevice(const PopulationData* hostPopulation)
{
    // Basic data check
    if (hostPopulation->chromosomeSize != mHostPopulationHandler.chromosomeSize)
    {
        throw std::out_of_range("Wrong chromosome size in GPUPopulation::copyToDevice function.");
    }

    if (hostPopulation->populationSize != mHostPopulationHandler.populationSize)
    {
        throw std::out_of_range("Wrong population size in GPUPopulation::copyToDevice function.");
    }

    if (hostPopulation->elitesSize != mHostPopulationHandler.elitesSize)
    {
        throw std::out_of_range("Wrong elite size in GPUPopulation::copyToDevice function.");
    }

    //- Copy chromosomes
    checkCudaErrors(cudaMemcpy(mHostPopulationHandler.population,
                               hostPopulation->population,
                               sizeof(Gene) * mHostPopulationHandler.chromosomeSize\
                                            * mHostPopulationHandler.populationSize,
                               cudaMemcpyHostToDevice));

    //- Copy fitness values
    checkCudaErrors(cudaMemcpy(mHostPopulationHandler.fitness,
                               hostPopulation->fitness,
                               sizeof(Fitness) * mHostPopulationHandler.populationSize,
                               cudaMemcpyHostToDevice));

    //- Copy elites indexes
    checkCudaErrors(cudaMemcpy(mHostPopulationHandler.elitesIdx,
                               hostPopulation->elitesIdx,
                               sizeof(ElitesIdx) * mHostPopulationHandler.elitesSize,
                               cudaMemcpyHostToDevice));
} // end of copyToDevice


/**
 * Copy data from GPU population structure to CPU.
 */
void GPUPopulation::copyFromDevice(PopulationData * hostPopulation)
{
    if (hostPopulation->chromosomeSize != mHostPopulationHandler.chromosomeSize)
    {
        throw std::out_of_range("Wrong chromosome size in GPUPopulation::copyFromDevice function.");
    }

    if (hostPopulation->populationSize != mHostPopulationHandler.populationSize)
    {
        throw std::out_of_range("Wrong population size in GPUPopulation::copyFromDevice function.");
    }

    if (hostPopulation->elitesSize != mHostPopulationHandler.elitesSize)
    {
        throw std::out_of_range("Wrong elite size in GPUPopulation::copyFromDevice function.");
    }

    //- Copy fitness values
    checkCudaErrors(cudaMemcpy(hostPopulation->population,
                               mHostPopulationHandler.population,
                               sizeof(Gene) * mHostPopulationHandler.chromosomeSize
                                            * mHostPopulationHandler.populationSize,
                               cudaMemcpyDeviceToHost));

    //- Copy fitness values
    checkCudaErrors(cudaMemcpy(hostPopulation->fitness,
                               mHostPopulationHandler.fitness,
                               sizeof(Fitness) * mHostPopulationHandler.populationSize,
                               cudaMemcpyDeviceToHost));

    //- Copy elites index values
    checkCudaErrors(cudaMemcpy(hostPopulation->elitesIdx,
                               mHostPopulationHandler.elitesIdx,
                               sizeof(ElitesIdx) * mHostPopulationHandler.elitesSize,
                               cudaMemcpyDeviceToHost));
} // end of copyFromDevice



/**
 * Copy data from different population (both on the same GPU).
 */
void GPUPopulation::copyOnDevice(const GPUPopulation* sourceDevicePopulation)
{
    if (sourceDevicePopulation->mHostPopulationHandler.chromosomeSize != mHostPopulationHandler.chromosomeSize)
    {
        throw std::out_of_range("Wrong chromosome size in GPUPopulation::copyOnDevice function.");
    }

    if (sourceDevicePopulation->mHostPopulationHandler.populationSize != mHostPopulationHandler.populationSize)
    {
        throw std::out_of_range("Wrong population size in GPUPopulation::copyOnDeivce function.");
    }

    if (sourceDevicePopulation->mHostPopulationHandler.elitesSize != mHostPopulationHandler.elitesSize)
    {
        throw std::out_of_range("Wrong elite size in GPUPopulation::copyOnDeivce function.");
    }

    // Copy chromosomes
    checkCudaErrors(cudaMemcpy(mHostPopulationHandler.population,
                               sourceDevicePopulation->mHostPopulationHandler.population,
                               sizeof(Gene) * mHostPopulationHandler.chromosomeSize
                                            * mHostPopulationHandler.populationSize,
                               cudaMemcpyDeviceToDevice));

    // Copy fitness values
    checkCudaErrors(cudaMemcpy(mHostPopulationHandler.fitness,
                               sourceDevicePopulation->mHostPopulationHandler.fitness,
                               sizeof(Fitness) * mHostPopulationHandler.populationSize,
                               cudaMemcpyDeviceToDevice));

    // Copy elites index values
    checkCudaErrors(cudaMemcpy(mHostPopulationHandler.elitesIdx,
                               sourceDevicePopulation->mHostPopulationHandler.elitesIdx,
                               sizeof(ElitesIdx) * mHostPopulationHandler.elitesSize,
                               cudaMemcpyDeviceToDevice));
} // end of copyOnDevice


/**
 * Copy a given individual from device to host
 */
void GPUPopulation::copyIndividualFromDevice(Gene* individual, int index)
{
    checkCudaErrors(
            cudaMemcpy(individual,
                       &(mHostPopulationHandler.population[index * mHostPopulationHandler.chromosomeSize]),
                       sizeof(Gene) * mHostPopulationHandler.chromosomeSize,
                       cudaMemcpyDeviceToHost));
} // end of copyIndividualFromDevice


//----- GPUPopulation     ------
//----- Protected methods ------
/**
 * Allocate GPU memory
 */
void GPUPopulation::allocateMemory()
{
    //- Allocate data structure
    checkCudaErrors(cudaMalloc<PopulationData>(&mDeviceData, sizeof(PopulationData)));

    //- Allocate Population data
    checkCudaErrors(cudaMalloc<Gene>(&(mHostPopulationHandler.population),
                                       sizeof(Gene) * mHostPopulationHandler.chromosomeSize
                                                    * mHostPopulationHandler.populationSize));

    //- Allocate Fitness data
    checkCudaErrors(cudaMalloc<Fitness>(&(mHostPopulationHandler.fitness),
                                          sizeof(Fitness) * mHostPopulationHandler.populationSize));

    //- Allocate ElitesIdx data
    checkCudaErrors(cudaMalloc<ElitesIdx>(&(mHostPopulationHandler.elitesIdx),
                                            sizeof(ElitesIdx) * mHostPopulationHandler.elitesSize));

    //- Copy structure to GPU
    checkCudaErrors(
            cudaMemcpy(mDeviceData, &mHostPopulationHandler, sizeof(PopulationData), cudaMemcpyHostToDevice)
            );
} // end of allocateMemory


/**
 * Free memory
 */
void GPUPopulation::freeMemory()
{
    // Free population data
    checkCudaErrors(cudaFree(mHostPopulationHandler.population));

    // Free fitness data
    checkCudaErrors(cudaFree(mHostPopulationHandler.fitness));

    // Free elitesIdx data
    checkCudaErrors(cudaFree(mHostPopulationHandler.elitesIdx));

    // Free whole structure
    checkCudaErrors(cudaFree(mDeviceData));
} // end of freeMemory



//----- CPUPopulation  ------
//----- Public methods ------
/**
 * Constructor of the class.
 */
CPUPopulation::CPUPopulation(const int populationSize,
                             const int chromosomeSize,
                             const int elitesSize)
{
    mHostData = new(PopulationData);
    mHostData->populationSize = populationSize;
    mHostData->chromosomeSize = chromosomeSize;
    mHostData->elitesSize     = elitesSize;

    allocateMemory();
} // end of CPUPopulation


/**
 * Destructor of the class.
 */
CPUPopulation::~CPUPopulation()
{
    freeMemory();

    delete mHostData;
} // end of ~CPUPopulation

std::int32_t CPUPopulation::getMax()
{
    return *std::max_element(mHostData->fitness,
                             mHostData->fitness + mHostData->populationSize);
}

std::int32_t CPUPopulation::getMin()
{
    return *std::min_element(mHostData->fitness,
                             mHostData->fitness + mHostData->populationSize);
}

double CPUPopulation::getMean()
{
    int fitnessSum = std::accumulate(mHostData->fitness,
                                     mHostData->fitness + mHostData->populationSize, 0);
    double fitnessAverage = static_cast<double>(fitnessSum) / mHostData->populationSize;
    return fitnessAverage;
}

//----- CPUPopulation     ------
//----- Protected methods ------

/**
 * Allocate memory
 */
void CPUPopulation::allocateMemory()
{
    // printf("num of population elements: %d\n", mHostData->chromosomeSize * mHostData->populationSize);
    //- Allocate Population on the host side
    checkCudaErrors(
            cudaHostAlloc<Gene>(&mHostData->population,
                                sizeof(Gene) * mHostData->chromosomeSize * mHostData->populationSize,
                                cudaHostAllocDefault)
            );

    //- Allocate fitness on the host side
    checkCudaErrors(
            cudaHostAlloc<Fitness>(&mHostData->fitness,
                                   sizeof(Fitness) * mHostData->populationSize,
                                   cudaHostAllocDefault)
            );
    //- Allocate Elites index on the host side
    checkCudaErrors(
            cudaHostAlloc<ElitesIdx>(&mHostData->elitesIdx,
                                     sizeof(ElitesIdx) * mHostData->elitesSize,
                                     cudaHostAllocDefault)
            );
} // end of allocateMemory


/**
 * Free memory.
 */
void CPUPopulation::freeMemory()
{
    // Free population on the host side
    checkCudaErrors(cudaFreeHost(mHostData->population));

    // Free fitness on the host side
    checkCudaErrors(cudaFreeHost(mHostData->fitness));

    // Free elitesIdx on the host side
    checkCudaErrors(cudaFreeHost(mHostData->elitesIdx));
} // end of freeMemory
