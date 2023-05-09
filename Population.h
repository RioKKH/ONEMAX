#ifndef POPULATION_H
#define POPULATION_H

#include <string>

// Data type for Gene.
typedef unsigned int Gene;
// Data type for fitness value
typedef unsigned int Fitness;
// Data type for the index of elites.
typedef unsigned int ElitesIdx;

/**
 * @struct PopulationData
 * @brief  Population data structure
 */
struct PopulationData
{
    //- Number of chromosomes.
    unsigned int populationSize;
    //- Size of chromosome in INFs.
    unsigned int chromosomeSize;
    //- Size of Elites.
    unsigned int elitesSize;

    //- 1D array of genes (chromosome-based encoding).
    Gene* population;
    //- 1D array of fitness value.
    Fitness* fitness;
    //- 1D array of index of selected elites
    ElitesIdx* elitesIdx;
}; // end of PopulationData


/**
 * @class GPUPopulation
 * @brief Population stored on the GPU.
 */
class GPUPopulation
{
public:
    //- Default constructor not allowed.
    GPUPopulation() = delete;
    //- Copy constructor not allowed.
    GPUPopulation(const GPUPopulation& orig) = delete;

    /**
     * Constructor
     * @param [in] populationSize - Number of chromosomes.
     * @param [in] chromosomeSize - Chromosome length.
     */
    GPUPopulation(const int populationSize,
                  const int chromosomeSize,
                  const int elitesSize);

    //- Destructor
    //- virutal destructor: 継承を使い、且つ動的に確保したオブジェクトを
    //  解放するときに、正しいデストラクターが呼ばれるようにするための機能
    virtual ~GPUPopulation();

    //- Get pointer to device population data.
    PopulationData* getDeviceData()
    {
        return mDeviceData;
    };

    //- Get pointer to device population data, const version.
    const PopulationData* getDeviceData() const
    {
        return mDeviceData;
    };


    /**
     * member function
     * @brief Copy data from CPU population structure to GPU.
     * Both population must have the same size (sizes not being copyied)!!
     *
     * @param [in] hostPopulation - Source of population data on the host side.
     */
    void copyToDevice(const PopulationData* hostPopulation);


    /**
     * member function
     * @brief Copy data from GPU population structure to CPU.
     * Both population must have the same size (sizes not copied)
     *
     * @param [out] hostPopulation - Source of population data on the host side.
     */
    void copyFromDevice(PopulationData* hostPopulation);


    /**
     * member function
     * @brief Copy data from different population (both on the same GPU)
     * No size check!!
     *
     * @param [in] sourceDevicePopulation - Source population.
     */
    void copyOnDevice(const GPUPopulation* sourceDevicePopulation);


    /**
     * member function
     * @brief Copy a given individual from device to host
     * @param [out] individual - Where to store an individual.
     * @param [in]  index      - Index of the individual in device population.
     */
    void copyIndividualFromDevice(Gene* individual, int index);

protected:
    //- Allocate memory.
    void allocateMemory();

    //- Free memory.
    void freeMemory();

private:
    //- Handler on the GPU data
    PopulationData* mDeviceData;

    // Host copy of population
    PopulationData mHostPopulationHandler;
}; // end of TGPU_Population


/**
 * @class CPUPopulation
 * @brief Population stored on the host side.
 */
class CPUPopulation
{
public:
    //- Default constructor not allowed.
    CPUPopulation() = delete;

    //- Default copy constructor not allowed.
    CPUPopulation(const CPUPopulation&) = delete;

    /**
     * Constructor
     * @param [in] populationSize - Number of chromosome.
     * @param [in] chromosomeSize - Chromosome length.
     */
    CPUPopulation(const int populationSize,
                  const int chromosomeSize,
                  const int elitesSize);

    //- Destructor
    virtual ~CPUPopulation();

    //- Get poointer to device population data.
    PopulationData* getDeviceData()
    {
        return mHostData;
    };

    //- Get pointer to device population data, const version.
    const PopulationData* getDeviceData() const
    {
        return mHostData;
    };

    std::int32_t getMax();
    std::int32_t getMin();
    double getMean();


protected:
    //- Allocate memory
    void allocateMemory();

    //- Free memory
    void freeMemory();

private:
    //- Host population data
    PopulationData* mHostData;
}; // end of CPUPopulation

#endif // POPULATION_H
