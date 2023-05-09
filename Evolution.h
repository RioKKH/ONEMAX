#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "Parameters.h"
#include "Population.h"

class GPUEvolution
{
public:
    /// Class constructor.
    GPUEvolution();
    GPUEvolution(Parameters* prms);

    /// Copy constructor is not allowed.
    GPUEvolution(const GPUEvolution&) = delete;

    /// Destructor
    /// クラスに仮想メンバー関数が存在する場合、そのクラスのデストラクタは
    /// virtualでなければならない
    virtual ~GPUEvolution();

    /// Run evolution
    void run(Parameters* prms);

protected:

    /// Initialize evolution.
    void initialize(Parameters* prms);

    /// Run evolution
    void runEvolutionCycle(Parameters* prms);

    /// Init random generator seed;
    void initRandomSeed();

    /// Get random generator seed and increment it.
    unsigned int getRandomSeed() { return mRandomSeed++; };


    /// Parameters of evolution
    // Parameters& mParams;

    /// Actual generation.
    // int mActGeneration;

    /// Number of SM on GPU.
    int mMultiprocessorCount;

    /// Device Id.
    int mDeviceIdx;

    /// Random Generator Seed.
    unsigned int mRandomSeed;

    // Population odd
    CPUPopulation* mHostParentPopulation;
    CPUPopulation* mHostOffspringPopulation;
    GPUPopulation* mDevParentPopulation;
    GPUPopulation* mDevOffspringPopulation;

    // Show the population
    void showPopulation(Parameters* prms, std::uint16_t generation);

}; // end of GPU_Evolution

#endif // EVOLUTION_H
