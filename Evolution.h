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
    /// 仮想メンバー関数とは派生クラスでオーバーライドされる関数のこと
    /// 仮想メンバー関数を持たないクラスのデストラクタはvirtualでなくてもよい
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
    CPUPopulation* mHostTempPopulation;
    CPUPopulation* mHostParentPopulation;
    CPUPopulation* mHostOffspringPopulation;

    GPUPopulation* mDevTempPopulation;
    GPUPopulation* mDevParentPopulation;
    GPUPopulation* mDevOffspringPopulation;

    // Show the summary of population
    void showSummary(const Parameters& prms, const float& elapsed_time, const int& generation);

    // Show the population
    void showPopulation(Parameters* prms);
    // void showPopulation(Parameters* prms, uint16_t type);
    // void showPopulation(Parameters* prms, std::uint16_t generation, std::uint16_t type);

    // int elitism(Parameters* prms);

}; // end of GPU_Evolution

#endif // EVOLUTION_H
