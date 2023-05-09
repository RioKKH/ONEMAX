#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>

//- Number of threads per block;
// constexpr int BLOCK_SIZE = 256;

//- Warp size
constexpr int WARP_SIZE  = 32;

//- Number of chromosome per block;
// constexpr int CHR_PER_BLOCK = (BLOCK_SIZE / WARP_SIZE);

/**
 * @struct EvolutionParameters
 * @brief  Parameters of the evolutionary process.
 */
typedef struct
{
    int POPSIZE;
    int CHROMOSOME_ACTUAL;
    int CHROMOSOME_PSEUDO;
    int NUM_OF_GENERATIONS;
    int NUM_OF_ELITE;
    int TOURNAMENT_SIZE;
    int NUM_OF_CROSSOVER_POINTS;
    float MUTATION_RATE;
    int N_ACTUAL;
    int N_PSEUDO;
    int Nbytes_ACTUAL;
    int Nbytes_PSEUDO;
} EvolutionParameters;

/**
 * @class Parameters
 * @blief Singleton class with Parameters maintaining them in CPU and GPU constant memory.
 */
class Parameters {
private:
    const std::string PARAMNAME = "onemax.prms";
    EvolutionParameters cpuEvoPrms;

public:
    explicit Parameters() {}
    ~Parameters() {}

    void loadParams(void);
    int getPopsize(void) const;
    int getChromosomeActual(void) const;
    int getChromosomePseudo(void) const;
    int getNumOfGenerations(void) const;
    int getNumOfElite(void) const;
    int getTournamentSize(void) const;
    int getNumOfCrossoverPoints(void) const;
    float getMutationRate(void) const;
    int getNActual(void) const;
    int getNPseudo(void) const;
    int getNbytesActual(void) const;
    int getNbytesPseudo(void) const;
    EvolutionParameters getEvoPrms(void) const;
    void copyToDevice();
    void showParams() const;
};

#endif // PARAMETERS_HPP
