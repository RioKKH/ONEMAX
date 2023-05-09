#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include "Parameters.h"
#include "GAregex.hpp"


void Parameters::loadParams()
{
    std::ifstream infile(PARAMNAME);
    std::string line;
    std::smatch results;

    while (getline(infile, line))
    {
        if (std::regex_match(line, results, rePOPSIZE))
        {
            cpuEvoPrms.POPSIZE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reCHROMOSOME))
        {
            cpuEvoPrms.CHROMOSOME_ACTUAL = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_GENERATIONS))
        {
            cpuEvoPrms.NUM_OF_GENERATIONS = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_ELITE))
        {
            cpuEvoPrms.NUM_OF_ELITE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reTOURNAMENT_SIZE))
        {
            cpuEvoPrms.TOURNAMENT_SIZE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_CROSSOVER_POINTS))
        {
            cpuEvoPrms.NUM_OF_CROSSOVER_POINTS = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reMUTATION_RATE))
        {
            cpuEvoPrms.MUTATION_RATE = std::stof(results[1].str());
        }
    }

    //- ここで確保する遺伝子メモリサイズを決定する
    for (int i = 5; i <= 10; ++i)
    {
        if (cpuEvoPrms.CHROMOSOME_ACTUAL == (1 << i))
        {
            cpuEvoPrms.CHROMOSOME_PSEUDO = (1 << i);
        }
        else if (( (1 << i) < cpuEvoPrms.CHROMOSOME_ACTUAL )
                && ( cpuEvoPrms.CHROMOSOME_ACTUAL < (1 << (i + 1)) ))
        {
            cpuEvoPrms.CHROMOSOME_PSEUDO = (1 << (i + 1));
        }
    }

    //-  総遺伝子長と総遺伝子サイズを設定する
    cpuEvoPrms.N_ACTUAL = cpuEvoPrms.POPSIZE * cpuEvoPrms.CHROMOSOME_ACTUAL;
    cpuEvoPrms.N_PSEUDO = cpuEvoPrms.POPSIZE * cpuEvoPrms.CHROMOSOME_PSEUDO;
    cpuEvoPrms.Nbytes_ACTUAL = cpuEvoPrms.N_ACTUAL * sizeof(int);
    cpuEvoPrms.Nbytes_PSEUDO = cpuEvoPrms.N_PSEUDO * sizeof(int);

    infile.close();

    return;
}

int Parameters::getPopsize() const { return cpuEvoPrms.POPSIZE; }
int Parameters::getChromosomeActual() const { return cpuEvoPrms.CHROMOSOME_ACTUAL; }
int Parameters::getChromosomePseudo() const { return cpuEvoPrms.CHROMOSOME_PSEUDO; }
int Parameters::getNumOfGenerations() const { return cpuEvoPrms.NUM_OF_GENERATIONS; }
int Parameters::getNumOfElite() const { return cpuEvoPrms.NUM_OF_ELITE; }
int Parameters::getTournamentSize() const { return cpuEvoPrms.TOURNAMENT_SIZE; }
int Parameters::getNumOfCrossoverPoints() const { return cpuEvoPrms.NUM_OF_CROSSOVER_POINTS; }
float Parameters::getMutationRate() const { return cpuEvoPrms.MUTATION_RATE; }
int Parameters::getNActual() const { return cpuEvoPrms.N_ACTUAL; }
int Parameters::getNPseudo() const { return cpuEvoPrms.N_PSEUDO; }
int Parameters::getNbytesActual() const { return cpuEvoPrms.Nbytes_ACTUAL; }
int Parameters::getNbytesPseudo() const { return cpuEvoPrms.Nbytes_PSEUDO; }
EvolutionParameters Parameters::getEvoPrms() const { return cpuEvoPrms; }

void Parameters::showParams() const
{
    std::cout << "POPSIZE: " << cpuEvoPrms.POPSIZE << std::endl;
    std::cout << "CHROMOSOME: " << cpuEvoPrms.CHROMOSOME_ACTUAL << std::endl;
    std::cout << "NUM_OF_GENERATIONS: " << cpuEvoPrms.NUM_OF_GENERATIONS << std::endl;
    std::cout << "NUM_OF_ELITE: " << cpuEvoPrms.NUM_OF_ELITE << std::endl;
    std::cout << "TOURNAMENT_SIZE: " << cpuEvoPrms.TOURNAMENT_SIZE << std::endl;
    std::cout << "NUM_OF_CROSSOVER_POINTS: " << cpuEvoPrms.NUM_OF_CROSSOVER_POINTS << std::endl;
    std::cout << "MUTATION_RATE: " << cpuEvoPrms.MUTATION_RATE << std::endl;
    std::cout << "N: " << cpuEvoPrms.N_ACTUAL << std::endl;
    std::cout << "Nbytes: " << cpuEvoPrms.Nbytes_ACTUAL << std::endl;
}

