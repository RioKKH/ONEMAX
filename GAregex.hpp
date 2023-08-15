#ifndef GAREGEX_HPP
#define GAREGEX_HPP

#include <regex>

const std::regex rePOPSIZE(R"(^POPSIZE\s+(\d+)$)");
const std::regex reCHROMOSOME(R"(^CHROMOSOME\s+(\d+)$)");
const std::regex reNUM_OF_GENERATIONS(R"(^NUM_OF_GENERATIONS\s+(\d+)$)");
const std::regex reNUM_OF_ELITE(R"(^NUM_OF_ELITE\s+(\d+)$)");
const std::regex reTOURNAMENT_SIZE(R"(^TOURNAMENT_SIZE\s+(\d+)$)");
const std::regex reNUM_OF_CROSSOVER_POINTS(R"(^NUM_OF_CROSSOVER_POINTS\s+(\d+)$)");
const std::regex reMUTATION_RATE(R"(^MUTATION_RATE\s+(\d\.\d+)$)");                                         

#endif // GAREGEX_HPP
