#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "Random123/philox.h"
#include "Random123/uniform.hpp"
#include "CUDAKernels.h"
#include "Parameters.h"
#include "Population.h"

typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;

__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

__constant__ int64_t RANDMAX = 4294967295;
__constant__ EvolutionParameters gpuEvoPrms;


void copyToDevice(EvolutionParameters cpuEvoPrms)
{
#ifdef _DEBUG
    printf("copyToDevice %d\n", cpuEvoPrms.POPSIZE);
#endif // _DEBUG
    cudaMemcpyToSymbol(gpuEvoPrms,
                       &cpuEvoPrms,
                       sizeof(EvolutionParameters));
}


void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber)
{
    const cudaError_t cudaError = cudaGetLastError();

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr,
                "Error in the CUDA routine: \"%s\"\nFile name: %s\nLine number: %d\n",
                cudaGetErrorString(cudaError),
                sourceFileName,
                sourceLineNumber);

        exit(EXIT_FAILURE);
    }
}


inline __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                          unsigned int counter)
{
    RNG_2x32 rng;
    return rng({0, counter}, {key});
} // end of TwoRandomINTs


__global__ void cudaKernelGenerateFirstPopulation(
        PopulationData* populationData,
        unsigned int    randomSeed)
{
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key
        = {{static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(blockIdx.x)}};
    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues;

    // offsetは遺伝子配列をPSEUDOの長さで扱わないといけない。
    uint32_t offset = blockIdx.x * gpuEvoPrms.CHROMOSOME_PSEUDO;
    // strideは実際にビット情報が入っている領域、つまりCHROMOSOME_ACTUALの長さで扱う。
    uint32_t stride = gpuEvoPrms.CHROMOSOME_ACTUAL / 4;

    // #pragma unroll // を使うこともできるが、具体的に書き下す事をここでは選択した
    // gpuEvoPrms.CHROMOSOME_ACTUALが4の倍数であることを
    // 前提としたループアンローリングを行う
    for (int i = 0; i < gpuEvoPrms.CHROMOSOME_ACTUAL / 4; i += 4)
    {
        // Iteration 1
        counter.incr();
        randomValues = rng_4x32(counter, key);
        populationData->population[offset +              i    ] = randomValues.v[0] % 2;
        populationData->population[offset + stride     + i    ] = randomValues.v[1] % 2;
        populationData->population[offset + stride * 2 + i    ] = randomValues.v[2] % 2;
        populationData->population[offset + stride * 3 + i    ] = randomValues.v[3] % 2;
        // Iteration 2
        counter.incr();
        randomValues = rng_4x32(counter, key);
        populationData->population[offset +              i + 1] = randomValues.v[0] % 2;
        populationData->population[offset + stride     + i + 1] = randomValues.v[1] % 2;
        populationData->population[offset + stride * 2 + i + 1] = randomValues.v[2] % 2;
        populationData->population[offset + stride * 3 + i + 1] = randomValues.v[3] % 2;
        // Iteration 3
        counter.incr();
        randomValues = rng_4x32(counter, key);
        populationData->population[offset +              i + 2] = randomValues.v[0] % 2;
        populationData->population[offset + stride     + i + 2] = randomValues.v[1] % 2;
        populationData->population[offset + stride * 2 + i + 2] = randomValues.v[2] % 2;
        populationData->population[offset + stride * 3 + i + 2] = randomValues.v[3] % 2;
        // Iteration 4
        counter.incr();
        randomValues = rng_4x32(counter, key);
        populationData->population[offset +              i + 3] = randomValues.v[0] % 2;
        populationData->population[offset + stride     + i + 3] = randomValues.v[1] % 2;
        populationData->population[offset + stride * 2 + i + 3] = randomValues.v[2] % 2;
        populationData->population[offset + stride * 3 + i + 3] = randomValues.v[3] % 2;
    }
    
    // フィットネスの初期化を全スレッドで行う
    populationData->fitness[blockIdx.x] = 0;
}


__global__ void evaluation(PopulationData* populationData)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx  = threadIdx.x;

    // 共有メモリの配列要素数をカーネル起動時に動的に決定
    extern __shared__ volatile int s_idata[];

    s_idata[tx] = populationData->population[idx];
    __syncthreads();

    // 以下はblockDim.xが32の倍数であることを前提として
    // ループアンローリングを用いる
    if (blockDim.x >= 1024 && tx < 512) { s_idata[tx] += s_idata[tx + 512]; __syncthreads(); }
    if (blockDim.x >=  512 && tx < 256) { s_idata[tx] += s_idata[tx + 256]; __syncthreads(); }
    if (blockDim.x >=  256 && tx < 128) { s_idata[tx] += s_idata[tx + 128]; __syncthreads(); }
    if (blockDim.x >=  128 && tx <  64) { s_idata[tx] += s_idata[tx +  64]; __syncthreads(); }

    if (tx < 32)
    {
        // ワープ内でのリダクション
        s_idata[tx] += s_idata[tx + 32];
        s_idata[tx] += s_idata[tx + 16];
        s_idata[tx] += s_idata[tx +  8];
        s_idata[tx] += s_idata[tx +  4];
        s_idata[tx] += s_idata[tx +  2];
        s_idata[tx] += s_idata[tx +  1];
    }

    if (tx == 0)
    {
        populationData->fitness[blockIdx.x] = s_idata[0];
    }
}

// __global__ void evaluation(PopulationData* populationData)
// {
//     int idx  = blockIdx.x * blockDim.x + threadIdx.x;
//     int tx   = threadIdx.x;
//     int stride;
// 
//     // printf("blockIdx.x:%d,threadIdx.x:%d,globalIdx.x:%d,\n", blockIdx.x, threadIdx.x, idx);
//     // 共有メモリの配列要素数をカーネル起動時に動的に決定
//     extern __shared__ volatile int s_idata[];
// 
//     s_idata[tx] = populationData->population[idx];
//     __syncthreads();
// 
//     for (stride = blockDim.x/2; stride >= 1; stride >>=1)
//     {
//         if (tx < stride)
//         {
//             s_idata[tx] += s_idata[tx + stride];
//         }
//         __syncthreads();
//     }
// 
//     if (tx == 0)
//     {
//         populationData->fitness[blockIdx.x] = s_idata[tx];
//     }
// }


__global__ void pseudo_elitism(PopulationData* populationData)
{
    int numOfEliteIdx     = blockIdx.x;  // index of elite
    int localFitnessIdx   = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    int globalFitnessIdx  = threadIdx.x + blockIdx.x * blockDim.x; // size of POPULATION x 2
    const int OFFSET      = blockDim.x;  // size of NUM_OF_ELITE

    extern __shared__ volatile int s_fitness[];

    // shared memoryにデータを読み込み
    s_fitness[localFitnessIdx]          = populationData->fitness[globalFitnessIdx];
    s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx;
    __syncthreads();

    // Warp単位でのリダクション
    if (localFitnessIdx < 32) {
        volatile int* warpShared = s_fitness + (localFitnessIdx / 32) * 32;
        if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 16]) {
            warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 16];
            warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 16 + OFFSET];
        }
        if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 8]) {
            warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 8];
            warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 8 + OFFSET];
        }
        if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 4]) {
            warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 4];
            warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 4 + OFFSET];
        }
        if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 2]) {
            warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 2];
            warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 2 + OFFSET];
        }
        if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 1]) {
            warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 1];
            warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 1 + OFFSET];
        }
    }
    __syncthreads();

    // ブロック単位でのリダクション
    for (int stride = OFFSET/2; stride >= 32; stride >>= 1)
    {
        if (localFitnessIdx < stride)
        {
            unsigned int index =
                (s_fitness[localFitnessIdx] >= s_fitness[localFitnessIdx + stride])
                ? localFitnessIdx : localFitnessIdx + stride;

            s_fitness[localFitnessIdx]          = s_fitness[index];
            s_fitness[localFitnessIdx + OFFSET] = s_fitness[index + OFFSET];
        }
        __syncthreads();
    }

    if (localFitnessIdx == 0 && blockIdx.x < gridDim.x)
    {
        populationData->elitesIdx[numOfEliteIdx] = s_fitness[localFitnessIdx + OFFSET];
    }
}


__global__ void pseudo_elitismOrg(PopulationData* populationData)
{
    int numOfEliteIdx     = blockIdx.x;  // index of elite
    int localFitnessIdx   = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    int globalFitnessIdx  = threadIdx.x + blockIdx.x * blockDim.x; // size of POPULATION x 2
    const int OFFSET      = blockDim.x;  // size of NUM_OF_ELITE

    extern __shared__ volatile int s_fitness[];

    // shared memoryにデータを読み込み
    // ブロック数はElite数の２倍。
    s_fitness[localFitnessIdx]          = populationData->fitness[globalFitnessIdx];
    s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx;
    __syncthreads();

    for (int stride = OFFSET/2; stride >= 1; stride >>= 1)
    {
        if (localFitnessIdx < stride)
        {
            unsigned int index =
                (s_fitness[localFitnessIdx] >= s_fitness[localFitnessIdx + stride])
                ? localFitnessIdx : localFitnessIdx + stride;

            s_fitness[localFitnessIdx]          = s_fitness[index];
            s_fitness[localFitnessIdx + OFFSET] = s_fitness[index + OFFSET];
        }
        __syncthreads();
    }

    if (localFitnessIdx == 0 && blockIdx.x < gridDim.x)
    {
        populationData->elitesIdx[numOfEliteIdx] = s_fitness[localFitnessIdx + OFFSET];
#ifdef _DEBUG
        printf("elitism: blockIdx:%d , eliteIndex:%d\n",
                blockIdx.x, s_fitness[localFitnessIdx + OFFSET]);
#endif // _DEBUG
    }
}

__global__ void replaceWithElites(
        PopulationData *parentPopulation,
        PopulationData *offspringPopulation)
{
    const uint32_t NUM_OF_ELITE = gpuEvoPrms.NUM_OF_ELITE;
    uint32_t geneIdx = threadIdx.x;
    uint32_t eliteIdx = blockIdx.x;

    uint32_t ELITE_INTERVAL = gpuEvoPrms.POPSIZE / NUM_OF_ELITE;
    uint32_t offspringIdx = eliteIdx * ELITE_INTERVAL;
    uint32_t ELITE_OFFSET
        = gpuEvoPrms.CHROMOSOME_PSEUDO * parentPopulation->elitesIdx[eliteIdx];
    uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * offspringIdx;

    offspringPopulation->population[OFFSET + geneIdx]
        = parentPopulation->population[ELITE_OFFSET + geneIdx];

    if (geneIdx == 0) {
        offspringPopulation->fitness[offspringIdx]
            = parentPopulation->fitness[parentPopulation->elitesIdx[eliteIdx]];
    }
}


__global__ void replaceWithElites2(
        PopulationData *parentPopulation,
        PopulationData *offspringPopulation)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * idx;

    uint32_t ELITE_INTERVAL = gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE;
    uint32_t ELITE_INDEX = idx / ELITE_INTERVAL;
    uint32_t ELITE_OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * parentPopulation->elitesIdx[ELITE_INDEX];

    bool shouldReplace = (idx % ELITE_INTERVAL == 0);
    // Use predicated execution to avoid warp divergence
    for (int i = 0; i < gpuEvoPrms.CHROMOSOME_PSEUDO; ++i)
    {
        offspringPopulation->population[OFFSET + i] = shouldReplace ?
            parentPopulation->population[ELITE_OFFSET + i] :
            offspringPopulation->population[OFFSET + i];
    }

    if (shouldReplace) {
        offspringPopulation->fitness[idx] = parentPopulation->fitness[parentPopulation->elitesIdx[ELITE_INDEX]];
    }
}


__global__ void replaceWithElitesNew1(
        PopulationData *parentPopulation,
        PopulationData *offspringPopulation)
{
    uint32_t tx = threadIdx.x;
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * threadIdx.x;
    // const uint32_t POP_PER_THR = gpuEvoPrms.POPSIZE / blockDim.x;

    // Shared memory to store elite indices
    extern __shared__ uint32_t sharedElites[];

    if (tx < gpuEvoPrms.NUM_OF_ELITE) {
        sharedElites[tx] = parentPopulation->elitesIdx[tx];
    }
    __syncthreads();

    uint32_t ELITE_INDEX = idx / (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE);
    uint32_t ELITE_OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * sharedElites[ELITE_INDEX];

    // Use predicated execution to avoid warp divergence
    bool shouldReplace = (idx % (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE) == 0);

    for (int i = 0; i < gpuEvoPrms.CHROMOSOME_PSEUDO; ++i)
    {
        offspringPopulation->population[OFFSET + i]
            = shouldReplace ? parentPopulation->population[ELITE_OFFSET + i]
                            : offspringPopulation->population[OFFSET + i];
    }
    if (shouldReplace) {
        offspringPopulation->fitness[idx]
            = parentPopulation->fitness[sharedElites[ELITE_INDEX]];
    }
    __syncthreads();
}

__global__ void replaceWithElitesOld(
        PopulationData *parentPopulation,
        PopulationData *offspringPopulation)
{
    // uint32_t tx  = threadIdx.x;
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * threadIdx.x;
    // const uint32_t POP_PER_THR = gpuEvoPrms.POPSIZE / blockDim.x;

    if (idx % (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE) == 0)
    {
        uint32_t ELITE_INDEX
            = idx / (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE);
        uint32_t ELITE_OFFSET
            = gpuEvoPrms.CHROMOSOME_PSEUDO * parentPopulation->elitesIdx[ELITE_INDEX];

        // std::uint32_t ELITE_OFFSET = gpuEvoPrms.CHROMOSOME_ACTUAL * parentPopulation->elitesIdx[ELITE_INDEX];
        //- エリートの遺伝子を子にコピーする
        for (int i = 0; i < gpuEvoPrms.CHROMOSOME_PSEUDO; ++i)
        // for (int i = 0; i < gpuEvoPrms.CHROMOSOME_ACTUAL; ++i)
        {
            offspringPopulation->population[OFFSET + i]
                = parentPopulation->population[ELITE_OFFSET + i];
        }
        //- エリートのFitnessをコピーする
        offspringPopulation->fitness[idx]
            = parentPopulation->fitness[parentPopulation->elitesIdx[ELITE_INDEX]];
    }
    __syncthreads();
}


__global__ void cudaKernelSelection(
        PopulationData* mParentPopulation,
        // PopulationData* mOffspringPopulation,
        uint32_t* selectedParents1, 
        uint32_t* selectedParents2,
        unsigned int    randomSeed)
{
    uint32_t PARENTIDX = threadIdx.x + blockIdx.x * blockDim.x;
    // const int CHR_PER_BLOCK = blockDim.x;

    // Ensure the index is within the population size
    // if (PARENTIDX >= mParentPopulation->populationSize) {
    if (PARENTIDX >= gpuEvoPrms.POPSIZE) {
        return;
    }

    // Init random number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {
        {
            static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(blockIdx.x)
        }
    };

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues1;
    RNG_4x32::ctr_type randomValues2;

    counter.incr();
    randomValues1 = rng_4x32(counter, key);
    counter.incr();
    randomValues2 = rng_4x32(counter, key);

    // 親1
    selectedParents1[PARENTIDX] = tournamentSelection(
            mParentPopulation,
            gpuEvoPrms.TOURNAMENT_SIZE,
            randomValues1.v[0],
            randomValues1.v[1],
            randomValues1.v[2],
            randomValues1.v[3]);
    // 親2
    selectedParents2[PARENTIDX] = tournamentSelection(
            mParentPopulation,
            gpuEvoPrms.TOURNAMENT_SIZE,
            randomValues2.v[0],
            randomValues2.v[1],
            randomValues2.v[2],
            randomValues2.v[3]);
}


__global__ void cudaKernelCrossover(
        PopulationData* parent,
        PopulationData* offspring,
        uint32_t* selectedParents1, 
        uint32_t* selectedParents2,
        unsigned int   randomSeed)
{
    uint32_t PARENTIDX = blockIdx.x;
    uint32_t CHROMOIDX = threadIdx.x;

    // // Init randome number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {
        {
            static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(blockIdx.x)
        }
    };

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues1;

    counter.incr();
    randomValues1 = rng_4x32(counter, key);
    printf("%e\n", randomValues1.v[0]);

    // ここで使われているchromosomeSizeはPseudoChromosomeSizeであることに注意
    uint32_t crossoveridx1 = randomValues1.v[0] % (gpuEvoPrms.CHROMOSOME_ACTUAL);
    uint32_t crossoveridx2 = randomValues1.v[1] % (gpuEvoPrms.CHROMOSOME_ACTUAL);
    // printf("crossoveridx1: %d, crossoveridx2: %d\n", crossoveridx1, crossoveridx2);

    swap(crossoveridx1, crossoveridx2);
    
    uint32_t parent1idx = selectedParents1[PARENTIDX];
    uint32_t parent2idx = selectedParents2[PARENTIDX];

    // Warpダイバージェンスを避けるための条件
    // この書き方をすれば、どのスレッドも同じ条件分岐を通ることになる為、
    // Warpダイバージェンスが発生しない
    bool isParent1 = (CHROMOIDX < crossoveridx1) || (CHROMOIDX >= crossoveridx2);

    if (CHROMOIDX < gpuEvoPrms.CHROMOSOME_PSEUDO)
    {
        offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX]
            = isParent1 ? parent->population[parent1idx * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX]
                        : parent->population[parent2idx * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX];
    }
}


__global__ void cudaKernelMutation(
        PopulationData* offspring,
        unsigned int   randomSeed)
{
    uint32_t PARENTIDX = blockIdx.x;
    uint32_t CHROMOIDX = threadIdx.x;
    // Ensure the index is within the population size
    // つまり1つのブロックには4スレッドだけ処理させることにする
    // if (offspringIdx >= offspring->populationSize || geneIdx >= 4) {
    //     return;
    // }

    // Init random number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {
        {
            static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(blockIdx.x)
        }
    };

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues;

    counter.incr();
    randomValues = rng_4x32(counter, key);
    printf("%f\n", randomValues.v[0]);

    // uint32_t genePosition = randomValues.v[CHROMOIDX] % (offspring->chromosomeSize);
    // uint32_t genePosition = randomValues.v[CHROMOIDX] % (gpuEvoPrms.CHROMOSOME_ACTUAL);
    // printf("randomValues.v[0]: %.12f, gpuEvoPrms.MUTATION_RATE: %f\n", randomValues.v[0], gpuEvoPrms.MUTATION_RATE);
    bool shouldMutate = randomValues.v[0] < gpuEvoPrms.MUTATION_RATE;

    // Warpダイバージェンスを避けるための条件
    bool isOriginal = !shouldMutate;
    // bool isMutated = shouldMutate;

    if (CHROMOIDX < gpuEvoPrms.CHROMOSOME_ACTUAL)
    {
        offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX]
            = isOriginal ?  offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX]
                         : ~offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX];
    }
}


inline __device__ int getBestIndividual(
        const PopulationData* mParentPopulation,
        const int& idx1,
        const int& idx2,
        const int& idx3,
        const int& idx4)
{
    int better1
        = mParentPopulation->fitness[idx1]
        > mParentPopulation->fitness[idx2] ? idx1 : idx2;
    int better2
        = mParentPopulation->fitness[idx3]
        > mParentPopulation->fitness[idx4] ? idx3 : idx4;
    int bestIdx
        = mParentPopulation->fitness[better1]
        > mParentPopulation->fitness[better2] ? better1 : better2;

    return bestIdx;
}


inline __device__ int tournamentSelection(
        const PopulationData* populationData,
        int tounament_size,
        const std::uint32_t& random1,
        const std::uint32_t& random2,
        const std::uint32_t& random3,
        const std::uint32_t& random4)
{
    // トーナメントサイズは4で固定とする。
    // これはrand123が一度に返すことが出来る乱数の最大個数が4のため。
    unsigned int idx1 = random1 % populationData->populationSize;
    unsigned int idx2 = random2 % populationData->populationSize;
    unsigned int idx3 = random3 % populationData->populationSize;
    unsigned int idx4 = random4 % populationData->populationSize;
    int bestIdx = getBestIndividual(populationData, idx1, idx2, idx3, idx4);
#ifdef _DEBUG
    printf("tournamentSelection: %d,%d,%d,%d,%d\n", bestIdx, idx1, idx2, idx3, idx4);
#endif // _DEBUG
    return bestIdx;
}

inline __device__ void swap(unsigned int& point1, unsigned int& point2)
{
    // Warpダイバージェンスが発生しないように以下のような実装に変更する

    // point1とpoint2が等しい場合、場合point2に1を加算する
    point2 += (point1 == point2);

    // point1とpoint2より大きい場合、値を交換する

    unsigned int tmp = (point1 > point2) ? point1 : point2;
    point1 = (point1 > point2) ? point2 : point1;
    point2 = tmp;
}

