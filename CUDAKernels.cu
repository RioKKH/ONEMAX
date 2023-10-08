#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "Random123/philox.h"
// #include "Random123/uniform.hpp"
#include "CUDAKernels.h"
#include "Parameters.h"
#include "Population.h"

using namespace r123;

typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;

__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

__constant__ uint32_t RANDMAX = 4294967295;
__constant__ EvolutionParameters gpuEvoPrms;



void copyToDevice(EvolutionParameters cpuEvoPrms)
{
#ifdef _DEBUG
    printf("copyToDevice %d\n", cpuEvoPrms.POPSIZE_ACTUAL);
    // printf("copyToDevice %d\n", cpuEvoPrms.POPSIZE);
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
    const int geneIdx = threadIdx.x;
    const int chromosomeIdx = threadIdx.x + blockIdx.x * blockDim.x;

    RNG_4x32  rng_4x32;
    RNG_4x32::key_type key
        = {{static_cast<unsigned int>(geneIdx), static_cast<unsigned int>(chromosomeIdx)}};
    RNG_4x32::ctr_type counter = {{0, 0, randomSeed ,0xbeeff00d}};
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
        // printf("randomValues: %u %u %u %u\n",
        //         randomValues.v[0], randomValues.v[1], randomValues.v[2], randomValues.v[3]);
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx  = threadIdx.x;

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
    const int OFFSET           = blockDim.x;  // Population size for each elite, 20
    const int EliteIdx         = blockIdx.x;  // index of elite
    const int localFitnessIdx  = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    const int globalFitnessIdx = threadIdx.x + blockIdx.x * blockDim.x; // Population size
    const int ACTUALPOPSIZE    = gpuEvoPrms.POPSIZE_ACTUAL;

    extern __shared__ volatile int s_fitness[];

    // shared memoryにデータを読み込み
    if (localFitnessIdx < gpuEvoPrms.POPSIZE_ACTUAL)
    {
        s_fitness[localFitnessIdx] = populationData->fitness[globalFitnessIdx];
        s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx;
    }
    else
    {
        s_fitness[localFitnessIdx] = 0;
        s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx;
    }
    __syncthreads();

    // グロック単位でのリダクション
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
        populationData->elitesIdx[EliteIdx] = s_fitness[OFFSET];
    }
}



__global__ void pseudo_elitism20231009(PopulationData* populationData)
{
    const int EliteIdx     = blockIdx.x;  // index of elite
    const int localFitnessIdx   = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    const int globalFitnessIdx  = threadIdx.x + blockIdx.x * blockDim.x; // Population size
    const int OFFSET            = blockDim.x;  // Population size for each elite, 20

    extern __shared__ volatile int s_fitness[];

    // shared memoryにデータを読み込み
    // printf("Offset:%d, localFitnessIdx:%d, globalFitnessIdx:%d\n", OFFSET, localFitnessIdx, globalFitnessIdx);

    //        0 ~ 19                                     0 ~ 7      20       0 ~ 19
    s_fitness[localFitnessIdx] = populationData->fitness[EliteIdx * OFFSET + localFitnessIdx];
    //        0 ~ 19             0~7        20       0 ~ 19
    s_fitness[localFitnessIdx] = EliteIdx * OFFSET + localFitnessIdx;
    // 上の配列の
    __syncthreads();


    // ブロック単位でのリダクション
    // for (int stride = OFFSET/2; stride >= 32; stride >>= 1)
    // これでは所望のリダクション処理になっていない
    //                20    /2               5, 2, 1 --> NG!
    for (int stride = OFFSET/2; stride >= 1; stride >>= 1)
    {
        // printf("stride:%d, localFitnessIdx:%d\n", stride, localFitnessIdx);
        //  < 20 = 160/8      8, 4, 2, 1
        if (localFitnessIdx < stride)
        {
            unsigned int index =
                //         < 20 = 160 / 8                < 20 = 160 / 8  + 8, 4, 2, 1
                (s_fitness[localFitnessIdx] >= s_fitness[localFitnessIdx + stride])
                ? localFitnessIdx : localFitnessIdx + stride;

            s_fitness[localFitnessIdx]          = s_fitness[index];
            s_fitness[localFitnessIdx + OFFSET] = s_fitness[index + OFFSET];
        }
        __syncthreads();
    }

    if (localFitnessIdx == 0 && blockIdx.x < gridDim.x)
    {
        //                                                                20
        populationData->elitesIdx[EliteIdx] = s_fitness[OFFSET];
        // populationData->elitesIdx[EliteIdx] = s_fitness[localFitnessIdx + OFFSET];
    }
}

    // // Warp単位でのリダクション
    // if (localFitnessIdx < 32) {
    //     volatile int* warpShared = s_fitness + (localFitnessIdx / 32) * 32;
    //     if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 16]) {
    //         warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 16];
    //         warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 16 + OFFSET];
    //     }
    //     if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 8]) {
    //         warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 8];
    //         warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 8 + OFFSET];
    //     }
    //     if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 4]) {
    //         warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 4];
    //         warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 4 + OFFSET];
    //     }
    //     if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 2]) {
    //         warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 2];
    //         warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 2 + OFFSET];
    //     }
    //     if (warpShared[localFitnessIdx] < warpShared[localFitnessIdx + 1]) {
    //         warpShared[localFitnessIdx] = warpShared[localFitnessIdx + 1];
    //         warpShared[localFitnessIdx + OFFSET] = warpShared[localFitnessIdx + 1 + OFFSET];
    //     }
    // }
    // __syncthreads();

__global__ void pseudo_elitismOrg(PopulationData* populationData)
{
    const int numOfEliteIdx     = blockIdx.x;  // index of elite
    const int localFitnessIdx   = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    const int globalFitnessIdx  = threadIdx.x + blockIdx.x * blockDim.x; // size of POPULATION x 2
    const int OFFSET            = blockDim.x;  // size of NUM_OF_ELITE

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
    const uint32_t geneIdx = threadIdx.x;
    const uint32_t eliteIdx = blockIdx.x;

    // 何個体ごとにエリートを選択するかを計算する。
    // こちらは実際の個体数で計算する。
    uint32_t ELITE_INTERVAL = gpuEvoPrms.POPSIZE_ACTUAL / NUM_OF_ELITE;
    // uint32_t ELITE_INTERVAL = gpuEvoPrms.POPSIZE / NUM_OF_ELITE;
    // Offspringの何個体目の個体をエリートで置き換えるかを計算する。
    uint32_t offspringIdx = eliteIdx * ELITE_INTERVAL;

    // 親の世代のエリートのオフセット値を計算する
    uint32_t PARENT_ELITE_OFFSET
        = gpuEvoPrms.CHROMOSOME_PSEUDO * parentPopulation->elitesIdx[eliteIdx];

    // 子の世代のオフセット値を計算する
    uint32_t OFFSPRING_OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * offspringIdx;

    // 子の世代の個体の遺伝子を親の世代のエリート個体の遺伝子で置き換える
    offspringPopulation->population[OFFSPRING_OFFSET + geneIdx]
        = parentPopulation->population[PARENT_ELITE_OFFSET + geneIdx];

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

    uint32_t ELITE_INTERVAL = gpuEvoPrms.POPSIZE_ACTUAL / gpuEvoPrms.NUM_OF_ELITE;
    // uint32_t ELITE_INTERVAL = gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE;
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

    uint32_t ELITE_INDEX = idx / (gpuEvoPrms.POPSIZE_ACTUAL / gpuEvoPrms.NUM_OF_ELITE);
    // uint32_t ELITE_INDEX = idx / (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE);
    uint32_t ELITE_OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * sharedElites[ELITE_INDEX];

    // Use predicated execution to avoid warp divergence
    bool shouldReplace = (idx % (gpuEvoPrms.POPSIZE_ACTUAL / gpuEvoPrms.NUM_OF_ELITE) == 0);
    // bool shouldReplace = (idx % (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE) == 0);

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

__global__ void replaceWithElitesOrg(
        PopulationData *parentPopulation,
        PopulationData *offspringPopulation)
{
    // uint32_t tx  = threadIdx.x;
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * threadIdx.x;
    // const uint32_t POP_PER_THR = gpuEvoPrms.POPSIZE / blockDim.x;

    if (idx % (gpuEvoPrms.POPSIZE_ACTUAL / gpuEvoPrms.NUM_OF_ELITE) == 0)
    // if (idx % (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE) == 0)
    {
        uint32_t ELITE_INDEX
            = idx / (gpuEvoPrms.POPSIZE_ACTUAL / gpuEvoPrms.NUM_OF_ELITE);
            // = idx / (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE);
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
    const uint32_t PARENTIDX = threadIdx.x + blockIdx.x * blockDim.x;
    // const int CHR_PER_BLOCK = blockDim.x;

    // Ensure the index is within the population size
    // if (PARENTIDX >= mParentPopulation->populationSize) {
    if (PARENTIDX >= gpuEvoPrms.POPSIZE_ACTUAL) {
    // if (PARENTIDX >= gpuEvoPrms.POPSIZE) {
        return;
    }

    // Init random number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {
        {
            static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(PARENTIDX)
        }
    };

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues1;
    RNG_4x32::ctr_type randomValues2;

    counter.incr();
    randomValues1 = rng_4x32(counter, key);
    counter.incr();
    randomValues2 = rng_4x32(counter, key);

    // printf("randomValues1: %u %u %u %u\n",
    //         randomValues1.v[0], randomValues1.v[1], randomValues1.v[2], randomValues1.v[3]);
    // printf("randomValues2: %u %u %u %u\n",
    //         randomValues2.v[0], randomValues2.v[1], randomValues2.v[2], randomValues2.v[3]);

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
    const uint32_t PARENTIDX = blockIdx.x;
    const uint32_t CHROMOIDX = threadIdx.x;

    // // Init randome number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {
        {
            static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(threadIdx.x + blockIdx.x * blockDim.x)
        }
    };

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues1;

    counter.incr();
    randomValues1 = rng_4x32(counter, key);
    // printf("randomValues1: %u %u %u %u\n",
    //         randomValues1.v[0], randomValues1.v[1], randomValues1.v[2], randomValues1.v[3]);

    // ここで使われているchromosomeSizeはPseudoChromosomeSizeであることに注意
    uint32_t crossoveridx1 = randomValues1.v[0] % (gpuEvoPrms.CHROMOSOME_ACTUAL);
    uint32_t crossoveridx2 = randomValues1.v[1] % (gpuEvoPrms.CHROMOSOME_ACTUAL);

    swap(crossoveridx1, crossoveridx2);
    // printf("crossoveridx1: %d, crossoveridx2: %d\n", crossoveridx1, crossoveridx2);
    
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
    const uint32_t PARENTIDX = blockIdx.x;
    const uint32_t CHROMOIDX = threadIdx.x;
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    const RNG_2x32::ctr_type randomValues = generateTwoRndValues(idx, randomSeed);

    bool shouldMutate = ((double)randomValues.v[0] / RANDMAX) < gpuEvoPrms.MUTATION_RATE;
    // printf("randomValues: %f %f %d\n",
    //         (double)randomValues.v[0] / RANDMAX, gpuEvoPrms.MUTATION_RATE, shouldMutate);

    // Warpダイバージェンスを避けるための条件
    bool isOriginal = !shouldMutate;
    // bool isMutated = shouldMutate;

    if (CHROMOIDX < gpuEvoPrms.CHROMOSOME_ACTUAL)
    {
        offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX]
            = isOriginal ? offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX]
                         : offspring->population[PARENTIDX * gpuEvoPrms.CHROMOSOME_PSEUDO + CHROMOIDX] ^ 1;
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
    unsigned int idx1 = random1 % gpuEvoPrms.POPSIZE_ACTUAL;
    unsigned int idx2 = random2 % gpuEvoPrms.POPSIZE_ACTUAL;
    unsigned int idx3 = random3 % gpuEvoPrms.POPSIZE_ACTUAL;
    unsigned int idx4 = random4 % gpuEvoPrms.POPSIZE_ACTUAL;
    // unsigned int idx1 = random1 % populationData->populationSize;
    // unsigned int idx2 = random2 % populationData->populationSize;
    // unsigned int idx3 = random3 % populationData->populationSize;
    // unsigned int idx4 = random4 % populationData->populationSize;
    int bestIdx = getBestIndividual(populationData, idx1, idx2, idx3, idx4);
    // printf("tournamentSelection: %d,%d,%d,%d,%d\n", bestIdx, idx1, idx2, idx3, idx4);
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

