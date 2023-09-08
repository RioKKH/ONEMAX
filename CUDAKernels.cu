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

// __constant__ long RANDMAX = 4294967295;
__constant__ int64_t RANDMAX = 4294967295;
// __constant__ std::int64_t RANDMAX = 4294967295;
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


__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationData,
                                                  unsigned int    randomSeed)
{
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key
        = {{static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(blockIdx.x)}};

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues;
    // RNG_4x32::ctr_type randomValues = rng_4x32(counter, key);

    uint32_t offset = blockIdx.x * gpuEvoPrms.CHROMOSOME_PSEUDO;
    uint32_t stride = gpuEvoPrms.CHROMOSOME_ACTUAL / 4;

    for (int i = 0; i < gpuEvoPrms.CHROMOSOME_ACTUAL / 4; ++i)
    {
        counter.incr();
        randomValues = rng_4x32(counter, key);
        populationData->population[offset + stride * 0 + i] = randomValues.v[0] % 2;
        populationData->population[offset + stride * 1 + i] = randomValues.v[1] % 2;
        populationData->population[offset + stride * 2 + i] = randomValues.v[2] % 2;
        populationData->population[offset + stride * 3 + i] = randomValues.v[3] % 2;
    }

    if (threadIdx.x == 0)
    {
        populationData->fitness[blockIdx.x] = 0;
    }

} // end of cudaGeneratePopulationKernel


__global__ void evaluation(PopulationData* populationData)
{
    // printf("gridmDim.x:%d, blockDim.x:%d, blockIdx.x:%d, threadIdx.x:%d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx   = threadIdx.x;
    int stride;

    // printf("blockIdx.x:%d,threadIdx.x:%d,globalIdx.x:%d,\n", blockIdx.x, threadIdx.x, idx);
    // 共有メモリの配列要素数をカーネル起動時に動的に決定
    extern __shared__ volatile int s_idata[];

    s_idata[tx] = populationData->population[idx];
    __syncthreads();

    for (stride = blockDim.x/2; stride >= 1; stride >>=1)
    {
        if (tx < stride)
        {
            s_idata[tx] += s_idata[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        populationData->fitness[blockIdx.x] = s_idata[tx];
    }
}


__global__ void pseudo_elitism(PopulationData* populationData)
{
    //<<<getNumOfElite(),
    //   getPopsize()/getNumOfElite(),
    //   getPopsize()/getNumOfElite()*2*sizeof(int)>>>
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

    // printf("### Pseudo elitism\n");
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

    // if (localFitnessIdx == 0 && blockIdx.x < gridDim.x/2)
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
    uint32_t tx  = threadIdx.x;
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * threadIdx.x;
    const uint32_t POP_PER_THR = gpuEvoPrms.POPSIZE / blockDim.x;

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
// #ifdef _DEBUG
//         printf("swap target:%d, src eindex:%d, src eoffset:%d\n", idx, ELITE_INDEX, ELITE_OFFSET);
// #endif // _DEBUG

__global__ void swapPopulation(PopulationData* parentPopulation,
                               PopulationData* offspringPopulation)
{
    uint32_t tx  = threadIdx.x;
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 遺伝子の入れ替え処理。ActualでもPseudoでもいけそうな気がするが、
    // 最初はPSEUDOで実装しておく
    uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_PSEUDO * threadIdx.x;
    // std::uint32_t OFFSET = gpuEvoPrms.CHROMOSOME_ACTUAL * threadIdx.x;
    const uint32_t POP_PER_THR = gpuEvoPrms.POPSIZE / blockDim.x;
    // printf("swapPopulation: %d, %d\n", OFFSET, idx);

    //- In case  of <<<1, 1>>>
    if (idx < gpuEvoPrms.CHROMOSOME_PSEUDO * gpuEvoPrms.POPSIZE)
    {
        for (int i = 0; i < gpuEvoPrms.CHROMOSOME_PSEUDO * gpuEvoPrms.POPSIZE; ++i)
        {
            parentPopulation->population[i] = offspringPopulation->population[i];
        }
    }
    __syncthreads();


    //- In case of <<<N, M>>>
    if (idx < gpuEvoPrms.CHROMOSOME_PSEUDO * gpuEvoPrms.POPSIZE)
    // if (idx < gpuEvoPrms.CHROMOSOME_ACTUAL * gpuEvoPrms.POPSIZE)
    {
        if (idx % (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE) == 0)
        {
            uint32_t ELITE_INDEX
                = idx / (gpuEvoPrms.POPSIZE / gpuEvoPrms.NUM_OF_ELITE);
            uint32_t ELITE_OFFSET
                = gpuEvoPrms.CHROMOSOME_PSEUDO * parentPopulation->elitesIdx[ELITE_INDEX];
            // std::uint32_t ELITE_OFFSET = gpuEvoPrms.CHROMOSOME_ACTUAL * parentPopulation->elitesIdx[ELITE_INDEX];
#ifdef _DEBUG
            printf("swap target:%d, src eindex:%d, src eoffset:%d\n", idx, ELITE_INDEX, ELITE_OFFSET);
#endif // _DEBUG
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
    }
    __syncthreads();

    if (idx < gpuEvoPrms.CHROMOSOME_PSEUDO * gpuEvoPrms.POPSIZE)
    // if (idx < gpuEvoPrms.CHROMOSOME_ACTUAL * gpuEvoPrms.POPSIZE)
    {
        for (int i = 0; i < gpuEvoPrms.CHROMOSOME_PSEUDO * POP_PER_THR; ++i)
        // for (int i = 0; i < gpuEvoPrms.CHROMOSOME_ACTUAL * POP_PER_THR; ++i)
        {
            parentPopulation->population[OFFSET + i]
                = offspringPopulation->population[OFFSET + i];
        }
    }
    __syncthreads();

    //- <<<1, 1>>> でも <<<N, M>>>でもFitnessのコピーは必要
    if (idx == 0)
    {
        for (int i = 0; i < gpuEvoPrms.POPSIZE; ++i)
        {
            parentPopulation->fitness[i] = offspringPopulation->fitness[i];
        }
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
    if (PARENTIDX >= mParentPopulation->populationSize) {
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
    uint32_t CHROMOIDX = threadIdx.x + blockIdx.x * blockDim.x;
    // Ensure the index is within the population size
    if (PARENTIDX >= parent->populationSize || CHROMOIDX >= parent->chromosomeSize) {
        return;
    }

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

    uint32_t crossoveridx1 = randomValues1.v[0] % (parent->chromosomeSize);
    uint32_t crossoveridx2 = randomValues1.v[1] % (parent->chromosomeSize);
    swap(crossoveridx1, crossoveridx2);
    
    uint32_t parent1idx = selectedParents1[PARENTIDX];
    uint32_t parent2idx = selectedParents2[PARENTIDX];

    // Warpダイバージェンスを避けるための条件
    // この書き方をすれば、どのスレッドも同じ条件分岐を通ることになる為、
    // Warpダイバージェンスが発生しない
    bool isParent1 = (CHROMOIDX < crossoveridx1) || (CHROMOIDX >= crossoveridx2);
    // bool isParent2 = (CHROMOIDX >= crossoveridx1) && (CHROMOIDX < crossoveridx2);

    if (CHROMOIDX < parent->chromosomeSize)
    {
        offspring->population[PARENTIDX * offspring->chromosomeSize + CHROMOIDX]
            = isParent1 ? parent->population[parent1idx * parent->chromosomeSize + CHROMOIDX]
                        : parent->population[parent2idx * parent->chromosomeSize + CHROMOIDX];
    }
}


__global__ void cudaKernelMutation(
        PopulationData* offspring,
        unsigned int   randomSeed)
{
    uint32_t offspringIdx = blockIdx.x;
    uint32_t geneIdx = threadIdx.x;
    // Ensure the index is within the population size
    // つまり1つのブロックには4スレッドだけ処理させることにする
    if (offspringIdx >= offspring->populationSize || geneIdx >= 4) {
        return;
    }

    // Init random number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {
        {static_cast<unsigned int>(threadIdx.x), static_cast<unsigned int>(blockIdx.x)}
    };

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues;

    counter.incr();
    randomValues = rng_4x32(counter, key);

    uint32_t genePosition = randomValues.v[geneIdx] % (offspring->chromosomeSize);
    bool shouldMutate = randomValues.v[geneIdx] < gpuEvoPrms.MUTATION_RATE;

    // Warpダイバージェンスを避けるための条件
    bool isOriginal = !shouldMutate;
    // bool isMutated = shouldMutate;

    offspring->population[offspringIdx * offspring->chromosomeSize + genePosition]
        = isOriginal ?  offspring->population[offspringIdx * offspring->chromosomeSize + genePosition]
                     : ~offspring->population[offspringIdx * offspring->chromosomeSize + genePosition];
}


#if 0
__global__ void cudaGeneticManipulationKernel(PopulationData* mParentPopulation,
                                              PopulationData* mOffspringPopulation,
                                              unsigned int    randomSeed)
{
    std::int32_t PARENTIDX = threadIdx.x + blockIdx.x * blockDim.x;
    const int CHR_PER_BLOCK = blockDim.x;

    // // Init randome number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key
        = {{static_cast<unsigned int>(threadIdx.x),
            static_cast<unsigned int>(blockIdx.x)}};

    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues1;
    RNG_4x32::ctr_type randomValues2;

    // Produce new offspring
    /* 共有メモリサイズを固定する場合
    __shared__ int parent1Idx[MAX_POP_SIZE];
    __shared__ int parent2Idx[MAX_POP_SIZE];
    __shared__ int randNums[CONST_TOURNAMENT_SIZE];
    */
    //共有メモリを動的に確保する場合
    extern __shared__ int s[];
    int *parent1Idx  = s;
    int *parent2Idx  = (int *)(&parent1Idx[gpuEvoPrms.POPSIZE]);
    int *randNums    = (int *)(&parent2Idx[gpuEvoPrms.POPSIZE]);

// #if 0
    //- selection
    // 1ブロックごとに32スレッドにしている。
    // つまり1ブロック毎に最大で親32体を処理し、子供32体を生成する
    // if ((threadIdx.y == 0) && (threadIdx.x < WARP_SIZE)) // <--ほぼ意味なし
    if (threadIdx.x < WARP_SIZE)
    {
        counter.incr();
        randomValues1 = rng_4x32(counter, key);
        counter.incr();
        randomValues2 = rng_4x32(counter, key);

        // 親1 : 0 ~ 31までのインデックス
        // parent1Idx[threadIdx.x] = tournamentSelection(mParentPopulation, gpuEvoPrms.TOURNAMENT_SIZE,
        parent1Idx[PARENTIDX] = tournamentSelection(mParentPopulation,
                                                    gpuEvoPrms.TOURNAMENT_SIZE,
                                                    randomValues1.v[0],
                                                    randomValues1.v[1],
                                                    randomValues1.v[2],
                                                    randomValues1.v[3]);

        // 親2 : 0 ~ 31までのインデックス
        // parent2Idx[threadIdx.x] = tournamentSelection(mParentPopulation, gpuEvoPrms.TOURNAMENT_SIZE,
        parent2Idx[PARENTIDX] = tournamentSelection(mParentPopulation,
                                                    gpuEvoPrms.TOURNAMENT_SIZE,
                                                    randomValues2.v[0],
                                                    randomValues2.v[1],
                                                    randomValues2.v[2],
                                                    randomValues2.v[3]);
    }
    __syncthreads();
// #endif


// #if 0
    //- crossover
    // if (1)
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    if (threadIdx.x < WARP_SIZE)
    {
        counter.incr();
        randomValues1 = rng_4x32(counter, key);
        doublepointsCrossover(mParentPopulation,
                              mOffspringPopulation,
                              // threadIdx.x, // offspring index
                              PARENTIDX, // offspring index
                              // parent1Idx[threadIdx.x], parent2Idx[threadIdx.x],
                              parent1Idx[PARENTIDX], parent2Idx[PARENTIDX],
                              randomValues1.v[0], randomValues1.v[1]);// ,
                              // randomValues1.v[2], randomValues2.v[3]);
    }
    __syncthreads();
// #endif

// #if 0
    //- mutation
    if (threadIdx.x < WARP_SIZE)
    {
        counter.incr();
        randomValues1 = rng_4x32(counter, key);

#ifdef _DEBUG
        printf("BitFlipMutation: %f,%f,%f,%f\n",
                r123::u01fixedpt<float>(randomValues1.v[0]),
                r123::u01fixedpt<float>(randomValues1.v[1]),
                r123::u01fixedpt<float>(randomValues1.v[2]),
                r123::u01fixedpt<float>(randomValues1.v[3]));
#endif // _DEBUG

        bitFlipMutation(mOffspringPopulation,
                        randomValues1.v[0],
                        randomValues1.v[1],
                        randomValues1.v[2],
                        randomValues1.v[3]);
    }
    __syncthreads();
// #endif
}
#endif

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

    /*
    const unsigned int tmp = point1;
    if (point1 > point2)
    {
        point1 = point2;
        point2 = tmp;
    }
    else if (point1 == point2)
    {
        point2 += 1;
    }
    */
}




inline __device__ void doublepointsCrossover(
        const PopulationData* parent,
        PopulationData* offspring,
        const unsigned int& offspringIdx, // threadIdx.x
        int& parent1Idx,
        int& parent2Idx,
        uint32_t& random1,
        uint32_t& random2)
        // unsigned int& random3,
        // unsigned int& random4)
{
    // 実際の遺伝子長を用いてクロスオーバーポイントを決定する
    uint32_t idx1 = random1 % (gpuEvoPrms.CHROMOSOME_ACTUAL);
    uint32_t idx2 = random2 % (gpuEvoPrms.CHROMOSOME_ACTUAL);
    // std::uint32_t idx1 = random1 % (parent->chromosomeSize);
    // std::uint32_t idx2 = random2 % (parent->chromosomeSize);
    swap(idx1, idx2); // random1 <= random2

    // オフセットを求める際の遺伝子長はPSEUDOのほうになる。
    // 一個体の遺伝子として確保した配列はCHROMOSOME_PSEUDOとなる
    std::uint32_t offset1 = parent1Idx         * gpuEvoPrms.CHROMOSOME_PSEUDO;
    std::uint32_t offset2 = parent2Idx         * gpuEvoPrms.CHROMOSOME_PSEUDO;
    std::uint32_t OFFSET1 = offspringIdx       * gpuEvoPrms.CHROMOSOME_PSEUDO;
    std::uint32_t OFFSET2 = (offspringIdx + 1) * gpuEvoPrms.CHROMOSOME_PSEUDO;

#ifdef _DEBUG
    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
            offspringIdx, parent1Idx, parent2Idx,
            idx1, idx2,
            offset1, offset2,
            OFFSET1, OFFSET2);
#endif // _DEBUG

    // std::uint32_t offset1 = 2 * offspringIdx * gpuEvoPrms.CHROMOSOME;
    // std::uint32_t offset2 = 2 * offspringIdx * gpuEvoPrms.CHROMOSOME + gpuEvoPrms.CHROMOSOME;

    int i = 0;
    if (offspringIdx % 2 == 0) // offspringIdxが偶数の場合
    {
        for (; i < idx1; ++i)
        {
            offspring->population[OFFSET1 + i] = parent->population[offset1 + i];
        }
        for (; i < idx2; ++i)
        {
            offspring->population[OFFSET1 + i] = parent->population[offset2 + i];
        }
        // 確保した遺伝子の最後までコピーをする。実際の遺伝子長を超える部分は0で埋められている
        // はずなので、2個目のクロスオーバーポイントから後ろはすべてコピーしてしまってよい。
        for (; i < gpuEvoPrms.CHROMOSOME_PSEUDO; ++i)
        // for (; i < gpuEvoPrms.CHROMOSOME; ++i)
        {
            offspring->population[OFFSET1 + i] = parent->population[offset1 + i];
        }
    }
    else if (offspringIdx % 2 == 1) // offspringIdxが奇数の場合
    {
        for (; i < idx1; ++i)
        {
            offspring->population[OFFSET1 + i] = parent->population[offset2 + i];
        }
        for (; i < idx2; ++i)
        {
            offspring->population[OFFSET1 + i] = parent->population[offset1 + i];
        }
        // 確保した遺伝子の最後までコピーをする。実際の遺伝子長を超える部分は0で埋められている
        // はずなので、2個目のクロスオーバーポイントから後ろはすべてコピーしてしまってよい。
        for (; i < gpuEvoPrms.CHROMOSOME_PSEUDO; ++i)
        // for (; i < gpuEvoPrms.CHROMOSOME; ++i)
        {
            offspring->population[OFFSET1 + i] = parent->population[offset2 + i];
        }
    }
}

inline __device__ void bitFlipMutation(
        PopulationData* offspring,
        uint32_t& random1, uint32_t& random2, uint32_t& random3, uint32_t& random4)
{
    // 実際の遺伝子長からmutation rateを決めるので、CHROMOSOME_ACTUALとなる
    const float MUTATION_RATE_ADJUSTED
        = gpuEvoPrms.MUTATION_RATE * gpuEvoPrms.CHROMOSOME_ACTUAL / 4.0f;

    // 一方でこちらは配列の全長がわかる必要があるのでCHROMOSOME_PSEUDOとなる
    const std::int32_t TOTALGENELENGTH
        = gpuEvoPrms.CHROMOSOME_PSEUDO * gpuEvoPrms.POPSIZE;

    std::uint32_t popidx  = 0;
    std::uint32_t geneidx = 0;
    std::uint32_t offset  = 0;

    if (r123::u01fixedpt<float>(random1) < MUTATION_RATE_ADJUSTED)
    {
        // mutationさせる親を選択→親の中でmutationさせる。この際に０で埋められた領域は使わない
        popidx  = random1 % offspring->populationSize;
        geneidx = (random1 + random2) % gpuEvoPrms.CHROMOSOME_ACTUAL;
        offset  = popidx * gpuEvoPrms.CHROMOSOME_PSEUDO;
        offspring->population[offset + geneidx] ^= 1;
    }
    if (r123::u01fixedpt<float>(random2) < MUTATION_RATE_ADJUSTED)
    {
        // mutationさせる親を選択→親の中でmutationさせる。この際に０で埋められた領域は使わない
        popidx  = random2 % offspring->populationSize;
        geneidx = (random2 + random3) % gpuEvoPrms.CHROMOSOME_ACTUAL;
        offset  = popidx * gpuEvoPrms.CHROMOSOME_PSEUDO;
        offspring->population[offset + geneidx] ^= 1;
    }
    if (r123::u01fixedpt<float>(random3) < MUTATION_RATE_ADJUSTED)
    {
        // mutationさせる親を選択→親の中でmutationさせる。この際に０で埋められた領域は使わない
        popidx  = random3 % offspring->populationSize;
        geneidx = (random3 + random4) % gpuEvoPrms.CHROMOSOME_ACTUAL;
        offset  = popidx * gpuEvoPrms.CHROMOSOME_PSEUDO;
        offspring->population[offset + geneidx] ^= 1;
    }
    if (r123::u01fixedpt<float>(random4) < MUTATION_RATE_ADJUSTED)
    {
        // mutationさせる親を選択→親の中でmutationさせる。この際に０で埋められた領域は使わない
        popidx  = random4 % offspring->populationSize;
        geneidx = (random4 + random1) % gpuEvoPrms.CHROMOSOME_ACTUAL;
        offset  = popidx * gpuEvoPrms.CHROMOSOME_PSEUDO;
        offspring->population[offset + geneidx] ^= 1;
    }
}
