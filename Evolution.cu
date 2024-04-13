#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Evolution.h"
#include "CUDAKernels.h"
#include "Parameters.h"

/**
 * Constructor of the class
 */
GPUEvolution::GPUEvolution()
    : mRandomSeed(0), mDeviceIdx(0)
{
}

GPUEvolution::GPUEvolution(Parameters* prms)
    : mRandomSeed(0),
      mDeviceIdx(0)
{
    //- Get parameters of the device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, mDeviceIdx);

    char hostname[1024];
    gethostname(hostname, 1024);

    // Create populations on CPU
    //- ここで確保する配列サイズはPSEUDOの方と思われる
    // 遺伝子配列はevaluationでカスケーディングを用いるため、
    // 2の冪乗のサイズにしておく必要がある

    mHostParentPopulation
        = new CPUPopulation(
                prms->getPopsizeActual(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    mHostOffspringPopulation
        = new CPUPopulation(
                prms->getPopsizeActual(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

#ifdef _DEBUG
    for (int i = 0; i < prms->getPopsizeActual(); ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < prms->getChromosomePseudo(); ++j)
        {
            printf("%d",
                    mHostParentPopulation->
                    getDeviceData()->
                    population[i * prms->getChromosomePseudo() + j]);

                    prms->getChromosomePseudo() + j;
        }
        printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
    }
#endif // _DEBUG

    // Create populations on GPU
    //- ここで確保する配列サイズもPSEUDOの方と思われる
    mDevParentPopulation
        = new GPUPopulation(
                prms->getPopsizeActual(),
                // prms->getChromosomeActual(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    mDevOffspringPopulation
        = new GPUPopulation(
                prms->getPopsizeActual(),
                // prms->getChromosomeActual(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    // Copy population from CPU to GPU
    // mDevParentPopulation->copyToDevice(mHostParentPopulation->getDeviceData());
    // mDevOffspringPopulation->copyToDevice(mHostOffspringPopulation->getDeviceData());

    mMultiprocessorCount = prop.multiProcessorCount;
    // Initialize Random seed
    initRandomSeed();
}; // end of GPUEvolution


/**
 * Destructor of the class
 */
GPUEvolution::~GPUEvolution()
{
    delete mHostParentPopulation;
    delete mHostOffspringPopulation;

    delete mDevParentPopulation;
    delete mDevOffspringPopulation;
} // end of Destructor


/**
 * Run Evolution
 */
void GPUEvolution::run(Parameters* prms)
{
    // 実行時間計測用
    float elapsed_time = 0.0f;
    // イベントを取り扱う変数
    cudaEvent_t start, end;
    // イベントのクリエイト
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    uint16_t generation = 0;
    // printf("### Initialize\n");
    initialize(prms);

    // 実行時間測定開始
    cudaEventRecord(start, 0);

    // printf("### EvoCycle\n");
    for (generation = 0; generation < prms->getNumOfGenerations(); ++generation)
    {

        // printf("### Generation: %d\n", generation);
        runEvolutionCycle(prms);
        // showSummary(*prms, elapsed_time, generation);

#ifdef _SHOWPOPULATION
        // showSummary(*prms, elapsed_time, generation);
#endif // SHOWPOPULATION
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
#ifdef _SHOWRESULT
    showSummary(*prms, elapsed_time, generation);
#endif // SHOWRESULT
}


void GPUEvolution::initRandomSeed()
{
    struct timeval tp1;
    gettimeofday(&tp1, nullptr);
    mRandomSeed = (tp1.tv_sec / (mDeviceIdx + 1)) * tp1.tv_usec;
#ifdef _DEBUG
    printf("mRandomSeed: %d\n", mRandomSeed);
#endif // _DEBUG
}

/**
 * Initialization of the GA
 */
void GPUEvolution::initialize(Parameters* prms)
{
    copyToDevice(prms->getEvoPrms());

    dim3 blocks;
    dim3 threads;


    //- 初期集団生成 ---------
    //- blocks.x  = prms->getPopsize() / 2;
    //- 1つのスレッドで1つの個体を初期化させるので、blocksには個体の数をそのまま登録する
    blocks.x  = prms->getPopsizeActual();
    blocks.y  = 1;
    blocks.z  = 1;

    //- イニシャライズでは本当のサイズの範囲のみだけが対象になるので、
    //- CHROMOSOME_ACTUALとしておく
    // threads.x = 1;
    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    cudaKernelGenerateFirstPopulation
        <<<blocks, threads>>>
        (mDevParentPopulation->getDeviceData(), getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);

#ifdef _ELITISM
    // printf("### Elitism_Init\n");
    cudaKernelGenerateFirstPopulation
        <<<blocks, threads>>>
        (mDevOffspringPopulation->getDeviceData(), getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);
#endif // _ELITISM

} // end of initialize


/**
 * Run evolutionary cycle for defined number of generations
 */
void GPUEvolution::runEvolutionCycle(Parameters* prms)
{
    dim3 blocks;
    dim3 threads;
    GPUPopulation* temp;

    // h_selectedParents1, 2のメモリを確保する
    uint32_t *h_selectedParents1 = new uint32_t[prms->getPopsizeActual()];
    uint32_t *h_selectedParents2 = new uint32_t[prms->getPopsizeActual()];

    // d_selectedParents1, 2のメモリを確保する
    uint32_t *d_selectedParents1, *d_selectedParents2;
    cudaMalloc(&d_selectedParents1, prms->getPopsizeActual() * sizeof(uint32_t));
    cudaMalloc(&d_selectedParents2, prms->getPopsizeActual() * sizeof(uint32_t));

    //- Fitness評価 ---------------------------------------
    blocks.x  = prms->getPopsizeActual();
    // blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    //- evaluation では遺伝子配列に対してカスケーディングを用いるためPSEUDOを用いる
    threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    // evaluationsingle
    //     <<< blocks, threads>>>
    //     (mDevParentPopulation->getDeviceData());
    evaluation
        <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);
    // mDevTempPopulation   = mDevParentPopulation;

    // セレクション ---------------------------------------
    // セレクションのブロックサイズとスレッドサイズは経験的に決める
    blocks.x = 32;
    blocks.y = 1;
    blocks.z = 1;

    threads.x = prms->getPopsizeActual() / blocks.x;
    // threads.x = prms->getPopsize() / blocks.x;
    threads.y = 1;
    threads.z = 1;

    cudaKernelSelection<<<blocks, threads>>>(
            mDevParentPopulation->getDeviceData(),
            d_selectedParents1,
            d_selectedParents2,
            getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);

    // クロスオーバー -------------------------------------
    blocks.x  = prms->getPopsizeActual();
    // blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    // クロスオーバーに使う遺伝子長はActualの方
    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    cudaKernelCrossover<<<blocks, threads>>>(
            mDevParentPopulation->getDeviceData(),
            mDevOffspringPopulation->getDeviceData(),
            d_selectedParents1,
            d_selectedParents2,
            getRandomSeed());

    checkAndReportCudaError(__FILE__, __LINE__);

    // ミューテーション -----------------------------------
    blocks.x  = prms->getPopsizeActual();
    // blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    cudaKernelMutation<<<blocks, threads>>>(
            mDevOffspringPopulation->getDeviceData(),
            getRandomSeed());

    checkAndReportCudaError(__FILE__, __LINE__);

#ifdef _SHOWPOPULATION
    printf("------------------------------------------------------------\n");
    printf("### showPopulation1 before elitism (with evaluation)       -\n");
    printf("------------------------------------------------------------\n");
    // showPopulation(prms);
    // showPopulationWithoutEvaluation(prms);
#endif // _SHOWPOPULATION

#ifdef _ELITISM
    //- エリート保存戦略 -----------------------------------
    // printf("### Elitism_elite\n");

    mDevParentPopulation->elitism(prms);
    checkAndReportCudaError(__FILE__, __LINE__);
#else
    // printf("### Elitism_pseudo_elite\n");
    //- 疑似エリート保存戦略 -------------------------------
    blocks.x  = prms->getNumOfElite();
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsizePseudo() / prms->getNumOfElite();
    // threads.x = prms->getPopsizeActual() / prms->getNumOfElite();
    threads.y = 1;
    threads.z = 1;

    pseudo_elitism
        <<<blocks, threads, threads.x * 2 * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);

    //- Elitesの差し込み -----------------------------------
    blocks.x  = prms->getNumOfElite();
    blocks.y  = 1;
    blocks.z  = 1;

    // threads.x = prms->getChromosomePseudo();
    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    replaceWithPseudoElites
        <<<blocks, threads>>>
        (mDevParentPopulation->getDeviceData(),
         mDevOffspringPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);
#endif // _ELITISM

#ifdef _SHOWPOPULATION
    printf("------------------------------------------------------------\n");
    printf("### showPopulation right after elitism (w/o  evaluation)   -\n");
    printf("------------------------------------------------------------\n");
    showPopulation(prms);
#endif // _SHOWPOPULATION

    //- Elitesの差し込み -----------------------------------
    blocks.x  = prms->getNumOfElite();
    blocks.y  = 1;
    blocks.z  = 1;

    // threads.x = prms->getChromosomePseudo();
    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    replaceWithElites
        <<<blocks, threads>>>
        (mDevParentPopulation->getDeviceData(),
         mDevOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    //- 現世代の子を次世代の親とする -----------------------------------
    temp = mDevParentPopulation;
    mDevParentPopulation = mDevOffspringPopulation;
    mDevOffspringPopulation = temp;
}

void GPUEvolution::showSummary(const Parameters& prms, const float& elapsedTime, const int& generation)
{
    dim3 blocks;
    dim3 threads;

    //- Fitness評価 ---------------------------------------
    blocks.x = prms.getPopsizeActual();
    blocks.y = 1;
    blocks.z = 1;

    //- evaluation では遺伝子配列に対してカスケーディングを用いるためPSEUDOを用いる
    threads.x = prms.getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    evaluation
        <<<blocks, threads, prms.getChromosomePseudo() * sizeof(int)>>>
        (mDevOffspringPopulation->getDeviceData());
    mDevOffspringPopulation->copyFromDevice(mHostOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    uint32_t* maxElementPtr
        = std::max_element(mHostOffspringPopulation->getDeviceData()->fitness,
                mHostOffspringPopulation->getDeviceData()->fitness + prms.getPopsizeActual());

    uint32_t* minElementPtr
        = std::min_element(mHostOffspringPopulation->getDeviceData()->fitness,
                mHostOffspringPopulation->getDeviceData()->fitness + prms.getPopsizeActual());

    double fitnessSum
        = std::accumulate(mHostOffspringPopulation->getDeviceData()->fitness,
                mHostOffspringPopulation->getDeviceData()->fitness + prms.getPopsizeActual(),
                0.0) / prms.getPopsizeActual();

    std::cout 
        << generation
        << "," << prms.getPopsizeActual()
        << "," << prms.getChromosomeActual()
        << "," << elapsedTime
        << "," << *maxElementPtr
        << "," << *minElementPtr
        << "," << fitnessSum
        << std::endl;

    return;
}

void GPUEvolution::showPopulationWithoutEvaluation(Parameters* prms)
{
    const int acsize = prms->getChromosomeActual();
    const int pcsize = prms->getChromosomePseudo();
    const int psize  = prms->getPopsizeActual();
    const int esize  = prms->getNumOfElite();

    mDevParentPopulation->copyFromDevice(mHostParentPopulation->getDeviceData());
    printf("=== Parent population ===\n");
    for (int i = 0; i < psize; ++i) // Population size
    {
        printf("%3d,", i);
        for (int j = 0; j < acsize; ++j) // Actual chromosome size
        {
            printf("%d", mHostParentPopulation->getDeviceData()->population[i * pcsize + j]);
        }
        printf(":%d",  mHostParentPopulation->getDeviceData()->fitness_index[i]);
        printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
        // printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness_sorted[i]);
    }

#ifdef _ELITISM
    printf("=== Parent population sorted ===\n");
    for (int i = 0; i < psize; ++i)
    {
        printf("fitness_index_sorted:%d ",
                mHostParentPopulation->getDeviceData()->fitness_index_sorted[i]);
        printf("fitness_sorted:%d\n", mHostParentPopulation->getDeviceData()->fitness_sorted[i]);
    }
#endif // _ELITISM

    printf("=== Offspring population ===\n");
    for (int i = 0; i < psize; ++i) // Population size
    {
        printf("%3d,", i);
        for (int j = 0; j < acsize; ++j) // Actual chromosome size
        {
            printf("%d", mHostOffspringPopulation->getDeviceData()->population[i * pcsize + j]);
        }
        printf(":%d", mHostOffspringPopulation->getDeviceData()->fitness_index[i]);
        printf(":%d\n", mHostOffspringPopulation->getDeviceData()->fitness[i]);
    }
}

void GPUEvolution::showPopulation(Parameters* prms)
// void GPUEvolution::showPopulation(Parameters* prms, uint16_t generation, uint16_t type)
{
    const int acsize = prms->getChromosomeActual();
    const int pcsize = prms->getChromosomePseudo();
    const int psize  = prms->getPopsizeActual();
    const int esize  = prms->getNumOfElite();

    dim3 blocks;
    dim3 threads;

    //- Fitness評価 ---------------------------------------
    blocks.x  = prms->getPopsizeActual();
    blocks.y  = 1;
    blocks.z  = 1;

    //- evaluation では遺伝子配列に対してカスケーディングを用いるためPSEUDOを用いる
    threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;
    
    // printf("### evaluationSingle ------------------------------\n");
    // evaluationsingle
    //     <<< blocks, threads>>>
    //     (mDevParentPopulation->getDeviceData());
    // mDevParentPopulation->copyFromDevice(mHostParentPopulation->getDeviceData());
    // checkAndReportCudaError(__FILE__, __LINE__);

    // 通常のevaluationを実行する場合には以下のコメントアウトを外す
    printf("### evaluation ------------------------------\n");
    evaluation
        <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());
    
    mDevParentPopulation->copyFromDevice(mHostParentPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);
    // // Print population in parent population
    // printf("=== Parent population ===\n");
    // for (int i = 0; i < psize; ++i) // Population size
    // {
    //     printf("%3d,", i);
    //     for (int j = 0; j < acsize; ++j) // Actual chromosome size
    //     {
    //         printf("%d", mHostParentPopulation->getDeviceData()->population[i * pcsize + j]);
    //     }
    //     printf(":%d",  mHostParentPopulation->getDeviceData()->fitness_index[i]);
    //     printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
    //     // printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness_sorted[i]);
    // }

#ifdef _ELITISM
    printf("=== Parent population sorted ===\n");
    for (int i = 0; i < psize; ++i)
    {
        printf("fitness_index_sorted:%d ",
                mHostParentPopulation->getDeviceData()->fitness_index_sorted[i]);
        printf("fitness_sorted:%d\n", mHostParentPopulation->getDeviceData()->fitness_sorted[i]);
    }
#endif // _ELITISM


    // Print elites in parent population
    int tempindex = 0;
    int tempvalue = 0;
    printf("\nElites in parent population\n");
    for (int k = 0; k < esize; ++k)
    {
        tempindex = mHostParentPopulation->getDeviceData()->fitness_index_sorted[psize - k - 1];
        tempvalue = mHostParentPopulation->getDeviceData()->fitness_sorted[psize - k - 1];
        // tempindex = mHostParentPopulation->getDeviceData()->elitesIdx[k];
        // tempvalue = mHostParentPopulation->getDeviceData()->fitness[tempindex];
        printf("elite%d : %d , %d\n", k, tempindex, tempvalue);
    }
    printf("\n");

    // evaluationsingle
    //     <<< blocks, threads>>>
    //     (mDevOffspringPopulation->getDeviceData());
    // mDevOffspringPopulation->copyFromDevice(mHostOffspringPopulation->getDeviceData());
    // checkAndReportCudaError(__FILE__, __LINE__);

    evaluation
        <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevOffspringPopulation->getDeviceData());
    mDevOffspringPopulation->copyFromDevice(mHostOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);
    // Print population in offspring population
    printf("=== Offspring population ===\n");
    for (int i = 0; i < psize; ++i) // Population size
    {
        printf("%3d,", i);
        for (int j = 0; j < acsize; ++j) // Actual chromosome size
        {
            printf("%d", mHostOffspringPopulation->getDeviceData()->population[i * pcsize + j]);
        }
        printf(":%d", mHostOffspringPopulation->getDeviceData()->fitness_index[i]);
        printf(":%d\n", mHostOffspringPopulation->getDeviceData()->fitness[i]);
    }
    printf("=== Offspring population sorted ===\n");
    for (int i = 0; i < psize; ++i)
    {
        printf("index_sorted:%d ",  mHostOffspringPopulation->getDeviceData()->fitness_index_sorted[i]);
        printf("fitness_sorted:%d\n", mHostOffspringPopulation->getDeviceData()->fitness_sorted[i]);
    }
}
