#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
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
    std::cout << "# Running on host: " << hostname << std::endl;
    std::cout << "# Device " << mDeviceIdx << ": " << prop.name << std::endl;
    std::cout << "# Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "# Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "# Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "# Number of SMs: " << prop.multiProcessorCount << std::endl;


    // Create populations on CPU
    //- ここで確保する配列サイズはPSEUDOの方と思われる
    mHostTempPopulation
        = new CPUPopulation(
                prms->getPopsize(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    mHostParentPopulation
        = new CPUPopulation(
                prms->getPopsize(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    mHostOffspringPopulation
        = new CPUPopulation(
                prms->getPopsize(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

#ifdef _DEBUG
    for (int i = 0; i < prms->getPopsize(); ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < prms->getChromosomePseudo(); ++j)
        // for (int j = 0; j < prms->getChromosome(); ++j)
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
    mDevTempPopulation
        = new GPUPopulation(
                prms->getPopsize(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    mDevParentPopulation
        = new GPUPopulation(
                prms->getPopsize(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    mDevOffspringPopulation
        = new GPUPopulation(
                prms->getPopsize(),
                prms->getChromosomePseudo(),
                prms->getNumOfElite());

    // Copy population from CPU to GPU
    mDevTempPopulation->copyToDevice(mHostTempPopulation->getDeviceData());
    mDevParentPopulation->copyToDevice(mHostParentPopulation->getDeviceData());
    mDevOffspringPopulation->copyToDevice(mHostOffspringPopulation->getDeviceData());

    mMultiprocessorCount = prop.multiProcessorCount;
    // Initialize Random seed
    initRandomSeed();
}; // end of GPUEvolution


/**
 * Destructor of the class
 */
GPUEvolution::~GPUEvolution()
{
    /**
    if (mHostTempPopulation != nullptr)
    {
        delete mHostTempPopulation;
    }
    if (mHostParentPopulation != nullptr)
    {
        delete mHostParentPopulation;
    }
    if (mHostOffspringPopulation != nullptr)
    {
        delete mHostOffspringPopulation;
    }
    */
    delete mHostTempPopulation;
    delete mHostParentPopulation;
    delete mHostOffspringPopulation;

    /**
    if (mDevTempPopulation != nullptr)
    {
        delete mDevTempPopulation;
    }
    if (mDevParentPopulation != nullptr)
    {
        delete mDevParentPopulation;
    }
    if (mDevOffspringPopulation != nullptr)
    {
        delete mDevOffspringPopulation;
    }
    */
    delete mDevTempPopulation;
    // delete mDevParentPopulation;
    // delete mDevOffspringPopulation;
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
    // showPopulation(prms, generation, 1);

    // 実行時間測定開始
    cudaEventRecord(start, 0);

    // printf("### EvoCycle\n");
    for (generation = 0; generation < prms->getNumOfGenerations(); ++generation)
    {
        // std::cout << "### Generation" << generation << std::endl;
        runEvolutionCycle(prms);
        // showPopulation(prms, generation, 0);
    }
    // std::cout << "End of EvoCycle" << std::endl;
    showPopulation(prms, generation, 2);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    showSummary(*prms, elapsed_time);
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
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    //- イニシャライズでは本当のサイズの範囲のみだけが対象になるので、
    //- CHROMOSOME_ACTUALとしておく
    // threads.x = 1;
    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    cudaGenerateFirstPopulationKernel
        <<<blocks, threads>>>
        (mDevParentPopulation->getDeviceData(), getRandomSeed());

    checkAndReportCudaError(__FILE__, __LINE__);
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
    uint32_t *h_selectedParents1 = new uint32_t[prms->getPopsize()];
    uint32_t *h_selectedParents2 = new uint32_t[prms->getPopsize()];

    // d_selectedParents1, 2のメモリを確保する
    uint32_t *d_selectedParents1, *d_selectedParents2;
    cudaMalloc(&d_selectedParents1, prms->getPopsize() * sizeof(uint32_t));
    cudaMalloc(&d_selectedParents2, prms->getPopsize() * sizeof(uint32_t));

    //- Fitness評価 ---------------------------------------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    //- evaluation では遺伝子配列に対してカスケーディングを用いるためPSEUDOを用いる
    threads.x = prms->getChromosomeActual();
    // threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    evaluation
        <<< blocks, threads, prms->getChromosomeActual() * sizeof(int)>>>
        // <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);
    mDevTempPopulation   = mDevParentPopulation;

    // セレクション ---------------------------------------
    blocks.x = 32;
    blocks.y = 1;
    blocks.z = 1;

    threads.x = prms->getPopsize() / blocks.x;
    threads.y = 1;
    threads.z = 1;

    cudaKernelSelection<<<blocks, threads>>>(
            mDevParentPopulation->getDeviceData(),
            d_selectedParents1,
            d_selectedParents2,
            getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);

#if 0
    // selectedParents1, 2の中身を表示してみる -------------
    printf("selectedParents1, 2\n");
    cudaMemcpy(h_selectedParents1, d_selectedParents1,
            prms->getPopsize() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_selectedParents2, d_selectedParents2, 
            prms->getPopsize() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < prms->getPopsize(); ++i)
    {
        printf("%4d, %4d\n", h_selectedParents1[i], h_selectedParents2[i]);
    }
#endif

    // クロスオーバー -------------------------------------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    // threads.x = prms->getChromosomePseudo();
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
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    cudaKernelMutation<<<blocks, threads>>>(
            mDevOffspringPopulation->getDeviceData(),
            getRandomSeed());

    checkAndReportCudaError(__FILE__, __LINE__);

    //- 疑似エリート保存戦略 -------------------------------
    blocks.x  = prms->getNumOfElite();
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsize() / prms->getNumOfElite();
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

    threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    replaceWithElites
        <<<blocks, threads>>>
        // <<<blocks, threads, sizeof(uint32_t) * prms->getNumOfElite()>>>
        (mDevParentPopulation->getDeviceData(),
         mDevOffspringPopulation->getDeviceData());

    // checkAndReportCudaError(__FILE__, __LINE__);

    //- 親と子の入れ替え -----------------------------------
    // swap population between parents and offsprings
    // temp = mDevParentPopulation;
    mDevParentPopulation = mDevOffspringPopulation;
    // mDevOffspringPopulation = temp;
}

void GPUEvolution::showSummary(const Parameters& prms, const float& elapsedTime)
{
    dim3 blocks;
    dim3 threads;

    //- Fitness評価 ---------------------------------------
    blocks.x = prms.getPopsize();
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
                mHostOffspringPopulation->getDeviceData()->fitness + prms.getPopsize());

    uint32_t* minElementPtr
        = std::min_element(mHostOffspringPopulation->getDeviceData()->fitness,
                mHostOffspringPopulation->getDeviceData()->fitness + prms.getPopsize());

    double fitnessSum
        = std::accumulate(mHostOffspringPopulation->getDeviceData()->fitness,
                mHostOffspringPopulation->getDeviceData()->fitness + prms.getPopsize(),
                0.0) / prms.getPopsize();

    std::cout 
        << prms.getPopsize()
        << "," << prms.getChromosomeActual()
        << "," << elapsedTime
        << "," << *maxElementPtr
        << "," << *minElementPtr
        << "," << fitnessSum
        << std::endl;

    return;
}


void GPUEvolution::showPopulation(Parameters* prms, uint16_t generation, uint16_t type)
{
    dim3 blocks;
    dim3 threads;

    //- Fitness評価 ---------------------------------------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    //- evaluation では遺伝子配列に対してカスケーディングを用いるためPSEUDOを用いる
    // printf("prms->getChromosomeActual():%d\n", prms->getChromosomeActual());
    // printf("prms->getChromosomePseudo():%d\n", prms->getChromosomePseudo());
    threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    if (type == 0) {
        evaluation
            <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
            (mDevTempPopulation->getDeviceData());
        mDevTempPopulation->copyFromDevice(mHostTempPopulation->getDeviceData());
    } else if (type == 1) {
        evaluation
            <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
            (mDevParentPopulation->getDeviceData());
        mDevParentPopulation->copyFromDevice(mHostParentPopulation->getDeviceData());
    } else if (type == 2) {
        evaluation
            <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
            (mDevOffspringPopulation->getDeviceData());
        mDevOffspringPopulation->copyFromDevice(mHostOffspringPopulation->getDeviceData());
    }
    checkAndReportCudaError(__FILE__, __LINE__);

    // Actualが見たい時もあるだろうし、Pseudoが見たい時もあると思う。
    // とりあえず最初はPSEUDOから確認しよう
    // int csize = prms->getChromosomePseudo();
    int csize = prms->getChromosomeActual();
    int psize = prms->getPopsize();
    int esize = prms->getNumOfElite();


    // printf("------------ Population: %d: ------------ \n", generation);
    if (generation == 0)
    {
        printf("Generation,Mean,Max,Min\n");
    }
    printf("%d,%f,%d,%d\n", generation, 
                            mHostParentPopulation->getMean(),
                            mHostParentPopulation->getMax(),
                            mHostParentPopulation->getMin());

    printf("============ Generation: %d ============ \n", generation);
    printf("------------ Parent:%d ------------ \n", generation);
    int tempindex = 0;
    int tempvalue = 0;
    for (int k = 0; k < esize; ++k)
    {
        if (type == 0) {
            tempindex = mHostTempPopulation->getDeviceData()->elitesIdx[k];
            tempvalue = mHostTempPopulation->getDeviceData()->fitness[tempindex];
        } else if (type == 1) {
            tempindex = mHostParentPopulation->getDeviceData()->elitesIdx[k];
            tempvalue = mHostParentPopulation->getDeviceData()->fitness[tempindex];
        } else if (type == 2) {
            tempindex = mHostOffspringPopulation->getDeviceData()->elitesIdx[k];
            tempvalue = mHostOffspringPopulation->getDeviceData()->fitness[tempindex];
        }
        printf("elite%d : %d , %d\n", k, tempindex, tempvalue);
    }
    printf("\n");

    for (int i = 0; i < psize; ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < csize; ++j)
        {
            if (type == 0) {
                printf("%d", mHostTempPopulation->getDeviceData()->population[i * csize + j]);
            } else if (type == 1) {
                printf("%d", mHostParentPopulation->getDeviceData()->population[i * csize + j]);
            } else if (type == 2) {
                printf("%d", mHostOffspringPopulation->getDeviceData()->population[i * csize + j]);
            }
        }
        if (type == 0) {
            printf(":%d\n", mHostTempPopulation->getDeviceData()->fitness[i]);
        } else if (type == 1) {
            printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
        } else if (type == 2) {
            printf(":%d\n", mHostOffspringPopulation->getDeviceData()->fitness[i]);
        }
    }
}


