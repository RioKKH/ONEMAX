#include <stdio.h>
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
    // printf("constructor\n");
    //- Select device
    // cudaSetDevice(mDeviceIdx);
    // checkAndReportCudaError(__FILE__, __LINE__);

    //- Get parameters of the device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, mDeviceIdx);

    // Create populations on CPU
    //- ここで確保する配列サイズはPSEUDOの方と思われる
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

    // mHostParentPopulation    = new CPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());
    // mHostOffspringPopulation = new CPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());

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

            // printf("%d", mHostParentPopulation->getDeviceData()->population[i * prms->getChromosome() + j]);
                    prms->getChromosomePseudo() + j]);
        }
        printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
    }
#endif // _DEBUG

    // Create populations on GPU
    //- ここで確保する配列サイズもPSEUDOの方と思われる
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

    // mDevParentPopulation    = new GPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());
    // mDevOffspringPopulation = new GPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());

    // Copy population from CPU to GPU
    mDevParentPopulation->copyToDevice(mHostParentPopulation->getDeviceData());
    mDevOffspringPopulation->copyToDevice(mHostOffspringPopulation->getDeviceData());

    mMultiprocessorCount = prop.multiProcessorCount;
    // mParams.setNumberOfDeviceSMs(prop.multiProcessorCount);

    // Create statistics
    // mStatistics = new GPUStatistics();

    // Initialize Random seed
    initRandomSeed();
    // printf("end of constructor\n");
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
    // showPopulation(prms, generation);

    // 実行時間測定開始
    cudaEventRecord(start, 0);
    // printf("### EvoCycle\n");
    for (generation = 0; generation < prms->getNumOfGenerations(); ++generation)
    {
        // printf("### Number of Generations : %d ###\n", generation);
        // printf("### Generations: %d\n", generation);
        runEvolutionCycle(prms);
        showPopulation(prms, generation);
    }
    printf("End of EvoCycle\n");
    // showPopulation(prms, generation);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    uint32_t popsize = static_cast<uint32_t>(prms->getPopsize());
    //- ここは結果を表示したいところなので、
    //  CHROMOSOME_ACTUALを表示するべきところと思われる
    std::cout
        << popsize << ","
        << prms->getChromosomeActual()
        << "," << elapsed_time << std::endl;
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
    // blocks.x  = prms->getPopsize() / 2;
    blocks.y  = 1; blocks.z  = 1;

    //- イニシャライズでは本当のサイズの範囲のみだけが対象になるので、
    //- CHROMOSOME_ACTUALとしておく
    threads.x = 1;
    // threads.x = prms->getChromosomeActual();
    threads.y = 1;
    threads.z = 1;

    cudaGenerateFirstPopulationKernel
        <<<blocks, threads>>>
        (mDevParentPopulation->getDeviceData(), getRandomSeed());

    checkAndReportCudaError(__FILE__, __LINE__);


    //- Fitness評価 ----------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    //- Fitness評価はカスケーディングを用いるため、
    //- 確保した配列全サイズで実施する必要がある。
    //- 従ってここではChromosomePseudoを用いる
    threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    evaluation
        <<<blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);

    evaluation
        <<<blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevOffspringPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);


    //- 疑似エリート保存戦略 -------------------------------
    blocks.x  = prms->getNumOfElite();
    // blocks.x  = prms->getNumOfElite() * 2;
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsize() / prms->getNumOfElite();
    threads.y = 1;
    threads.z = 1;

#ifdef _DEBUG
    printf("blocks.x:%d, threads.x:%d, offset:%d, shared_memory_size:%d\n",
            blocks.x, threads.x, blocks.x * threads.x / 2, prms->getPopsize() * 2);
#endif // _DEBUG

    pseudo_elitism
        <<<blocks, threads, threads.x * 2 * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);

    pseudo_elitism
        <<<blocks, threads, threads.x * 2 * sizeof(int)>>>
        (mDevOffspringPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);


} // end of initialize


/**
 * Run evolutionary cycle for defined number of generations
 */
void GPUEvolution::runEvolutionCycle(Parameters* prms)
{
    dim3 blocks;
    dim3 threads;

    uint32_t *selectedParents1;
    uint32_t *selectedParents2;
    GPUPopulation *temp;

    // selectedParents1, 2のメモリを確保する
    cudaMalloc(&selectedParents1, prms->getPopsize() * sizeof(int));
    cudaMalloc(&selectedParents2, prms->getPopsize() * sizeof(int));

    //- Selection, Crossover, and Mutation -----------------
    int CHR_PER_BLOCK = (prms->getPopsize() % WARP_SIZE == 0)
                         ? prms->getPopsize() / WARP_SIZE
                         : prms->getPopsize() / WARP_SIZE + 1;

    // blocks.x = CHR_PER_BLOCK;
    blocks.x = 32;
    blocks.y = 1;
    blocks.z = 1;

    // threads.x = (prms->getPopsize() > WARP_SIZE) ? WARP_SIZE : prms->getPopsize();
    threads.x = 32;
    // threads.x = (prms->getPopsize() > WARP_SIZE) ? WARP_SIZE : prms->getPopsize() / 2;
    threads.y = 1;
    threads.z = 1;

    // /* 共有メモリを固定サイズにして毎回確保するのをやめてみる
    int shared_memory_size =   prms->getPopsize()        * sizeof(int)
                             + prms->getPopsize()        * sizeof(int)
                             + prms->getTournamentSize() * sizeof(int);
    // */


#ifdef _DEBUG
    printf("Start of cudaGeneticManipulationKernel\n");
    printf("GA: blocks: %d, threads: %d\n", blocks.x, threads.x);
#endif // _DEBUG
    // /* 共有メモリを可変サイズにする
    // cudaGeneticManipulationKernel<<<blocks, threads, shared_memory_size>>> 
    // /* 共有メモリを固定サイズにする
    // cudaGeneticManipulationKernel<<<blocks, threads>>> 
    //                              (mDevParentPopulation->getDeviceData(),
    //                               mDevOffspringPopulation->getDeviceData(),
    //                               getRandomSeed());

    // セレクション ---------------------------------------
    blocks.x = 32; blocks.y = 1; blocks.z = 1;
    threads.x = prms->getPopsize() / blocks.x; threads.y = 1; threads.z = 1;
    cudaKernelSelection<<<blocks, threads>>>(
            mDevParentPopulation->getDeviceData(),
            selectedParents1,
            selectedParents2,
            getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);

    // クロスオーバー -------------------------------------
    blocks.x = prms->getPopsize(); blocks.y = 1; blocks.z = 1;
    threads.x = prms->getChromosomeActual(); threads.y = 1; threads.z = 1;
    cudaKernelCrossover<<<blocks, threads>>>(
            mDevParentPopulation->getDeviceData(),
            mDevOffspringPopulation->getDeviceData(),
            selectedParents1,
            selectedParents2,
            getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);

    // ミューテーション -----------------------------------
    blocks.x = prms->getPopsize(); blocks.y = 1; blocks.z = 1;
    threads.x = prms->getChromosomeActual(); threads.y = 1; threads.z = 1;
    cudaKernelMutation<<<blocks, threads>>>(
            mDevOffspringPopulation->getDeviceData(),
            getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);


#ifdef _DEBUG
    printf("End of cudaGeneticManipulationKernel\n");
#endif // _DEBUG


    //- Fitness評価 ---------------------------------------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    //- evaluation では遺伝子配列に対してカスケーディングを用いるためPSEUDOを用いる
    threads.x = prms->getChromosomePseudo();
    threads.y = 1;
    threads.z = 1;

    evaluation
        <<< blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);

    evaluation
        <<<blocks, threads, prms->getChromosomePseudo() * sizeof(int)>>>
        (mDevOffspringPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);


    //- 疑似エリート保存戦略 -------------------------------
    blocks.x  = prms->getNumOfElite();
    // blocks.x  = prms->getNumOfElite() * 2;
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsize() / prms->getNumOfElite();
    threads.y = 1;
    threads.z = 1;
#ifdef _DEBUG
    printf("blocks.x:%d, threads.x:%d, offset:%d, shared_memory_size:%d\n",
            blocks.x, threads.x, blocks.x * threads.x / 2, prms->getPopsize() * 2);
#endif // _DEBUG

    pseudo_elitism
        <<<blocks, threads, threads.x * 2 * sizeof(int)>>>
        (mDevParentPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);

    pseudo_elitism
        <<<blocks, threads, threads.x * 2 * sizeof(int)>>>
        (mDevOffspringPopulation->getDeviceData());

    checkAndReportCudaError(__FILE__, __LINE__);


    //- 親と子の入れ替え & Elitesの差し込み --------------------------------------------------------
#ifdef _DEBUG
    printf("Copy population from offspring to parent, then insert elites in it.\n");
#endif // _DEBUG
    blocks.x = 1; // gridDim.x
    // blocks.x = CHR_PER_BLOCK; // gridDim.x
    blocks.y = 1;
    blocks.z = 1;

    // threads.x = 1; // blockDim.x
    threads.x = prms->getPopsize();
    // threads.x = 1;
    // threads.x = prms->getPopsize() / CHR_PER_BLOCK; // blockDim.x
    threads.y = 1;
    threads.z = 1;

    // swap population between parents and offsprings
    temp = mDevParentPopulation;
    mDevParentPopulation = mDevOffspringPopulation;
    mDevOffspringPopulation = temp;

    blocks.x  = prms->getNumOfElite();
    // blocks.x  = prms->getNumOfElite() * 2;
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsize() / prms->getNumOfElite();
    threads.y = 1;
    threads.z = 1;

#if 1
    replaceWithElites
        <<<blocks, threads>>>
        (mDevParentPopulation->getDeviceData(),
         mDevOffspringPopulation->getDeviceData());
#endif

    // swapPopulation<<<blocks, threads>>>(mDevParentPopulation->getDeviceData(),
    //                                    mDevOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);
}


void GPUEvolution::showPopulation(Parameters* prms, uint16_t generation)
{
    // Actualが見たい時もあるだろうし、Pseudoが見たい時もあると思う。
    // とりあえず最初はPSEUDOから確認しよう
    int csize = prms->getChromosomePseudo();
    // int csize = prms->getChromosomeActual();
    int psize = prms->getPopsize();
    int esize = prms->getNumOfElite();

    mDevParentPopulation->copyFromDevice(mHostParentPopulation->getDeviceData());
    mDevOffspringPopulation->copyFromDevice(mHostOffspringPopulation->getDeviceData());

    // printf("------------ Population: %d: ------------ \n", generation);
    if (generation == 0)
    {
        printf("Generation,Mean,Max,Min\n");
    }
    printf("%d,%f,%d,%d\n", generation, 
                            mHostParentPopulation->getMean(),
                            mHostParentPopulation->getMax(),
                            mHostParentPopulation->getMin());
    // for (int k = 0; k < psize; ++k)
    for (int k = 0; k < esize; ++k)
    {
        int tempindex = mHostParentPopulation->getDeviceData()->elitesIdx[k];
        // printf("elite%d : %d\n", k, mHostParentPopulation->getDeviceData()->elitesIdx[k]);
        printf("elite%d : %d , %d\n", 
                k, tempindex, mHostParentPopulation->getDeviceData()->fitness[tempindex]);
    }
    printf("\n");

    /*
    for (int i = 0; i < psize; ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < csize; ++j)
        {
            printf("%d", mHostParentPopulation->getDeviceData()->population[i * csize + j]);
        }
        printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
    }
    */

    /*
    printf("------------ Offspring:%d ------------ \n", generation);
    for (int k = 0; k < esize; ++k)
    {
        printf("elite%d : %d\n", k, mHostOffspringPopulation->getDeviceData()->elitesIdx[k]);
    }
    printf("\n");
    
    for (int i = 0; i < psize; ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < csize; ++j)
        {
            printf("%d", mHostOffspringPopulation->getDeviceData()->population[i * csize + j]);
        }
        printf(":%d\n", mHostOffspringPopulation->getDeviceData()->fitness[i]);
    }
    */
}


