#include <iostream>
#include <cuda_runtime.h>
#include "Evolution.h"
#include "Parameters.h"

static int show_summary(const float& elapsed_time, const Parameters& prms);

/**
 * The main function
 */
int main(int argc, char** argv)
{
#ifndef _OFFLOAD
    float elapsed_time = 0.0f;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

    Parameters* prms = new Parameters();
    prms->loadParams();
    // printf("%d\n", prms->getChromosome());
    // prms->showParams();

    GPUEvolution GPU_Evolution(prms);
    GPU_Evolution.run(prms);

#ifndef _OFFLOAD
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    show_summary(elapsed_time, *prms);
#endif // _OFFLOAD

    return 0;
} // end of main


static int show_summary(const float& elapsed_time, const Parameters& prms)
{
    std::cout
        << prms.getNumOfGenerations()
        << "," << prms.getPopsizeActual()
        << "," << prms.getChromosomeActual()
        << "," << elapsed_time
        << std::endl;
    return 0;
}


