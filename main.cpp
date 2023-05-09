#include <stdio.h>

#include "Evolution.h"
#include "Parameters.h"

/**
 * The main function
 */
int main(int argc, char** argv)
{
    Parameters* prms = new Parameters();
    prms->loadParams();
    // printf("%d\n", prms->getChromosome());
    // prms->showParams();

    GPUEvolution GPU_Evolution(prms);
    GPU_Evolution.run(prms);

    return 0;
} // end of main


