#!/bin/bash

CWD=$(pwd); readonly CWD

#RUNALL=1
RUNALL=15
#RUNALL=30
#POPULATION=$(seq 32 1 32); readonly POPULATION
#CHROMOSOME=$(seq 32 1 32); readonly CHROMOSOME
POPULATION=$(seq 32 32 1024); readonly POPULATION
CHROMOSOME=$(seq 32 32 1024); readonly CHROMOSOME
DATETIME=$(date +%Y%m%d-%H%M%S); readonly DATETIME
readonly PARAMSFILE=${CWD}/onemax.prms
readonly BACKUPFILE=${CWD}/result_${DATETIME}.csv
readonly RESULTFILE=${CWD}/result.csv

if [[ -f ${RESULTFILE} ]]; then
	mv "${RESULTFILE}" "${BACKUPFILE}"
	rm "${RESULTFILE}"
fi
for num in $(seq 1 1 ${RUNALL}); do
	for pop in ${POPULATION}; do
		sed -i "s/^POPSIZE.*$/POPSIZE                   ${pop}/" "${PARAMSFILE}"
		for chr in ${CHROMOSOME}; do
			sed -i "s/^CHROMOSOME.*$/CHROMOSOME                ${chr}/" "${PARAMSFILE}"
			echo "${pop} ${chr}" "${num}"
			./gpuonemax >> "${BACKUPFILE}"
		done
	done
done



