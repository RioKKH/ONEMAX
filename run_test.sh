#!/bin/bash

CWD=$(pwd); readonly CWD

RUNALL=1
POPULATION=$(seq 32 32 1024); readonly POPULATION
CHROMOSOME=$(seq 32 32 1024); readonly CHROMOSOME
DATETIME=$(date +%Y%m%d-%H%M%S); readonly DATETIME
readonly PARAMSFILE=${CWD}/onemax.prms
readonly BACKUPFILE=${CWD}/result_${DATETIME}.csv
readonly RESULTFILE=${CWD}/result.csv

run()
{
	for _ in $(seq 1 1 ${RUNALL}); do
		./gpuonemax >> "${BACKUPFILE}"
	done
}

run_all()
{
	if [[ -f ${RESULTFILE} ]]; then
		mv "${RESULTFILE}" "${BACKUPFILE}"
		rm "${RESULTFILE}"
	fi

	for pop in ${POPULATION}; do
		sed -i "s/^POPSIZE.*$/POPSIZE                   ${pop}/" "${PARAMSFILE}"
		for chr in ${CHROMOSOME}; do
			echo "${pop} ${chr}"
			sed -i "s/^CHROMOSOME.*$/CHROMOSOME                ${chr}/" "${PARAMSFILE}"
			run
		done
	done
}

run_all

