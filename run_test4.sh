#!/bin/bash

CWD=$(pwd); readonly CWD

RUNALL=10
POPULATION=$(seq 128 128 1024); readonly POPULATION
CHROMOSOME=$(seq 128 128 1024); readonly CHROMOSOME
DATETIME=$(date +%Y%m%d-%H%M%S); readonly DATETIME
readonly PARAMSFILE=${CWD}/onemax.prms

for pop in ${POPULATION}; do
	sed -i "s/^POPSIZE.*$/POPSIZE                   ${pop}/" "${PARAMSFILE}"
	for chr in ${CHROMOSOME}; do
		sed -i "s/^CHROMOSOME.*$/CHROMOSOME                ${chr}/" "${PARAMSFILE}"
		for num in $(seq 1 1 ${RUNALL}); do
			echo "${pop} ${chr}" "${num}"
			./gpuonemax >> ${CWD}/fitnesstrend_${DATETIME}_${pop}_${chr}_${num}.csv
		done
	done
done



