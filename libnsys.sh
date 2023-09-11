#!/bin/bash

# -t : APIs to be tracked
# --stats : Outputs profiling information similar to nvprof
# -f : Overview the outputs
# -o : Output filename
# -t cuda,osrt,nvtx,cudnn,cublas \
run_nsight_systems()
{
	local PROGRAM=$1
	local OUTPUT=$2
	nsys profile \
		-t cuda,osrt,nvtx,cudnn,cublas \
		--stats=true \
		--force-overwrite true \
		-o "${OUTPUT}" \
		-w true \
		./${PROGRAM}
}

run_nsight_compute()
{
	local PROGRAM=$1
	local OUTPUT=$2
	ncu -o "${OUTPUT}" ./"${PROGRAM}"
}

run_nsight_compute_one_kernel()
{
	local PROGRAM=$1
	local KERNEL=$2
	local OUTPUT=$3
	ncu -o "${OUTPUT}" \
		--kernels "${KERNEL}" \
		./"${PROGRAM}"
}

run_nsight_systems $1 $2
#run_nsight_compute $1 $2
