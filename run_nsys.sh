#!/bin/bash

nsys profile \
	   -t cuda,osrt,nvtx,cudnn,cublas \
		 --stats=true \
		 -f true \
		 -o "${1}" \
		 true ./gpuonemax

# -t : APIs to be tracked
# --stats : Outputs profiling information similar to nvprof
# -f : Overview the outputs
# -o : Output filename
