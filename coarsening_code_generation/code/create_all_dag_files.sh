#!/bin/bash
for i in 32 64 128 256
do
        for n in edlenet yololite
        do
		echo $i\_$n
		python fusion_overhead_profiling.py $n $i interference

	done
done
