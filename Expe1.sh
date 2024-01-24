#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage is bash Expe1.sh workload day1 day2 mode"
    exit
fi

# Get arguments
WORKLOAD=$1
WORKLOAD_TP=${WORKLOAD:27}
CLUSTER="inputs/clusters/cluster_450_128_32_256_4_1024.txt"
CLUSTER_TP=${CLUSTER:24}
CLUSTER_TP=${CLUSTER_TP::-4}
CONTRAINTES_TAILLES=0
BUSY_CLUSTER_THRESHOLD=100
day1=$2
day2=$3
mode=$4

make save -C src/

OUTPUT_FILE=data/Results_FCFS_Score_Backfill_${WORKLOAD_TP}_${CLUSTER_TP}.csv
echo "Scheduler,Number of jobs,Maximum queue time,Mean queue time,Total queue time,Maximum flow,Mean flow,Total flow,Transfer time,Makespan,Core time used, Waiting for a load time, Total waiting for a load time and transfer time, Mean Stretch, Mean Stretch With a Minimum, Max Stretch, Max Stretch With a Minimum, Nb Upgraded Jobs, Nb jobs large queue time, Mean flow stretch 128 jobs, Mean flow stretch 256 jobs, Mean flow stretch 1024 jobs, Mean flow stretch with a minimum 128 jobs, Mean flow stretch with a minimum 256 jobs, Mean flow stretch with a minimum 1024 jobs, Number of data reuse" > ${OUTPUT_FILE}

for ((i=1; i<=5; i++))
do
	if [ $((i)) == 1 ]; then SCHEDULER="Fcfs"; BACKFILL_MODE=0
	elif [ $((i)) == 2 ]; then SCHEDULER="Fcfs_with_a_score_x1_x0_x0_x0"; BACKFILL_MODE=0
	elif [ $((i)) == 3 ]; then SCHEDULER="Fcfs_with_a_score_x500_x1_x0_x0"; BACKFILL_MODE=0
	elif [ $((i)) == 4 ]; then SCHEDULER="Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x500_x1_x0_x0"; BACKFILL_MODE=0
	elif [ $((i)) == 5 ]; then SCHEDULER="Fcfs_with_a_score_mixed_strategy_x500_x1_x0_x0"; BACKFILL_MODE=0
	fi
	
	./src/main $WORKLOAD $CLUSTER $SCHEDULER $CONTRAINTES_TAILLES $OUTPUT_FILE $BACKFILL_MODE $BUSY_CLUSTER_THRESHOLD
done

python3 src/plot_boxplot.py ${day1} ${day2} byuser NO_BF stretch ${mode} small_hist
