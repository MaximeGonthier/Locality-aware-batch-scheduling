#!/bin/bash

CLUSTER="inputs/clusters/cluster_450_128_32_256_4_1024.txt"
CLUSTER_TP=${CLUSTER:24}
CLUSTER_TP=${CLUSTER_TP::-4}
CONTRAINTES_TAILLES=0
BUSY_CLUSTER_THRESHOLD=100

make save -C src/

ndays=12

for ((i=1; i<=$((ndays)); i++))
do
	if [ $((i)) == 1 ]; then
		day1="07-11"
		day2="07-17"
	elif [ $((i)) == 2 ]; then
		day1="09-05"
		day2="09-11"
	elif [ $((i)) == 3 ]; then
		day1="10-10"
		day2="10-16"
	elif [ $((i)) == 4 ]; then
		day1="01-10"
		day2="01-16"
	elif [ $((i)) == 5 ]; then
		day1="02-07"
		day2="02-13"
	elif [ $((i)) == 6 ]; then
		day1="03-28"
		day2="04-03"
	elif [ $((i)) == 7 ]; then
		day1="10-24"
		day2="10-30"
	elif [ $((i)) == 8 ]; then
		day1="10-17"
		day2="10-23"
	elif [ $((i)) == 9 ]; then
		day1="12-12"
		day2="12-18"
	elif [ $((i)) == 10 ]; then
		day1="10-03"
		day2="10-09"
	elif [ $((i)) == 11 ]; then
		day1="01-03"
		day2="01-09"
	elif [ $((i)) == 12 ]; then
		day1="02-21"
		day2="02-27"
	fi
	WORKLOAD="inputs/workloads/converted/2022-${day1}->2022-${day2}_V10000_anonymous"
	WORKLOAD_TP=${WORKLOAD:27}

	OUTPUT_FILE=data/Results_FCFS_Score_Backfill_${WORKLOAD_TP}_${CLUSTER_TP}.csv
	echo "Scheduler,Number of jobs,Maximum queue time,Mean queue time,Total queue time,Maximum flow,Mean flow,Total flow,Transfer time,Makespan,Core time used, Waiting for a load time, Total waiting for a load time and transfer time, Mean Stretch, Mean Stretch With a Minimum, Max Stretch, Max Stretch With a Minimum, Nb Upgraded Jobs, Nb jobs large queue time, Mean flow stretch 128 jobs, Mean flow stretch 256 jobs, Mean flow stretch 1024 jobs, Mean flow stretch with a minimum 128 jobs, Mean flow stretch with a minimum 256 jobs, Mean flow stretch with a minimum 1024 jobs, Number of data reuse" > ${OUTPUT_FILE}

	for ((j=1; j<=10; j++))
	do
		if [ $((j)) == 1 ]; then SCHEDULER="Fcfs"; BACKFILL_MODE=0
		elif [ $((j)) == 2 ]; then SCHEDULER="Fcfs_conservativebf"; BACKFILL_MODE=2
		elif [ $((j)) == 3 ]; then SCHEDULER="Fcfs_with_a_score_x1_x0_x0_x0"; BACKFILL_MODE=0
		elif [ $((j)) == 4 ]; then SCHEDULER="Fcfs_with_a_score_conservativebf_x1_x0_x0_x0"; BACKFILL_MODE=2
		elif [ $((j)) == 5 ]; then SCHEDULER="Fcfs_with_a_score_x500_x1_x0_x0"; BACKFILL_MODE=0
		elif [ $((j)) == 6 ]; then SCHEDULER="Fcfs_with_a_score_conservativebf_x500_x1_x0_x0"; BACKFILL_MODE=2
		elif [ $((j)) == 7 ]; then SCHEDULER="Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x500_x1_x0_x0"; BACKFILL_MODE=0
		elif [ $((j)) == 8 ]; then SCHEDULER="Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_conservativebf_x500_x1_x0_x0"; BACKFILL_MODE=2
		elif [ $((j)) == 9 ]; then SCHEDULER="Fcfs_with_a_score_mixed_strategy_x500_x1_x0_x0"; BACKFILL_MODE=0
		elif [ $((j)) == 10 ]; then SCHEDULER="Fcfs_with_a_score_mixed_strategy_conservativebf_x500_x1_x0_x0"; BACKFILL_MODE=2
		fi
		
		./src/main $WORKLOAD $CLUSTER $SCHEDULER $CONTRAINTES_TAILLES $OUTPUT_FILE $BACKFILL_MODE $BUSY_CLUSTER_THRESHOLD
	done
done

common_output="data/Stretch_times_2022-all->2022-all_"
common_input="data/Stretch_times_2022-"
year="->2022-"
fcfs="Fcfs.csv"
eft="Fcfs_with_a_score_x1_x0_x0_x0.csv"
lea="Fcfs_with_a_score_x500_x1_x0_x0.csv"
leo="Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x500_x1_x0_x0.csv"
lem="Fcfs_with_a_score_mixed_strategy_x500_x1_x0_x0.csv"
fcfs_bf="Fcfs_conservativebf.csv"
eft_bf="Fcfs_with_a_score_conservativebf_x1_x0_x0_x0.csv"
lea_bf="Fcfs_with_a_score_conservativebf_x500_x1_x0_x0.csv"
leo_bf="Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_conservativebf_x500_x1_x0_x0.csv"
lem_bf="Fcfs_with_a_score_mixed_strategy_conservativebf_x500_x1_x0_x0.csv"

day1="02-21"
day2="02-27"
echo "Week 1:" ${day1} ${day2}
cat ${common_input}${day1}${year}${day2}_${fcfs} > ${common_output}${fcfs}
cat ${common_input}${day1}${year}${day2}_${eft} > ${common_output}${eft}
cat ${common_input}${day1}${year}${day2}_${lea} > ${common_output}${lea}
cat ${common_input}${day1}${year}${day2}_${leo} > ${common_output}${leo}
cat ${common_input}${day1}${year}${day2}_${lem} > ${common_output}${lem}
cat ${common_input}${day1}${year}${day2}_${fcfs_bf} > ${common_output}${fcfs_bf}
cat ${common_input}${day1}${year}${day2}_${eft_bf} > ${common_output}${eft_bf}
cat ${common_input}${day1}${year}${day2}_${lea_bf} > ${common_output}${lea_bf}
cat ${common_input}${day1}${year}${day2}_${leo_bf} > ${common_output}${leo_bf}
cat ${common_input}${day1}${year}${day2}_${lem_bf} > ${common_output}${lem_bf}

for ((i=1; i<=$((ndays-1)); i++))
do
	if [ $((i)) == 1 ]; then
		day1="07-11"
		day2="07-17"
	elif [ $((i)) == 2 ]; then
		day1="09-05"
		day2="09-11"
	elif [ $((i)) == 3 ]; then
		day1="10-10"
		day2="10-16"
	elif [ $((i)) == 4 ]; then
		day1="01-10"
		day2="01-16"
	elif [ $((i)) == 5 ]; then
		day1="02-07"
		day2="02-13"
	elif [ $((i)) == 6 ]; then
		day1="03-28"
		day2="04-03"
	elif [ $((i)) == 7 ]; then
		day1="10-24"
		day2="10-30"
	elif [ $((i)) == 8 ]; then
		day1="10-17"
		day2="10-23"
	elif [ $((i)) == 9 ]; then
		day1="12-12"
		day2="12-18"
	elif [ $((i)) == 10 ]; then
		day1="10-03"
		day2="10-09"
	elif [ $((i)) == 11 ]; then
		day1="01-03"
		day2="01-09"
	fi
	echo "Week "$((i+1))":" ${day1} ${day2}
	cat ${common_input}${day1}${year}${day2}_${fcfs} >> ${common_output}${fcfs}
	cat ${common_input}${day1}${year}${day2}_${eft} >> ${common_output}${eft}
	cat ${common_input}${day1}${year}${day2}_${lea} >> ${common_output}${lea}
	cat ${common_input}${day1}${year}${day2}_${leo} >> ${common_output}${leo}
	cat ${common_input}${day1}${year}${day2}_${lem} >> ${common_output}${lem}
	cat ${common_input}${day1}${year}${day2}_${fcfs_bf} >> ${common_output}${fcfs_bf}
	cat ${common_input}${day1}${year}${day2}_${eft_bf} >> ${common_output}${eft_bf}
	cat ${common_input}${day1}${year}${day2}_${lea_bf} >> ${common_output}${lea_bf}
	cat ${common_input}${day1}${year}${day2}_${leo_bf} >> ${common_output}${leo_bf}
	cat ${common_input}${day1}${year}${day2}_${lem_bf} >> ${common_output}${lem_bf}
done

python3 src/plot_boxplot.py all all byuser NO_BF stretch 1 boxplot
python3 src/plot_boxplot.py all all byuser BF stretch 1 boxplot
python3 src/plot_boxplot.py all all byuser NO_BF stretch 1 ecdf
