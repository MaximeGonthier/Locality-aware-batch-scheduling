# Data-Driven Locality-Aware Batch Scheduling

You can find in this depot the source code of the batch simulator and the schedulers used in the paper Data-Driven Locality-Aware Batch Scheduling.

Results presented in the paper can be reproduced.

All the workloads used for our evaluations are in the folder inputs/workloads/converted/

The model of the cluster used is in inputs/clusters/

The source code is in src/

The generated Figures will be in plot/

The generated data to plot the Figures will be in data/

You can also find the methodology we followed to randomly select 12 weeks to evaluate our scheduler in the file random_week_selection.py. Using python3.6.9 you can re-create the random generation by doing python3 random_week_selection.py

## Source code

The source code presented here is a simplified version of our simulator and schedulers.
It contains only the functions needed to run the 5 schedulers presented in the paper.
All other schedulers and settings tested and not presented in the paper, as well as tools 
to visualize our strategies are not present.

The code is constructed as follows:
- main.c calls functions to initialize the simulator and contains the loop that iterate while all jobs have not been scheduled
- scheduler.c contains the 5 schedulers presented in the paper
- basic_functions.c gather functions that can be called by any scheduler or by the main
- read_input_files.c initialize nodes and jobs list from input files
- print_functions.c contains functions used to print in an output file the results of the execution
- linked_list_functions gather basic functions for managing linked list
- backfill_functions.c regroup functions used when using conservative backfilling
- scheduler_calling.c contain the function that start a scheduler depending on the scheduler chosen by the user

The python file is used to plot the figures.

## Reproducible experiments

You can re-create Figures 2 and 6 showing the stretch's improvement by doing the following steps:

```bash
~/$ bash Expe1.sh inputs/workloads/converted/2022-10-03-\>2022-10-09_V10000_anonymous 10-03 10-09 1
```

```bash
~/$ bash Expe1.sh inputs/workloads/converted/2022-10-24-\>2022-10-30_V10000_anonymous 10-24 10-30 1
```
The plots will be in plot/Boxplot/byuser

To compute the 12 weeks and plot the aggregated results (Figures 7, 8 and 9) do:
```bash
~/$ bash Expe2.sh
```
The boxplots will be in plot/Boxplot/byuser and the ecdf in plot/ECDF/byuser
