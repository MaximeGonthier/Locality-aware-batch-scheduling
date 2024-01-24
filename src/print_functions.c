#include <main.h>

/** 
 * This file contains various functions used to
 * print in terminal or output files.
 * The printed information in output files have 
 * been used to plot figures and better understand 
 * the behavior of our schedulers.
**/

void print_cores_in_specific_node(struct Node* n)
{
	printf("Cores in node %d are: ", n->unique_id);
	for (int i = 0; i < 19; i++)
	{
		printf("%d(%d),", n->cores[i]->unique_id, n->cores[i]->available_time);
	}
	printf("%d(%d).\n", n->cores[19]->unique_id, n->cores[19]->available_time);
}

/** Print in a csv file information on a job that just finished. **/
void to_print_job_csv(struct Job* job, int time)
{
	int time_used = job->end_time - job->start_time;
	
	/* Plotting stats on choosen method. */
	#ifdef PLOT_STATS
	FILE* f_stats_choosen_method = fopen("outputs/choosen_methods.txt", "a");
	if (!f_stats_choosen_method)
	{
		perror("Error opening file outputs/choosen_methods.txt.");
		exit(EXIT_FAILURE);
	}
	fprintf(f_stats_choosen_method, "%d,%d\n", job->unique_id, job->last_choosen_method);
	fclose(f_stats_choosen_method);
	#endif
	
	/* Only for evaluated jobs */
	if (job->workload == 1)
	{
		struct To_Print* new = (struct To_Print*) malloc(sizeof(struct To_Print));
		new->next = NULL;
		new->job_unique_id = job->unique_id;
		new->job_subtime = job->subtime;
		new->time = time;
		new->time_used = time_used;
		new->transfer_time = job->transfer_time;
		new->job_start_time = job->start_time;
		new->job_end_time = job->end_time;
		new->job_cores = job->cores;
		new->waiting_for_a_load_time = job->waiting_for_a_load_time;
		new->empty_cluster_time = job->delay + (job->data_size)/0.1;
		new->data_type = job->index_node_list;
		new->job_data_size = job->data_size;
		
		new->user = job->user;
		new->input_file = job->data;
		
		if (job->node_used->index_node_list > job->index_node_list)
		{
			new->upgraded = 1;
		}
		else
		{
			new->upgraded = 0;
		}
				
		insert_tail_to_print_list(jobs_to_print_list, new);
	}
		
	#ifdef PRINT_GANTT_CHART
	int i = 0;
	char* file_to_open = malloc(100*sizeof(char));
	strcpy(file_to_open, "outputs/Results_all_jobs_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".csv");
	FILE* f = fopen(file_to_open, "a");
	if (!f)
	{
		printf("Error opening file Results_all_jobs_scheduler.csv\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	if (job->transfer_time > 0)
	{
		if (job->transfer_time > job->walltime) { job->transfer_time = job->walltime; }
		fprintf(f, "I/O-%d,%d,delay,%f,%d,%d,1,COMPLETED_SUCCESSFULLY,%d,%d,%d,%d,%d,%d,", job->data, job->unique_id, 0.0, job->cores, job->walltime, job->start_time - first_subtime_day_0, job->transfer_time, job->start_time - first_subtime_day_0 + job->transfer_time, job->start_time - first_subtime_day_0, job->end_time - first_subtime_day_0, 1);
		/* Printing the cores used. */
		if (job->cores > 1)
		{
			for (i = 0; i < job->cores; i++)
			{
				if (i == job->cores - 1)
				{
					fprintf(f, "%d", job->node_used->unique_id*20 + job->cores_used[i]);
				}
				else
				{
					fprintf(f, "%d ", job->node_used->unique_id*20 + job->cores_used[i]);
				}
			 }
		}
		else
		{
			fprintf(f, "%d", job->node_used->unique_id*20 + job->cores_used[0]);
		}
		fprintf(f, ",-1,\"\"\n");
	}
	
	fprintf(f, "%d,%d,delay,%f,%d,%d,1,COMPLETED_SUCCESSFULLY,%d,%d,%d,%d,%d,%d,", job->unique_id, job->unique_id, 0.0, job->cores, job->walltime, job->start_time - first_subtime_day_0 + job->transfer_time, time_used - job->transfer_time, job->end_time - first_subtime_day_0, job->start_time - first_subtime_day_0, job->end_time - first_subtime_day_0, 1);
	if (job->cores > 1)
	{
		for (i = 0; i < job->cores; i++)
		{
			if (i == job->cores - 1)
			{
				fprintf(f, "%d", job->node_used->unique_id*20 + job->cores_used[i]);
			}
			else
			{
				fprintf(f, "%d ", job->node_used->unique_id*20 + job->cores_used[i]);
			}
		 }
	}
	else
	{
		fprintf(f, "%d", job->node_used->unique_id*20 + job->cores_used[0]);
	}
	fprintf(f, ",-1,\"\"\n");
	
	fclose(f);
	free(file_to_open);
	#endif
	
	#ifdef PRINT_DISTRIBUTION_QUEUE_TIMES
	char* file_to_open = malloc(100*sizeof(char));
	strcpy(file_to_open, "outputs/Distribution_queue_times_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".txt");
	FILE* f = fopen(file_to_open, "a");
	if (!f)
	{
		perror("Error opening file.\n");
		exit(EXIT_FAILURE);
	}
	fprintf(f, "%d\n", job->start_time - job->subtime);
	fclose(f);
	free(file_to_open);
	#endif
}

/** Print in a file the final results. Only called once at the end of the simulation. **/
void print_csv(struct To_Print* head_to_print)
{
	printf("nb_call_finished_jobs: %d - nb_call_new_jobs: %d\n", nb_call_finished_jobs, nb_call_new_jobs); fflush(stdout);
	
	int size_file_to_open = 300;

	/* Stretch times for boxplots on all jobs */
	char* file_to_open = malloc(size_file_to_open*sizeof(char));
	char* day = malloc(22*sizeof(char));
	int i = 0;
	for (i = 0; i < 22; i++)
	{
		day[i] = input_job_file[i + 27];
	}
	file_to_open = malloc(size_file_to_open*sizeof(char));
	strcpy(file_to_open, "data/Stretch_times_");
	strcat(file_to_open, day);
	strcat(file_to_open, "_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".csv");
	FILE* f_stretch = fopen(file_to_open, "w");
	if (!f_stretch)
	{
		perror("Error opening file.\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}

	if (strcmp(scheduler, "Fcfs") == 0)
	{
		scheduler = "FCFS";
	}
	else if (strcmp(scheduler, "Fcfs_conservativebf") == 0)
	{
		if (backfill_mode == 0)
		{
			scheduler = "FCFS CONSERVATIVE BF MODE 0";
		}
		else if (backfill_mode == 1)
		{
			scheduler = "FCFS CONSERVATIVE BF MODE 1";
		}
		else  if (backfill_mode == 2)
		{
			scheduler = "FCFS CONSERVATIVE BF MODE 2";
		}
		else
		{
			printf("Error mauvais backfill mode dnas print.\n");
			exit(1);
		}
	}
	else if (strcmp(scheduler, "Fcfs_easybf") == 0)
	{
		scheduler = "FCFS EASY BF";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_conservativebf_x1_x0_x0_x0") == 0 || strcmp(scheduler, "Eft") == 0)
	{
		if (backfill_mode == 0)
		{
			scheduler = "EFT CONSERVATIVE BF MODE 0";
		}
		else if (backfill_mode == 1)
		{
			scheduler = "EFT CONSERVATIVE BF MODE 1";
		}
		else  if (backfill_mode == 2)
		{
			scheduler = "EFT CONSERVATIVE BF MODE 2";
		}
		else
		{
			printf("Error mauvais backfill mode dans print.\n");
			exit(1);
		}
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_x1_x0_x0_x0") == 0)
	{
		scheduler = "EFT";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_x500_x1_x0_x0") == 0)
	{
		scheduler = "SCORE";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_conservativebf_x500_x1_x0_x0") == 0)
	{
		if (backfill_mode == 0)
		{
			scheduler = "SCORE CONSERVATIVE BF MODE 0";
		}
		else if (backfill_mode == 1)
		{
			scheduler = "SCORE CONSERVATIVE BF MODE 1";
		}
		else  if (backfill_mode == 2)
		{
			scheduler = "SCORE CONSERVATIVE BF MODE 2";
		}
		else
		{
			printf("Error mauvais backfill mode dnas print.\n");
			exit(1);
		}
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_easybf_x500_x1_x0_x0") == 0)
	{
		scheduler = "SCORE EASY BF";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x500_x1_x0_x0") == 0)
	{
		scheduler = "OPPORTUNISTIC-SCORE MIX";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_easybf_x500_x1_x0_x0") == 0)
	{
		scheduler = "OPPORTUNISTIC-SCORE MIX EASY BF";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_conservativebf_x500_x1_x0_x0") == 0)
	{
		if (backfill_mode == 0)
		{
			scheduler = "OPPORTUNISTIC-SCORE MIX CONSERVATIVE BF MODE 0";
		}
		else if (backfill_mode == 1)
		{
			scheduler = "OPPORTUNISTIC-SCORE MIX CONSERVATIVE BF MODE 1";
		}
		else  if (backfill_mode == 2)
		{
			scheduler = "OPPORTUNISTIC-SCORE MIX CONSERVATIVE BF MODE 2";
		}
		else
		{
			printf("Error mauvais backfill mode dnas print.\n");
			exit(1);
		}
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_mixed_strategy_x500_x1_x0_x0") == 0)
	{
		scheduler = "EFT-SCORE MIX";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_mixed_strategy_easybf_x500_x1_x0_x0") == 0)
	{
		scheduler = "EFT-SCORE MIX EASY BF";
	}
	else if (strcmp(scheduler, "Fcfs_with_a_score_mixed_strategy_conservativebf_x500_x1_x0_x0") == 0)
	{
		if (backfill_mode == 0)
		{
			scheduler = "EFT-SCORE MIX CONSERVATIVE BF MODE 0";
		}
		else if (backfill_mode == 1)
		{
			scheduler = "EFT-SCORE MIX CONSERVATIVE BF MODE 1";
		}
		else  if (backfill_mode == 2)
		{
			scheduler = "EFT-SCORE MIX CONSERVATIVE BF MODE 2";
		}
		else
		{
			printf("Error mauvais backfill mode dnas print.\n");
			exit(1);
		}
	}
	
	#ifdef PLOT_STATS
	FILE* f_nb_backfilled_jobs = fopen("outputs/nb_backfilled_jobs.txt", "w");
	if (!f_nb_backfilled_jobs)
	{
		perror("Error opening file outputs/nb_backfilled_jobs.txt.");
		exit(EXIT_FAILURE);
	}
	fprintf(f_nb_backfilled_jobs, "%s: %d\n", scheduler, number_of_backfilled_jobs);
	fclose(f_nb_backfilled_jobs);
	
	FILE* tie_breaks_fcfs_score = fopen("outputs/tie_breaks_fcfs_score.txt", "w");
	if (!tie_breaks_fcfs_score)
	{
		perror("Error opening file outputs/tie_breaks_fcfs_score.txt.");
		exit(EXIT_FAILURE);
	}
	fprintf(tie_breaks_fcfs_score, "%s: %d/%d\n", scheduler, number_of_tie_breaks_before_computing_evicted_files_fcfs_score, total_number_of_scores_computed);
	fclose(tie_breaks_fcfs_score);
	
	FILE* f_data_persistence_exploited = fopen("outputs/data_persistence_exploited.txt", "w");
	if (!f_data_persistence_exploited)
	{
		perror("Error opening file outputs/data_persistence_exploited.txt.");
		exit(EXIT_FAILURE);
	}
	fprintf(f_data_persistence_exploited, "%s: %d\n", scheduler, data_persistence_exploited);
	fclose(f_data_persistence_exploited);
	#endif
		
	#ifdef PRINT_DISTRIBUTION_QUEUE_TIMES
	/* For distribution of flow and queue times on each job. */
	//~ char* file_to_open = malloc(size_file_to_open*sizeof(char));
	strcpy(file_to_open, "outputs/Queue_times_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".txt");
	FILE* f_queue = fopen(file_to_open, "w");
	if (!f_queue)
	{
		perror("Error opening file.\n");
		exit(EXIT_FAILURE);
	}
	
	
	
	file_to_open = malloc(size_file_to_open*sizeof(char));
	strcpy(file_to_open, "data/Flow_times_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".txt");
	FILE* f_flow = fopen(file_to_open, "w");
	if (!f_flow)
	{
		perror("Error opening file.\n");
		exit(EXIT_FAILURE);
	}
		
	file_to_open = malloc(size_file_to_open*sizeof(char));
	strcpy(file_to_open, "outputs/Stretch_times_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".txt");
	FILE* f_stretch2 = fopen(file_to_open, "w");
	if (!f_stretch2)
	{
		perror("Error opening file.\n");
		exit(EXIT_FAILURE);
	}
	
	file_to_open = malloc(size_file_to_open*sizeof(char));
	strcpy(file_to_open, "outputs/Bounded_Stretch_times_");
	strcat(file_to_open, scheduler);
	strcat(file_to_open, ".txt");
	FILE* f_bounded_stretch = fopen(file_to_open, "w");
	if (!f_bounded_stretch)
	{
		perror("Error opening file.\n");
		exit(EXIT_FAILURE);
	}
	#endif
		
	/* Values evaluated. */
	float max_queue_time = 0;
	float mean_queue_time = 0;
	float total_queue_time = 0;
	float max_flow = 0;
	float mean_flow = 0;
	float total_flow = 0;
	float total_transfer_time = 0;
	float total_waiting_for_a_load_time = 0;
	float total_waiting_for_a_load_time_and_transfer_time = 0;
	float makespan = 0;
	float total_flow_stretch = 0;
	float total_flow_stretch_128 = 0;
	float total_flow_stretch_256 = 0;
	float total_flow_stretch_1024 = 0;
	float max_flow_stretch = 0;
	float total_flow_stretch_with_a_minimum = 0;
	float total_flow_stretch_with_a_minimum_128 = 0;
	float total_flow_stretch_with_a_minimum_256 = 0;
	float total_flow_stretch_with_a_minimum_1024 = 0;
	float max_flow_stretch_with_a_minimum = 0;
	float mean_flow_stretch = 0;
	float mean_flow_stretch_128 = 0;
	float mean_flow_stretch_256 = 0;
	float mean_flow_stretch_1024 = 0;
	float mean_flow_stretch_with_a_minimum = 0;
	float mean_flow_stretch_with_a_minimum_128 = 0;
	float mean_flow_stretch_with_a_minimum_256 = 0;
	float mean_flow_stretch_with_a_minimum_1024 = 0;
	float core_time_used = 0;
	float denominator_bounded_stretch = 0;
	int nb_upgraded_jobs = 0;
	int what_is_a_large_queue_time = 25000;
	int nb_large_queue_times = 0;
		
	int verify_nb_job_to_evaluate = 0;
	
	while (head_to_print != NULL)
	{
		verify_nb_job_to_evaluate += 1;
		
		nb_upgraded_jobs += head_to_print->upgraded;
				
		/* Flow stretch */
		total_flow_stretch += (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time;
		if (head_to_print->job_data_size <= 128)
		{
			total_flow_stretch_128 += (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time;
		}
		else if (head_to_print->job_data_size <= 256)
		{
			total_flow_stretch_256 += (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time;
		}
		else if (head_to_print->job_data_size <= 1024)
		{
			total_flow_stretch_1024 += (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time;
		}
		else
		{
			printf("error print csv data type %f.\n", head_to_print->job_data_size); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		/* Bounded flow stretch */
		if (head_to_print->empty_cluster_time > 300)
		//~ if (head_to_print->empty_cluster_time > 60)
		{
			denominator_bounded_stretch = head_to_print->empty_cluster_time;
		}
		else
		{
			denominator_bounded_stretch = 300;
			//~ denominator_bounded_stretch = 60;
		}
		
		if (denominator_bounded_stretch == 0)
		{
			printf("error denomitaro == 0\n"); fflush(stdout);
			exit(EXIT_FAILURE);
		}
		
		total_flow_stretch_with_a_minimum += (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch;
		if (head_to_print->job_data_size <= 128)
		{
			total_flow_stretch_with_a_minimum_128 += (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch;
		}
		else if (head_to_print->job_data_size <= 256)
		{
			total_flow_stretch_with_a_minimum_256 += (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch;
		}
		else if (head_to_print->job_data_size <= 1024)
		{
			total_flow_stretch_with_a_minimum_1024 += (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch;
		}
		else
		{
			printf("error print csv data type %f.\n", head_to_print->job_data_size); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		/* Maximum flow stretch */
		if (max_flow_stretch < (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time)
		{
			max_flow_stretch = (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time;
		}
		if (max_flow_stretch_with_a_minimum < (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch)
		{
			max_flow_stretch_with_a_minimum = (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch;
		}
			
		core_time_used += head_to_print->time_used*head_to_print->job_cores;

		if (head_to_print->job_start_time - head_to_print->job_subtime < 0)
		{
			printf("Error queue time is %d for job %d.\n", head_to_print->job_start_time - head_to_print->job_subtime, head_to_print->job_unique_id); fflush(stdout);
			exit(EXIT_FAILURE);
		}
		
		total_queue_time += head_to_print->job_start_time - head_to_print->job_subtime;
		
		if (head_to_print->job_start_time - head_to_print->job_subtime > what_is_a_large_queue_time)
		{
			nb_large_queue_times += 1;
		}
		
		if (max_queue_time < head_to_print->job_start_time - head_to_print->job_subtime)
		{
			max_queue_time = head_to_print->job_start_time - head_to_print->job_subtime;
		}
		total_flow += head_to_print->job_end_time - head_to_print->job_subtime;
		if (max_flow < head_to_print->job_end_time - head_to_print->job_subtime)
		{
			max_flow = head_to_print->job_end_time - head_to_print->job_subtime;
		}
		total_transfer_time += head_to_print->transfer_time;
		total_waiting_for_a_load_time += head_to_print->waiting_for_a_load_time;
		total_waiting_for_a_load_time_and_transfer_time += head_to_print->transfer_time + head_to_print->waiting_for_a_load_time;
		if (makespan < head_to_print->job_end_time)
		{
			makespan = head_to_print->job_end_time;
		}
		
		#ifdef PRINT_DISTRIBUTION_QUEUE_TIMES
		/* For distribution of flow and queue times on each job to show VS curves */
		fprintf(f_queue, "%d %d %d %d %d\n", head_to_print->job_unique_id, head_to_print->job_start_time - head_to_print->job_subtime, head_to_print->data_type, head_to_print->job_end_time - head_to_print->job_start_time, head_to_print->job_subtime);
		fprintf(f_flow, "%d %d %d %d %d\n", head_to_print->job_unique_id, head_to_print->job_end_time - head_to_print->job_subtime, head_to_print->data_type, head_to_print->job_end_time - head_to_print->job_start_time, head_to_print->job_subtime);
		fprintf(f_stretch2, "%d %f %d %d %d\n", head_to_print->job_unique_id, (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time, head_to_print->data_type, head_to_print->job_end_time - head_to_print->job_start_time, head_to_print->job_subtime);
		fprintf(f_bounded_stretch, "%d %f %d %d %d\n", head_to_print->job_unique_id, (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch, head_to_print->data_type, head_to_print->job_end_time - head_to_print->job_start_time, head_to_print->job_subtime);
		#endif
		
		/* Id Stretch Datatype Length Subtime, Ncores, TransferTime, user, input_file, core_time_used, bounded stretch 1min */
		fprintf(f_stretch, "%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f\n", head_to_print->job_unique_id, (head_to_print->job_end_time - head_to_print->job_subtime)/head_to_print->empty_cluster_time, head_to_print->data_type, head_to_print->job_end_time - head_to_print->job_start_time, head_to_print->job_subtime, head_to_print->job_cores, head_to_print->transfer_time + head_to_print->waiting_for_a_load_time, head_to_print->user, head_to_print->input_file, head_to_print->time_used*head_to_print->job_cores, (head_to_print->job_end_time - head_to_print->job_subtime)/denominator_bounded_stretch); fflush(stdout);
		
		head_to_print = head_to_print->next;
	}
	
	if (verify_nb_job_to_evaluate != nb_job_to_evaluate)
	{
		printf("Error verify_nb_job_to_evaluate %d != nb_job_to_evaluate %d\n", verify_nb_job_to_evaluate, nb_job_to_evaluate); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	else
	{
		printf("verify_nb_job_to_evaluate %d == nb_job_to_evaluate %d\n", verify_nb_job_to_evaluate, nb_job_to_evaluate); fflush(stdout);
	}
	
	fclose(f_stretch);
	
	#ifdef PRINT_DISTRIBUTION_QUEUE_TIMES
	fclose(f_queue);
	fclose(f_flow);
	fclose(f_stretch2);
	fclose(f_bounded_stretch);
	#endif
	
	/* Compute mean values */
	mean_queue_time = total_queue_time/nb_job_to_evaluate;
	mean_flow = total_flow/nb_job_to_evaluate;
	mean_flow_stretch = total_flow_stretch/nb_job_to_evaluate;
	mean_flow_stretch_128 = total_flow_stretch_128/nb_job_to_evaluate;
	mean_flow_stretch_256 = total_flow_stretch_256/nb_job_to_evaluate;
	mean_flow_stretch_1024 = total_flow_stretch_1024/nb_job_to_evaluate;
	mean_flow_stretch_with_a_minimum = total_flow_stretch_with_a_minimum/nb_job_to_evaluate;
	mean_flow_stretch_with_a_minimum_128 = total_flow_stretch_with_a_minimum_128/nb_job_to_evaluate;
	mean_flow_stretch_with_a_minimum_256 = total_flow_stretch_with_a_minimum_256/nb_job_to_evaluate;
	mean_flow_stretch_with_a_minimum_1024 = total_flow_stretch_with_a_minimum_1024/nb_job_to_evaluate;
	
	/* Main file of results */
	char* file_to_open_2 = malloc(size_file_to_open*sizeof(char));
	strcpy(file_to_open_2, output_file);
	FILE* f = fopen(file_to_open_2, "w");
	if (!f)
	{
		perror("Error opening file.\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}

	printf("Scheduler: %s, Number of jobs evaluated: %d, Max queue time: %f, Mean queue time: %f, Total queue time: %f, Max flow: %f, Mean flow: %f, Total flow: %f, Transfer time: %f, Makespan: %f, Core time: %f, Waiting for a load time: %f, Transfer + waiting time: %f, Mean flow stretch: %f, Mean bounded flow stretch: %f, Max flow stretch: %f, Max bounded flow stretch: %f, Nb of upgraded jobs: %d, Nb large queue times (>%d): %d, Mean flow stretch 128 256 1024: %f %f %f, Mean flow stretch with a minimum 128 256 1024: %f %f %f, Number of data reuse: %d\n\n", scheduler, nb_job_to_evaluate, max_queue_time, mean_queue_time, total_queue_time, max_flow, mean_flow, total_flow, total_transfer_time, makespan, core_time_used, total_waiting_for_a_load_time, total_waiting_for_a_load_time_and_transfer_time, mean_flow_stretch, mean_flow_stretch_with_a_minimum, max_flow_stretch, max_flow_stretch_with_a_minimum, nb_upgraded_jobs, what_is_a_large_queue_time, nb_large_queue_times, mean_flow_stretch_128, mean_flow_stretch_256,mean_flow_stretch_1024, mean_flow_stretch_with_a_minimum_128, mean_flow_stretch_with_a_minimum_256, mean_flow_stretch_with_a_minimum_1024, nb_data_reuse);
	
	fprintf(f, "%s,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f,%d\n", scheduler, nb_job_to_evaluate, max_queue_time, mean_queue_time, total_queue_time, max_flow, mean_flow, total_flow, total_transfer_time, makespan, core_time_used, total_waiting_for_a_load_time, total_waiting_for_a_load_time_and_transfer_time, mean_flow_stretch, mean_flow_stretch_with_a_minimum, max_flow_stretch, max_flow_stretch_with_a_minimum, nb_upgraded_jobs, nb_large_queue_times, mean_flow_stretch_128, mean_flow_stretch_256,mean_flow_stretch_1024, mean_flow_stretch_with_a_minimum_128, mean_flow_stretch_with_a_minimum_256, mean_flow_stretch_with_a_minimum_1024, nb_data_reuse);
	fclose(f);
	
	free(file_to_open);
	free(file_to_open_2);
}

/** Save state of an execution
 * Manque new_jobs ?
 * Manque end_of_file_load
 * Manque les intervalles **/
void save_state(int t, int old_finished_jobs, int next_submit_time, char* input_job_file)
{
	printf("Saving the state of the execution..\n"); fflush(stdout);

	char* to_open = malloc(120*sizeof(char));
	strcpy(to_open, "outputs/saved_state_");
	strncat(to_open, input_job_file + 32, 5);
	strcat(to_open, "-");
	strncat(to_open, input_job_file + 44, 5);
	strcat(to_open, "_");
	strcat(to_open, scheduler);
	printf("File to save: %s\n", to_open); fflush(stdout);
	FILE *f = fopen(to_open, "w");
	free(to_open);
	int i = 0;
	int j = 0;
	
	fprintf(f, "t: %d\n", t);
	fprintf(f, "next_submit_time: %d\n", next_submit_time);
	fprintf(f, "planned_or_ratio: %d\n", planned_or_ratio);
	fprintf(f, "constraint_on_sizes: %d\n", constraint_on_sizes);
	fprintf(f, "nb_cores: %d\n", nb_cores);
	fprintf(f, "nb_job_to_evaluate: %d\n", nb_job_to_evaluate);
	fprintf(f, "nb_data_reuse: %d\n", nb_data_reuse);
	fprintf(f, "finished_jobs: %d\n", finished_jobs);
	fprintf(f, "old_finished_jobs: %d\n", old_finished_jobs);
	fprintf(f, "total_number_jobs: %d\n", total_number_jobs);
	fprintf(f, "total_number_nodes: %d\n", total_number_nodes);
	fprintf(f, "running_cores: %d\n", running_cores);
	fprintf(f, "running_nodes: %d\n", running_nodes);
	fprintf(f, "nb_job_to_schedule: %d\n", nb_job_to_schedule);
	fprintf(f, "nb_cores_to_schedule: %d\n", nb_cores_to_schedule);
	fprintf(f, "total_queue_time: %d\n", total_queue_time);
	fprintf(f, "first_subtime_day_0: %d\n", first_subtime_day_0);
	fprintf(f, "global_nb_non_available_cores_at_time_t: %d\n", global_nb_non_available_cores_at_time_t);
	fprintf(f, "Planned_Area[3][3]:"); 
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			fprintf(f, " %lld", Planned_Area[i][j]);
		}
	}
	fprintf(f, "\n");
	fprintf(f, "number_node_size[3]:");
	for (i = 0; i < 3; i++)
	{
		fprintf(f, " %d", number_node_size[i]);
	}
	fprintf(f, "\n");
	fprintf(f, "busy_cluster: %d\n", busy_cluster);
	fprintf(f, "backfill_mode: %d\n", backfill_mode);
	fprintf(f, "Allocated_Area[3][3]:"); 
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			fprintf(f, " %lld", Allocated_Area[i][j]);
		}
	}
	fprintf(f, "\n");


	fprintf(f, "nb_job_to_evaluate_started: %d\n", nb_job_to_evaluate_started);
	//~ print_cores_in_specific_node(node_list[0]->head); exit(1);
	fprintf(f, "\nnode_list:\n");
	for (i = 0; i < 3; i++)
	{
		struct Node* n = node_list[i]->head;
		
		while (n != NULL)
		{
			fprintf(f, "unique_id: %d, memory: %d, bandwidth: %f, data: [", n->unique_id, n->memory, n->bandwidth);
			
			/* Data */
			struct Data* d = n->data->head;
			while (d != NULL)
			{
				fprintf(f, " unique_id: %d, start_time: %d, end_time: %d, ", d->unique_id, d->start_time, d->end_time);
				
				#ifndef DATA_PERSISTENCE
				fprintf(f, "nb_task_using_it: %d, ", d->nb_task_using_it); 
				#endif
				
				fprintf(f, "intervals: null, size: %f", d->size);
				d = d->next;
			}
			
			fprintf(f, " ], cores: [");
			for (j = 0; j < 20; j++)
			{
				fprintf(f, " unique_id: %d, available_time: %d, running_job: %d, running_job_end: %d", n->cores[j]->unique_id, n->cores[j]->available_time, n->cores[j]->running_job, n->cores[j]->running_job_end);
			}
			fprintf(f, " ], ");
			//~ fprintf(f, "n_available_cores: %d, index_node_list: %d, number_cores_in_a_hole: %d, cores_in_a_hole: null", n->n_available_cores, n->index_node_list, n->number_cores_in_a_hole);
			fprintf(f, "n_available_cores: %d, index_node_list: %d, number_cores_in_a_hole: %d, cores_in_a_hole: [", n->n_available_cores, n->index_node_list, n->number_cores_in_a_hole);
			struct Core_in_a_hole* c = n->cores_in_a_hole->head;
			while (c != NULL)
			{
				fprintf(f, " unique_id: %d, start_time_of_the_hole: %d",c->unique_id, c->start_time_of_the_hole);
				c = c->next;
			}
			fprintf(f, " ]");
			
			#ifdef DATA_PERSISTENCE
			fprintf(f, ", data_occupation: %d, temp_data: null, ", n->data_occupation);
			#endif
			
			fprintf(f, ", end_of_file_load: %d\n", n->end_of_file_load);
			
			n = n->next;
		}
	}	
	
	struct Job* job = job_list_to_start_from_history->head;
	if (job != NULL)
	{
		printf("Error job_list_to_start_from_history->head != NULL not dealt with in save state!\n");
		exit(1);
	}
	
	job = job_list->head;
	fprintf(f, "\njob_list:\n");
	while (job != NULL)
	{
		fprintf(f, "unique_id: %d, subtime: %d, delay: %d, walltime: %d, cores: %d, data: %d, data_size: %f, index_node_list: %d, start_time: %d, end_time: %d, end_before_walltime: %d, node_used: null, cores_used: null, transfer_time: %d, waiting_for_a_load_time: %d, workload: %d, start_time_from_history: %d, node_from_history: %d, user: %d\n", job->unique_id, job->subtime, job->delay, job->walltime, job->cores, job->data, job->data_size, job->index_node_list, job->start_time, job->end_time, job->end_before_walltime, job->transfer_time, job->waiting_for_a_load_time, job->workload, job->start_time_from_history, job->node_from_history, job->user);
		job = job->next;
	}
	
	job = new_job_list->head;
	fprintf(f, "\nnew_job_list:\n");
	while (job != NULL)
	{
		fprintf(f, "unique_id: %d, subtime: %d, delay: %d, walltime: %d, cores: %d, data: %d, data_size: %f, index_node_list: %d, start_time: %d, end_time: %d, end_before_walltime: %d, node_used: null, cores_used: null, transfer_time: %d, waiting_for_a_load_time: %d, workload: %d, start_time_from_history: %d, node_from_history: %d, user: %d\n", job->unique_id, job->subtime, job->delay, job->walltime, job->cores, job->data, job->data_size, job->index_node_list, job->start_time, job->end_time, job->end_before_walltime, job->transfer_time, job->waiting_for_a_load_time, job->workload, job->start_time_from_history, job->node_from_history, job->user);
		job = job->next;
	}
	
	job = scheduled_job_list->head;
	fprintf(f, "\nscheduled_job_list:\n");
	while (job != NULL)
	{
		fprintf(f, "unique_id: %d, subtime: %d, delay: %d, walltime: %d, cores: %d, data: %d, data_size: %f, index_node_list: %d, start_time: %d, end_time: %d, end_before_walltime: %d, node_used: null, cores_used: null, transfer_time: %d, waiting_for_a_load_time: %d, workload: %d, start_time_from_history: %d, node_from_history: %d, user: %d\n", job->unique_id, job->subtime, job->delay, job->walltime, job->cores, job->data, job->data_size, job->index_node_list, job->start_time, job->end_time, job->end_before_walltime, job->transfer_time, job->waiting_for_a_load_time, job->workload, job->start_time_from_history, job->node_from_history, job->user);
		job = job->next;
	}
	
	job = running_jobs->head;
	fprintf(f, "\nrunning_jobs:\n");
	while (job != NULL)
	{
		fprintf(f, "unique_id: %d, subtime: %d, delay: %d, walltime: %d, cores: %d, data: %d, data_size: %f, index_node_list: %d, start_time: %d, end_time: %d, end_before_walltime: %d, node_used: %d, cores_used: [ ", job->unique_id, job->subtime, job->delay, job->walltime, job->cores, job->data, job->data_size, job->index_node_list, job->start_time, job->end_time, job->end_before_walltime, job->node_used->unique_id);
		for (i = 0; i < job->cores - 1; i++)
		{
			fprintf(f, "%d, ", job->cores_used[i]);
		}
		fprintf(f, "%d ], ", job->cores_used[job->cores - 1]);
		fprintf(f, "transfer_time: %d, waiting_for_a_load_time: %d, workload: %d, start_time_from_history: %d, node_from_history: %d, user: %d\n", job->transfer_time, job->waiting_for_a_load_time, job->workload, job->start_time_from_history, job->node_from_history, job->user);
		job = job->next;
	}
	
	struct To_Print* job_to_print = jobs_to_print_list->head;
	fprintf(f, "\njob_to_print:\n");
	while (job_to_print != NULL)
	{
		fprintf(f, "job_unique_id: %d, job_subtime: %d, time: %d, time_used: %d, transfer_time: %d, job_start_time: %d, job_end_time: %d, job_cores: %d, waiting_for_a_load_time: %d, empty_cluster_time: %f, data_type: %d, job_data_size: %f, upgraded: %d, user: %d, input_file: %d\n", job_to_print->job_unique_id, job_to_print->job_subtime, job_to_print->time, job_to_print->time_used, job_to_print->transfer_time, job_to_print->job_start_time, job_to_print->job_end_time, job_to_print->job_cores, job_to_print->waiting_for_a_load_time, job_to_print->empty_cluster_time, job_to_print->data_type, job_to_print->job_data_size, job_to_print->upgraded, job_to_print->user, job_to_print->input_file);
		job_to_print = job_to_print->next;
	}
	
	struct Next_Time* next_start_time = start_times->head;
	fprintf(f, "\nnext_start_time:\n");
	while (next_start_time != NULL)
	{
		fprintf(f, "time: %d\n", next_start_time->time);
		next_start_time = next_start_time->next;
	}

	struct Next_Time* next_end_time = end_times->head;
	fprintf(f, "\nnext_end_time:\n");
	while (next_end_time != NULL)
	{
		fprintf(f, "time: %d\n", next_end_time->time);
		next_end_time = next_end_time->next;
	}
	
	fprintf(f, "\nend_of_file\n");

	printf("Successfully saved state.\n"); fflush(stdout);
	fclose(f);
}

/** Resume execution **/
void resume_state(int* t, int* old_finished_jobs, int* next_submit_time, char* input_job_file)
{
	job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	job_list->head = NULL;
	job_list->tail = NULL;
	job_list_to_start_from_history = (struct Job_List*) malloc(sizeof(struct Job_List));
	job_list_to_start_from_history->head = NULL;
	job_list_to_start_from_history->tail = NULL;
	scheduled_job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	scheduled_job_list->head = NULL;
	scheduled_job_list->tail = NULL;
	running_jobs = (struct Job_List*) malloc(sizeof(struct Job_List));
	running_jobs->head = NULL;
	running_jobs->tail = NULL;
	new_job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	new_job_list->head = NULL;
	new_job_list->tail = NULL;
		
	char* to_open = malloc(120*sizeof(char));
	strcpy(to_open, "outputs/saved_state_");
	strncat(to_open, input_job_file + 32, 5);
	strcat(to_open, "-");
	strncat(to_open, input_job_file + 44, 5);
	strcat(to_open, "_");
	strcat(to_open, scheduler);
	printf("File to resume: %s\n", to_open); fflush(stdout);
	FILE *f = fopen(to_open, "r");
	free(to_open);
	
	int i = 0;
	char str[100];
	int end_before_walltime_bool_converter = 0;
	
	if(!fscanf(f, "%s %d\n", str, t))
	{
		perror("Error if(!fscanf t");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT		
	printf("%s %d\n", str, *t);	
	#endif
		
	if(!fscanf(f, "%s %d\n", str, next_submit_time))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("%s %d\n", str, *next_submit_time);
	#endif
	
	if(!fscanf(f, "%s %d\n", str, &planned_or_ratio))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("%s %d\n", str, planned_or_ratio);
	#endif
	
	if(!fscanf(f, "%s %d\n", str, &constraint_on_sizes))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &nb_cores))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &nb_job_to_evaluate))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &nb_data_reuse))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("%s %d\n", str, nb_job_to_evaluate);
	#endif
	
	if(!fscanf(f, "%s %d\n", str, &finished_jobs))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("%s %d\n", str, finished_jobs);
	#endif
	
	if(!fscanf(f, "%s %d\n", str, old_finished_jobs))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("%s %d\n", str, *old_finished_jobs);
	#endif
	
	if(!fscanf(f, "%s %d\n", str, &total_number_jobs))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &total_number_nodes))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &running_cores))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &running_nodes))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &nb_job_to_schedule))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("%s %d\n", str, nb_job_to_schedule);
	#endif
	
	if(!fscanf(f, "%s %d\n", str, &nb_cores_to_schedule))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
		
	if(!fscanf(f, "%s %d\n", str, &total_queue_time))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d\n", str, &first_subtime_day_0))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	//~ printf("%d\n", first_subtime_day_0);
	
	if(!fscanf(f, "%s %d\n", str, &global_nb_non_available_cores_at_time_t))
	{
		perror("Error if(!fscanf global_nb_non_available_cores_at_time_t");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld", str, &Planned_Area[0][0], &Planned_Area[0][1], &Planned_Area[0][2], &Planned_Area[1][0], &Planned_Area[1][1], &Planned_Area[1][2], &Planned_Area[2][0], &Planned_Area[2][1], &Planned_Area[2][2]))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	if(!fscanf(f, "%s %d %d %d", str, &number_node_size[0], &number_node_size[1], &number_node_size[2]))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}

	if(!fscanf(f, "%s %d", str, &busy_cluster))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("busy_cluster: %d\n", busy_cluster);
	#endif
	
	if(!fscanf(f, "%s %d", str, &backfill_mode))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("backfill_mode: %d\n", backfill_mode);
	#endif
	
	if(!fscanf(f, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld", str, &Allocated_Area[0][0], &Allocated_Area[0][1], &Allocated_Area[0][2], &Allocated_Area[1][0], &Allocated_Area[1][1], &Allocated_Area[1][2], &Allocated_Area[2][0], &Allocated_Area[2][1], &Allocated_Area[2][2]))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}

	if(!fscanf(f, "%s %d", str, &nb_job_to_evaluate_started))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	#ifdef PRINT
	printf("nb_job_to_evaluate_started: %d\n", nb_job_to_evaluate_started);
	#endif
	
	// Nodes
	if(!fscanf(f, "%s", str)) // node_list:
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	
	node_list = (struct Node_List**) malloc(3*sizeof(struct Node_List));
	for (i = 0; i < 3; i++)
	{
		node_list[i] = (struct Node_List*) malloc(sizeof(struct Node_List));
		node_list[i]->head = NULL;
		node_list[i]->tail = NULL;
	}
	
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf str");
		exit(EXIT_FAILURE);
	}
	
	char str1[100];
	char str2[100];
	char str3[100];
	char str4[100];
	char str5[100];
	char str6[100];
	char str7[100];
	
	//~ printf("str before node id: %s\n", str);
	while (strcmp(str, "job_list:") != 0)
	{
		//~ printf("new node\n"); fflush(stdout);
		struct Node *new_node = (struct Node*) malloc(sizeof(struct Node));
		if(!fscanf(f, "%d %s %s %d %s %s %f %s %s %s", &new_node->unique_id, str1, str2, &new_node->memory, str3, str4, &new_node->bandwidth, str5, str6, str7))
		//~ new_node->memory = 128;
		//~ new_node->bandwidth = 0.100000;
		//~ if(!fscanf(f, "%d %s %s %d %s %s %f %s %s %s", &new_node->unique_id, str1, str2, &new_node->memory, str3, str4, &new_node->bandwidth, str5, str6, str7))
		{
			printf("%d %d %f\n", new_node->unique_id, new_node->memory, new_node->bandwidth);
			printf("%s\n", str1);
			printf("%s\n", str2);
			printf("%s\n", str3);
			printf("%s\n", str4);
			printf("%s\n", str5);
			printf("%s\n", str6);
			printf("%s\n", str7);
			perror("Error if(!fscanf debut node list");
			exit(EXIT_FAILURE);
		}
		
		#ifdef PRINT
		printf("%d %d %f\n", new_node->unique_id, new_node->memory, new_node->bandwidth);
		#endif
		
		new_node->data = (struct Data_List*) malloc(sizeof(struct Data_List));
		new_node->data->head = NULL;
		new_node->data->tail = NULL;
		
		if(!fscanf(f, "%s", str))
		{
			perror("Error if(!fscanf -2");
			exit(EXIT_FAILURE);
		}
	
		while (strcmp(str, "],") != 0)
		{
			struct Data *new_data = (struct Data*) malloc(sizeof(struct Data));
			new_data->next = NULL;
			
			#ifndef DATA_PERSISTENCE
			if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %d %s %s %s %s %f", &new_data->unique_id, str, str, &new_data->start_time, str ,str, &new_data->end_time, str, str, &new_data->nb_task_using_it, str, str, str, str, &new_data->size))
			{
				//~ printf("%d\n", new_data->unique_id);
				//~ printf("%f\n", new_data->size);
				perror("Error if(!fscanf -1");
				exit(EXIT_FAILURE);
			}
			//~ printf("%d\n", new_data->unique_id);
			//~ printf("%f\n", new_data->size);
			#else
			if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %s %s %f", &new_data->unique_id, str, str, &new_data->start_time, str ,str, &new_data->end_time, str, str, str, str, &new_data->size))
			{
				perror("Error if(!fscanf 0");
				exit(EXIT_FAILURE);
			}
			#endif
			
			new_data->intervals = (struct Interval_List*) malloc(sizeof(struct Interval_List));
			new_data->intervals->head = NULL;
			
			insert_tail_data_list(new_node->data, new_data);
			if(!fscanf(f, "%s", str))
			{
				perror("Error if(!fscanf 1");
				exit(EXIT_FAILURE);
			}
			//~ printf("k\n");
		}
		
		int running_job_bool_converter = false;
		if(!fscanf(f, "%s %s", str, str))
		{
			perror("Error if(!fscanf 2");
			exit(EXIT_FAILURE);
		}
		new_node->cores = (struct Core**) malloc(20*sizeof(struct Core));
		for (i = 0; i < 20; i++)
		{
			//~ printf("i\n");
			new_node->cores[i] = (struct Core*) malloc(sizeof(struct Core));
			if(!fscanf(f, "%s %d %s %s %d %s %s %d %s %s %d", str, &new_node->cores[i]->unique_id, str, str, &new_node->cores[i]->available_time, str, str, &running_job_bool_converter, str, str, &new_node->cores[i]->running_job_end))
			{
				perror("Error if(!fscanf 3");
				exit(EXIT_FAILURE);
			}
			//~ printf("core %d\n", new_node->cores[i]->unique_id);
			if (running_job_bool_converter == 0)
			{
				new_node->cores[i]->running_job = false;
			}
			else
			{
				new_node->cores[i]->running_job = true;
			}
		}
		if(!fscanf(f, "%s", str)) // ],
		{
			perror("Error if(!fscanf 4");
			exit(EXIT_FAILURE);
		}
		
		//~ #ifdef PRINT
		//~ print_cores_in_specific_node(new_node);
		//~ #endif
		
		new_node->number_cores_in_a_hole = 0;
		new_node->cores_in_a_hole = malloc(sizeof(*new_node->cores_in_a_hole));
		new_node->cores_in_a_hole->head = NULL;
		new_node->cores_in_a_hole->tail = NULL;
		
		#ifndef DATA_PERSISTENCE
		//~ if(!fscanf(f, "%s %d %s %s %d %s %s %d %s %s %s %s %s %d", str, &new_node->n_available_cores, str, str, &new_node->index_node_list, str, str, &new_node->number_cores_in_a_hole, str, str, str, str, str, &new_node->end_of_file_load))
		//~ printf("la\n");
		if(!fscanf(f, "%s %d %s %s %d %s %s %d %s %s %s", str, &new_node->n_available_cores, str, str, &new_node->index_node_list, str, str, &new_node->number_cores_in_a_hole, str, str, str))
		{
			perror("Error if(!fscanf before cores in a hole");
			exit(EXIT_FAILURE);
		}
		//~ printf("%s\n", str);
		int lll = 0;
		for (lll = 0; lll < new_node->number_cores_in_a_hole; lll++)
		{
			//~ printf("lll: %d\n", lll);
			struct Core_in_a_hole* new_core_in_a_hole = malloc(sizeof(struct Core_in_a_hole));
			if(!fscanf(f, "%s %d %s %s %d", str, &new_core_in_a_hole->unique_id, str, str, &new_core_in_a_hole->start_time_of_the_hole))
			{
				perror("Error if(!fscanf core in a hole");
				exit(EXIT_FAILURE);
			}
			new_core_in_a_hole->next = NULL;
			if (new_node->cores_in_a_hole == NULL)
			{
							initialize_cores_in_a_hole(new_node->cores_in_a_hole, new_core_in_a_hole);
						}
						else
						{
							if (backfill_mode == 2) /* Favorise les jobs backfill car se met sur le coeurs qui a le temps le plus petit possible. */
							{
								insert_cores_in_a_hole_list_sorted_increasing_order(new_node->cores_in_a_hole, new_core_in_a_hole);
							}
							else
							{
								insert_cores_in_a_hole_list_sorted_decreasing_order(new_node->cores_in_a_hole, new_core_in_a_hole);
							}
						}
		}
		if(!fscanf(f, "%s %s %d", str, str, &new_node->end_of_file_load))
		{
				perror("Error if(!fscanf core in a hole end get ]");
				exit(EXIT_FAILURE);
		}
		//~ printf("%s %d\n", str, new_node->end_of_file_load);
		#else
		if(!fscanf(f, "%s %d %s %s %d %s %s %d %s %s %s %s %d %s %s %s", str, &new_node->n_available_cores, str, str, &new_node->index_node_list, str, str, &new_node->number_cores_in_a_hole, str, str, str, str, &new_node->data_occupation, str, str, str))
		{
			perror("Error if(!fscanf after cores in a hole");
			exit(EXIT_FAILURE);
		}
		#endif
		
		#ifdef DATA_PERSISTENCE
		new_node->temp_data = malloc(sizeof(*new_node->temp_data));
		new_node->temp_data->head = NULL;
		new_node->temp_data->tail = NULL;
		#endif
		
		//~ A FAIRE
		//~ fprintf(f, "end_of_file_load: %d\n", n->end_of_file_load);
		//~ new_node->end_of_file_load = end_of_file_load
		
		new_node->next = NULL;
		if (new_node->memory == 128)
		{
			insert_tail_node_list(node_list[0], new_node);
		}
		else if (new_node->memory == 256)
		{
			insert_tail_node_list(node_list[1], new_node);
		}
		else if (new_node->memory == 1024)
		{
			insert_tail_node_list(node_list[2], new_node);
		}
		else
		{
			printf("Error cluster\n");
			exit(EXIT_FAILURE);
		}
		
		if(!fscanf(f, "%s", str))
		{
			perror("Error if(!fscanf end");
			exit(EXIT_FAILURE);
		}
		//~ printf("last: %s\n", str);
	}
	//~ printf("la\n");
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	//~ printf("%s\n", str);
	while (strcmp(str, "new_job_list:") != 0)
	{
		#ifdef PRINT
		printf("Add a job in job_list\n");
		#endif
		
		struct Job* new_job = (struct Job*) malloc(sizeof(struct Job));
		if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %f %s %s %d %s %s %d %s %s %d %s %s %d %s %s %s %s %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d", &new_job->unique_id, str, str, &new_job->subtime, str, str, &new_job->delay, str, str, &new_job->walltime, str, str, &new_job->cores, str, str, &new_job->data, str, str, &new_job->data_size, str, str, &new_job->index_node_list, str, str, &new_job->start_time, str, str, &new_job->end_time, str, str, &end_before_walltime_bool_converter, str, str, str, str, str, str, &new_job->transfer_time, str, str, &new_job->waiting_for_a_load_time, str, str, &new_job->workload, str, str, &new_job->start_time_from_history, str, str, &new_job->node_from_history, str, str, &new_job->user))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
		//~ printf("la2\n");
		if (end_before_walltime_bool_converter == 0)
		{
			new_job->end_before_walltime = false;
		}
		else
		{
			new_job->end_before_walltime = true;
		}
		//~ printf("la2\n");
		new_job->node_used = (struct Node*) malloc(sizeof(struct Node));
		new_job->node_used = NULL;
		new_job->cores_used = (int*) malloc(new_job->cores*sizeof(int));
		new_job->next = NULL;
		//~ printf("la2\n");
		#ifdef PRINT
		printf("unique_id: %d, subtime: %d, delay: %d, walltime: %d, cores: %d, data: %d, data_size: %f, index_node_list: %d start_time: %d end_time: %d end_before_walltime: %d transfer_time: %d waiting_for_a_load_time: %d workload: %d start_time_from_history: %d node_from_history: %d\n", new_job->unique_id, new_job->subtime, new_job->delay, new_job->walltime, new_job->cores, new_job->data, new_job->data_size, new_job->index_node_list, new_job->start_time, new_job->end_time, new_job->end_before_walltime, new_job->transfer_time, new_job->waiting_for_a_load_time, new_job->workload, new_job->start_time_from_history, new_job->node_from_history);
		#endif
		//~ printf("la2\n");
		insert_tail_job_list(job_list, new_job);
		//~ printf("la2\n");
		//~ printf("la2\n);
		if(!fscanf(f, "%s", str))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
		//~ printf("last: %s\n", str);
	}
	
	//~ #ifdef PRINT
	//~ printf("Job list:\n");
	//~ print_job_list(job_list->head);
	//~ printf("\n");
	//~ #endif
	
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	while (strcmp(str, "scheduled_job_list:") != 0)
	{
		//~ #ifdef PRINT
		//~ printf("Add a job in new_job_list\n");
		//~ #endif
		
		struct Job* new_job = (struct Job*) malloc(sizeof(struct Job));
		if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %f %s %s %d %s %s %d %s %s %d %s %s %d %s %s %s %s %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d", &new_job->unique_id, str, str, &new_job->subtime, str, str, &new_job->delay, str, str, &new_job->walltime, str, str, &new_job->cores, str, str, &new_job->data, str, str, &new_job->data_size, str, str, &new_job->index_node_list, str, str, &new_job->start_time, str, str, &new_job->end_time, str, str, &end_before_walltime_bool_converter, str, str, str, str, str, str, &new_job->transfer_time, str, str, &new_job->waiting_for_a_load_time, str, str, &new_job->workload, str, str, &new_job->start_time_from_history, str, str, &new_job->node_from_history, str, str, &new_job->user))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
		
		if (end_before_walltime_bool_converter == 0)
		{
			new_job->end_before_walltime = false;
		}
		else
		{
			new_job->end_before_walltime = true;
		}
		new_job->node_used = (struct Node*) malloc(sizeof(struct Node));
		new_job->node_used = NULL;
		new_job->cores_used = (int*) malloc(new_job->cores*sizeof(int));
		new_job->next = NULL;
				
		insert_tail_job_list(new_job_list, new_job);
		
		if(!fscanf(f, "%s", str))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
	}
		
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	while (strcmp(str, "running_jobs:") != 0)
	{		
		struct Job* new_job = (struct Job*) malloc(sizeof(struct Job));
		if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %f %s %s %d %s %s %d %s %s %d %s %s %d %s %s %s %s %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d", &new_job->unique_id, str, str, &new_job->subtime, str, str, &new_job->delay, str, str, &new_job->walltime, str, str, &new_job->cores, str, str, &new_job->data, str, str, &new_job->data_size, str, str, &new_job->index_node_list, str, str, &new_job->start_time, str, str, &new_job->end_time, str, str, &end_before_walltime_bool_converter, str, str, str, str, str, str, &new_job->transfer_time, str, str, &new_job->waiting_for_a_load_time, str, str, &new_job->workload, str, str, &new_job->start_time_from_history, str, str, &new_job->node_from_history, str, str, &new_job->user))
			{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
		
		if (end_before_walltime_bool_converter == 0)
		{
			new_job->end_before_walltime = false;
		}
		else
		{
			new_job->end_before_walltime = true;
		}
		new_job->node_used = (struct Node*) malloc(sizeof(struct Node));
		new_job->node_used = NULL;
		new_job->cores_used = (int*) malloc(new_job->cores*sizeof(int));
		new_job->next = NULL;
				
		insert_tail_job_list(scheduled_job_list, new_job);
		
		if(!fscanf(f, "%s", str))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
	}
	
	//~ #ifdef PRINT
	//~ printf("Scheduled job list:\n");
	//~ print_job_list(scheduled_job_list->head);
	//~ printf("\n");
	//~ #endif
	
	struct Node* n = (struct Node*) malloc(sizeof(struct Node));
	int node_used_id = 0;
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	while (strcmp(str, "job_to_print:") != 0)
	{
		//~ #ifdef PRINT
		//~ printf("Add a job in running_jobs\n");
		//~ #endif
		
		struct Job* new_job = (struct Job*) malloc(sizeof(struct Job));
		if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %f %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d", &new_job->unique_id, str, str, &new_job->subtime, str, str, &new_job->delay, str, str, &new_job->walltime, str, str, &new_job->cores, str, str, &new_job->data, str, str, &new_job->data_size, str, str, &new_job->index_node_list, str, str, &new_job->start_time, str, str, &new_job->end_time, str, str, &end_before_walltime_bool_converter, str, str, &node_used_id))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
		
		// Node used
		new_job->node_used = (struct Node*) malloc(sizeof(struct Node));
		for (i = 0; i < 3; i++)
		{
			n = node_list[i]->head;
			while (n != NULL)
			{
				if (n->unique_id == node_used_id)
				{
					new_job->node_used = n;
					i = 3;
					break;
				}
				n = n->next;
			}
		}
		
		//~ #ifdef PRINT
		//~ printf("Node used: %d\n", new_job->node_used->unique_id);
		//~ #endif
		
		new_job->cores_used = (int*) malloc(new_job->cores*sizeof(int));
		if(!fscanf(f, "%s %s %s", str, str, str))
		{
			perror("Error if(!fscanf");
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < new_job->cores; i++)
		{
			if(!fscanf(f, "%d", &new_job->cores_used[i]))
			{
				perror("Error if(!fscanf");
				exit(EXIT_FAILURE);
			}
			if(!fscanf(f, "%s", str))
			{
				perror("Error if(!fscanf");
				exit(EXIT_FAILURE);
			}
		}
		
		if(!fscanf(f, "%s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d", str, &new_job->transfer_time, str, str, &new_job->waiting_for_a_load_time, str, str, &new_job->workload, str, str, &new_job->start_time_from_history, str, str, &new_job->node_from_history, str, str, &new_job->user))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}

		if (end_before_walltime_bool_converter == 0)
		{
			new_job->end_before_walltime = false;
		}
		else
		{
			new_job->end_before_walltime = true;
		}

		new_job->next = NULL;
				
		insert_tail_job_list(running_jobs, new_job);
		
		if(!fscanf(f, "%s", str))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	}
		
	jobs_to_print_list = malloc(sizeof(*jobs_to_print_list));
	jobs_to_print_list->head = NULL;
	jobs_to_print_list->tail = NULL;
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	while (strcmp(str, "next_start_time:") != 0)
	{
		//~ #ifdef PRINT
		//~ printf("Add a job in job_to_print\n");
		//~ #endif
		
		struct To_Print* new_job = (struct To_Print*) malloc(sizeof(struct To_Print));

		if(!fscanf(f, "%d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %d %s %s %f %s %s %d %s %s %f %s %s %d %s %s %d %s %s %d", &new_job->job_unique_id, str, str, &new_job->job_subtime, str, str, &new_job->time, str, str, &new_job->time_used, str, str, &new_job->transfer_time, str, str, &new_job->job_start_time, str, str, &new_job->job_end_time, str, str, &new_job->job_cores, str, str, &new_job->waiting_for_a_load_time, str, str, &new_job->empty_cluster_time, str, str, &new_job->data_type, str, str, &new_job->job_data_size, str, str, &new_job->upgraded, str, str, &new_job->user, str, str, &new_job->input_file))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
		
		//~ #ifdef PRINT
		//~ printf("%d %d %d %d %d %d %d %d %d %f %d %f %d", new_job->job_unique_id, new_job->job_subtime, new_job->time, new_job->time_used, new_job->transfer_time, new_job->job_start_time, new_job->job_end_time, new_job->job_cores, new_job->waiting_for_a_load_time, new_job->empty_cluster_time, new_job->data_type, new_job->job_data_size, new_job->upgraded);
		//~ #endif
		
		new_job->next = NULL;
				
		insert_tail_to_print_list(jobs_to_print_list, new_job);
		
		if(!fscanf(f, "%s", str))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	}
	//~ #ifdef PRINT
	//~ printf("\n");
	//~ #endif
	
	start_times = malloc(sizeof(*start_times));
	start_times->head = NULL;
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	int new_time = 0;
	while (strcmp(str, "next_end_time:") != 0)
	{
		//~ #ifdef PRINT
		//~ printf("Add a time in next_start_time\n");
		//~ #endif
		
		if(!fscanf(f, "%d", &new_time))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
						
		insert_next_time_in_sorted_list(start_times, new_time);
		
		if(!fscanf(f, "%s", str))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	}
	//~ #ifdef PRINT
	//~ print_time_list(start_times->head, 0);
	//~ printf("\n");
	//~ #endif
	
	end_times = malloc(sizeof(*end_times));
	end_times->head = NULL;
	if(!fscanf(f, "%s", str))
	{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	while (strcmp(str, "end_of_file") != 0)
	{
		//~ #ifdef PRINT
		//~ printf("Add a time in next_end_time\n");
		//~ #endif
		
		if(!fscanf(f, "%d", &new_time))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
						
		insert_next_time_in_sorted_list(end_times, new_time);
		
		if(!fscanf(f, "%s", str))
		{
		perror("Error if(!fscanf");
		exit(EXIT_FAILURE);
	}
	}
	//~ #ifdef PRINT
	//~ print_time_list(end_times->head, 1);
	//~ printf("\n");
	//~ #endif
	
	printf("Successfully read state.\n"); fflush(stdout);
	fclose(f);
}
