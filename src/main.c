#include <main.h>

int first_node_size_to_choose_from;
int last_node_size_to_choose_from;

char* input_job_file;

int planned_or_ratio; /* O = planned, 1 = ratio */
int constraint_on_sizes;
int nb_cores;
int nb_job_to_evaluate;
int finished_jobs;
int total_number_jobs;
int total_number_nodes;
struct Job_List* job_list; /* All jobs not available yet */
struct Job_List* new_job_list; /* New available jobs */
struct Job_List* job_list_to_start_from_history; /* With -2 and before start */
struct Job_List* scheduled_job_list; /* Scheduled or available */
//~ struct Job_List* new_job_list; /* Scheduled or available */
struct Job_List* running_jobs; /* Started */
struct Node_List** node_list;
struct To_Print_List* jobs_to_print_list;
int running_cores;
int running_nodes;
#ifdef PRINT_CLUSTER_USAGE
int running_cores_from_workload_1;
int running_nodes_workload_1;
int mixed_mode;
#endif
int total_queue_time;
int first_subtime_day_0;
int nb_job_to_schedule;
int nb_cores_to_schedule;
char* scheduler;
char* output_file;
struct Next_Time_List* end_times;
struct Next_Time_List* start_times;
int nb_job_to_evaluate_started;
long long Allocated_Area[3][3];
long long Planned_Area[3][3];
int number_node_size[3];
int busy_cluster;
int backfill_mode;
#ifdef PLOT_STATS
int number_of_backfilled_jobs;
int number_of_tie_breaks_before_computing_evicted_files_fcfs_score;
int total_number_of_scores_computed;
int data_persistence_exploited;
#endif
int biggest_hole;
int biggest_hole_unique_id;
int global_nb_non_available_cores_at_time_t;
int nb_data_reuse;
int busy_cluster_threshold;

#ifdef NB_HOUR_MAX
int nb_h_scheduled;
#endif

int nb_call_finished_jobs;
int nb_call_new_jobs;


int main(int argc, char *argv[])
{
	#ifdef NB_HOUR_MAX
	nb_h_scheduled = 1;
	#endif
	
	nb_data_reuse = 0;
	if (argc != 8 && argc != 9 && argc != 10)
	{
		printf("Error: args must be 8, 9 or 10!\n");
		exit(EXIT_FAILURE);
	}
	
	bool need_to_resume_state = false;
	#ifdef SAVE
	bool need_to_save_state = false;
	int time_to_save = 0;
			
	/* If you add save or resume as the last argument */
	if (argc > 8)
	{
		if (strcmp(argv[8], "save") == 0) 
		{
			need_to_save_state = true;
			time_to_save = atoi(argv[9]);
			printf("Save after %d jobs.\n", time_to_save); fflush(stdout);
		}
		else if (strcmp(argv[8], "save_and_resume") == 0)
		{
			need_to_resume_state = true;
			need_to_save_state = true;
			time_to_save = atoi(argv[9]);
			printf("Time to save_and_resume is %d.\n", time_to_save); fflush(stdout);
		}
		else if (strcmp(argv[8], "resume") == 0)
		{
			need_to_resume_state = true;
			if (argc > 9)
			{
				printf("Error: no arg must be after resume.\n"); fflush(stdout);
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			printf("Error: argv[8] must be save or resume.\n");
			exit(EXIT_FAILURE);
		}
	}
	#endif
	
	new_job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	new_job_list->head = NULL;
	new_job_list->tail = NULL;
	
	#ifdef PLOT_STATS
	number_of_backfilled_jobs = 0;
	number_of_tie_breaks_before_computing_evicted_files_fcfs_score = 0;
	total_number_of_scores_computed = 0;
	data_persistence_exploited = 0;
	#endif
	
	/* random seed init. */
	busy_cluster = 0; /* Not busy initially */
	srand(time(NULL));
	planned_or_ratio = 0;
	/* Init global variables */
	finished_jobs = 0;
	total_number_jobs = 0;
	running_cores = 0;
	running_nodes = 0;
	#ifdef PRINT_CLUSTER_USAGE
	running_nodes_workload_1 = 0;
	mixed_mode = 0;
	int nodes_loading_a_file = 0;
	int cores_loading_a_file = 0;
	#endif
	
	first_node_size_to_choose_from = 0;
	last_node_size_to_choose_from = 2;
	
	nb_job_to_evaluate_started = 0;
	total_queue_time = 0;
	struct Job* job_pointer = (struct Job*) malloc(sizeof(struct Job));
	struct Job* temp = (struct Job*) malloc(sizeof(struct Job));
	
	end_times = malloc(sizeof(*end_times));
	end_times->head = NULL;
	start_times = malloc(sizeof(*start_times));
	start_times->head = NULL;
	jobs_to_print_list = malloc(sizeof(*jobs_to_print_list));
	jobs_to_print_list->head = NULL;
	jobs_to_print_list->tail = NULL;
	
	int old_finished_jobs = 0;
	
	/** Args **/
	input_job_file = argv[1];
	char* input_node_file = argv[2];
	scheduler = argv[3]; /* malloc ? */
	constraint_on_sizes = atoi(argv[4]); /* To add or remove the constraint that some jobs can't be executed on certain nodes. 0 for no constraint, 1 for constraint, 2 for constraint but we don't consider transfer time. 3 for constraint and you can only execute on your specific size. */
	output_file = argv[5];
	if (output_file == NULL)
	{
		printf("Need file output\n");
		exit(1);
	}
	
	backfill_mode = atoi(argv[6]);
	if (backfill_mode < 0 || backfill_mode > 3)
	{
		printf("Error, backfill_mode = %d not dealt with.\n", backfill_mode);
		exit(1);
	}
	
	#ifdef DATA_PERSISTENCE
	if (constraint_on_sizes != 0)
	{
		printf("Constraint on sizes not dealt with if it's not 0 with data persistence (because of the way I look at the memory usage of a node.\n");
		exit(1);
	}
	#endif
	
	printf("Workload: %s\n", input_job_file);
	printf("Scheduler: %s - Backfill mode: %d\n", scheduler, backfill_mode);
	if (constraint_on_sizes != 0)
	{
		printf("Error constraint on size.\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	
	/* Read cluster */
	if (need_to_resume_state == false)
	{
		read_cluster(input_node_file);
	}
	
	/* Read workload */
	if (need_to_resume_state == false)
	{
		read_workload(input_job_file, constraint_on_sizes);	
		nb_job_to_evaluate = get_nb_job_to_evaluate(job_list->head);
		first_subtime_day_0 = get_first_time_day_0(job_list->head);
	}
	
	#ifdef PRINT_CLUSTER_USAGE
	write_in_file_first_times_all_day(job_list->head, first_subtime_day_0);
	#endif

	int next_submit_time = first_subtime_day_0;
	int t = first_subtime_day_0;

	/* First start jobs from rackham's history. First need to sort it by start time */
	if (need_to_resume_state == false) 
	{
		if (job_list_to_start_from_history->head != NULL)
		{
			get_state_before_day_0_scheduler(job_list_to_start_from_history->head, node_list, t);
		}
	}

	/* getting the number of jobs we need to schedule */
	if (need_to_resume_state == false) {
		job_pointer = scheduled_job_list->head;
		nb_job_to_schedule = 0;
		nb_cores_to_schedule = 0;
		while(job_pointer != NULL)
		{
			nb_job_to_schedule += 1;
			nb_cores_to_schedule += job_pointer->cores;
			job_pointer = job_pointer->next;
		}
		/* Just for -2 jobs here */
		if (scheduled_job_list->head != NULL)
		{
			start_jobs(t, scheduled_job_list->head);
		}
		//~ printf("Start jobs before day 0 done.\n");
	}

	#ifdef PRINT_SCORES_DATA
	FILE* f_fcfs_score = fopen("outputs/Scores_data.txt", "w");
	if (!f_fcfs_score)
	{
		perror("fopen in main");
        exit(EXIT_FAILURE);
	}
	fclose(f_fcfs_score);
	#endif
	
	/* Initializings stats on choosen method. */
	#ifdef PLOT_STATS
	FILE* f_stats_choosen_method = fopen("outputs/choosen_methods.txt", "w");
	if (!f_stats_choosen_method)
	{
		perror("Error opening file outputs/choosen_methods.txt.");
		exit(EXIT_FAILURE);
	}
	fprintf(f_stats_choosen_method, "Job_id,Chosen_method\n");
	fclose(f_stats_choosen_method);
	#endif
	
	#ifdef PRINT_CLUSTER_USAGE
	/* Fichier complet */
	char* title = malloc(100*sizeof(char));
	strcpy(title, "outputs/Stats_");
	strcat(title, scheduler);
	strcat(title, ".csv");
	FILE* f_stats = fopen(title, "w");
	if (!f_stats)
	{
		perror("fopen in main");
        exit(EXIT_FAILURE);
	}
	fprintf(f_stats, "Time,Used cores,Used nodes,Scheduled jobs,Used nodes workload 1,Cores required in queue,Cores required from evaluated jobs in queue, Nodes loading a file, Used cores from workload 1, Cores loading a file\n");
	//~ free(title);
	int nb_jobs_in_queue = 0;
	int nb_cores_in_queue = 0;
	int nb_cores_from_workload_1_in_queue = 0;
	
	free(title);
	#endif
	
	int i = 0;
	int j = 0;
	int multiplier_file_to_load = 0;
	int multiplier_file_evicted = 0;
	int multiplier_nb_copy = 0;
	int multiplier_area_bigger_nodes = 0;
	int adaptative_multiplier = 0; /* 0 = no, 1 = yes */
	int penalty_on_job_sizes = 0; /* 0 = no, 1 = yes */
	int backfill_big_node_mode = 0;
	bool use_bigger_nodes = true;
	float (*Ratio_Area)[3] = malloc(sizeof(float[3][3]));
	int start_immediately_if_EAT_is_t = 0;
	int mixed_strategy = 0;
	
	/* Getting informations for certain schedulers. */
	if (strncmp(scheduler, "Fcfs_with_a_score_x", 19) == 0 || strncmp(scheduler, "Fcfs_with_a_score_easybf_x", 26) == 0 || strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_95th_percentile_x", 54) == 0 || strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_weighted_random_x", 54) == 0 || strncmp(scheduler, "Fcfs_area_filling_with_a_score_x", 32) == 0 || strncmp(scheduler, "Fcfs_area_filling_omniscient_with_a_score_x", 43) == 0 || strncmp(scheduler, "Fcfs_area_filling_with_ratio_with_a_score_x", 43) == 0 || strncmp(scheduler, "Fcfs_area_filling_omniscient_with_ratio_with_a_score_x", 54) == 0 || strncmp(scheduler, "Fcfs_area_filling_with_ratio_7_days_earlier_with_a_score_x", 58) == 0 || strncmp(scheduler, "Fcfs_with_a_score_area_factor_x", 31) == 0 || strncmp(scheduler, "Fcfs_with_a_score_area_factor_with_omniscient_planned_area_x", 60) == 0 || strncmp(scheduler, "Fcfs_with_a_score_area_factor_with_planned_area_x", 49) == 0 || strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_gain_loss_tradeoff_x", 57) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_x", 41) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_3_x", 43) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_4_x", 43) == 0 || strncmp(scheduler, "Fcfs_with_a_score_penalty_on_big_jobs_x", 39) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_x", 34) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_not_EFT_x", 43) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_adaptative_multiplier_x", 56) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x", 53) == 0 || strncmp(scheduler, "Fcfs_with_a_score_conservativebf_x", 34) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_conservativebf_x", 49) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_conservativebf_x", 68) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_easybf_x", 60) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_easybf_x", 41) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_non_dynamic_x", 46) == 0)
	{
		if (strncmp(scheduler, "Fcfs_with_a_score_x", 19) == 0)
		{
			i = 19;
			j = 19;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_easybf_x", 26) == 0)
		{
			i = 26;
			j = 26;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_95th_percentile_x", 54) == 0)
		{
			i = 54;
			j = 54;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_weighted_random_x", 54) == 0)
		{
			i = 54;
			j = 54;
		}
		else if (strncmp(scheduler, "Fcfs_area_filling_with_a_score_x", 32) == 0)
		{
			i = 32;
			j = 32;
		}
		else if (strncmp(scheduler, "Fcfs_area_filling_omniscient_with_a_score_x", 43) == 0)
		{
			i = 43;
			j = 43;
		}
		else if (strncmp(scheduler, "Fcfs_area_filling_with_ratio_with_a_score_x", 43) == 0)
		{
			i = 43;
			j = 43;
		}
		else if (strncmp(scheduler, "Fcfs_area_filling_omniscient_with_ratio_with_a_score_x", 54) == 0)
		{
			i = 54;
			j = 54;
		}
		else if (strncmp(scheduler, "Fcfs_area_filling_with_ratio_7_days_earlier_with_a_score_x", 58) == 0)
		{
			i = 58;
			j = 58;
		}		
		else if (strncmp(scheduler, "Fcfs_with_a_score_area_factor_x", 31) == 0)
		{
			i = 31;
			j = 31;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_area_factor_with_omniscient_planned_area_x", 60) == 0)
		{
			i = 60;
			j = 60;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_area_factor_with_planned_area_x", 49) == 0)
		{
			i = 49;
			j = 49;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_gain_loss_tradeoff_x", 57) == 0)
		{
			i = 57;
			j = 57;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_x", 41) == 0)
		{
			i = 41;
			j = 41;
			adaptative_multiplier = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_3_x", 43) == 0)
		{
			i = 43;
			j = 43;
			adaptative_multiplier = 3;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_4_x", 43) == 0)
		{
			i = 43;
			j = 43;
			adaptative_multiplier = 4;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_penalty_on_big_jobs_x", 39) == 0)
		{
			i = 39;
			j = 39;
			penalty_on_job_sizes = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_x", 34) == 0)
		{
			i = 34;
			j = 34;
			mixed_strategy = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_non_dynamic_x", 46) == 0)
		{
			i = 46;
			j = 46;
			mixed_strategy = 2;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_not_EFT_x", 43) == 0)
		{
			i = 43;
			j = 43;
			mixed_strategy = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_adaptative_multiplier_x", 56) == 0)
		{
			i = 56;
			j = 56;
			adaptative_multiplier = 2;
			mixed_strategy = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x", 53) == 0)
		{
			i = 53;
			j = 53;
			start_immediately_if_EAT_is_t = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_conservativebf_x", 68) == 0)
		{
			i = 68;
			j = 68;
			start_immediately_if_EAT_is_t = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_easybf_x", 60) == 0)
		{
			i = 60;
			j = 60;
			start_immediately_if_EAT_is_t = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_conservativebf_x", 34) == 0)
		{
			i = 34;
			j = 34;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_conservativebf_x", 49) == 0)
		{
			i = 49;
			j = 49;
			mixed_strategy = 1;
		}
		else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_easybf_x", 41) == 0)
		{
			i = 41;
			j = 41;
			mixed_strategy = 1;
		}
		else
		{
			printf("Error.\n");
			exit(EXIT_FAILURE);
		}
		
		int number_of_char = 0;
		
		/* Multiplier 1 */
		while (scheduler[i] != '_')
		{
			i += 1;
			number_of_char += 1;
		}
		char to_copy1[number_of_char];
		for (j = 0; j < number_of_char; j++)
		{
			to_copy1[j] = scheduler[i - number_of_char + j];
		}
		multiplier_file_to_load = (int) strtol(to_copy1, NULL, 10);
		i = i + 2;
		
		/* Multiplier 2 */
		number_of_char = 0;
		while (scheduler[i] != '_')
		{
			i += 1;
			number_of_char += 1;
		}
		char to_copy2[number_of_char];
		for (j = 0; j < number_of_char; j++)
		{
			to_copy2[j] = scheduler[i - number_of_char + j];
		}
		multiplier_file_evicted = (int) strtol(to_copy2, NULL, 10);
		i = i + 2;
				
		/* Multiplier 3 */
		number_of_char = 0;
		while (scheduler[i] != '_')
		{
			i += 1;
			number_of_char += 1;
		}
		char to_copy3[number_of_char];
		for (j = 0; j < number_of_char; j++)
		{
			to_copy3[j] = scheduler[i - number_of_char + j];
		}
		multiplier_nb_copy = (int) strtol(to_copy3, NULL, 10);
		i = i + 2;
				
		/* Multiplier 4 */
		number_of_char = 0;
		while (scheduler[i])
		{
			i += 1;
			number_of_char += 1;
		}
		char to_copy4[number_of_char];
		for (j = 0; j < number_of_char; j++)
		{
			to_copy4[j] = scheduler[i - number_of_char + j];
		}
		multiplier_area_bigger_nodes = (int) strtol(to_copy4, NULL, 10);
						
		printf("Multiplier file to load: %d / Multiplier file evicted: %d / Multiplier nb of copy: %d / Multiplier area bigger nodes: %d / Adaptative multiplier : %d / Penalty on sizes : %d / Start immeditaly if EAT is t: %d / Mixed strategy: %d.\n", multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy, multiplier_area_bigger_nodes, adaptative_multiplier, penalty_on_job_sizes, start_immediately_if_EAT_is_t, mixed_strategy);
		
		/* Error I have sometimes when the int is not what I putted */
		if (multiplier_file_to_load > 500 || multiplier_file_evicted > 500 || multiplier_nb_copy > 500 || multiplier_area_bigger_nodes > 500)
		{
			printf("############################## Error multiplier. 500, 1, 0, 0 affected. ##############################\n");
			multiplier_file_to_load = 500;
			multiplier_file_evicted = 1;
			multiplier_nb_copy = 0;
			multiplier_area_bigger_nodes = 0;
		}
	}
	

	busy_cluster_threshold = atoi(argv[7]);
	

	int division_by_planned_area = 0;
	
	bool new_jobs = false;
	/* For the schedulers dealing with size constraint I need to sort scheduled_job_list by file size (biggest to smallest) now but
	 * also do it when new jobs are added to scheduled_job_list. */
	bool sort_by_file_size = false;
	if (
	(strcmp(scheduler, "Fcfs_big_job_first") == 0) || (strcmp(scheduler, "Fcfs_backfill_big_nodes_0_big_job_first") == 0) || (strcmp(scheduler, "Fcfs_backfill_big_nodes_1_big_job_first") == 0) || (strcmp(scheduler, "Fcfs_area_filling_big_job_first") == 0) || (strcmp(scheduler, "Fcfs_area_filling_with_ratio_big_job_first") == 0) || (strcmp(scheduler, "Fcfs_area_filling_omniscient_big_job_first") == 0) || (strcmp(scheduler, "Fcfs_area_filling_omniscient_with_ratio_big_job_first") == 0))
	{
		printf("Sorting job list by file's size.\n");
		
		/* To sort by file size for certain schedulers. */
		sort_by_file_size = true;
		sort_job_list_by_file_size(&scheduled_job_list->head);
		
		#ifdef PRINT
		printf("Job list after sort byt file's size:\n");
		print_job_list(scheduled_job_list->head);
		#endif
	}

	/* Besoin de calculer le nombre de nodes dans chaque catégorie pour certains algos. */
	int number_node_size_128_and_more = 0;
	int number_node_size_256_and_more = 0;
	int number_node_size_1024 = 0;
	if (strncmp(scheduler, "Fcfs_with_a_score_backfill_big_nodes_95th_percentile_x", 54) == 0)
	{
		for (i = 0; i <= 2; i++)
		{
			struct Node* n = node_list[i]->head;
			while (n != NULL)
			{
				number_node_size_128_and_more += 1;
				if (i == 1)
				{
					number_node_size_256_and_more += 1;
				}
				else if (i == 2)
				{
					number_node_size_256_and_more += 1;
					number_node_size_1024 += 1;
				}
				n = n->next;
			}
		}
		printf("There are %d nodes of size 128 and more, %d of size 256 and more, %d of size 1024.\n", number_node_size_128_and_more, number_node_size_256_and_more, number_node_size_1024);
	}
		
	
	#ifdef SAVE
	if (need_to_resume_state == true)
	{
		//~ printf("T = %d\n", t); fflush(stdout);
		need_to_resume_state = false;
		resume_state(&t, &old_finished_jobs, &next_submit_time, input_job_file);
		//~ printf("next end = %d\n", end_times->head->time); fflush(stdout);
				old_finished_jobs = finished_jobs;
		if (end_times->head != NULL && end_times->head->time == t)
		{
			end_jobs(running_jobs->head, t);
		}

		
			reset_cores(node_list, t);
			
			/* Reset planned starting times. */
			free_next_time_linked_list(&start_times->head);
						
			#ifdef PRINT
			printf("Reschedule.\n");
			#endif
			
			call_scheduler(scheduler, scheduled_job_list, t, use_bigger_nodes, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy, adaptative_multiplier, penalty_on_job_sizes, start_immediately_if_EAT_is_t, backfill_mode, number_node_size_128_and_more, number_node_size_256_and_more, number_node_size_1024, Ratio_Area, multiplier_area_bigger_nodes, division_by_planned_area, backfill_big_node_mode, mixed_strategy);
							
			#ifdef PRINT	
			printf("End of reschedule.\n");
			#endif
			
			/* Get started jobs. */
			if (start_times->head != NULL)
			{
				if (start_times->head->time == t)
				{
					start_jobs(t, scheduled_job_list->head);
				}
			}
	}
	#endif
		
	/** START OF SIMULATION **/
	
	nb_call_finished_jobs = 0;
	nb_call_new_jobs = 0;
	
	int last_scheduler_call = t;
	
	while (nb_job_to_evaluate != nb_job_to_evaluate_started)
	{
		#ifdef SAVE
		if (need_to_save_state == true && finished_jobs >= time_to_save) /* Avec le nb de jobs terminés */
		{
			#ifdef PLOT_SATS
			printf("Cas pas géré #ifdef PLOT_SATS avec asave_state\n");
			exit(1);
			#endif
			save_state(t, old_finished_jobs, next_submit_time, input_job_file);
			exit(1);
		}
		#endif
						
		/* Get ended job. */
		old_finished_jobs = finished_jobs;
		if (end_times->head != NULL && end_times->head->time == t)
		{
			end_jobs(running_jobs->head, t);
		}
		
		/* Get started jobs. */
		if (start_times->head != NULL)
		{
			if (start_times->head->time == t)
			{
				start_jobs(t, scheduled_job_list->head);
			}
		}
				
		new_jobs = false;
		/* Get the set of available jobs at time t */
		/* Jobs are already sorted by subtime so I can simply stop with a break */
		if (next_submit_time == t) /* We have new jobs need to schedule them. */
		{
			new_jobs = true;
			
			#ifdef PRINT
			printf("We have new jobs at time %d.\n", t);
			#endif
			
			/* Copy in scheduled_job_list jobs and delete from job_list. */
			job_pointer = job_list->head;
			while (job_pointer != NULL)
			{
				if (job_pointer->subtime <= t)
				{
					#ifdef PRINT
					printf("New job %d.\n", job_pointer->unique_id);
					#endif
					
					/* Update nb of jobs to schedule (not running but available) */
					nb_job_to_schedule += 1;
					nb_cores_to_schedule += job_pointer->cores;
					
					temp = job_pointer->next;
					if (sort_by_file_size == true)
					{
						if (old_finished_jobs == finished_jobs) /* Que des nouveaux jobs, donc liste séparé */
						{
							#ifdef PRINT
							printf("Copy in new job list\n");
							#endif
							
							copy_delete_insert_job_list_sorted_by_file_size(job_list, new_job_list, job_pointer);
						}
						else /* Il y a aussi eu des libérations, je met tout dans scheduled_job_list */
						{
							copy_delete_insert_job_list_sorted_by_file_size(job_list, scheduled_job_list, job_pointer);
						}
					}
					else
					{
						if (old_finished_jobs == finished_jobs) /* Que des nouveaux jobs, donc liste séparé */
						{
							#ifdef PRINT
							printf("Copy in new job list\n");
							#endif
							
							copy_delete_insert_job_list(job_list, new_job_list, job_pointer);
						}
						else /* Il y a aussi eu des libérations, je met tout dans scheduled_job_list */
						{
							copy_delete_insert_job_list(job_list, scheduled_job_list, job_pointer);
						}
					}
					job_pointer = temp;
				}
				else
				{
					break;
				}
			}
							
			if (job_pointer != NULL)
			{
				next_submit_time = job_pointer->subtime;
			}
			else
			{
				next_submit_time = -1;
			}
		}
		
		if (last_scheduler_call <= t - 30) /* Version ou j'attends X secondes avant de re schedule */
		{
		last_scheduler_call = t;
		
		if (old_finished_jobs < finished_jobs && scheduled_job_list->head != NULL)
		{			
			#ifdef PRINT
			printf("Core(s) liberated. Need to free them.\n"); fflush(stdout);
			#endif
			
			/* Reset all cores and jobs. */
			reset_cores(node_list, t);
			
			/* Reset planned starting times. */
			free_next_time_linked_list(&start_times->head);
						
			#ifdef PRINT
			printf("Reschedule.\n");
			#endif
			
			call_scheduler(scheduler, scheduled_job_list, t, use_bigger_nodes, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy, adaptative_multiplier, penalty_on_job_sizes, start_immediately_if_EAT_is_t, backfill_mode, number_node_size_128_and_more, number_node_size_256_and_more, number_node_size_1024, Ratio_Area, multiplier_area_bigger_nodes, division_by_planned_area, backfill_big_node_mode, mixed_strategy);
							
			#ifdef PRINT	
			printf("End of reschedule.\n");
			#endif
			
			/* Get started jobs. */
			if (start_times->head != NULL)
			{
				if (start_times->head->time == t)
				{
					start_jobs(t, scheduled_job_list->head);
				}
			}
		}
		else if (new_jobs == true) /* Pas de jobs fini mais des nouveaux jobs à schedule. */
		{			
			#ifdef PRINT
			printf("Schedule only new jobs.\n");
			#endif
			
			call_scheduler(scheduler, new_job_list, t, use_bigger_nodes, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy, adaptative_multiplier, penalty_on_job_sizes, start_immediately_if_EAT_is_t, backfill_mode, number_node_size_128_and_more, number_node_size_256_and_more, number_node_size_1024, Ratio_Area, multiplier_area_bigger_nodes, division_by_planned_area, backfill_big_node_mode, mixed_strategy);

			job_pointer = new_job_list->head;
			if (sort_by_file_size == true)
			{
				while (job_pointer != NULL)
				{
					temp = job_pointer->next;
					/* Copie des jobs de new job list dans scheduled job list */
					copy_delete_insert_job_list_sorted_by_file_size(new_job_list, scheduled_job_list, job_pointer);
					job_pointer = temp;
				}
			}
			else
			{
				while (job_pointer != NULL)
				{
					temp = job_pointer->next;
					/* Copie des jobs de new job list dans scheduled job list */
					copy_delete_insert_job_list(new_job_list, scheduled_job_list, job_pointer);
					job_pointer = temp;
				}	
			}
			
			/* Get started jobs. */
			if (start_times->head != NULL)
			{
				if (start_times->head->time == t)
				{
					start_jobs(t, scheduled_job_list->head);
				}
			}
		}
		
		}
						
		#ifdef PRINT_CLUSTER_USAGE
		get_length_job_list(scheduled_job_list->head, &nb_jobs_in_queue, &nb_cores_in_queue, &nb_cores_from_workload_1_in_queue);
				
		get_nb_nodes_and_cores_loading_a_file(node_list, t, &nodes_loading_a_file, &cores_loading_a_file);
		
		fprintf(f_stats, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", t, running_cores, running_nodes*20, nb_jobs_in_queue, running_nodes_workload_1*20, nb_cores_in_queue + 486*20, nb_cores_from_workload_1_in_queue + 486*20, nodes_loading_a_file*20, running_cores_from_workload_1, cores_loading_a_file);		
		#endif
				
		if (start_times->head != NULL && t > start_times->head->time)
		{
			printf("ERROR in main.c, next start time is %d and t is %d.\n", start_times->head->time, t); fflush(stdout);
			exit(EXIT_FAILURE);
		}
		
		/* Time is advancing. */
		t += 1;
	}
	
	#ifdef PRINT_CLUSTER_USAGE
	fclose(f_stats);
	#endif
		
	/* Delete all running and scheduled jobs */
	job_pointer = running_jobs->head;
	while (job_pointer != NULL)
	{
		struct Job* temp = job_pointer->next;
		delete_job_linked_list(running_jobs, job_pointer->unique_id);
		job_pointer = temp;
	}
	job_pointer = scheduled_job_list->head;
	while (job_pointer != NULL)
	{
		struct Job* temp = job_pointer->next;
		delete_job_linked_list(scheduled_job_list, job_pointer->unique_id);
		job_pointer = temp;
	}
	
	printf("\nComputing and writing results...\n");
	print_csv(jobs_to_print_list->head);

	return 1;
}
