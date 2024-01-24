	#include <main.h>

void call_scheduler(char* scheduler, struct Job_List* liste, int t, int use_bigger_nodes, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t, int backfill_mode, int number_node_size_128_and_more, int number_node_size_256_and_more, int number_node_size_1024, float (*Ratio_Area)[3], int multiplier_area_bigger_nodes, int division_by_planned_area, int backfill_big_node_mode, int mixed_strategy)
{	
	if (strncmp(scheduler, "Fcfs_with_a_score_x", 19) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_x", 41) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_3_x", 43) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_4_x", 43) == 0 || strncmp(scheduler, "Fcfs_with_a_score_penalty_on_big_jobs_x", 39) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_x", 53) == 0)
	{
		fcfs_with_a_score_scheduler(liste->head, node_list, t, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy, adaptative_multiplier, penalty_on_job_sizes, start_immediately_if_EAT_is_t, 0);
	}
	else if (strcmp(scheduler, "Fcfs") == 0)
	{
		fcfs_scheduler(liste->head, node_list, t, use_bigger_nodes);
	}
	else if (strncmp(scheduler, "Fcfs_with_a_score_conservativebf_x", 34) == 0 || strncmp(scheduler, "Fcfs_with_a_score_adaptative_multiplier_if_EAT_is_t_conservativebf_x", 68) == 0) /* Ok avec DATA_PERSISTENCE */
	{
		fcfs_with_a_score_conservativebf_scheduler(liste->head, node_list, t, multiplier_file_to_load, multiplier_file_evicted, adaptative_multiplier, start_immediately_if_EAT_is_t, backfill_mode, mixed_strategy);
	}
	else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_conservativebf_x", 49) == 0) /* Ok avec DATA_PERSISTENCE */
	{	
		fcfs_with_a_score_conservativebf_scheduler(liste->head, node_list, t, multiplier_file_to_load, multiplier_file_evicted, adaptative_multiplier, start_immediately_if_EAT_is_t, backfill_mode, mixed_strategy);
	}
	else if (strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_x", 34) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_adaptative_multiplier_x", 56) == 0 || strncmp(scheduler, "Fcfs_with_a_score_mixed_strategy_non_dynamic_x", 46) == 0) /* Ok avec DATA_PERSISTENCE */
	{
		fcfs_with_a_score_scheduler(liste->head, node_list, t, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy, adaptative_multiplier, penalty_on_job_sizes, start_immediately_if_EAT_is_t, mixed_strategy);
	}
	else if (strcmp(scheduler, "Fcfs_conservativebf") == 0)
	{
		fcfs_conservativebf_scheduler(liste->head, node_list, t, backfill_mode);
	}
	else
	{
		printf("Error: wrong scheduler in arguments.\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
}
