#include <main.h>

void get_state_before_day_0_scheduler(struct Job* j2, struct Node_List** n, int t)
{
	int i = 0;
	int nb_job_to_delete = 0;		
	/* Get number of node in each size category */
	struct Node *temp = (struct Node*) malloc(sizeof(struct Node));
	int* nb_node = malloc(3*sizeof(int));
	for (i = 0; i < 3; i++)
	{
		nb_node[i] = 0;
		temp = node_list[i]->head;
		while (temp != NULL)
		{
			temp = temp->next;
			nb_node[i] += 1;
		}
	}
	
	#ifdef PRINT
	printf("%d nodes of size 128, %d of size 256 and %d of size 1024.\n", nb_node[0], nb_node[1], nb_node[2]);
	#endif
	
	free(temp);

	struct Job* j = j2;
	while (j != NULL)
	{		
		int time_since_start = t - j->start_time_from_history;
		j->delay -= time_since_start;
		if(j->delay <= 0)
		{
			j->delay = 1;
		}
		j->walltime -= time_since_start;
		if(j->walltime <= 0)
		{
			j->walltime = 1;
		}
		
		struct Node *choosen_node = (struct Node*) malloc(sizeof(struct Node));
		int index_node = (j->node_from_history - 1)%(nb_node[0] + nb_node[1] + nb_node[2]);
		if (index_node >= nb_node[0])
		{
			if (index_node >= nb_node[0] + nb_node[1])
			{
				choosen_node = node_list[2]->head;
				for (i = 0; i < index_node - nb_node[0] - nb_node[1]; i++)
				{
					choosen_node = choosen_node->next;
				}
			}
			else
			{
				choosen_node = node_list[1]->head;
				for (i = 0; i < index_node - nb_node[0]; i++)
				{
					choosen_node = choosen_node->next;
				}
			}
		}
		else
		{
			choosen_node = node_list[0]->head;
			for (i = 0; i < index_node; i++)
			{
				choosen_node = choosen_node->next;
			}
		}
				
		schedule_job_specific_node_at_earliest_available_time(j, choosen_node, t);
		nb_job_to_delete += 1;
		
		/* Add in list of starting times. */		
		insert_next_time_in_sorted_list(start_times, j->start_time);
		
		#ifdef PRINT
		print_decision_in_scheduler(j);
		#endif
		
		j = j->next;
	}
	
	/* Remove from job list to start and put it in scheduled job list. */
	for (i = 0; i < nb_job_to_delete; i++)
	{
		j = job_list_to_start_from_history->head;
		copy_delete_insert_job_list(job_list_to_start_from_history, scheduled_job_list, j);
	}

	free(nb_node);
}

/* FCFS */
void fcfs_scheduler(struct Job* head_job, struct Node_List** head_node, int t, bool use_bigger_nodes)
{
	#ifdef PRINT
	printf("Start fcfs scheduler. Use bigger nodes: %d.\n", use_bigger_nodes);
	#endif
		
	int nb_non_available_cores = get_nb_non_available_cores(node_list, t);

	struct Job* j = head_job;
	while (j != NULL)
	{
		if (nb_non_available_cores < nb_cores)
		{
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			#endif
			
			nb_non_available_cores = schedule_job_on_earliest_available_cores(j, head_node, t, nb_non_available_cores, use_bigger_nodes);
			
			/* test reduced complexity */
			if (j->start_time >= t)
			{
				insert_next_time_in_sorted_list(start_times, j->start_time);
			}
			
			j = j->next;
		}
		else
		{
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			#endif
			
			/* Need to put -1 at remaining start times of jobs to avoid error in n_vailable_cores. */
			while (j != NULL)
			{
				j->start_time = -1;
				j = j->next;
			}
			
			break;
		}
	}
}

/**
 * Schedule normalement.
 * Si je vois que un trou va se créer avec EAT == t, je le marque dans la liste des trous dans la struct node.
 * Dans la struct node je met la liste des cores ainsi que leurs nombre qui forment le trou au temps t pour aller plus vite.
 * Attention il faut reset cette liste de trou au moment du reset du reschedule.
 * Je ne check que les trou au temps t car sinon ca change rien avec le reschedule normalement.
 * Pour les jobs suivant je check node par node le EAT et les trou de la node. Si je rentre dans le trou je me schedule la pour fcfs.
 * nb_non_available_cores ne vaut que quand c'est t qui est recouvert, pas plus loin.
 **/
void fcfs_conservativebf_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int backfill_mode)
{
	//~ int nb_cores_rescheduled = 0; /* 486*20 = 9720 */
	int nb_non_available_cores = get_nb_non_available_cores(node_list, t);
	int nb_non_available_cores_at_time_t = global_nb_non_available_cores_at_time_t;
	//~ printf("biggest hole is %d.\n", biggest_hole);
	//~ biggest_hole = 0;
	
	#ifdef PRINT
	printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
	printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
	printf("Start FCFS CONSERVATIVE BF at time %d. backfill mode is %d.\n", t, backfill_mode);
	#endif

	struct Job* j = head_job;
	while (j != NULL)
	{
		if (nb_non_available_cores < nb_cores)
		{
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
			printf("Schedule and backfill.\n");
			#endif
			
			schedule_job_on_earliest_available_cores_with_conservative_backfill(j, head_node, t, backfill_mode, &nb_non_available_cores, &nb_non_available_cores_at_time_t);
			
			if (j->start_time >= t)
			{
				insert_next_time_in_sorted_list(start_times, j->start_time);
			}
			
			j = j->next;
		}
		else if (nb_non_available_cores_at_time_t < nb_cores) // Ajouter un && nb_non_available_cores_at_time_t > 20 car ça peut tout bloquer pour rien ?
		{
			/* Only check backfill */
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
			printf("Only check backfill.\n");
			printf("Biggest hole is %d on node %d.\n", biggest_hole, biggest_hole_unique_id);
			#endif
						
			if (j->cores <= biggest_hole && only_check_conservative_backfill(j, head_node, t, backfill_mode, &nb_non_available_cores_at_time_t) == true)
			{
				if (j->start_time < t)
				{
					printf("Error: j->start_time < t\n");
					exit(1);
				}
					
				insert_next_time_in_sorted_list(start_times, j->start_time);
			}
			else
			{
				j->start_time = -1;
			}
			
			j = j->next;
		}
		else
		{
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
			printf("Break scheduling.\n");
			#endif
			
			/* Need to put -1 at remaining start times of jobs to avoid error in n_vailable_cores. */
			while (j != NULL)
			{
				j->start_time = -1;
				j = j->next;
			}
			break;
		}
	}
	
	global_nb_non_available_cores_at_time_t = nb_non_available_cores_at_time_t;
}

void fcfs_with_a_score_conservativebf_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int adaptative_multiplier, int start_immediately_if_EAT_is_t, int backfill_mode, int mixed_strategy)
{
	//~ int nb_cores_rescheduled = 0; /* 486*20 = 9720 */
	int nb_non_available_cores = get_nb_non_available_cores(node_list, t);
	int nb_non_available_cores_at_time_t = global_nb_non_available_cores_at_time_t;
	//~ biggest_hole = 0;
	
	#ifdef PRINT
	printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
	printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
	printf("Start SCORE CONSERVATIVE BF at time %d. backfill mode is %d.\n", t, backfill_mode);
	#endif
	
	/* Get intervals of data. */ 
	get_current_intervals(head_node, t);
	
	/* For mixed */
	int temp_running_nodes = running_nodes;
	
	struct Job* j = head_job;
	
	while (j != NULL)
	{		
		//~ if (nb_non_available_cores < nb_cores) /* Vrai version */
		if (j->cores <= nb_cores - nb_non_available_cores)
		{
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
			printf("Schedule and backfill.\n");
			#endif
			
			if (mixed_strategy == 1)
			{
				if ((temp_running_nodes*100)/486 < busy_cluster_threshold)
				{
					schedule_job_fcfs_score_with_conservative_backfill(j, head_node, t, 1, 0, adaptative_multiplier, start_immediately_if_EAT_is_t, backfill_mode, &nb_non_available_cores, &nb_non_available_cores_at_time_t, mixed_strategy, &temp_running_nodes);
				}
				else
				{
					schedule_job_fcfs_score_with_conservative_backfill(j, head_node, t, multiplier_file_to_load, multiplier_file_evicted, adaptative_multiplier, start_immediately_if_EAT_is_t, backfill_mode, &nb_non_available_cores, &nb_non_available_cores_at_time_t, mixed_strategy, &temp_running_nodes);
				}
			}
			else
			{
				schedule_job_fcfs_score_with_conservative_backfill(j, head_node, t, multiplier_file_to_load, multiplier_file_evicted, adaptative_multiplier, start_immediately_if_EAT_is_t, backfill_mode, &nb_non_available_cores, &nb_non_available_cores_at_time_t, mixed_strategy, &temp_running_nodes);
			}
						
			//~ if (j->start_time < t)
			//~ {
				//~ printf("Error: j->start_time < t\n");
				//~ exit(EXIT_FAILURE);
			//~ }
			
			/* Test complexité réduite */
			if (j->start_time >= t)
			{
				insert_next_time_in_sorted_list(start_times, j->start_time);
			}
			
			j = j->next;
		}
		else if (nb_non_available_cores_at_time_t < nb_cores) // Ajouter un && nb_non_available_cores_at_time_t > 20 car ça peut tout bloquer pour rien ?
		{
			/* Only check backfill */
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
			printf("Only check backfill for job %d.\n", j->unique_id);
			printf("Biggest hole is %d on node %d.\n", biggest_hole, biggest_hole_unique_id);
			#endif
				
			//~ printf("%d running nodes.\n", temp_running_nodes);

			if (mixed_strategy == 1)
			{
				if ((temp_running_nodes*100)/486 < busy_cluster_threshold)
				{
					//~ printf("Call hole not busy.\n");
					if (j->cores <= biggest_hole && only_check_conservative_backfill_with_a_score(j, head_node, t, backfill_mode, &nb_non_available_cores_at_time_t, 1, 0, start_immediately_if_EAT_is_t) == true)
					{
						if (j->start_time < t)
						{
							printf("Error: j->start_time < t\n");
							exit(1);
						}
						insert_next_time_in_sorted_list(start_times, j->start_time);
					}
					else
					{
						j->start_time = -1;
					}
				}
				else
				{
					//~ printf("Call hole busy.\n");
					if (j->cores <= biggest_hole && only_check_conservative_backfill_with_a_score(j, head_node, t, backfill_mode, &nb_non_available_cores_at_time_t, multiplier_file_to_load, multiplier_file_evicted, start_immediately_if_EAT_is_t) == true)
					{
						if (j->start_time < t)
						{
							printf("Error: j->start_time < t\n");
							exit(1);
						}
						insert_next_time_in_sorted_list(start_times, j->start_time);
					}
					else
					{
						j->start_time = -1;
					}
				}
			}
			else
			{
				if (j->cores <= biggest_hole && only_check_conservative_backfill_with_a_score(j, head_node, t, backfill_mode, &nb_non_available_cores_at_time_t, multiplier_file_to_load, multiplier_file_evicted, start_immediately_if_EAT_is_t) == true)
				{
					if (j->start_time < t)
					{
						printf("Error: j->start_time < t\n");
						exit(1);
					}
					insert_next_time_in_sorted_list(start_times, j->start_time);
				}
				else
				{
					j->start_time = -1;
				}
			}
							
			j = j->next;
		}
		else
		{
			#ifdef PRINT
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);
			printf("There are %d/%d available cores at time t.\n", nb_cores - nb_non_available_cores_at_time_t, nb_cores);
			printf("Break scheduling.\n");
			#endif
			
			/* Need to put -1 at remaining start times of jobs to avoid error in n_vailable_cores. */
			while (j != NULL)
			{
				j->start_time = -1;
				j = j->next;
			}
			break;
		}
	}
	
	global_nb_non_available_cores_at_time_t = nb_non_available_cores_at_time_t;
}

void fcfs_with_a_score_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t, int mixed_strategy)
{
	#ifdef PRINT
	printf("\nFcfs_score\n");
	#endif
		
	int nb_non_available_cores = get_nb_non_available_cores(node_list, t);		
	int i = 0;
	int min_score = -1;
	int earliest_available_time = 0;
	float time_to_load_file = 0;
	bool is_being_loaded = false;
	float time_to_reload_evicted_files = 0;
	int nb_copy_file_to_load = 0;
	//~ int time_or_data_already_checked = 0;
	int score = 0;
	int min_time = 0;
	int choosen_time_to_load_file = 0;
	bool found = false;
				
	/* temp multiplier pour le cas avec if EAT is t start now */
	int temp_multiplier_file_to_load = multiplier_file_to_load;
	int temp_multiplier_file_evicted = multiplier_file_evicted;
	int temp_multiplier_nb_copy = multiplier_nb_copy;

		int temp_running_nodes = running_nodes;
	
	/* Get intervals of data. */ 
	get_current_intervals(head_node, t);
	
	#ifdef PRINT
	printf("AFTER.\n");
	print_data_intervals(head_node, t);
	#endif
	
	#ifdef PRINT_SCORES_DATA
	FILE* f_fcfs_score = fopen("outputs/Scores_data.txt", "a");
	#endif
	
			/* Cas non dynamique */
			if (mixed_strategy == 2)
			{
				//~ if (temp_running_nodes < 486)
				if ((temp_running_nodes*100)/486 < busy_cluster_threshold)
				{
					multiplier_file_to_load = 1;
					multiplier_file_evicted = 0;
					multiplier_nb_copy = 0;
				}
				#ifdef PRINT
				printf("At time %d, running nodes is %d. Multiplier are %d %d %d.\n", t, temp_running_nodes, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy);
				#endif
			}

	/* 1. Loop on available jobs. */
	struct Job* j = head_job;
	while (j != NULL)
	{		
		if (j->cores <= nb_cores - nb_non_available_cores)
		{
			#ifdef PRINT
			printf("\nNeed to schedule job %d using file %d. T = %d\n", j->unique_id, j->data, t); fflush(stdout);
			printf("There are %d/%d available cores.\n", nb_cores - nb_non_available_cores, nb_cores);			
			#endif
			
			/* cas if EAT is t reset multipliers */
			if (start_immediately_if_EAT_is_t == 1)
			{
				multiplier_file_to_load = temp_multiplier_file_to_load;
				multiplier_file_evicted = temp_multiplier_file_evicted;
				multiplier_nb_copy = temp_multiplier_nb_copy;
			}
			
			if (mixed_strategy == 1)
			{
				if ((temp_running_nodes*100)/486 < busy_cluster_threshold)
				{
					multiplier_file_to_load = 1;
					multiplier_file_evicted = 0;
					multiplier_nb_copy = 0;
				}
				else
				{
					multiplier_file_to_load = temp_multiplier_file_to_load;
					multiplier_file_evicted = temp_multiplier_file_evicted;
					multiplier_nb_copy = temp_multiplier_nb_copy;
				}
				
				#ifdef PRINT
				printf("At time %d, running nodes is %d. Multiplier are %d %d %d.\n", t, temp_running_nodes, multiplier_file_to_load, multiplier_file_evicted, multiplier_nb_copy);
				#endif
				
			}
			
			/* 2. Choose a node. */		
			/* Reset some values. */					
			min_score = -1;
			earliest_available_time = 0;
			is_being_loaded = false;
			time_to_reload_evicted_files = 0;
			nb_copy_file_to_load = 0;
			
			for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
			{
				struct Node* n = head_node[i]->head;
				while (n != NULL)
				{
					#ifdef PRINT
					printf("On node %d?\n", n->unique_id); fflush(stdout);
					#endif
										
					/* 2.1. A = Get the earliest available time from the number of cores required by the job and add it to the score. */
					earliest_available_time = n->cores[j->cores - 1]->available_time; /* -1 because tab start at 0 */
					
					/* Test complexité réduite */
					#ifdef NB_HOUR_MAX
					if (earliest_available_time > t + 3600*nb_h_scheduled)
					{
						goto next_node;
					}
					#endif
					
					if (earliest_available_time < t) /* A core can't be available before t. This happens when a node is idling. */				
					{
						earliest_available_time = t;
					}
					
					if (start_immediately_if_EAT_is_t == 1 && earliest_available_time == t) /* Ou dans une fenêtre ? */
					{
						multiplier_file_to_load = 1;
						multiplier_file_evicted = 0;
						multiplier_nb_copy = 0;
					}
										
					#ifdef PRINT
					printf("A: EAT is: %d.\n", earliest_available_time); fflush(stdout);
					#endif
					
					if (min_score == -1 || earliest_available_time < min_score)
					{						
						/* 2.2. B = Compute the time to load all data. For this look at the data that will be available at the earliest available time of the node. */
						if (j->data == 0)
						{
							time_to_load_file = 0;
						}
						else
						{
							time_to_load_file = is_my_file_on_node_at_certain_time_and_transfer_time(earliest_available_time, n, t, j->data, j->data_size, &is_being_loaded); /* Use the intervals in each data to get this info. */
							
						}
						
						#ifdef PRINT
						printf("B: Time to load file: %f. Is being loaded? %d.\n", time_to_load_file, is_being_loaded); fflush(stdout);
						#endif
											
						if (min_score == -1 || earliest_available_time + multiplier_file_to_load*time_to_load_file < min_score)
						{
							/* 2.5. Get the amount of files that will be lost because of this load by computing the amount of data that end at the earliest time only on the supposely choosen cores, excluding current file of course. */
							if (multiplier_file_evicted == 0)
							{
								time_to_reload_evicted_files = 0;
							}
							else
							{
								time_to_reload_evicted_files = time_to_reload_percentage_of_files_ended_at_certain_time(earliest_available_time, n, j->data, (float) j->cores/20);
							}
							
							#ifdef PRINT
							printf("C: Time to reload evicted files %f.\n", time_to_reload_evicted_files); fflush(stdout);
							#endif
							
							if (min_score == -1 || earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files < min_score)
							{
								nb_copy_file_to_load = 0;
								
								#ifdef PRINT
								printf("Nb of copy for data %d at time %d on node %d is %d.\n", j->data, earliest_available_time, n->unique_id, nb_copy_file_to_load); fflush(stdout);
								#endif
								
								/* Compute node's score. */
								score = earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files + nb_copy_file_to_load*time_to_load_file*multiplier_nb_copy;
																
								#ifdef PRINT	
								printf("Score for job %d is %d (EAT: %d + TL %d*%f + TRL %d*%f + NCP %d*%d*%f) with node %d.\n", j->unique_id, score, earliest_available_time, multiplier_file_to_load, time_to_load_file, multiplier_file_evicted, time_to_reload_evicted_files, nb_copy_file_to_load, multiplier_nb_copy, time_to_load_file, n->unique_id); fflush(stdout);
								#endif
																					
								/* 2.6. Get minimum score/ */
								if (min_score == -1)
								{
									min_time = earliest_available_time;
									min_score = score;
									j->node_used = n;
									choosen_time_to_load_file = time_to_load_file;
								}
								else if (min_score > score)
								{
									min_time = earliest_available_time;
									min_score = score;
									j->node_used = n;
									choosen_time_to_load_file = time_to_load_file;
								}
							}
						}
					}
					
					#ifdef PRINT_SCORES_DATA
					fprintf(f_fcfs_score, "Node: %d EAT: %d C: %f CxX: %f Score: %f\n", n->unique_id, earliest_available_time, time_to_reload_evicted_files, time_to_reload_evicted_files*multiplier_file_evicted, earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files);
					#endif
					
					#ifdef NB_HOUR_MAX
					/* Test complexité réduite */
					next_node: ;
					#endif
					
					n = n->next;
				}
			}
			
			
			/* Test complexité réduite */
			/* Update info if job was successfully scheduled. Else I put -1 at start_time to avoid any issue. */
			if (min_score != -1)
			{				
				j->transfer_time = choosen_time_to_load_file;
						
				/* Get start time and update available times of the cores. */
				j->start_time = min_time;
				j->end_time = min_time + j->walltime;
				
				/* Cas mixte pour LEO */
				if (mixed_strategy == 1 && j->node_used->n_available_cores == 20 && j->start_time == t)
				{
					#ifdef PRINT
					printf("Used node is %d.\n", j->node_used->unique_id);
					#endif
					
					bool new_running_node = true;
					for (int v = 0; v < 20; v++)
					{
						if (j->node_used->cores[v]->available_time > t)
						{
							new_running_node = false;
							break;
						}
					}
					if (new_running_node == true)
					{
						temp_running_nodes += 1;
					}
				}
				
				for (int k = 0; k < j->cores; k++)
				{
					j->cores_used[k] = j->node_used->cores[k]->unique_id;
					if (j->node_used->cores[k]->available_time <= t)
					{
						nb_non_available_cores += 1;
					}
					j->node_used->cores[k]->available_time = min_time + j->walltime;
					
					/* Maybe I need job queue or not not sure. TODO. */
				}

				/* Need to add here intervals for current scheduling. */
				found = false;
				
				#ifdef DATA_PERSISTENCE
				struct Data* d = j->node_used->temp_data->head;
				#else
				struct Data* d = j->node_used->data->head;
				#endif
							
				while (d != NULL)
				{
					if (d->unique_id == j->data)
					{
						found = true;
						create_and_insert_tail_interval_list(d->intervals, j->start_time);
						create_and_insert_tail_interval_list(d->intervals, j->start_time + j->transfer_time);
						create_and_insert_tail_interval_list(d->intervals, j->end_time);
						break;
					}
					d = d->next;
				}
				
				if (found == false)
				{					
					#ifdef PRINT
					printf("Need to create a data and intervals for the node %d data %d.\n", j->node_used->unique_id, j->data); fflush(stdout);
					#endif
					
					/* Create a class Data for this node. */
					struct Data* new = (struct Data*) malloc(sizeof(struct Data));
					new->next = NULL;
					new->unique_id = j->data;
					new->start_time = -1;
					new->end_time = -1;
					
					#ifndef DATA_PERSISTENCE
					new->nb_task_using_it = 0;
					#endif
					
					new->intervals = (struct Interval_List*) malloc(sizeof(struct Interval_List));
					new->intervals->head = NULL;
					new->intervals->tail = NULL;
					create_and_insert_tail_interval_list(new->intervals, j->start_time);
					create_and_insert_tail_interval_list(new->intervals, j->start_time + j->transfer_time);
					create_and_insert_tail_interval_list(new->intervals, j->end_time);
					new->size = j->data_size;
					
					#ifdef DATA_PERSISTENCE
					insert_tail_data_list(j->node_used->temp_data, new);
					#else
					insert_tail_data_list(j->node_used->data, new);
					#endif
				}			
				
				/* Need to sort cores after each schedule of a job. */
				sort_cores_by_available_time_in_specific_node(j->node_used);
											
				#ifdef PRINT
				print_decision_in_scheduler(j);
				#endif
										
				/* Insert in start times. */
				insert_next_time_in_sorted_list(start_times, j->start_time);
							
			}
			else /* Could not schedule the job in the time frame t + 1h */
			{
				j->start_time = -1;
			}
			j = j->next;
		}				
		else
		{
			#ifdef PRINT
			printf("No more available cores.\n"); fflush(stdout);
			#endif
			
			/* Need to put -1 at remaining start times of jobs to avoid error in n_vailable_cores. */
			while (j != NULL)
			{
				j->start_time = -1;
				j = j->next;
			}
			
			break;
		}
	}
	
	#ifdef PRINT_SCORES_DATA
	fclose(f_fcfs_score);
	#endif
}
