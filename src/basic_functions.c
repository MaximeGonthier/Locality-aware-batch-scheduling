#include <main.h>

/** 
 * This file contains functions that can be called
 * by any scheduler. They return jobs depending
 * on the parameters that want to be maximized, return
 * values from a node list, etc... 
**/

/** Return the smallest time that can be used to compute a job of size
 * nb_cores cores. Do not deal with backfilling. **/
int get_min_EAT(struct Node_List** head_node, int first_node_size_to_choose_from, int last_node_size_to_choose_from, int nb_cores, int t)
{
	int min_EAT = INT_MAX;
	int i = 0;
	
	for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			if (n->cores[nb_cores - 1]->available_time < min_EAT)
			{
				if (n->cores[nb_cores - 1]->available_time <= t)	
				{
					#ifdef PRINT
					printf("EAT == t.\n");
					#endif
					
					return t;
				}
				else
				{
					min_EAT = n->cores[nb_cores - 1]->available_time;
				}
			}
			n = n->next;
		}
	}
	if (min_EAT == INT_MAX)
	{
		perror("EAT is INT_MAX in get_min_EAT");
		exit(EXIT_FAILURE);
	}
	return min_EAT;
}

/** Schedule a job on the node with the smallest start time.
 * Does not deal with backfilling. **/
int schedule_job_on_earliest_available_cores(struct Job* j, struct Node_List** head_node, int t, int nb_non_available_cores, bool use_bigger_nodes)
{
	int i = 0;
	int min_time = -1;
	int earliest_available_time = 0;

	/* Finding the node with the earliest available time. */
	for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			earliest_available_time = n->cores[j->cores - 1]->available_time;
			
			//~ if (j->unique_id == 9396) { printf("EAT on node %d is %d\n", n->unique_id, earliest_available_time); fflush(stdout); }
			
			#ifdef NB_HOUR_MAX
			/* Test complexité réduite */
			if (earliest_available_time > t + 3600*nb_h_scheduled)
			{
				goto next_node;
			}
			#endif
			
			if (earliest_available_time < t) /* A core can't be available before t. This happens when a node is idling. */				
			{
				earliest_available_time = t;
			}
			if (min_time == -1 || min_time > earliest_available_time)
			{
				min_time = earliest_available_time;
				j->node_used = n;
				
				if (min_time == t)
				{
					i = last_node_size_to_choose_from + 1;
					break;
				}
			}
			
			#ifdef NB_HOUR_MAX
			next_node: ;
			#endif
			
			n = n->next;
		}
	}
	
	/* Test complexité réduite */
	if (min_time != -1)
	{
		/* Update infos on the job and on cores. */
		j->start_time = min_time;
		j->end_time = min_time + j->walltime;
			
		for (i = 0; i < j->cores; i++)
		{
			j->cores_used[i] = j->node_used->cores[i]->unique_id;
			if (j->node_used->cores[i]->available_time <= t)
			{
				nb_non_available_cores += 1;
			}
			j->node_used->cores[i]->available_time = min_time + j->walltime;
		}
		
		#ifdef PRINT
		print_decision_in_scheduler(j);
		#endif

		sort_cores_by_available_time_in_specific_node(j->node_used);
	}
	else
	{
		j->start_time = -1;
	}
	
	return nb_non_available_cores;
}

/** Explications des différents modes de backfilling
 * 0: Ne minimise pas la création de trous
 * 1: Minimise la création de trous. Si il trouve un trou, se met sur les coeurs qui ont un job qui commence dans le plus longtemps pour favoriser les jobs schedule en cas de terminaison de walltime plus tôt.
 * 2: Minimise la création de trous. Si il trouve un trou, se met sur les coeurs qui ont un job qui commence le plus tôt pour favoriser les jobs backfill.
 **/
/** **/
void schedule_job_on_earliest_available_cores_with_conservative_backfill(struct Job* j, struct Node_List** head_node, int t, int backfill_mode, int* nb_non_available_cores, int* nb_non_available_cores_at_time_t)
{
	/* NEW core selection conservative bf only */
	int nb_cores_from_hole = 0;
	int nb_cores_from_outside = 0;
	//~ int nb_cores_from_outside_remembered = 0;
	/* End of NEW core selection conservative bf only */
	
	/* OLD */
	//~ bool can_fit = false;
	/* OLD */
	
	#ifdef PRINT
	printf("\nScheduling job %d at time %d. Backfill mode is %d.\n", j->unique_id, t, backfill_mode);
	#endif

	//~ int parcours_des_nodes = 0;
	//~ int start_index = 0;
	int i = 0;
	//~ int k = 0;
	int min_time = -1;
	int earliest_available_time = 0;
	// int first_node_size_to_choose_from
	// int // last_node_size_to_choose_from = 0;
	bool backfilled_job = false;
	//~ struct Core_in_a_hole* c = (struct Core_in_a_hole*) malloc(sizeof(struct Core_in_a_hole));
	
	/* In which node size I can pick. */
	// get_node_size_to_choose_from(j->index_node_list, &first_node_size_to_choose_from, &last_node_size_to_choose_from);

	/* Finding the node with the earliest available time. */
	//~ for (parcours_des_nodes = 0; parcours_des_nodes < 2; parcours_des_nodes++) /* Pour faire nodes puis trou ou l'inverse. */
	//~ {
		for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
		{
			struct Node* n = head_node[i]->head;
			while (n != NULL)
			{
				//~ if ((parcours_des_nodes == 0 && (backfill_mode == 0 || backfill_mode == 2 || backfill_mode == 3)) || (parcours_des_nodes == 1 && (backfill_mode == 1)))
				//~ {
					#ifdef PRINT
					printf("Checking node %d.\n", n->unique_id);
					#endif
					
					earliest_available_time = n->cores[j->cores - 1]->available_time;
					
					#ifdef NB_HOUR_MAX
					/* Test complexité réduite */
					if (earliest_available_time > t + 3600*nb_h_scheduled)
					{
						goto next_node;
					}
					#endif
					
					if (earliest_available_time < t)	
					{
						earliest_available_time = t;
					}
								
					if (min_time == -1 || min_time > earliest_available_time)
					{
						min_time = earliest_available_time;
						j->node_used = n;
						
						if (min_time == t)
						{
							#ifdef PRINT
							printf("min_time == t, break.\n");
							#endif
							
							i = last_node_size_to_choose_from + 1;
							break;
						}
					}

					#ifdef NB_HOUR_MAX
					next_node: ;
					#endif
					
					backfilled_job = can_it_get_backfilled(j, n, t, &nb_cores_from_hole, &nb_cores_from_outside);
					if (backfilled_job == true)
					{
						min_time = t;
						j->node_used = n;
						i = last_node_size_to_choose_from + 1;
						//~ parcours_des_nodes = 2;
						break;
					}
				//~ }
				n = n->next;
			}
		}
	//~ }
		
	if (min_time != -1)
	{
		/* Update infos on the job and on cores. */
		j->start_time = min_time;
		j->end_time = min_time + j->walltime;
				 
		if (backfilled_job == true)
		{
			update_cores_for_backfilled_job(j, t, nb_cores_from_hole, nb_cores_from_outside);
			//~ exit(1);
			*nb_non_available_cores_at_time_t += j->cores;
			*nb_non_available_cores += nb_cores_from_outside;
			
			if (j->node_used->unique_id == biggest_hole_unique_id)
			{
				get_new_biggest_hole(head_node);
			}
		}
		 
		
		else /* backfilled_job == false */
		{			
			if (j->start_time == t)
			{
				*nb_non_available_cores_at_time_t += j->cores;
			}
			
			if (backfill_mode == 1 || backfill_mode == 2)
			{
				#ifdef PRINT
				printf("fill_cores_minimize_holes\n");
				#endif
				
				fill_cores_minimize_holes(j, true, backfill_mode, t, nb_non_available_cores);
				
				if (j->node_used->unique_id == biggest_hole_unique_id)
				{
					get_new_biggest_hole(head_node);
				}
			}
			else
			{
				for (i = 0; i < j->cores; i++)
				{
							j->cores_used[i] = j->node_used->cores[i]->unique_id;
							
							if (j->node_used->cores[i]->available_time <= t)
							{
								*nb_non_available_cores += 1;
							}
							
							/* Spécifique au cas avec backfilling */
							if (j->node_used->cores[i]->available_time <= t && j->start_time > t)
							{
								#ifdef PRINT
								printf("Il va y avoir un trou sur node %d core %d.\n", j->node_used->unique_id, j->node_used->cores[i]->unique_id); fflush(stdout);
								#endif
								
								j->node_used->number_cores_in_a_hole += 1;
															
								struct Core_in_a_hole* new = (struct Core_in_a_hole*) malloc(sizeof(struct Core_in_a_hole));
								//~ new->unique_id = j->node_used->cores[k]->unique_id;
								new->unique_id = j->node_used->cores[i]->unique_id;
								new->start_time_of_the_hole = min_time;
								new->next = NULL;
								
								if (j->node_used->cores_in_a_hole == NULL)
								{
									initialize_cores_in_a_hole(j->node_used->cores_in_a_hole, new);
								}
								else
								{
									insert_cores_in_a_hole_list_sorted_decreasing_order(j->node_used->cores_in_a_hole, new);
								}
							}
							/* Fin de spécifique au cas avec backfilling */
							
							j->node_used->cores[i]->available_time = j->start_time + j->walltime;
				}
				
				if (j->node_used->number_cores_in_a_hole > biggest_hole)
				{
					biggest_hole = j->node_used->number_cores_in_a_hole;
					biggest_hole_unique_id = j->node_used->unique_id;
				}
			}
		}

			if (backfilled_job == false || backfill_mode > 0 || nb_cores_from_outside > 0)
			{
				sort_cores_by_available_time_in_specific_node(j->node_used);
			}
		
		#ifdef PRINT
		print_decision_in_scheduler(j);
		print_cores_in_specific_node(j->node_used);
		#endif
	}
	else
	{
		j->start_time = -1;
	}
}

void schedule_job_fcfs_score_with_conservative_backfill(struct Job* j, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int adaptative_multiplier, int start_immediately_if_EAT_is_t, int backfill_mode, int* nb_non_available_cores, int* nb_non_available_cores_at_time_t, int mixed_strategy, int* temp_running_nodes)
{
	#ifdef PLOT_STATS
	total_number_of_scores_computed += 1;
	bool tie = false;
	#endif
	
	
	/* NEW core selection conservative bf only */
	int nb_cores_from_hole = 0;
	int nb_cores_from_outside = 0;
	int nb_cores_from_outside_remembered = 0;
	int choosen_nb_cores_from_hole = 0;
	int choosen_nb_cores_from_outside = 0;
	/* End of NEW core selection conservative bf only */

	/* OLD */
	//~ bool can_fit = false;
	/* OLD */
	
	#ifdef PRINT
	printf("\nScheduling job %d at time %d backfill mode %d.\n", j->unique_id, t, backfill_mode);
	#endif
	
	int i = 0;
	//~ int k = 0;
	int min_time = -1;
	int earliest_available_time = 0;
	// int first_node_size_to_choose_from
	// int // last_node_size_to_choose_from = 0;
	bool backfilled_job = false;
	int min_score = -1;
	float time_to_load_file = 0;
	bool is_being_loaded = false;
	float time_to_reload_evicted_files = 0;
	int score = 0;
	int choosen_time_to_load_file = 0;
	bool found = false;
	
	/* Pour garder les temps si ETA == t dans le cas du check des trous */
	bool want_to_save_times_for_backfill = false;
	float time_to_load_file_saved = 0;
	float time_to_reload_evicted_files_saved = 0;
		
	/* In which node size I can pick. */
	// get_node_size_to_choose_from(j->index_node_list, &first_node_size_to_choose_from, &last_node_size_to_choose_from);
	
	/* Finding the node with the earliest available time. */
	//~ for (parcours_des_nodes = 0; parcours_des_nodes < 2; parcours_des_nodes++) /* Pour faire nodes puis trou ou l'inverse. */
	//~ {
		for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
		{
			struct Node* n = head_node[i]->head;
			while (n != NULL)
			{
				want_to_save_times_for_backfill = false;
				earliest_available_time = n->cores[j->cores - 1]->available_time; /* -1 because tab start at 0 */
				
				#ifdef NB_HOUR_MAX
				/* Test complexité réduite */
				if (earliest_available_time > t + 3600*nb_h_scheduled)
				{
					goto next_node;
				}
				#endif
				
				if (earliest_available_time <= t) /* A core can't be available before t. This happens when a node is idling. */				
				{
					earliest_available_time = t;
					want_to_save_times_for_backfill = true; /* pour ne pas répeter les calculs lors du backfill */
					time_to_load_file_saved = -1;
					time_to_reload_evicted_files_saved = -1;
				}
					
				if (start_immediately_if_EAT_is_t == 1 && earliest_available_time == t) /* Ou dans une fenêtre ? */
				{
					multiplier_file_to_load = 1;
					multiplier_file_evicted = 0;
				}

				#ifdef PRINT
				printf("On node %d?\n", n->unique_id);
				#endif
					
				/* Computing score on the node outside of holes. */
				#ifdef PRINT
				printf("A: EAT is %d.\n", earliest_available_time);
				#endif
					
				if (min_score == -1 || earliest_available_time < min_score)
				{
					if (j->data == 0)
					{
						time_to_load_file = 0;
					}
					else
					{
						time_to_load_file = is_my_file_on_node_at_certain_time_and_transfer_time(earliest_available_time, n, t, j->data, j->data_size, &is_being_loaded);
					}
					
					if (want_to_save_times_for_backfill == true)
					{
						time_to_load_file_saved = time_to_load_file;
					}
					
					#ifdef PRINT
					printf("B: Time to load file: %f. Is being loaded? %d.\n", time_to_load_file, is_being_loaded);
					#endif
						
					//~ printf("Score for job %d is %f (EAT: %d + TL %d*%f) with node %d.\n", j->unique_id, earliest_available_time + multiplier_file_to_load*time_to_load_file, earliest_available_time, multiplier_file_to_load, time_to_load_file, n->unique_id);
						
					#ifdef PLOT_STATS
					if (min_score != -1 && min_score == earliest_available_time + multiplier_file_to_load*time_to_load_file)
					{
						tie = true;
					}
					else
					{
						tie = false;
					}
					#endif
		
					if (min_score == -1 || earliest_available_time + multiplier_file_to_load*time_to_load_file < min_score)
					{
						if (multiplier_file_evicted == 0)
						{
							time_to_reload_evicted_files = 0;
						}
						else
						{
							time_to_reload_evicted_files = time_to_reload_percentage_of_files_ended_at_certain_time(earliest_available_time, n, j->data, (float) j->cores/20);
						}
						
						if (want_to_save_times_for_backfill == true)
						{
							time_to_reload_evicted_files_saved = time_to_reload_evicted_files;
						}
						
						#ifdef PRINT
						printf("C: Time to reload evicted files %f.\n", time_to_reload_evicted_files);
						#endif	
													
						score = earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files;
						
						#ifdef PRINT	
						printf("Score for job %d is %d (EAT: %d + TL %d*%f + TRL %d*%f) with node %d.\n", j->unique_id, score, earliest_available_time, multiplier_file_to_load, time_to_load_file, multiplier_file_evicted, time_to_reload_evicted_files, n->unique_id);
						#endif
																						
						if (min_score == -1 || min_score > score)
						{
							min_time = earliest_available_time;
							min_score = score;
							j->node_used = n;
							choosen_time_to_load_file = time_to_load_file;
							backfilled_job = false; /* On met à false car ça a pu mettre à true par un trou dans une node précédente. */
								
							if (min_time == t && min_score == t) /* Temps de début est t et pas de temps de chargements du tout */
							{		
								#ifdef PRINT
								printf("min_time == t and no file to load/evict, break.\n");
								printf("Score for job %d is %d (EAT: %d + TL %d*%f + TRL %d*%f) with node %d.\n", j->unique_id, score, earliest_available_time, multiplier_file_to_load, time_to_load_file, multiplier_file_evicted, time_to_reload_evicted_files, n->unique_id);
								#endif
									
								i = last_node_size_to_choose_from + 1;
								//~ parcours_des_nodes = 2;
								break;
							}
						}
					}
				}
				
				#ifdef NB_HOUR_MAX
				/* Test complexité réduite */
				next_node: ;
				#endif
				
				#ifdef PRINT
				printf("Can I backfill on node %d?\n", n->unique_id);
				#endif
					
				if (can_it_get_backfilled(j, n, t, &nb_cores_from_hole, &nb_cores_from_outside) == true)
				{
					#ifdef PRINT
					printf("Yes.\n");
					#endif
						
					earliest_available_time = t;
					/* Calcul du score dans le trou de la node en question. */
						
					#ifdef PRINT
					printf("A: EAT is %d.\n", earliest_available_time);
					#endif
						
					if (min_score == -1 || earliest_available_time < min_score)
					{
						if (j->data == 0)
						{
							time_to_load_file = 0;
						}
						else if (want_to_save_times_for_backfill == true && time_to_load_file_saved != -1)
						{
							time_to_load_file = time_to_load_file_saved;
						}
						else
						{
							time_to_load_file = is_my_file_on_node_at_certain_time_and_transfer_time(earliest_available_time, n, t, j->data, j->data_size, &is_being_loaded);
						}
						#ifdef PRINT
						printf("B: Time to load file: %f. Is being loaded? %d.\n", time_to_load_file, is_being_loaded);
						#endif
														
						#ifdef PLOT_STATS
						if (min_score != -1 && min_score == earliest_available_time + multiplier_file_to_load*time_to_load_file)
						{
							tie = true;
						}
						else
						{
							tie = false;
						}
						#endif
			
						if (min_score == -1 || earliest_available_time + multiplier_file_to_load*time_to_load_file < min_score)
						{	
							//~ if (multiplier_file_evicted == 0 || time_to_load_file == 0) /* Attention ça avec mon calcul de evicted file en % c'est complètement faux! */
							if (multiplier_file_evicted == 0)
							{
								time_to_reload_evicted_files = 0;
							}
							else if (want_to_save_times_for_backfill == true && time_to_reload_evicted_files_saved != -1)
							{
								time_to_reload_evicted_files = time_to_reload_evicted_files_saved;
							}
							else
							{
								time_to_reload_evicted_files = time_to_reload_percentage_of_files_ended_at_certain_time(earliest_available_time, n, j->data, (float) j->cores/20);
							}
								
							#ifdef PRINT
							printf("C: Time to reload evicted files %f.\n", time_to_reload_evicted_files);
							#endif
								
							/* Je ne veux backfill que si je n'évince pas de fichiers */
							//~ if (time_to_load_file == 0 || time_to_reload_evicted_files == 0 || is_being_loaded == true)
							//~ {
								score = earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files;
									
								#ifdef PRINT
								printf("Score for job %d is %d (EAT: %d + TL %d*%f + TRL %d*%f) with node %d.\n", j->unique_id, score, earliest_available_time, multiplier_file_to_load, time_to_load_file, multiplier_file_evicted, time_to_reload_evicted_files, n->unique_id);
								#endif
																								
								if (min_score == -1 || min_score > score)
								{
									min_time = earliest_available_time;
									min_score = score;
									j->node_used = n;
									choosen_time_to_load_file = time_to_load_file;
									backfilled_job = true; /* On met à false car ça a pu mettre à true par un trou dans une node précédente. */
										
									/* maj choosen numbers is important for fcs score. */
									choosen_nb_cores_from_hole = nb_cores_from_hole;
									choosen_nb_cores_from_outside = nb_cores_from_outside;
										
									if (min_time == t && min_score == t) /* Temps de début est t et pas de temps de chargements du tout */
									{
										#ifdef PRINT
										printf("min_time == t and no file to load/evict, break.\n");
										printf("Score for job %d is %d (EAT: %d + TL %d*%f + TRL %d*%f) with node %d.\n", j->unique_id, score, earliest_available_time, multiplier_file_to_load, time_to_load_file, multiplier_file_evicted, time_to_reload_evicted_files, n->unique_id);
										#endif
											
										i = last_node_size_to_choose_from + 1;
										//~ parcours_des_nodes = 2;
										break;
									}
								}
							//~ }
						}
					}
				}
				n = n->next;
			}
		}
	
	#ifdef PLOT_STATS
	if (tie == true)
	{
		number_of_tie_breaks_before_computing_evicted_files_fcfs_score += 1;
	}
	#endif
	
	if (min_score != -1)
	{
		/* Update infos on the job and on cores. */
		j->start_time = min_time;
		j->end_time = min_time + j->walltime;
		j->transfer_time = choosen_time_to_load_file;
		/* Need to add here intervals for current scheduling. */
		found = false;
	
		/* Cas mixte */
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
				*temp_running_nodes += 1;
			}
		}
	
		/* TODO: crash malloc corrupted top size içi ? */
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
		
		/* 2 cas en fonction du choix */
		if (backfilled_job == true)
		{
			#ifdef PLOT_STATS
			number_of_backfilled_jobs+= 1;
			#endif
			
			update_cores_for_backfilled_job(j, t, choosen_nb_cores_from_hole, choosen_nb_cores_from_outside);
			*nb_non_available_cores_at_time_t += j->cores;
			*nb_non_available_cores += choosen_nb_cores_from_outside;
					
			if (j->node_used->unique_id == biggest_hole_unique_id)
			{
				get_new_biggest_hole(head_node);
			}
		}
		else /* backfilled_job == false */
		{
			if (j->start_time == t)
			{
				*nb_non_available_cores_at_time_t += j->cores;
			}
					
			if (backfill_mode == 1 || backfill_mode == 2)
			{
				fill_cores_minimize_holes(j, true, backfill_mode, t, nb_non_available_cores);
				
				if (j->node_used->unique_id == biggest_hole_unique_id)
				{
					get_new_biggest_hole(head_node);
				}
			}
			else
			{
				for (i = 0; i < j->cores; i++)
				{
					j->cores_used[i] = j->node_used->cores[i]->unique_id;

					if (j->node_used->cores[i]->available_time <= t)
					{
						*nb_non_available_cores += 1;
					}
								
					/* Est-ce que je créé un trou ? Si oui je le rajoute dans les infos de la node. */
					if (j->node_used->cores[i]->available_time <= t && min_time > t)
					{
						#ifdef PRINT
						printf("Il va y avoir un trou sur node %d core %d.\n", j->node_used->unique_id, j->node_used->cores[i]->unique_id); fflush(stdout);
						#endif
						
						j->node_used->number_cores_in_a_hole += 1;
						struct Core_in_a_hole* new = (struct Core_in_a_hole*) malloc(sizeof(struct Core_in_a_hole));
						new->unique_id = j->node_used->cores[i]->unique_id;
						new->start_time_of_the_hole = min_time;
						new->next = NULL;
						if (j->node_used->cores_in_a_hole == NULL)
						{
							initialize_cores_in_a_hole(j->node_used->cores_in_a_hole, new);
						}
						else
						{
							insert_cores_in_a_hole_list_sorted_decreasing_order(j->node_used->cores_in_a_hole, new);
						}
					}
					
					/* NEW core selection */
					j->node_used->cores[i]->available_time = min_time + j->walltime;
					/* End of NEW core selection */
				}
				
				if (j->node_used->number_cores_in_a_hole > biggest_hole)
				{
					biggest_hole = j->node_used->number_cores_in_a_hole;
					biggest_hole_unique_id = j->node_used->unique_id;
				}
				
			}
		}
		#ifdef PRINT
		print_decision_in_scheduler(j);
		#endif
		
		if (backfilled_job == false || backfill_mode > 0 || nb_cores_from_outside_remembered > 0)
		{
			sort_cores_by_available_time_in_specific_node(j->node_used);
		}
	}
	else
	{
		j->start_time = -1;
	}

	//~ return nb_non_available_cores;
}

/* Pour easybf fcfs */
int schedule_job_on_earliest_available_cores_return_running_cores(struct Job* j, struct Node_List** head_node, int t, int nb_running_cores, bool use_bigger_nodes)
{
	int i = 0;
	int min_time = -1;
	int earliest_available_time = 0;
		
	/* Finding the node with the earliest available time. */
	for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			//~ if (j->unique_id == 1020 && n->unique_id == 0)
			//~ {
				//~ for (int k = 0; k < 20; k++)
				//~ {
					//~ printf("avail time of core %d = %d.\n", n->cores[k]->unique_id, n->cores[k]->available_time);
				//~ }
				//~ print_cores_in_specific_node(n);
			//~ }
			
			
			//~ struct Node* n = head_node[i]->head;
			earliest_available_time = n->cores[j->cores - 1]->available_time; /* -1 because tab start at 0 */
			if (earliest_available_time < t) /* A core can't be available before t. This happens when a node is idling. */				
			{
				earliest_available_time = t;
			}
			if (min_time == -1 || min_time > earliest_available_time)
			{
				min_time = earliest_available_time;
				j->node_used = n;
				
				if (min_time == t)
				{
					i = last_node_size_to_choose_from + 1;
					break;
				}
			}
			
			//~ #ifdef PRINT
			//~ printf("EAT on node %d is %d.\n", n->unique_id, earliest_available_time);
			//~ #endif
			
			n = n->next;
		}
	}
	
	//~ #ifdef PRINT
	//~ if (j->node_used->unique_id == 183){
	//~ print_cores_in_specific_node(j->node_used);}
	//~ #endif
	
	/* Update infos on the job and on cores. */
	j->start_time = min_time;
	j->end_time = min_time + j->walltime;
	for (i = 0; i < j->cores; i++)
	{
		j->cores_used[i] = j->node_used->cores[i]->unique_id;
		if (min_time <= t)
		{
			nb_running_cores += 1;
		}
		j->node_used->cores[i]->available_time = min_time + j->walltime;
		
		/* Maybe I need job queue or not not sure. TODO. */
		//~ copy_job_and_insert_tail_job_list(n->cores[i]->job_queue, j);
	}
		
	#ifdef PRINT
	print_decision_in_scheduler(j);
	#endif

	/* Need to sort cores after each schedule of a job. */
	sort_cores_by_available_time_in_specific_node(j->node_used);
		
	return nb_running_cores;
}

int schedule_job_on_earliest_available_cores_specific_sublist_node(struct Job* j, struct Node_List* head_node_size_i, int t, int nb_non_available_cores)
{
	int i = 0;
	int min_time = -1;
	int earliest_available_time = 0;
			
	/* Finding the node with the earliest available time. */
	//~ for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
	//~ {
		struct Node* n = head_node_size_i->head;
		while (n != NULL)
		{			
			earliest_available_time = n->cores[j->cores - 1]->available_time; /* -1 because tab start at 0 */
			if (earliest_available_time < t) /* A core can't be available before t. This happens when a node is idling. */				
			{
				earliest_available_time = t;
			}
			if (min_time == -1 || min_time > earliest_available_time)
			{
				min_time = earliest_available_time;
				j->node_used = n;
				
				if (min_time == t)
				{
					//~ printf("break.\n");
					break;
				}
			}
						
			n = n->next;
		}
		
	/* Update infos on the job and on cores. */
	j->start_time = min_time;
	j->end_time = min_time + j->walltime;
	for (i = 0; i < j->cores; i++)
	{
		j->cores_used[i] = j->node_used->cores[i]->unique_id;
		if (j->node_used->cores[i]->available_time <= t)
		{
			nb_non_available_cores += 1;
		}
		j->node_used->cores[i]->available_time = min_time + j->walltime;
		
		/* Maybe I need job queue or not not sure. TODO. */
		//~ copy_job_and_insert_tail_job_list(n->cores[i]->job_queue, j);
	}
		
	#ifdef PRINT
	print_decision_in_scheduler(j);
	#endif
		
	/* Need to sort cores after each schedule of a job. */
	sort_cores_by_available_time_in_specific_node(j->node_used);
		
	return nb_non_available_cores;
}

/* Pour first job of easybf fcfs with a score */
int schedule_job_fcfs_score_return_running_cores(struct Job* j, struct Node_List** head_node, int t, int nb_running_cores, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t)
{
	#ifdef PRINT
	printf("\nschedule_job_fcfs_score_return_running_cores job %d\n", j->unique_id); fflush(stdout);
	#endif
	
	int i = 0;
	int min_score = -1;
	int earliest_available_time = 0;
	// int first_node_size_to_choose_from
	// int // last_node_size_to_choose_from = 0;
	float time_to_load_file = 0;
	bool is_being_loaded = false;
	float time_to_reload_evicted_files = 0;
	int nb_copy_file_to_load = 0;
	int time_or_data_already_checked = 0;
	int score = 0;
	int min_time = 0;
	int choosen_time_to_load_file = 0;
	bool found = false;
	
	/** 1 = gives the number of running nodes as a multiplier.
	 *  2 = gives the number of jobs to schedule divided by the total number of nodes as a multiplier.
	 *  3 = gives the number of running nodes as a multiplier but put only 1 if the number of runing nodes is inferior to 75%.
	 *  4 = gives the number of running nodes as a multiplier but put only 1 if the number of runing nodes is inferior to 75% and the queue of jobs to schedule is too important.
	 **/
	if (adaptative_multiplier == 1) 
	{
		if (multiplier_file_to_load != 0)
		{
			multiplier_file_to_load = running_nodes;
		}
	}
	else if (adaptative_multiplier == 2)
	{
		if (multiplier_file_to_load != 0)
		{
			multiplier_file_to_load = (int) ceil ((float) nb_job_to_schedule/486);
		}
	}
	else if (adaptative_multiplier == 3)
	{
		if (running_nodes < 454)
		{
			multiplier_file_to_load = 1;
			multiplier_file_evicted = 0;
			multiplier_nb_copy = 0;
		}
		else
		{
			if (multiplier_file_to_load != 0)
			{
				multiplier_file_to_load = running_nodes;
			}
		}
	}
	else if (adaptative_multiplier == 4)
	{
		if (486*20 - running_cores >= nb_cores_to_schedule)
		{
			multiplier_file_to_load = 1;
			multiplier_file_evicted = 0;
			multiplier_nb_copy = 0;
		}
		else
		{
			if (multiplier_file_to_load != 0)
			{
				multiplier_file_to_load = running_nodes;
			}
		}
	}
			
	/* temp multiplier pour le cas avec if EAT is t start now */
	int temp_multiplier_file_to_load = multiplier_file_to_load;
	int temp_multiplier_file_evicted = multiplier_file_evicted;
	int temp_multiplier_nb_copy = multiplier_nb_copy;

	
	/* Get intervals of data. */ /* Je le fais pas la car je l'ai fais avant dans le scheduler de easybf fcfs wit ha score */
	//~ get_current_intervals(head_node, t);
	
	#ifdef PRINT
	print_data_intervals(head_node, t);
	#endif
	
	#ifdef PRINT_SCORES_DATA
	FILE* f_fcfs_score = fopen("outputs/Scores_data.txt", "a");
	#endif
	
	/* --- Reduced complexity nb of copy --- */	
		struct Time_or_Data_Already_Checked_Nb_of_Copy_List* time_or_data_already_checked_nb_of_copy_list = (struct Time_or_Data_Already_Checked_Nb_of_Copy_List*) malloc(sizeof(struct Time_or_Data_Already_Checked_Nb_of_Copy_List));
		time_or_data_already_checked_nb_of_copy_list->head = NULL;

	/* 1. Loop on available jobs. */
		
			#ifdef PRINT
			printf("\nNeed to schedule job %d using file %d. T = %d\n", j->unique_id, j->data, t); fflush(stdout);
			#endif
			
			/* cas if EAT is t reset multipliers */
			if (start_immediately_if_EAT_is_t == 1)
			{
				multiplier_file_to_load = temp_multiplier_file_to_load;
				multiplier_file_evicted = temp_multiplier_file_evicted;
				multiplier_nb_copy = temp_multiplier_nb_copy;
			}
									
			/* --- Reduced complexity nb of copy --- */
			if (multiplier_nb_copy != 0)
			{
				time_or_data_already_checked = was_time_or_data_already_checked_for_nb_copy(j->data, time_or_data_already_checked_nb_of_copy_list);
			}

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
						/* Update the dividor of the multiplier in function of the file size; */
						
						/* 2.2. B = Compute the time to load all data. For this look at the data that will be available at the earliest available time of the node. */
						if (j->data == 0)
						{
							time_to_load_file = 0;
						}
						else
						{
							time_to_load_file = is_my_file_on_node_at_certain_time_and_transfer_time(earliest_available_time, n, t, j->data, j->data_size, &is_being_loaded); /* Use the intervals in each data to get this info. */
							
							/* Cas avec pénalité sur les gros jobs */
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
								/* 2.5bis Get number of copy of the file we want to load on other nodes (if you need to load a file that is) at the time that is predicted to be used. So if a file is already loaded on a lot of node, you have a penalty if you want to load it on a new node. */
								if (time_to_load_file != 0 && is_being_loaded == false && multiplier_nb_copy != 0)
								{
									/* --- Reduced complexity nb of copy --- */
									if (time_or_data_already_checked == -1)
									{
										#ifdef PRINT
										printf("Need to compute nb of copy it was never done.\n");
										#endif
										nb_copy_file_to_load = get_nb_valid_copy_of_a_file(t, head_node, j->data);
										create_and_insert_head_time_or_data_already_checked_nb_of_copy_list(time_or_data_already_checked_nb_of_copy_list, j->data, nb_copy_file_to_load);
										time_or_data_already_checked = nb_copy_file_to_load;
										#ifdef PRINT
										printf("Compute nb of copy done, it's %d.\n", nb_copy_file_to_load);
										#endif
									}
									else
									{
										nb_copy_file_to_load = time_or_data_already_checked;
										#ifdef PRINT
										printf("Already done for job %d at time %d so nb of copy is %d.\n", j->unique_id, t, nb_copy_file_to_load);
										#endif
									}
								}
								else
								{
									nb_copy_file_to_load = 0;
								}
								
								#ifdef PRINT
								printf("Nb of copy for data %d at time %d on node %d is %d.\n", j->data, earliest_available_time, n->unique_id, nb_copy_file_to_load); fflush(stdout);
								#endif
								
								/* Compute node's score. */
								score = earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files + nb_copy_file_to_load*time_to_load_file*multiplier_nb_copy;
																
								#ifdef PRINT	
								printf("Score for job %d is %d (EAT: %d + TL %d*%f + TRL %d*%f + NCP %d*%d*%f) with node %d.\n", j->unique_id, score, earliest_available_time, multiplier_file_to_load, time_to_load_file, multiplier_file_evicted, time_to_reload_evicted_files, nb_copy_file_to_load, multiplier_nb_copy, time_to_load_file, n->unique_id); fflush(stdout);
								#endif
																					
								/* 2.6. Get minimum score/ */
								/* TODO : simpliefier la complexité: si EAT est t et TL et TLE == 0 alors break */
								if (min_score == -1 || min_score > score)
								{
									min_time = earliest_available_time;
									min_score = score;
									j->node_used = n;
									choosen_time_to_load_file = time_to_load_file;
									
									if (min_time == t && min_score == t)
									{
										i = last_node_size_to_choose_from + 1;
										break;
									}
								}
							}
						}
					}
					
					#ifdef PRINT_SCORES_DATA
					fprintf(f_fcfs_score, "Node: %d EAT: %d C: %f CxX: %f Score: %f\n", n->unique_id, earliest_available_time, time_to_reload_evicted_files, time_to_reload_evicted_files*multiplier_file_evicted, earliest_available_time + multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files);
					#endif
					
					n = n->next;
				}
			}
			
			j->transfer_time = choosen_time_to_load_file;
					
			/* Get start time and update available times of the cores. */
			j->start_time = min_time;
			j->end_time = min_time + j->walltime;
			
			for (int k = 0; k < j->cores; k++)
			{
				j->cores_used[k] = j->node_used->cores[k]->unique_id;
				if (min_time <= t)
				{
					nb_running_cores += 1;
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
			
			/* --- Normal complexity nb of copy --- */
			/* Free time already checked. */
			
			/* --- Normal complexity nb of copy --- */
			/* Increment nb of copy for current file if we scheduled at time t the current job. */
			if (multiplier_nb_copy != 0 && j->start_time == t)
			{
				increment_time_or_data_nb_of_copy_specific_time_or_data(time_or_data_already_checked_nb_of_copy_list, j->data);
			}
				
	/* --- Reduced complexity nb of copy --- */
	/* Free time already checked. */
	if (multiplier_nb_copy != 0)
	{
		free_time_or_data_already_checked_nb_of_copy_linked_list(&time_or_data_already_checked_nb_of_copy_list->head);
	}

	#ifdef PRINT_SCORES_DATA
	fclose(f_fcfs_score);
	#endif

	return nb_running_cores;
}

/* Called at the beggining of each callof fcfs with a score. 
 * Fill the node's data list with intervals of current data.
 * For data persistency, I put it in the temp_data part of nodes. */
void get_current_intervals(struct Node_List** head_node, int t)
{
	#ifdef DATA_PERSISTENCE
	free_and_copy_data_and_intervals_in_temp_data(head_node, t);
	#else
	int i = 0;
	
	/* Des free se font a des moments ou c'est pas necessaire ? Ce n'st pas vraiment à NULL ou alors il y a un elmeent a vide ? */
	for (i = 0; i < 3; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			struct Data* d = n->data->head;
			while (d != NULL)
			{
				
					//~ struct Interval* current = *head_ref;
					   struct Interval* next;
					//~ if (d->intervals->head != NULL) 
					//~ {
					   while (d->intervals->head != NULL)
					   {
						   next = d->intervals->head->next;
						   free(d->intervals->head);
						   d->intervals->head = next;
					   }
					   free(d->intervals);
				   //~ }
									
					/* OLD */
					d->intervals = (struct Interval_List*) malloc(sizeof(struct Interval_List));	
					//~ d->intervals = realloc(d->intervals, sizeof(struct Interval_List));	
					d->intervals->head = NULL;
					d->intervals->tail = NULL;
									
					if (d->nb_task_using_it > 0)
					{
						create_and_insert_tail_interval_list(d->intervals, t);
						if (d->start_time < t)
						{
							create_and_insert_tail_interval_list(d->intervals, t);
						}
						else
						{
							create_and_insert_tail_interval_list(d->intervals, d->start_time);
						}
						create_and_insert_tail_interval_list(d->intervals, d->end_time);
					}
					else if (d->end_time >= t) /* Cas finis et re enchaine */
					{
						create_and_insert_tail_interval_list(d->intervals, t);
						create_and_insert_tail_interval_list(d->intervals, t);
						create_and_insert_tail_interval_list(d->intervals, t);
					}					
				d = d->next;
			}
			n = n->next;
		}
	}
	#endif
}

int get_nb_non_available_cores(struct Node_List** n, int t)
{
	int nb_non_available_cores = 0;
	int i = 0;
	int j = 0;
	for (i = 0; i < 3; i++)
	{
		struct Node* temp = n[i]->head;
		while (temp != NULL)
		{
			for (j = 0; j < 20; j++)
			{
				if (temp->cores[j]->available_time > t)
				{
					nb_non_available_cores += 1;
				}
			}			
			temp = temp->next;
		}
	}
	return nb_non_available_cores;
}

void schedule_job_specific_node_at_earliest_available_time(struct Job* j, struct Node* n, int t)
{
	int i = 0;
	//~ struct Job* j = j2;

	int earliest_available_time = n->cores[j->cores - 1]->available_time;
	if (earliest_available_time < t)
	{
		earliest_available_time = t;
	}
	// choosen_core = node.cores[0:cores_asked]		

	j->node_used = n;
		
	j->start_time = earliest_available_time;
	j->end_time = earliest_available_time + j->walltime;
	for (i = 0; i < j->cores; i++)
	{
		j->cores_used[i] = n->cores[i]->unique_id;
		
		
		n->cores[i]->available_time = earliest_available_time + j->walltime;
		
	}
		
	/* Need to sort cores after each schedule of a job. */
	sort_cores_by_available_time_in_specific_node(n);
}

void add_data_in_node (int data_unique_id, float data_size, struct Node* node_used, int t, int end_time, int* transfer_time, int* waiting_for_a_load_time, int delay, int walltime, int start_time, int cores)
{
	//~ printf("1.\n"); fflush(stdout);
	//~ printf("%d\n", node_used->unique_id); fflush(stdout);
	#ifdef PRINT
	printf("\nChecking data %d on node %d at time %d.\n", data_unique_id, node_used->unique_id, t); fflush(stdout);
	#endif
	
	bool data_is_on_node = false;
	/* Let's try to find it in the node */
	struct Data* d = (struct Data*) malloc(sizeof(struct Data));
	d = node_used->data->head;
	while (d != NULL)
	{
		//~ printf("check data %d\n", d->unique_id); fflush(stdout);
		if (data_unique_id == d->unique_id) /* It is already on node */
		{
			//~ printf("On node\n"); fflush(stdout);
			#ifdef DATA_PERSISTENCE
			//~ if (d->nb_task_using_it > 0 || d->end_time == t) /* And is still valid! */
			//~ {
				if (d->start_time > t) /* The job will have to wait for the data to be loaded by another job before starting */
				{
					*waiting_for_a_load_time = d->start_time - t;
				}
				else
				{
					*transfer_time = 0; /* No need to wait to start the job, data is already fully loaded */
					
					#ifdef PLOT_STATS
					if (d->end_time < t)
					{ 
						printf("data persistence was exploited\n");
						data_persistence_exploited++;
					}
					#endif
				}
			
			data_is_on_node = true;
			
			int min_between_delay_and_walltime = 0;
			if (delay + *waiting_for_a_load_time + *transfer_time < walltime)
			{
				min_between_delay_and_walltime = delay + *waiting_for_a_load_time + *transfer_time;
			}
			else
			{
				min_between_delay_and_walltime = walltime;
			}
			end_time = start_time + min_between_delay_and_walltime;
			
			if (d->end_time < end_time)
			{
				d->end_time = end_time;
			}
			break;			
			#else
			if (d->nb_task_using_it > 0 || d->end_time == t) /* And is still valid! */
			{
				if (d->start_time > t) /* The job will have to wait for the data to be loaded by another job before starting */
				{
					*waiting_for_a_load_time = d->start_time - t;
				}
				else
				{
					*transfer_time = 0; /* No need to wait to start the job, data is already fully loaded */
				}
			}
			else /* Need to reload it */
			{
				*transfer_time = data_size/node_used->bandwidth;
				d->start_time = t + *transfer_time;
			}
			
			data_is_on_node = true;
			d->nb_task_using_it += 1;
			
			if (d->end_time < end_time)
			{
				d->end_time = end_time;
			}
			break;
			#endif
		}
		d = d->next;
	}
	//~ printf("is it on node: %d\n", data_is_on_node); fflush(stdout);
	if (data_is_on_node == false) /* Need to load it */
	{
		#ifdef PRINT
		printf("*transfer_time %f = data_size %f /node_used->bandwidth %f\n", data_size/node_used->bandwidth, data_size, node_used->bandwidth); fflush(stdout);
		#endif
		
		*transfer_time = data_size/node_used->bandwidth;
		/* Create a class Data for this node */
		struct Data* new = (struct Data*) malloc(sizeof(struct Data));
		new->unique_id = data_unique_id;
		new->start_time = t + *transfer_time;
		
		#ifndef DATA_PERSISTENCE
		new->end_time = end_time;
		#else
		int min_between_delay_and_walltime = 0;
		if (delay + *waiting_for_a_load_time + *transfer_time < walltime)
		{
			min_between_delay_and_walltime = delay + *waiting_for_a_load_time + *transfer_time;
		}
		else
		{
			min_between_delay_and_walltime = walltime;
		}
		end_time = start_time + min_between_delay_and_walltime;
		new->end_time = end_time;
		#endif
		
		#ifndef DATA_PERSISTENCE
		new->nb_task_using_it = 1;
		#endif
		
		new->size = data_size;
		new->next = NULL;
		new->intervals = (struct Interval_List*) malloc(sizeof(struct Interval_List));
		new->intervals->head = NULL;
		insert_tail_data_list(node_used->data, new);
		
		#ifdef DATA_PERSISTENCE
		node_used->data_occupation += cores;
		#endif
		
		#ifdef PRINT
		printf("Created data %d starttime %d on node %d.\n", new->unique_id, new->start_time, node_used->unique_id);
		#endif
	}
	
	#ifdef DATA_PERSISTENCE
	while (node_used->data_occupation > 20) /* Need an eviction */
	{
		#ifdef PRINT
		printf("node_used->data_occupation %d > 20. Current data to load is %d\n", node_used->data_occupation, data_unique_id);
		#endif
		
		struct Data* d_temp = node_used->data->head;
		struct Data* data_to_evict = (struct Data*) malloc(sizeof(struct Data));
		int min_end_time = INT_MAX;
		while (d_temp != NULL)
		{
			#ifdef PRINT
			printf("Testing data %d (end_time %d) to evict.\n", d_temp->unique_id, d_temp->end_time);
			#endif
			
			if (d_temp->unique_id != data_unique_id)
			{
				if (min_end_time > d_temp->end_time)
				{
					min_end_time = d_temp->end_time;
					data_to_evict = d_temp;
				}
			}
			d_temp = d_temp->next;
		}
		if (data_to_evict == NULL)
		{
			printf("Error data_to_evict NULL.\n"); fflush(stdout);
			exit(1);
		}
		
		//~ data_to_evict->end_time = t - 1; /* A quoi ca servait ? */
		
		#ifdef PRINT
		printf("Evicting data %d size %f on node %d.\n", data_to_evict->unique_id, data_to_evict->size, node_used->unique_id);
		#endif
		
		node_used->data_occupation -= data_to_evict->size;
		delete_specific_data_from_node(node_used->data, data_to_evict->unique_id);
	}
	
	#ifdef PRINT
	printf("After checking data, occupation is %d.\n", node_used->data_occupation);
	#endif
	#endif
}

/* Pas utile en cas data persistence */
void remove_data_from_node(struct Job* j, int t)
{
	#ifndef DATA_PERSISTENCE
	struct Data* d = j->node_used->data->head;
	while (d != NULL)
	{
		if (j->data == d->unique_id)
		{
			d->nb_task_using_it -= 1;
				
			if (d->nb_task_using_it == 0) /* modif ? */
			{
				d->end_time = t;
			}
			break;
		}
		d = d->next;
	}
	#endif
}

/* Go through schedule jobs to find finished jobs. */
void start_jobs(int t, struct Job* head)
{
	int i = 0;
	int k = 0;
	int overhead_of_load = 0;
	int min_between_delay_and_walltime = 0;
	int transfer_time = 0;
	int waiting_for_a_load_time = 0;
	
	#ifdef PRINT
	printf("Start of start_jobs at time %d.\n", t); fflush(stdout);
	#endif
	
	struct Job* j = head;
	
	while (j != NULL)
	{
		if (j->start_time == t)
		{
			/* Update nb of jobs to schedule */
			nb_job_to_schedule -= 1;
			nb_cores_to_schedule -= j->cores;
			
			if (nb_job_to_schedule < 0)
			{
				printf("Error nb_job_to_schedule = %d\n", nb_job_to_schedule);
				exit(EXIT_FAILURE);
			}
			
			if (j->delay <= 0)
			{
				printf("Error delay is %d for job %d.\n", j->delay, j->unique_id);
				exit(EXIT_FAILURE);
			}
			
			/* Remove from list of starting times. */
			if (start_times->head != NULL && start_times->head->time == t)
			{
				delete_next_time_linked_list(start_times, t);
			}
			
			/* For constraint on sizes only. TODO : remove it or put it in an ifdef if I don't have this constraint to gain some time ? */
			total_queue_time += j->start_time - j->subtime;
			
			transfer_time = 0;
			waiting_for_a_load_time = 0;
			
			if (j->data != 0 && constraint_on_sizes != 2)
			{
				/* Let's look if a data transfer is needed */
				add_data_in_node(j->data, j->data_size, j->node_used, t, j->end_time, &transfer_time, &waiting_for_a_load_time, j->delay, j->walltime, j->start_time, j->cores);
			}
			//~ printf("la\n"); fflush(stdout);
			
			j->transfer_time = transfer_time;
			j->waiting_for_a_load_time = waiting_for_a_load_time;
			
			/* Pour compter le nombre de fois qu'on reutilise des données (ou du moins que 2 jobs utilisant le même fichier ont été schedule en même temps sur la même node. Que pour les jobs du workload évalué. */
			if (j->workload == 1 && (j->transfer_time == 0 || j->waiting_for_a_load_time != 0))
			{
				nb_data_reuse += 1;
			}
						
			overhead_of_load = 0;

			if (transfer_time == 0)
			{
				overhead_of_load = waiting_for_a_load_time;
			}
			else if (waiting_for_a_load_time == 0)
			{
				overhead_of_load = transfer_time;
			}
			else
			{
				printf("Error calcul transfer time.\n");
				exit(EXIT_FAILURE);
			}
			
			#ifdef PRINT
			printf("For job %d (delay = %d): %d transfer time and %d waiting for a load time. Overhead is %d\n", j->unique_id, j->delay, transfer_time, waiting_for_a_load_time, overhead_of_load); fflush(stdout);
			#endif
			
			if (j->delay + overhead_of_load < j->walltime)
			{
				min_between_delay_and_walltime = j->delay + overhead_of_load;
				j->end_before_walltime = true;
			}
			else
			{
				min_between_delay_and_walltime = j->walltime;
				j->end_before_walltime = false;
			}
			j->end_time = j->start_time + min_between_delay_and_walltime; /* Attention le j->end time est mis a jour la! */
			
			if (j->end_time <= j->start_time)
			{
				printf("Error end time job %d workload %d: %d -> %d\n min_between_delay_and_walltime is %d, walltime was %d, delay was %d\n", j->unique_id, j->workload, j->start_time, j->end_time, min_between_delay_and_walltime, j->walltime, j->delay);
				exit(EXIT_FAILURE);
			}
			
			insert_next_time_in_sorted_list(end_times, j->end_time);
			
			#ifdef PRINT
			printf("==> Job %d %d cores start at time %d on node %d and will end at time %d before walltime: %d transfer time is %d data was %d.\n", j->unique_id, j->cores, t, j->node_used->unique_id, j->end_time, j->end_before_walltime, transfer_time, j->data);
			#endif
			
			/* For easy bf */
			running_cores += j->cores;
			
			#ifdef PRINT
			printf("Updating running cores on job %d: %d/20 -> %d/20.\n", j->unique_id, j->node_used->n_available_cores, j->node_used->n_available_cores - j->cores);
			#endif
			
			/** Defining cluster usage **/
			if (j->node_used->n_available_cores == 20)
			{
				/* Just for PRINT_CLUSTER_USAGE or I use it elsewhere ? */
				running_nodes += 1;
				
				// if (j->workload == -2)
				// {
					// running_nodes_workload_minus_2 += 1;
				//~}
			}
			
			#ifdef PRINT_CLUSTER_USAGE
			if (j->workload == 1)
			{
				if (j->node_used->nb_jobs_workload_1 == 0)
				{
					running_nodes_workload_1 += 1;
				}
				j->node_used->nb_jobs_workload_1 += 1;
				running_cores_from_workload_1 += j->cores;
			}
			
			if (overhead_of_load > 0 && j->node_used->end_of_file_load < t + overhead_of_load)
			{
				j->node_used->end_of_file_load = t + overhead_of_load;
			}
			
			for (i = 0; i < j->cores; i++)
			{
				if (overhead_of_load > 0 && j->node_used->cores[j->cores_used[i]]->end_of_file_load < t + overhead_of_load)
				{
					j->node_used->cores[j->cores_used[i]]->end_of_file_load = t + overhead_of_load;
				}
			}		
			
			#endif
			
			j->node_used->n_available_cores -= j->cores;
			if (j->node_used->n_available_cores < 0 || j->node_used->n_available_cores > 20)
			{
				printf("ERROR ERROR start_jobs %d available cores at time %d.\n", j->node_used->n_available_cores, t); 
				print_cores_in_specific_node(j->node_used);
				exit(1);
			}
			/** End of defining cluster usage **/
						
			for (i = 0; i < j->cores; i++)
			{
				for (k = 0; k < 20; k++)
				{
					if (j->node_used->cores[k]->unique_id == j->cores_used[i])
					{
						//~ j->cores[i]->running_job = j;
						j->node_used->cores[k]->running_job = true;
						j->node_used->cores[k]->running_job_end = j->start_time + j->walltime;
						break;
					}
				}
			}
			
			/* Test with finish in start jobs instead of end jobs. */
			if (j->workload == 1)
			{
				nb_job_to_evaluate_started += 1;
			}
			to_print_job_csv(j, t);
		}
		j = j->next;
	}
	
	/* Copy and delete. */
	j = scheduled_job_list->head;
	while (j != NULL)
	{
		if (j->start_time == t)
		{
			struct Job* temp = j->next;
			copy_delete_insert_job_list(scheduled_job_list, running_jobs, j);
			j = temp;
		}
		else
		{
			j = j->next;
		}
	}
}

/* Go through running jobs to find finished jobs. */
void end_jobs(struct Job* job_list_head, int t)
{
	#ifdef PRINT
	printf("Start of end_jobs at time %d.\n", t); fflush(stdout);
	#endif
	
	int i = 0;
	int k = 0;
	struct Job* j= job_list_head;
	while(j != NULL)
	{
		if (j->end_time == t) /* A job has finished, let's remove it from the cores, write its results and figure out if we need to fill */
		//~ if (j->end_time >= t) /* A job has finished, let's remove it from the cores, write its results and figure out if we need to fill */
		{
			/* Remove from list of ending times. */
			if (end_times->head != NULL && end_times->head->time == t)
			{				
				delete_next_time_linked_list(end_times, t);
			}
				
			finished_jobs += 1;
			
			#ifdef PRINT
			printf("==> Job %d %d cores finished at time %d on node %d.\n", j->unique_id, j->cores, t, j->node_used->unique_id);
			#endif
			
			/* Just printing, can remove */
			if (finished_jobs%1000 == 0)
			{
				printf("Evaluated jobs: %d/%d | All jobs: %d/%d | T = %d.\n", nb_job_to_evaluate_started, nb_job_to_evaluate, finished_jobs, total_number_jobs, t); fflush(stdout);
			}
			
			/* For easybf */
			running_cores -= j->cores;				
			
			#ifdef PRINT
			printf("Updating running cores on job %d: %d/20 -> %d/20.\n", j->unique_id, j->node_used->n_available_cores, j->node_used->n_available_cores + j->cores);
			#endif
			
			/** Defining cluster usage **/
			j->node_used->n_available_cores += j->cores;
			if (j->node_used->n_available_cores == 20)
			{
				running_nodes -= 1;
			}

			if (j->node_used->n_available_cores < 0 || j->node_used->n_available_cores > 20)
			{
				printf("ERROR ERROR end jobs\n");
				exit(1); 
			}
			
			#ifdef PRINT_CLUSTER_USAGE
			if (j->workload == 1)
			{
				if (j->node_used->nb_jobs_workload_1 == 1)
				{
					running_nodes_workload_1 -= 1;
				}
				j->node_used->nb_jobs_workload_1 -= 1;
				running_cores_from_workload_1 -= j->cores;
			}
			
			#endif
			/** End of defining cluster usage **/
									
			for (i = 0; i < j->cores; i++)
			{
				for (k = 0; k < 20; k++)
				{
					if (j->node_used->cores[k]->unique_id == j->cores_used[i])
					{
						//~ j->cores[i]->running_job = j;
						j->node_used->cores[k]->running_job = false;
						j->node_used->cores[k]->running_job_end = -1;
						//~ printf("Running false for job %d.\n", j->unique_id);
						break;
					}
				}
			}
			
			/* Dans le cas data persistence plus besoin de nb_task_using_it */
			#ifndef DATA_PERSISTENCE
			if (j->data != 0)
			{
				remove_data_from_node(j, t);
			}
			#endif
		}
		j = j->next;
	}				
		
	/* Delete from running jobs. */
	j = job_list_head;
	while (j != NULL)
	{
		if (j->end_time == t)
		{
			struct Job* temp = j->next;
			delete_job_linked_list(running_jobs, j->unique_id);
			j = temp;
		}
		else
		{
			j = j->next;
		}
	}
}

/* Reset available times by going through the cores in each node. */
void reset_cores(struct Node_List** l, int t)
{
	#ifdef PRINT
	printf("reset_cores.\n");
	#endif
	
	biggest_hole = 0;
	global_nb_non_available_cores_at_time_t = 0;
	
	int i = 0;
	int j = 0;
	for (i = 0; i < 3; i++)
	{
		struct Node* n = l[i]->head;
		while (n != NULL)
		{
			/* Reset aussi les trou pour conservative bf. */
			//~ if(n->cores_in_a_hole != NULL)
			//~ {
				//~ printf("Free all holes node %d.\n", n->unique_id);
				//~ n->number_cores_in_a_hole = 0;
				//~ n->cores_in_a_hole = NULL;
				//~ n->start_time_of_the_hole = NULL;
				//~ n->cores_in_a_hole = NULL; /* free plutot ? */
				if (n->number_cores_in_a_hole != 0)
				{
					//~ printf("Free in reset_cores.\n"); fflush(stdout);
					free_cores_in_a_hole(&n->cores_in_a_hole->head);
					n->number_cores_in_a_hole = 0;
				}
			//~ }
			
			for (j = 0; j < 20; j++)
			{
				if (n->cores[j]->running_job == false)
				{
					n->cores[j]->available_time = t;
				}
				else
				{
					if (n->cores[j]->running_job_end == -1)
					{
						perror("error reset cores.\n");
						exit(EXIT_FAILURE);
					}
					n->cores[j]->available_time = n->cores[j]->running_job_end;
				}
			}
			
			/* Need to sort cores after a reset as well. */
			sort_cores_by_available_time_in_specific_node(n);
			
			n = n->next;
		}
	}
}

//~ int is_my_file_on_node_at_certain_time_and_transfer_time(int predicted_time, struct Node* n, int t, int current_data, int current_data_size, bool* is_being_loaded)
float is_my_file_on_node_at_certain_time_and_transfer_time(int predicted_time, struct Node* n, int t, int current_data, float current_data_size, bool* is_being_loaded)
{
	#ifdef DATA_PERSISTENCE
	struct Data* d = n->temp_data->head;
	#else
	struct Data* d = n->data->head;
	#endif
	
	//~ struct Interval* i = malloc(sizeof(struct Interval));
	
	int* temp_interval_usage_time = malloc(3*sizeof(int));
	while (d != NULL)
	{
		#ifdef PRINT
		printf("Data %d is on node %d.\n", d->unique_id, n->unique_id); fflush(stdout);
		#endif
		
		//~ struct Interval* i = malloc(sizeof(struct Interval));
		//~ i = d->intervals->head;
		struct Interval* i = d->intervals->head;
		if (d->unique_id == current_data && i != NULL)
		{
			#ifdef PRINT
			printf("Interval not empty, but is it on the node at time %d ?\n", predicted_time);
			#endif
			
			while (i != NULL)
			{
				temp_interval_usage_time[0] = i->time;
				i = i->next;
				temp_interval_usage_time[1] = i->time;
				i = i->next;
				temp_interval_usage_time[2] = i->time;
				
				#ifdef PRINT
				printf("Checking %d / %d / %d.\n", temp_interval_usage_time[0], temp_interval_usage_time[1], temp_interval_usage_time[2]);
				#endif
				
				#ifdef DATA_PERSISTENCE
				if (temp_interval_usage_time[0] <= predicted_time && temp_interval_usage_time[1] <= predicted_time)
				{
					*is_being_loaded = false;
					free(temp_interval_usage_time);
					return 0;
				}
				else if (temp_interval_usage_time[0] <= predicted_time)
				{
					*is_being_loaded = true;
					int temp = temp_interval_usage_time[1];
					free(temp_interval_usage_time);
					return temp - t;
				}
				#else
				if (temp_interval_usage_time[0] <= predicted_time && temp_interval_usage_time[1] <= predicted_time && predicted_time <= temp_interval_usage_time[2])
				{
					*is_being_loaded = false;
					free(temp_interval_usage_time);
					return 0;
				}
				else if (temp_interval_usage_time[0] <= predicted_time && predicted_time <= temp_interval_usage_time[2])
				{
					*is_being_loaded = true;
					int temp = temp_interval_usage_time[1];
					free(temp_interval_usage_time);
					return temp - t;
				}
				#endif
				
				i = i->next;
			}
			break;
		}
		d = d->next;
	}
	*is_being_loaded = false;
	//~ free(i);
	free(temp_interval_usage_time);
	return current_data_size/n->bandwidth;
}

/* % of space you will take and thus time it will take to reload evicted data. */
//~ float time_to_reload_percentage_of_files_ended_at_certain_time(int predicted_time, struct Node* n, int current_data, int percentage_occupied)
float time_to_reload_percentage_of_files_ended_at_certain_time(int predicted_time, struct Node* n, int current_data, float percentage_occupied)
{
	//~ int size_file_ended = 0;
	float size_file_ended = 0;

	#ifdef DATA_PERSISTENCE
	struct Data* d = n->temp_data->head;
	#else
	struct Data* d = n->data->head;
	#endif
	
	while (d != NULL)
	{
		if (d->unique_id != current_data && d->intervals->head != NULL)
		{
			#ifdef PRINT
			printf("Checking tail of the interval of data %d: %d->\n", d->unique_id, d->intervals->tail->time);
			#endif
			
			if (predicted_time >= d->intervals->tail->time)
			{
				size_file_ended += d->size;
				
				#ifdef PRINT
				printf("Add size %f->\n", d->size); fflush(stdout);
				#endif
			}
		}
		d = d->next;
		//~ free(i);
	}
	
	#ifdef PRINT
	printf("Total size of data on node ending before my EAT is: %f but I return (%f*%f)/%f = %f.\n", size_file_ended, percentage_occupied, size_file_ended, n->bandwidth, (size_file_ended*percentage_occupied)/n->bandwidth); fflush(stdout);
	#endif
		
	return (size_file_ended*percentage_occupied)/n->bandwidth;
}

int get_nb_valid_copy_of_a_file(int predicted_time, struct Node_List** head_node, int current_data)
{
	int nb_of_copy = 0;
	int j = 0;
	int* temp_interval_usage_time = malloc(2*sizeof(int));
	
	for (j = 0; j < 3; j++)
	{
		struct Node* n = head_node[j]->head;
		while(n != NULL)
		{
			#ifdef DATA_PERSISTENCE
			struct Data* d = n->temp_data->head;
			#else
			struct Data* d = n->data->head;
			#endif
			
			while (d != NULL)
			{
				struct Interval* i = d->intervals->head;
				if (d->unique_id == current_data && i != NULL)
				{
					#ifdef PRINT
					printf("Data %d is on node %d but at predicted time %d?\n", d->unique_id, n->unique_id, predicted_time);
					#endif
					
					while (i != NULL)
					{
						temp_interval_usage_time[0] = i->time;
						i = i->next;
						i = i->next;
						temp_interval_usage_time[1] = i->time;
						
						#ifdef PRINT
						printf("Checking %d / %d.\n", temp_interval_usage_time[0], temp_interval_usage_time[1]);
						#endif
						
						#ifdef DATA_PERSISTENCE
						if (temp_interval_usage_time[0] <= predicted_time)
						{
							nb_of_copy += 1;
							break;
						}
						#else
						if (temp_interval_usage_time[0] <= predicted_time && predicted_time <= temp_interval_usage_time[1])
						{
							nb_of_copy += 1;
							break;
						}
						#endif
						
						i = i->next;
					}
					break;
				}
				d = d->next;
			}
			n = n->next;
		}
	}
	free(temp_interval_usage_time);
	return nb_of_copy;
}
					
int was_time_or_data_already_checked_for_nb_copy(int t_or_d, struct Time_or_Data_Already_Checked_Nb_of_Copy_List* list)
{
	struct Time_or_Data_Already_Checked_Nb_of_Copy* a = list->head;
	while (a != NULL)
	{
		if (a->time_or_data == t_or_d)
		{
			return a->nb_of_copy;
		}
		a = a->next;
	}
	return -1;
}

int schedule_job_to_start_immediatly_on_specific_node_size(struct Job* j, struct Node_List* head_node_size_i, int t, int backfill_big_node_mode, int total_queue_time, int nb_finished_jobs, int nb_non_available_cores, bool* result)
{
	int mean_queue_time = 0;
	int earliest_available_time = 0;
	int threshold_for_a_start = 0;
	int i = 0;
	
	if (nb_finished_jobs == 0)
	{
		mean_queue_time = 0;
	}
	else
	{
		mean_queue_time = total_queue_time/nb_finished_jobs;
	}
	
	struct Node* n = head_node_size_i->head;
	while(n != NULL)
	{
		earliest_available_time = n->cores[j->cores - 1]->available_time; /* -1 because tab start at 0 */
		
		if (backfill_big_node_mode == 0)
		{
			threshold_for_a_start = t;
		}
		else if (backfill_big_node_mode == 1)
		{
			if (mean_queue_time - (t - j->subtime) > 0)
			{
				threshold_for_a_start = t + mean_queue_time - (t - j->subtime);
			}
			else
			{
				threshold_for_a_start = t;
			}
		}
		else
		{
			perror("Error on backfill_big_node_mode, must be 0 or 1.\n");
			exit(EXIT_FAILURE);
		}
			
		if (earliest_available_time <= threshold_for_a_start) /* Ok I can start immediatly, schedule job and return true. */
		{
			/* Update infos on the job and on cores. */
			j->node_used = n;
			j->start_time = earliest_available_time;
			j->end_time = earliest_available_time + j->walltime;
			for (i = 0; i < j->cores; i++)
			{
				j->cores_used[i] = j->node_used->cores[i]->unique_id;
				if (j->node_used->cores[i]->available_time <= t)
				{
					nb_non_available_cores += 1;
				}
				j->node_used->cores[i]->available_time = earliest_available_time + j->walltime;
				
			}
		
			#ifdef PRINT
			print_decision_in_scheduler(j);
			#endif
		
			/* Need to sort cores after each schedule of a job. */
			sort_cores_by_available_time_in_specific_node(j->node_used);
			
			*result = true;
			return nb_non_available_cores;
		}
		n = n->next;
	}
	return nb_non_available_cores;
}

int try_to_start_job_immediatly_without_delaying_j1(struct Job* j, struct Job* j1, struct Node_List** head_node, int nb_running_cores, bool* result, bool use_bigger_nodes, int t)
{
	int earliest_available_time = 0;
	int i = 0;
	int k = 0;
	bool ok_on_this_node = false;
	
	/* TODO: To try and reduce complexity a little I look at the available cores on each node and if it's inferior to j->cores I don't bother looking. Is it worth it ? Idk. */
	
	int l = 0;
	
	bool need_to_break = false;
	
	/* Finding the node with the earliest available time. */
	for (l = first_node_size_to_choose_from; l <= last_node_size_to_choose_from; l++)
	{
		struct Node* n = head_node[l]->head;
		while(n != NULL)
		{
			ok_on_this_node = false;
			earliest_available_time = n->cores[j->cores - 1]->available_time; /* -1 because tab start at 0 */
						
			if (earliest_available_time <= t) /* Ok I can start immediatly, schedule job and return true. */
			{
				ok_on_this_node = true;
				earliest_available_time = t;
				/* But is it the same node as j1 ? If yes I need to be careful. */
				if (n->unique_id == j1->node_used->unique_id)
				{
					if (earliest_available_time + j->walltime > j1->start_time) /* It will finish later so I need to check if it's the same cores. If yes I can't do it. */
					{
						need_to_break = false;
						for (i = 0; i < j->cores; i++)
						{
							for (k = 0; k < j1->cores; k++)
							{
								//~ printf("Testing %d == %d ?\n", n->cores[i]->unique_id, j1->cores_used[k]);
								if (n->cores[i]->unique_id == j1->cores_used[k])
								{
									/* Need to exit. */
									ok_on_this_node = false;
									//~ printf("%d == %d.\n", n->cores[i]->unique_id, j1->cores_used[k]);
									need_to_break = true;
									break;
								}
							}
							if (need_to_break == true)
							{
								break;
							}
						}
					}
				}
				
				if (ok_on_this_node == true)
				{
					/* Update infos on the job and on cores. */
					j->node_used = n;
					j->start_time = earliest_available_time;
					j->end_time = earliest_available_time + j->walltime;
					nb_running_cores += j->cores;
					for (i = 0; i < j->cores; i++)
					{
						j->cores_used[i] = j->node_used->cores[i]->unique_id;
						j->node_used->cores[i]->available_time = earliest_available_time + j->walltime;
						
						/* Maybe I need job queue or not not sure. TODO. */
						//~ copy_job_and_insert_tail_job_list(n->cores[i]->job_queue, j);
					}
				
					#ifdef PRINT
					print_decision_in_scheduler(j);
					#endif
				
					/* Need to sort cores after each schedule of a job. */
					sort_cores_by_available_time_in_specific_node(j->node_used);
					
					*result = true;
					return nb_running_cores;
				}
			}
			n = n->next;
		}
	}
	return nb_running_cores;
}

int get_earliest_available_time_specific_sublist_node(int nb_cores_asked, struct Node_List* head_node_size_i, struct Node** choosen_node, int t)
{
	int min_time = -1;
	int earliest_available_time = 0;
	struct Node* n = head_node_size_i->head;
	while (n != NULL)
	{			
		earliest_available_time = n->cores[nb_cores_asked - 1]->available_time; /* -1 because tab start at 0 */
		if (earliest_available_time < t) /* A core can't be available before t. This happens when a node is idling. */				
		{
			earliest_available_time = t;
		}
		if (min_time == -1 || min_time > earliest_available_time)
		{
			min_time = earliest_available_time;
			*choosen_node = n;
		}			
		n = n->next;
	}
				
	return min_time;
}

void sort_tab_of_int_decreasing_order(long long arr[], int n)
{
    int i, j, min_idx;
 
    for (i = 0; i < n - 1; i++) {
 
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] > arr[min_idx])
                min_idx = j;
 
        swap(&arr[min_idx], &arr[i]);
        
    }
}

void swap(long long* xp, long long* yp)
{
    long long temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void get_node_size_to_choose_from(int index, int* first_node_size_to_choose_from, int* last_node_size_to_choose_from)
{
	if (constraint_on_sizes != 3)
	{
		if (index == 0)
		{
			*first_node_size_to_choose_from = 0;
			*last_node_size_to_choose_from = 2;
		}
		else if (index == 1)
		{
			*first_node_size_to_choose_from = 1;
			*last_node_size_to_choose_from = 2;
		}
		else if (index == 2)
		{
			*first_node_size_to_choose_from = 2;
			*last_node_size_to_choose_from = 2;
		}
		else
		{
			printf("Error index value in schedule_job_on_earliest_available_cores.\n");  fflush(stdout);
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		*first_node_size_to_choose_from = index;
		*last_node_size_to_choose_from = index;
	}
}

void get_nb_nodes_and_cores_loading_a_file(struct Node_List** head_node, int t, int* nodes_loading_a_file, int* cores_loading_a_file)
{
	#ifdef PRINT_CLUSTER_USAGE
	int i = 0;
	int j = 0;
	*nodes_loading_a_file = 0;
	*cores_loading_a_file = 0;
	
	for (i = 0; i < 3; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			if (n->end_of_file_load > t)
			{
				*nodes_loading_a_file += 1;
			}
			
			for (j = 0; j < 20; j++)
			{
				if (n->cores[j]->end_of_file_load > t)
				{
					*cores_loading_a_file += 1;
				}
			}
			
			n = n->next;
		}
	}
	#endif
}
