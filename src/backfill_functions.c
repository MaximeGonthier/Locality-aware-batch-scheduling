#include <main.h>

bool can_it_get_backfilled (struct Job* j, struct Node* n, int t, int* nb_cores_from_hole, int* nb_cores_from_outside)
{
	*nb_cores_from_hole = 0;
	*nb_cores_from_outside = 0;
	int k = 0;
	struct Core_in_a_hole* c = n->cores_in_a_hole->head;
	
	#ifdef PRINT
	printf("Can we backfill job %d on node %d?\n", j->unique_id, n->unique_id); fflush(stdout);
	#endif
	
	
	if (n->number_cores_in_a_hole != 0 && (j->cores <= n->number_cores_in_a_hole || n->cores[j->cores - 1 - n->number_cores_in_a_hole]->available_time <= t))
	{
		#ifdef PRINT
		printf("It could maybe fit or partially fit in the %d cores composing the hole of node %d.\n", n->number_cores_in_a_hole, n->unique_id); fflush(stdout);
		#endif
		
		
		c = n->cores_in_a_hole->head;
		
				
		for (k = 0; k < n->number_cores_in_a_hole; k++)
		{
			//~ printf("Checking core %d next start time %d.\n", c->unique_id, c->start_time_of_the_hole); fflush(stdout);
			
			if (t + j->walltime <= c->start_time_of_the_hole)
			{
				*nb_cores_from_hole += 1;
			}
			
			//~ if (c->next != NULL)
			//~ {
			//~ printf("next\n"); fflush(stdout);
			
			c = c->next;
			
			//~ printf("next done\n"); fflush(stdout);
			
			//~ }
			//~ else
			//~ {
				//~ break;
			//~ }
			if (j->cores == *nb_cores_from_hole)
			{
				//~ printf("break\n"); fflush(stdout);
				break;
			}
		}
		//~ exit(1);
		
		//~ #ifdef PRINT
		//~ printf("nb_cores_from_hole = %d.\n", *nb_cores_from_hole);
		//~ #endif
					
		if (nb_cores_from_hole > 0)
		{
			*nb_cores_from_outside = j->cores - *nb_cores_from_hole;
			//~ nb_cores_from_outside = nb_cores_from_outside;
		
			#ifdef PRINT
			printf("Number of cores from the hole/outside: %d/%d.\n", *nb_cores_from_hole, *nb_cores_from_outside); fflush(stdout);
			#endif
						
			//~ k = 0;
			//~ while (nb_cores_from_outside != 0)
			for (k = 0; k < *nb_cores_from_outside; k++)
			{
				if (n->cores[k]->available_time > t)
				{
					//~ nb_cores_from_outside--;
					//~ k++;
				//~ }
				//~ else
				//~ {
					#ifdef PRINT
					printf("Could not fit in the hole because outside cores don't match. Return false.\n"); fflush(stdout);
					#endif
					
					return false;
				}
			}
			//~ exit(1);
			
			//~ if (nb_cores_from_outside == 0)
			if (k == *nb_cores_from_outside)
			{
				/* On break et on met à vrai le booleen pour dire que le remplissage des cores sera différent et qu'il faut pas sort les cores de la node et qu'il faut mettre à jour le nombre de cores dans un trou de la node et mettre à jour le nb de non available cores. */
				#ifdef PRINT
				printf("Can fit in the hole and start at time t = %d. Return true.\n", t); fflush(stdout);
				#endif
				
				//~ backfilled_job = true;	
				//~ min_time = t;
				//~ j->node_used = n;
				//~ i = last_node_size_to_choose_from + 1;						
				//~ break;
				return true;
			}
		}
	}
	#ifdef PRINT
	printf("Could not fit or times not available. Return false.\n"); fflush(stdout);
	#endif
	return false;
}							

void update_cores_for_backfilled_job(struct Job* j, int t, int nb_cores_from_hole, int nb_cores_from_outside)
{
	//~ struct Core_in_a_hole* c = (struct Core_in_a_hole*) malloc(sizeof(struct Core_in_a_hole));
	int i = 0;
	int k = 0;
	
	#ifdef PLOT_STATS
	number_of_backfilled_jobs += 1;
	#endif
	
	/* Ca ajoute des unavailable cores puisque c'est à t. */
	//~ nb_non_available_cores_at_time_t += j->cores;
	
	/* Mettre les cores dans le job depuis ceux du trou. */
	struct Core_in_a_hole* c = j->node_used->cores_in_a_hole->head;
	
	if (nb_cores_from_hole > j->cores || nb_cores_from_hole > j->node_used->number_cores_in_a_hole)
	{ 
		printf("Error cores: %d cores taken from hole, %d cores on the job, %d cores in a hole of the node.\n", nb_cores_from_hole, j->cores, j->node_used->number_cores_in_a_hole); 
		fflush(stdout); 
		exit(EXIT_FAILURE); 
	}
	
	for (k = 0; k < nb_cores_from_hole;)
	{
		if (t + j->walltime <= c->start_time_of_the_hole)
		{
			j->cores_used[i] = c->unique_id;
			i++;
			k++;
			
			//~ #ifdef PRINT
			//~ if (j->unique_id == 17269)
			//~ {
				//~ printf("Adding %d in cores used from hole of node %d (next start time is %d) at i = %d.\n", c->unique_id, j->node_used->unique_id, c->start_time_of_the_hole, i);
			//~ }
			//~ #endif
		}
		//~ else if (j->unique_id == 17269) { printf("t + walltime = %d. Could not use hole %d that start nextat time %d.\n", t + j->walltime, c->unique_id, c->start_time_of_the_hole); }
	
		/* Je rentre la dedans ? je crois pas. Niormalement non, c'est un bug lié à un appel de cet fonction alors que le trou est pas remplissable par le job. */
		if (c->next == NULL && k != nb_cores_from_hole)
		{
			printf("Error: Next is null at time %d.\nJob %d uses %d cores.\ni = %d.\nk = %d.\nnb_cores_from_hole = %d.\nNode %d has %d cores in a hole.\n", t, j->unique_id, j->cores, i, k, nb_cores_from_hole, j->node_used->unique_id, j->node_used->number_cores_in_a_hole); fflush(stdout);
			exit(EXIT_FAILURE);
		}
		
		c = c->next;
	}
	
	for (k = 0; k < nb_cores_from_outside; k++)
	{
		#ifdef PRINT
		printf("Adding core from outside %d.\n", j->node_used->cores[k]->unique_id);
		#endif
				
		j->cores_used[i] = j->node_used->cores[k]->unique_id;
		j->node_used->cores[k]->available_time = t + j->walltime;
		i++;
	}		
	
	j->node_used->number_cores_in_a_hole -= nb_cores_from_hole;

	if (j->node_used->number_cores_in_a_hole < 0 || j->node_used->number_cores_in_a_hole > 19)
	{
		printf("Error nb core in hole %d on node %d.\n", j->node_used->number_cores_in_a_hole, j->node_used->unique_id); 
		fflush(stdout);
		exit(EXIT_FAILURE);
	}
	
	if (j->node_used->number_cores_in_a_hole == 0)
	{
		#ifdef PRINT
		printf("Deleting all the cores in the hole cause we use them all.\n");
		#endif
				
		free_cores_in_a_hole(&j->node_used->cores_in_a_hole->head);
	}
	else
	{
		for (i = 0; i < nb_cores_from_hole; i++)
		{
			#ifdef PRINT
			printf("Deleting core %d.\n", j->cores_used[i]);
			#endif
				
			delete_core_in_hole_specific_core(j->node_used->cores_in_a_hole, j->cores_used[i]);
		}
	}
	
	#ifdef PRINT
	printf("Holes on node %d after this backfill are:\n", j->node_used->unique_id);
	print_holes_specific_node(j->node_used);
	#endif
	
	//~ return nb_non_available_cores_at_time_t;
}

void fill_cores_minimize_holes (struct Job* j, bool backfill_activated, int backfill_mode, int t, int* nb_non_available_cores)
{
	int i = 0;
	int k = 0;
	sort_cores_by_unique_id_in_specific_node(j->node_used); /* Attention, il faut faire gaffe que dans le scheduler c'est ensuite re sort dans le sens des temps disponible le plus tôt. */
		
	for (k = 0; k < 20; k++)
	{
		//~ while()
		//~ {
			if (j->node_used->cores[k]->available_time <= j->start_time) /* Si c'est un core que je peux sélectionner */
			{
				j->cores_used[i] = j->node_used->cores[k]->unique_id;
				
				i++;
				
				if (j->node_used->cores[k]->available_time <= t)
				{
					*nb_non_available_cores += 1;
				}
						
				if (backfill_activated == true)
				{
					/* Spécifique au cas avec backfilling */
					if (j->node_used->cores[k]->available_time <= t && j->start_time > t)
					{
						#ifdef PRINT
						printf("Il va y avoir un trou sur node %d core %d.\n", j->node_used->unique_id, j->node_used->cores[k]->unique_id); fflush(stdout);
						#endif
						
						j->node_used->number_cores_in_a_hole += 1;
						struct Core_in_a_hole* new = (struct Core_in_a_hole*) malloc(sizeof(struct Core_in_a_hole));
						new->unique_id = j->node_used->cores[k]->unique_id;
						new->start_time_of_the_hole = j->start_time;
						new->next = NULL;
						
						if (j->node_used->cores_in_a_hole == NULL)
						{
							initialize_cores_in_a_hole(j->node_used->cores_in_a_hole, new);
						}
						else
						{
							if (backfill_mode == 2) /* Favorise les jobs backfill car se met sur le coeurs qui a le temps le plus petit possible. */
							{
								insert_cores_in_a_hole_list_sorted_increasing_order(j->node_used->cores_in_a_hole, new);
							}
							else
							{
								insert_cores_in_a_hole_list_sorted_decreasing_order(j->node_used->cores_in_a_hole, new);
							}
						}
					}
					/* Fin de spécifique au cas avec backfilling */
				}

				j->node_used->cores[k]->available_time = j->start_time + j->walltime;
				//~ k++; /* k est pas censé valoir 20 la quand même */
				//~ break;
			}
			//~ k++;
			// if (k > 20) { break; }
			if (i == j->cores)
			{
				break;
			}
		//~ }
	}
	
	if (i != j->cores) { printf("Error fill_cores_minimize_holes. i = %d job %d j->cores = %d j->start time is %d\n", i, j->unique_id, j->cores, j->start_time); printf("Dans l'erreur\n"); print_cores_in_specific_node(j->node_used); exit(1); }
}

/** 
 * Only check if I can backfill a job. If I can I do.
 * There is an option for a locality backfil that only backfill if no data is evicted.
 * I don't need to check cores from outside car dans ce cas tout est recouvert déjà.
 */
bool only_check_conservative_backfill(struct Job* j, struct Node_List** head_node, int t, int backfill_mode, int* nb_non_available_cores_at_time_t)
{
	/* NEW core selection conservative bf only */
	int nb_cores_from_hole = 0;
	int nb_cores_from_outside = 0;
	/* End of NEW core selection conservative bf only */
		
	#ifdef PRINT
	printf("\nScheduling job %d at time %d. Backfill mode is %d.\n", j->unique_id, t, backfill_mode);
	#endif

	int i = 0;
	int min_time = -1;
	//~ int first_node_size_to_choose_from = 0;
	//~ int last_node_size_to_choose_from = 0;
	bool backfilled_job = false;
	//~ bool is_being_loaded = false;
	
	for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			#ifdef PRINT
			printf("Checking node %d.\n", n->unique_id);
			#endif
					
			//~ backfilled_job = can_it_get_backfilled(j, n, t, &nb_cores_from_hole, &nb_cores_from_outside);
			// OR pas besoin de can it get backfilled car je peux juste vérifier le trou, y aura pas de cores from outside de toute façon dans ce cas car tout est couvert déjà! Le seul cas à gérer c'est que je peux avoir deux trou de tailles différentes!
			if (n->number_cores_in_a_hole > 0 && j->cores <= n->number_cores_in_a_hole)
			{
				/* Dans le cas localité utilisé par fcfs with a score je veux backfill que si j'évince pas une donnée; Pas utilisé dans le cas fcfs ou eft. */
				//~ if (locality_backfill == false || is_my_file_on_node_at_certain_time_and_transfer_time(t, n, t, j->data, j->data_size, &is_being_loaded) == 0 || is_being_loaded == true || time_to_reload_percentage_of_files_ended_at_certain_time(t, n, j->data, (float) j->cores/20) == 0)
				//~ {
					nb_cores_from_hole = 0;
					struct Core_in_a_hole* c = n->cores_in_a_hole->head;
					for (int k = 0; k < n->number_cores_in_a_hole; k++)
					{
						if (t + j->walltime <= c->start_time_of_the_hole)
						{
							nb_cores_from_hole += 1;
						}
						
						if (j->cores == nb_cores_from_hole)
						{
							backfilled_job = true;
							break;
						}
						c = c->next;
					}
				//~ }
			}
					
			if (backfilled_job == true)
			{
				min_time = t;
				j->node_used = n;
				i = last_node_size_to_choose_from + 1;
				j->start_time = min_time;
				j->end_time = min_time + j->walltime;
				update_cores_for_backfilled_job(j, t, nb_cores_from_hole, nb_cores_from_outside);
				*nb_non_available_cores_at_time_t += j->cores;
					
				/* NEW core selection */
				if (nb_cores_from_outside > 0)
				{
					sort_cores_by_available_time_in_specific_node(j->node_used);
				}
				/* End of NEW core selection */
			
				if (j->node_used->unique_id == biggest_hole_unique_id)
				{
					get_new_biggest_hole(head_node);
				}
			
				#ifdef PRINT
				print_decision_in_scheduler(j);
				print_cores_in_specific_node(j->node_used);
				#endif
						
				return true;
			}
			n = n->next;
		}
	}
	return false;
}

bool only_check_conservative_backfill_with_a_score(struct Job* j, struct Node_List** head_node, int t, int backfill_mode, int* nb_non_available_cores_at_time_t, int multiplier_file_to_load, int multiplier_file_evicted, int start_immediately_if_EAT_is_t)
{
	/* NEW core selection conservative bf only */
	int nb_cores_from_hole = 0;
	int nb_cores_from_outside = 0;
	/* End of NEW core selection conservative bf only */
		
	#ifdef PRINT
	printf("\nScheduling job %d at time %d. Backfill mode is %d.\n", j->unique_id, t, backfill_mode);
	#endif

	int i = 0;
	//~ int min_time = -1;
	//~ int first_node_size_to_choose_from = 0;
	//~ int last_node_size_to_choose_from = 0;
	bool backfilled_job = false;
	bool can_backfill_on_this_node = false;
	
	int min_score = -1;
	float time_to_load_file = 0;
	bool is_being_loaded = false;
	float time_to_reload_evicted_files = 0;
	int score = 0;
	int choosen_time_to_load_file = 0;
		
	if (start_immediately_if_EAT_is_t == 1)
	{
		multiplier_file_to_load = 1;
		multiplier_file_evicted = 0;
	}

	for (i = first_node_size_to_choose_from; i <= last_node_size_to_choose_from; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			can_backfill_on_this_node = false;
			
			#ifdef PRINT
			printf("Checking node %d.\n", n->unique_id);
			#endif
			
			if (n->number_cores_in_a_hole > 0 && j->cores <= n->number_cores_in_a_hole)
			{
				nb_cores_from_hole = 0;
				struct Core_in_a_hole* c = n->cores_in_a_hole->head;
				for (int k = 0; k < n->number_cores_in_a_hole; k++)
				{
					if (t + j->walltime <= c->start_time_of_the_hole)
					{
						nb_cores_from_hole += 1;
					}
						
					if (j->cores == nb_cores_from_hole)
					{
						can_backfill_on_this_node = true;
						break;
					}
					c = c->next;
				}
			}
					
			if (can_backfill_on_this_node == true)
			{
				backfilled_job = true;
				//~ if (min_score == -1 || earliest_available_time < min_score)
				//~ {
					if (j->data == 0)
					{
						time_to_load_file = 0;
					}
					else
					{
						time_to_load_file = is_my_file_on_node_at_certain_time_and_transfer_time(t, n, t, j->data, j->data_size, &is_being_loaded);
					}
																	
					if (min_score == -1 || multiplier_file_to_load*time_to_load_file < min_score)
					{
						if (multiplier_file_evicted == 0)
						{
							time_to_reload_evicted_files = 0;
						}
						else
						{
							time_to_reload_evicted_files = time_to_reload_percentage_of_files_ended_at_certain_time(t, n, j->data, (float) j->cores/20);
						}
																
						score = multiplier_file_to_load*time_to_load_file + multiplier_file_evicted*time_to_reload_evicted_files;
																													
						if (min_score == -1 || min_score > score)
						{
							min_score = score;
							j->node_used = n;
							choosen_time_to_load_file = time_to_load_file;
										
							/* maj choosen numbers is important for fcs score. */
							//~ choosen_nb_cores_from_hole = nb_cores_from_hole;
							//~ choosen_nb_cores_from_outside = nb_cores_from_outside;
										
							if (min_score == 0) /* Temps de début est t et pas de temps de chargements du tout */
							{			
								//~ printf("break.\n");
								i = last_node_size_to_choose_from + 1;
								break;
							}
						}
					}
				//~ }
			}
			n = n->next;
		}
	}
	
	if(backfilled_job == true)
	{
		//~ j->node_used = n;
		j->start_time = t;
		j->end_time = t + j->walltime;
		j->transfer_time = choosen_time_to_load_file;
		//~ update_cores_for_backfilled_job(j, t, nb_cores_from_hole, nb_cores_from_outside);
		update_cores_for_backfilled_job(j, t, j->cores, 0);
		*nb_non_available_cores_at_time_t += j->cores;
					
		/* NEW core selection */
		if (nb_cores_from_outside > 0)
		{
			sort_cores_by_available_time_in_specific_node(j->node_used);
		}
		/* End of NEW core selection */
			
		if (j->node_used->unique_id == biggest_hole_unique_id)
		{
			get_new_biggest_hole(head_node);
		}
			
		#ifdef PRINT
		print_decision_in_scheduler(j);
		print_cores_in_specific_node(j->node_used);
		#endif
		
		return true;
	}

	return false;
}

void get_new_biggest_hole(struct Node_List** head_node)
{
	biggest_hole = -1;
	for (int i = 0; i < 3; i++)
	{
		struct Node* n = head_node[i]->head;
		while (n != NULL)
		{
			if (n->number_cores_in_a_hole > biggest_hole)
			{
				biggest_hole = n->number_cores_in_a_hole;
				biggest_hole_unique_id = n->unique_id;
			}
			n = n->next;
		}
	}
}
