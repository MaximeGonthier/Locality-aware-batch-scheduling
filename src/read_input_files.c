#include <main.h>

void read_cluster(char* input_node_file)
{
	node_list = (struct Node_List**) malloc(3*sizeof(struct Node_List));
	for (int i = 0; i < 3; i++)
	{
		node_list[i] = (struct Node_List*) malloc(sizeof(struct Node_List));
		node_list[i]->head = NULL;
		node_list[i]->tail = NULL;
	}
	total_number_nodes = 0;
	
	FILE *f = fopen(input_node_file, "r");

	if (!f)
	{
		perror("fopen");
        exit(EXIT_FAILURE);
	}
    
	number_node_size[0] = 0;
	number_node_size[1] = 0;
	number_node_size[2] = 0;
    
    char s[100];
    char id[100];
    char memory[100];
    char bandwidth[100];
    char core[100];
    int index_node = 0;
    while (fscanf(f, "%s %s %s %s %s %s %s %s %s %s", s, s, id, s, memory, s, bandwidth, s, core, s) == 10)
	{
		struct Node *new = (struct Node*) malloc(sizeof(struct Node));
		new->unique_id = atoi(id);
		new->memory = atoi(memory);
		new->bandwidth = atof(bandwidth);
		new->n_available_cores = 20;
		
		if (constraint_on_sizes != 0)
		{
			if (new->memory == 128)
			{
				number_node_size[0] += 1;
				index_node = 0;
			}
			else if (new->memory == 256)
			{
				number_node_size[1] += 1;
				index_node = 1;
			}
			else if (new->memory == 1024)
			{
				number_node_size[2] += 1;
				index_node = 2;
			}
			else
			{
				perror("Error memory size in read_cluster"); fflush(stdout);
				exit(EXIT_FAILURE);
			}
		}
		new->index_node_list = index_node;
		
		new->data = malloc(sizeof(*new->data));
		new->data->head = NULL;
		new->data->tail = NULL;
		
		//~ new->cores = malloc(20*sizeof(*new->cores));
		new->cores = (struct Core**) malloc(20*sizeof(struct Core));
		for (int i = 0; i < 20; i++)
		{
			//~ new->cores[i] = malloc(sizeof(*new->cores));
			new->cores[i] = (struct Core*) malloc(sizeof(struct Core));
			new->cores[i]->unique_id = i;
			//~ new->cores[i]->job_queue = malloc(sizeof(*new->cores[i]->job_queue));
			//~ new->cores[i]->job_queue->head = NULL;
			//~ new->cores[i]->job_queue->tail = NULL;
			new->cores[i]->available_time = 0;
			new->cores[i]->running_job = false;
			new->cores[i]->running_job_end = -1;
			
			#ifdef PRINT_CLUSTER_USAGE
			new->end_of_file_load = 0; /* Used to then get the total number of cores running a load */
			#endif
		}
		
		/* For conservative bf */
		new->number_cores_in_a_hole = 0;
		new->cores_in_a_hole = malloc(sizeof(*new->cores_in_a_hole));
		new->cores_in_a_hole->head = NULL;
		new->cores_in_a_hole->tail = NULL;
		
		#ifdef DATA_PERSISTENCE
		new->data_occupation = 0; /* From 0 to 128 */
		new->temp_data = malloc(sizeof(*new->temp_data));
		new->temp_data->head = NULL;
		new->temp_data->tail = NULL;
		#endif
		
		#ifdef PRINT_CLUSTER_USAGE
		new->nb_jobs_workload_1 = 0;
		new->end_of_file_load = 0;
		#endif

		new->next = NULL;
		if (new->memory == 128)
		{
			insert_tail_node_list(node_list[0], new);
			//~ node_list[0]->number_of_node += 1;
		}
		else if (new->memory == 256)
		{
			insert_tail_node_list(node_list[1], new);
			//~ node_list[1]->number_of_node += 1;
		}
		else if (new->memory == 1024)
		{
			insert_tail_node_list(node_list[2], new);
			//~ node_list[2]->number_of_node += 1;
		}
		else
		{
			printf("Error cluster\n");
			exit(EXIT_FAILURE);
		}
		total_number_nodes += 1;
		
		nb_cores += 20;
	}
 	fclose(f);
}

//~ struct Job* read_workload(char* input_job_file, int constraint_on_sizes)
void read_workload(char* input_job_file, int constraint_on_sizes)
{
	//~ job_list = malloc(sizeof(*job_list));
	job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	
	//~ /** For mixed decreasing strategy **/
	//~ data_list = (struct Data_List*) malloc(sizeof(struct Data_List));
	//~ /** For mixed decreasing strategy **/
	
	job_list->head = NULL;
	job_list->tail = NULL;
	//~ job_list_to_start_from_history = malloc(sizeof(*job_list_to_start_from_history));
	job_list_to_start_from_history = (struct Job_List*) malloc(sizeof(struct Job_List));
	job_list_to_start_from_history->head = NULL;
	job_list_to_start_from_history->tail = NULL;
	//~ scheduled_job_list = malloc(sizeof(*scheduled_job_list));
	scheduled_job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	scheduled_job_list->head = NULL;
	scheduled_job_list->tail = NULL;
	//~ running_jobs = malloc(sizeof(*running_jobs));
	running_jobs = (struct Job_List*) malloc(sizeof(struct Job_List));
	running_jobs->head = NULL;
	running_jobs->tail = NULL;
	//~ new_job_list = malloc(sizeof(*new_job_list));
	new_job_list = (struct Job_List*) malloc(sizeof(struct Job_List));
	new_job_list->head = NULL;
	new_job_list->tail = NULL;
	
	FILE *f = fopen(input_job_file, "r");
	if (!f)
	{
		perror("fopen");
        exit(EXIT_FAILURE);
	}
	
	int index_node = 0;
	char s[100];
    char id[100];
    char subtime[100];
    char delay[100];
    char walltime[100];
    char cores[100];
    char data[100];
    char data_size[100];
    char workload[100];
    char start_time_from_history[100];
    char start_node_from_history[100];
    
    int user_id = 0;
    char* current_user = malloc(sizeof(char)*9);
    char* last_user = malloc(sizeof(char)*9);
    //~ current_user = "";
    //~ last_user = "";
    
    //~ /** For mixed decreasing strategy **/
    //~ struct Data* d = data_list->head;
    //~ /** For mixed decreasing strategy **/
    
    while (fscanf(f, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s", s, s, id, s, subtime, s, delay, s, walltime, s, cores, s, current_user, s, data, s, data_size, s, workload, s, start_time_from_history, s, start_node_from_history, s) == 24)
	{
		total_number_jobs += 1;
		
		struct Job *new = (struct Job*) malloc(sizeof(struct Job));
		new->unique_id = atoi(id);
		new->subtime = atoi(subtime);
		new->delay = atoi(delay);	
		new->walltime = atoi(walltime);
		new->cores = atoi(cores);
		//~ /** For mixed decreasing strategy **/
		new->data = atoi(data);
		new->data_size = atof(data_size);
		//~ /** For mixed decreasing strategy **/
		new->workload = atoi(workload);
		new->start_time_from_history = atoi(start_time_from_history);
		new->node_from_history = atoi(start_node_from_history);
		
		//~ printf("%s\n", current_user); fflush(stdout);
		if (strcmp(current_user, last_user) == 0)
		{
			new->user = user_id;
		}
		else
		{
			strcpy(last_user, current_user);
			user_id++;
			new->user = user_id;
		}
		
		
		/* Get index_node */
		if (constraint_on_sizes != 0)
		{
			if ((atof(data_size)*10)/(atoi(cores)*10) == 0.0)
			{
				index_node = 0;
			}
			else if ((atof(data_size)*10)/(atoi(cores)*10) == 6.4)
			{
				index_node = 0;
			}
			else if ((atof(data_size)*10)/(atoi(cores)*10) == 12.8)
			{
				index_node = 1;
			}
			else if ((atof(data_size)*10)/(atoi(cores)*10) == 51.2)
			{
				index_node = 2;
			}
			else
			{
				printf("Job %d: %f x 10 divided by %d x 10 = %f\n", atoi(id), atof(data_size), atoi(cores), (atof(data_size)*10)/(atoi(cores)*10)); fflush(stdout);
				perror("Error data size in read_workload"); fflush(stdout);
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			index_node = 0;
		}
		new->index_node_list = index_node;
		
		new->start_time = -1; 
		new->end_time = 0; 
		new->end_before_walltime = false;
		
		new->node_used = (struct Node*) malloc(sizeof(struct Node));
		new->node_used = NULL;
		
		/* OLD */
		//~ new->cores_used = malloc(new->cores*sizeof(int));
		/* NEW */
		new->cores_used = (int*) malloc(new->cores*sizeof(int));
		
		//~ new->cores_used = malloc(new->cores*sizeof(struct Core*));
		//~ for (int i = 0; i < new->cores; i++)
		//~ {
			//~ new->cores_used[i] = malloc(sizeof(struct Core*));
		//~ }
		//~ printf("Mallocs ok.\n"); fflush(stdout);
		
		new->transfer_time = 0;
		new->waiting_for_a_load_time = 0;
		new->next = NULL;
		
		/* Add in job list or job to start from history */
		if (new->workload != -2)
		{
			//~ printf("Insert.\n"); fflush(stdout);
			insert_tail_job_list(job_list, new);
			//~ printf("Insert Ok.\n"); fflush(stdout);
		}
		else
		{
			//~ printf("Insert -2.\n"); fflush(stdout);
			
			// NEW
			insert_job_in_sorted_list(job_list_to_start_from_history, new);
			
		}
	}
	fclose(f);
	
	free(current_user);
	free(last_user);
}

int get_nb_job_to_evaluate(struct Job* l)
{
	struct Job *j = l;
	int count = 0;
	while (j != NULL)
	{
		if (j->workload == 1)
		{
			count += 1;
		}
		j = j->next;
	}
	return count;
}

int get_first_time_day_0(struct Job* l)
{
	struct Job *j = l;
	while (j != NULL && j->workload != 0) /* Attention il faut mettre le j != NULL avant car sinon le j->workload sur un maillon NULL fais un segfault. */
	{
		j = j->next;
	}
	/* Cas où il n'y a pas de jobs de catégorie 0, comme dans les tests. */
	if (j == NULL)
	{
		printf("No jobs of category 0. First subtime day 0 is set to 0.\n");
		return 0;
	}
	//~ printf("First time day 0 is %d.\n", j->subtime);
	return j->subtime;
}

void write_in_file_first_times_all_day(struct Job* l, int first_subtime_day_0)
{
	struct Job *j = l;
	bool first_before_0 = false;
	bool first_day_1 = false;
	bool first_day_2 = false;
	int first_subtime_before_0 = 0;
	int first_subtime_day_1 = 0;
	int first_subtime_day_2 = 0;
	
	while (j != NULL)
	{
		if (j->workload == -1 && first_before_0 == false)
		{
			first_before_0 = true;
			first_subtime_before_0 = j->subtime;
		}
		else if (j->workload == 1 && first_day_1 == false)
		{
			first_day_1 = true;
			first_subtime_day_1 = j->subtime - first_subtime_day_0;
		}
		else if (j->workload == 2 && first_day_2 == false)
		{
			first_day_2 = true;
			first_subtime_day_2 = j->subtime - first_subtime_day_0;
		}
		j = j->next;
	}
	
	FILE *f = fopen("outputs/Start_end_evaluated_slice.txt", "w");
	if (!f)
	{
		perror("fopen");
        exit(EXIT_FAILURE);
	}
	fprintf(f, "%d %d %d %d", first_subtime_before_0, 0, first_subtime_day_1, first_subtime_day_2);
	//~ printf("%d %d %d %d\n", first_subtime_before_0, 0, first_subtime_day_1, first_subtime_day_2);
	fclose(f);
}
