#include <main.h>

void insert_head_job_list(struct Job_List* liste, struct Job* j)
{
	if (liste->head == NULL)
	{
		liste->head = j;
		liste->tail = j;
	}
	else
	{
		j->next = liste->head;
		liste->head = j;
	}
}

void insert_tail_job_list(struct Job_List* liste, struct Job* j)
{
    //if head is NULL, it is an empty list
    if(liste->head == NULL)
    {
         liste->head = j;
	 }
    //Otherwise, find the last node and add the newNode
    else
    {
        struct Job *lastNode = liste->head;
        

        //last node's next address will be NULL.
        while(lastNode->next != NULL)
        {
            lastNode = lastNode->next;
        }

        //add the newNode at the end of the linked list
        lastNode->next = j;
    }
}

void insert_tail_node_list(struct Node_List* liste, struct Node* n)
{
	if (liste->head == NULL)
	{
		liste->head = n;
		liste->tail = n;
	}
	else
	{
		liste->tail->next = n;
		liste->tail = n;
	}
}

void initialize_cores_in_a_hole(struct Core_in_a_hole_List* liste, struct Core_in_a_hole* c)
{
	liste = (struct Core_in_a_hole_List*) malloc(sizeof(struct Core_in_a_hole_List));
	liste->head = c;
}

void insert_cores_in_a_hole_list_sorted_decreasing_order(struct Core_in_a_hole_List* liste, struct Core_in_a_hole* c)
{
	//~ printf("Inserting core %d time %d.\n", c->unique_id, c->start_time_of_the_hole);
	if (liste->head == NULL || liste->head->start_time_of_the_hole < c->start_time_of_the_hole)
	//~ if (liste->head == NULL || liste->head->start_time_of_the_hole >= c->start_time_of_the_hole)
	{
		//~ printf("1.\n");
		c->next = liste->head;
        liste->head = c;
    }
    else if (liste->head->start_time_of_the_hole == c->start_time_of_the_hole && liste->head->unique_id > c->unique_id)
    {
		//~ printf("2.\n");
		c->next = liste->head;
        liste->head = c;
	}
    else
    {
		 struct Core_in_a_hole* current;
        /* Locate the node before
the point of insertion */
        current = liste->head;
        //~ while (current->next != NULL && current->next->start_time_of_the_hole >= c->start_time_of_the_hole) {
        while (current->next != NULL && current->next->start_time_of_the_hole >= c->start_time_of_the_hole) {
			//~ printf("next.\n");
            current = current->next;
        }
        
        c->next = current->next;
        current->next = c;
    }
}

void insert_cores_in_a_hole_list_sorted_increasing_order(struct Core_in_a_hole_List* liste, struct Core_in_a_hole* c)
{
	if (liste->head == NULL || liste->head->start_time_of_the_hole > c->start_time_of_the_hole)
	{
		c->next = liste->head;
        liste->head = c;
    }
    else if (liste->head->start_time_of_the_hole == c->start_time_of_the_hole && liste->head->unique_id > c->unique_id)
    {
		//~ printf("2.\n");
		c->next = liste->head;
        liste->head = c;
	}
    else {
		 struct Core_in_a_hole* current;
        /* Locate the node before
the point of insertion */
        current = liste->head;
        while (current->next != NULL && current->next->start_time_of_the_hole <= c->start_time_of_the_hole) {
            current = current->next;
        }
        
        c->next = current->next;
        current->next = c;
    }
}

void delete_core_in_hole_from_head(struct Core_in_a_hole_List* liste, int nb_cores_to_delete)
	//~ void delete_job_linked_list(struct Job_List* liste, int unique_id_to_delete)
{
    if (liste == NULL)
    {
		printf("Error list empty.\n"); fflush(stdout);
        exit(EXIT_FAILURE);
    }
	//~ printf("%d holes to delete.\n",nb_cores_to_delete); fflush(stdout);
	//~ struct Core_in_a_hole* temp = liste->head;
	struct Core_in_a_hole* temp = NULL;
	//~ struct Job* prev = liste->head;
	int i = 0;
	// If head node itself holds the key to be deleted
    //~ if (temp != NULL && temp->unique_id == unique_id_to_delete) {
    for (i = 0; i < nb_cores_to_delete; i++)
    {
		temp = liste->head;
		//~ printf("changing head.\n"); fflush(stdout);
        liste->head = temp->next; // Changed head
        free(temp); // free old head
    }
}

void delete_core_in_hole_specific_core(struct Core_in_a_hole_List* liste, int unique_id_to_delete)
{
	struct Core_in_a_hole* temp = liste->head;
	struct Core_in_a_hole* prev = liste->head;
	
	// If head node itself holds the key to be deleted
    if (temp != NULL && temp->unique_id == unique_id_to_delete) {
        liste->head = temp->next; // Changed head
        //~ if (unique_id_to_delete == 11) {
			//~ printf("Free the head.\n"); fflush(stdout); }
        free(temp); // free old head
        //~ if (unique_id_to_delete == 11) {
        //~ printf("Free the head Ok!\n"); fflush(stdout); }
        return;
    }
	
	while (temp->unique_id != unique_id_to_delete && temp != NULL)
	{
		//~ if (unique_id_to_delete == 11) {
		//~ printf("prev (%d) get temp.\n", prev->unique_id); fflush(stdout); }
		prev = temp;
		//~ if (unique_id_to_delete == 11) {
		//~ printf("temp is %d and next is %d.\n", temp->unique_id, temp->next->unique_id); fflush(stdout); }
		temp = temp->next;
	}
	//~ if (unique_id_to_delete == 11) {
	//~ printf("temp is %d. Next is %d\n", temp->unique_id, temp->next->unique_id); fflush(stdout); }
	if (temp == NULL)
	{
		printf("Error, deletion of core %d failed.\n", unique_id_to_delete); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	//~ if (unique_id_to_delete == 11) {
		//~ printf("Next.\n"); fflush(stdout); }
	prev->next = temp->next;
	//~ if (unique_id_to_delete == 11) {
	//~ printf("Will free temp = %d.\n", temp->unique_id); fflush(stdout); }
	
	if (unique_id_to_delete != temp->unique_id)
	{
		printf("ERROR: unique_id_to_delete != temp->unique_id!\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	
	/* OLD */
	//~ free(temp);
	/* NEW */
	// free(temp->node_used);
	//~ free(temp->cores_used);
	free(temp);
	
	//~ if (unique_id_to_delete == 11) {
	//~ printf("free temp ok!.\n"); fflush(stdout); }
}

void create_and_insert_head_time_or_data_already_checked_nb_of_copy_list(struct Time_or_Data_Already_Checked_Nb_of_Copy_List* liste, int time_or_data_to_insert, int nb_of_copy_to_insert)
{
	struct Time_or_Data_Already_Checked_Nb_of_Copy* a = (struct Time_or_Data_Already_Checked_Nb_of_Copy*) malloc(sizeof(struct Time_or_Data_Already_Checked_Nb_of_Copy));
	a->time_or_data = time_or_data_to_insert;
	a->nb_of_copy = nb_of_copy_to_insert;
	a->next = NULL;
	if (liste->head == NULL)
	{
		liste->head = a;
	}
	else
	{
		a->next = liste->head;
		liste->head = a;
	}
}

void create_and_insert_tail_interval_list(struct Interval_List* liste, int time_to_insert)
{
	struct Interval* i = (struct Interval*) malloc(sizeof(struct Interval));
	i->time = time_to_insert;
	i->next = NULL;
	
	if (liste->head == NULL)
	{
		liste->head = i;
		liste->tail = i;
	}
	else
	{
		liste->tail->next = i;
		liste->tail = i;
	}
}

void insert_tail_to_print_list(struct To_Print_List* liste, struct To_Print* tp)
{
	if (liste->head == NULL)
	{
		liste->head = tp;
		liste->tail = tp;
	}
	else
	{
		liste->tail->next = tp;
		liste->tail = tp;
	}
}

void insert_tail_data_list(struct Data_List* liste, struct Data* d)
{
	if (liste->head == NULL)
	{
		liste->head = d;
		liste->tail = d;
	}
	else
	{
		liste->tail->next = d;
		liste->tail = d;
	}
}

/* Insert so it's sorted by growing times. */
void insert_next_time_in_sorted_list(struct Next_Time_List* liste, int time_to_insert)
{
	//~ void sortedInsert(struct Node** head_ref,
                  //~ struct Node* new_node)
//~ {
    struct Next_Time* current;
    /* Special case for the head end */
    if (liste->head == NULL || liste->head->time >= time_to_insert) {
		
		/* I don't want the same time twice. So I add this condition. */
		if (liste->head != NULL)
		{
			if (liste->head->time == time_to_insert)
			{
				return;
			}
		}
		
		struct Next_Time* new = (struct Next_Time*) malloc(sizeof(struct Next_Time));
		new->next = liste->head;
		new->time = time_to_insert;
        //~ new_node->next = *head_ref;
        liste->head = new;
    }
    else {
        /* Locate the node before
the point of insertion */
        current = liste->head;
        while (current->next != NULL && current->next->time <= time_to_insert) {
            current = current->next;
        }
        
 		/* I don't want the same time twice. So I addthis condition. */
 		if (current != NULL)
 		{
			if (current->time == time_to_insert)
			{
				return;
			}   
		}
        
        struct Next_Time* new = (struct Next_Time*) malloc(sizeof(struct Next_Time));
        new->time = time_to_insert;
        new->next = current->next;
        current->next = new;
    }
//~ }
	//~ if (liste->head == NULL)
	//~ {
		//~ struct Next_Time* new = (struct Next_Time*) malloc(sizeof(struct Next_Time));
		//~ new->next = NULL;
		//~ new->time = time_to_insert;
		//~ liste->head = new;
	//~ }
	//~ else
	//~ {
		//~ if (liste->head->time == time_to_insert)
		//~ {
			//~ return;
		//~ }
		//~ else if (liste->head->time > time_to_insert) /* Need to insert at head. */
		//~ {
			//~ printf("Dans le else.\n");
			//~ struct Next_Time* new = (struct Next_Time*) malloc(sizeof(struct Next_Time));
			//~ new->next = liste->head;
			//~ new->time = time_to_insert;
			//~ liste->head = new;
			//~ return;
		//~ }
		//~ printf("In insert time 0. Head is %d\n", liste->head->time);
		//~ struct Next_Time* temp = liste->head;
		//~ struct Next_Time* prev = liste->head;
		//~ while (temp->time < time_to_insert && temp != NULL)
		//~ {
			//~ printf("Next.\n");
			//~ prev = temp;
			//~ temp = temp->next;
			//~ printf("Next ok.\n");
			//~ if (temp == NULL) { printf("Temp is null.\n"); break; }
		//~ }
		//~ printf("In insert time 0.5.\n"); fflush(stdout);
		//~ if (temp == NULL) /* We are at the end. */
		//~ {
			//~ struct Next_Time* new = (struct Next_Time*) malloc(sizeof(struct Next_Time));
			//~ new->next = NULL;
			//~ new->time = time_to_insert;
			//~ temp = new;
			//~ return;
		//~ }
		//~ printf("In insert time 1.\n");
		//~ if (temp->time == time_to_insert)
		//~ {
			//~ return;
		//~ }
		//~ struct Next_Time* new = (struct Next_Time*) malloc(sizeof(struct Next_Time));
		//~ new->next = temp;
		//~ new->time = time_to_insert;
		//~ prev = new;
	//~ }
}

/* Insert so it's sorted by start times from history. */
void insert_job_in_sorted_list(struct Job_List* liste, struct Job* j)
{
    struct Job* current;
    /* Special case for the head end */
    if (liste->head == NULL || liste->head->start_time_from_history >= j->start_time_from_history) {	
		j->next = liste->head;
        liste->head = j;
    }
    else {
        /* Locate the node before
the point of insertion */
        current = liste->head;
        while (current->next != NULL && current->next->start_time_from_history <= j->start_time_from_history) {
            current = current->next;
        }
        
        j->next = current->next;
        current->next = j;
    }
}

/* Delete all time corresponding to this time. The list is sorted so it should be easy. */
void delete_next_time_linked_list(struct Next_Time_List* liste, int time_to_delete)
{
    if (liste == NULL)
    {
		printf("Error list empty.\n");
        exit(EXIT_FAILURE);
    }

	struct Next_Time* temp = liste->head;
	struct Next_Time* prev = liste->head;
	
	 // If head node itself holds the key to be deleted
    if (temp != NULL && temp->time == time_to_delete) {
        liste->head = temp->next; // Changed head
        free(temp); // free old head
        return;
    }
	
	while (temp->time != time_to_delete && temp != NULL)
	{
		prev = temp;
		temp = temp->next;
	}
	if (temp == NULL)
	{
		printf("Error, deletion of time %d failed.\n", time_to_delete);
		exit(EXIT_FAILURE);
	}
	prev->next = temp->next;
	free(temp);
}

//~ /* Attention might need to malloc here for Data and other struct !!! */
//~ void copy_job_and_insert_tail_job_list(struct Job_List* liste, struct Job* j)
//~ {
	//~ struct Job* new = (struct Job*) malloc(sizeof(struct Job));
	
	//~ new->next = NULL;
	//~ new->unique_id = j->unique_id;
	//~ new->subtime = j->subtime;
	//~ new->delay = j->delay;
	//~ new->walltime = j->walltime;
	//~ new->cores = j->cores;
	//~ new->data = j->data;
	//~ new->data_size = j->data_size;
	//~ new->index_node_list = j->index_node_list;
	//~ new->start_time = j->start_time;
	//~ new->end_time = j->end_time;
	//~ new->end_before_walltime = j->end_before_walltime;
	//~ new->node_used = j->node_used;
	//~ new->cores_used = j->cores_used;
	//~ new->transfer_time = j->transfer_time;
	//~ new->waiting_for_a_load_time = j->waiting_for_a_load_time;
	//~ new->workload = j->workload;
	//~ new->start_time_from_history = j->start_time_from_history;
	//~ new->node_from_history = j->node_from_history;
   
	//~ if (liste->head == NULL)
	//~ {
		//~ liste->head = new;
		//~ liste->tail = new;
	//~ }
	//~ else
	//~ {
		//~ liste->tail->next = new;
		//~ liste->tail = new;
	//~ }
//~ }

void delete_job_linked_list(struct Job_List* liste, int unique_id_to_delete)
{
    if (liste == NULL)
    {
		printf("Error list empty.\n"); fflush(stdout);
        exit(EXIT_FAILURE);
    }

	struct Job* temp = liste->head;
	struct Job* prev = liste->head;
	
	// If head node itself holds the key to be deleted
    if (temp != NULL && temp->unique_id == unique_id_to_delete)
    {
        liste->head = temp->next; // Changed head
        //~ if (unique_id_to_delete == 11) {
			//~ printf("Free the head.\n"); fflush(stdout); }
        
        /* OLD */
        //~ free(temp); // free old head
        /* NEW */
		//~ if (temp != NULL)
		//~ {
			//~ if (temp->node_used != NULL)
			//~ {
				//~ free(temp->node_used);
			//~ }
			if (temp->cores_used != NULL)
			{
				free(temp->cores_used);
			}
			free(temp);
		//~ } else { printf("NULL\n"); }
        
        //~ if (unique_id_to_delete == 11) {
        //~ printf("Free the head Ok!\n"); fflush(stdout); }
        return;
    }
	
	while (temp->unique_id != unique_id_to_delete && temp != NULL)
	{
		//~ if (unique_id_to_delete == 11) {
		//~ printf("prev (%d) get temp.\n", prev->unique_id); fflush(stdout); }
		prev = temp;
		//~ if (unique_id_to_delete == 11) {
		//~ printf("temp is %d and next is %d.\n", temp->unique_id, temp->next->unique_id); fflush(stdout); }
		temp = temp->next;
	}
	//~ if (unique_id_to_delete == 11) {
	//~ printf("temp is %d. Next is %d\n", temp->unique_id, temp->next->unique_id); fflush(stdout); }
	if (temp == NULL)
	{
		printf("Error, deletion of job %d failed.\n", unique_id_to_delete); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	//~ if (unique_id_to_delete == 11) {
		//~ printf("Next.\n"); fflush(stdout); }
	prev->next = temp->next;
	//~ if (unique_id_to_delete == 11) {
	//~ printf("Will free temp = %d.\n", temp->unique_id); fflush(stdout); }
	
	if (unique_id_to_delete != temp->unique_id)
	{
		printf("ERROR!\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	
	/* OLD */
	//~ free(temp->cores_used);
	/* NEW */
	//~ if (temp != NULL)
	//~ {
		//~ if (temp->node_used != NULL)
		//~ {
			//~ free(temp->node_used);
		//~ }
		//~ if (temp->cores_used != NULL)
		//~ {
			//~ free(temp->cores_used);
		//~ }
		//~ free(temp);
	//~ } else { printf("NULL\n"); }
				if (temp->cores_used != NULL)
			{
				free(temp->cores_used);
			}
			free(temp);

	//~ if (unique_id_to_delete == 11) {
	//~ printf("free temp ok!.\n"); fflush(stdout); }
}

void delete_specific_data_from_node(struct Data_List* liste, int unique_id_to_delete)
{
    if (liste == NULL)
    {
		printf("Error list empty.\n"); fflush(stdout);
        exit(EXIT_FAILURE);
    }

	struct Data* temp = liste->head;
	struct Data* prev = liste->head;
	
	// If head node itself holds the key to be deleted
    if (temp != NULL && temp->unique_id == unique_id_to_delete) {
        liste->head = temp->next; // Changed head
        //~ if (unique_id_to_delete == 11) {
			//~ printf("Free the head.\n"); fflush(stdout); }
        free(temp); // free old head
        //~ if (unique_id_to_delete == 11) {
        //~ printf("Free the head Ok!\n"); fflush(stdout); }
        return;
    }
	
	while (temp->unique_id != unique_id_to_delete && temp != NULL)
	{
		//~ if (unique_id_to_delete == 11) {
		//~ printf("prev (%d) get temp.\n", prev->unique_id); fflush(stdout); }
		prev = temp;
		//~ if (unique_id_to_delete == 11) {
		//~ printf("temp is %d and next is %d.\n", temp->unique_id, temp->next->unique_id); fflush(stdout); }
		temp = temp->next;
	}
	//~ if (unique_id_to_delete == 11) {
	//~ printf("temp is %d. Next is %d\n", temp->unique_id, temp->next->unique_id); fflush(stdout); }
	if (temp == NULL)
	{
		printf("Error, deletion of data %d failed.\n", unique_id_to_delete); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	//~ if (unique_id_to_delete == 11) {
		//~ printf("Next.\n"); fflush(stdout); }
	prev->next = temp->next;
	//~ if (unique_id_to_delete == 11) {
	//~ printf("Will free temp = %d.\n", temp->unique_id); fflush(stdout); }
	
	if (unique_id_to_delete != temp->unique_id)
	{
		printf("ERROR!\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	
	/* OLD */
	//~ free(temp);
	/* NEW */
	// free(temp->node_used);
	//~ free(temp->cores_used);
	free(temp);
	
	//~ if (unique_id_to_delete == 11) {
	//~ printf("free temp ok!.\n"); fflush(stdout); }
}

/* Copy a job, delete it from list 1 and add it in tail of list 2. */
void copy_delete_insert_job_list(struct Job_List* to_delete_from, struct Job_List* to_append_to, struct Job* j)
{
	/* If empty can't delete. */
	if (to_delete_from == NULL)
    {
		printf("Error list empty.\n");
        exit(EXIT_FAILURE);
    }
        
    /* New copy from j */
	struct Job* new = (struct Job*) malloc(sizeof(struct Job));
	new->next = NULL;
	new->unique_id = j->unique_id;
	new->subtime = j->subtime;
	new->delay = j->delay;
	new->walltime = j->walltime;
	new->cores = j->cores;
	new->data = j->data;
	new->data_size = j->data_size;
	new->index_node_list = j->index_node_list;
	new->start_time = j->start_time;
	new->end_time = j->end_time;
	new->end_before_walltime = j->end_before_walltime;
	//~ new->node_used = j->node_used;
	//~ new->cores_used = j->cores_used;
	new->transfer_time = j->transfer_time;
	new->waiting_for_a_load_time = j->waiting_for_a_load_time;
	new->workload = j->workload;
	new->start_time_from_history = j->start_time_from_history;
	new->node_from_history = j->node_from_history;

	new->node_used = (struct Node*) malloc(sizeof(struct Node));
	new->node_used = j->node_used;
	
	new->user = j->user;
	
	new->cores_used = (int*) malloc(new->cores*sizeof(int));
	for (int i = 0; i < new->cores; i++)
	{
		new->cores_used[i] = j->cores_used[i];
	}

	delete_job_linked_list(to_delete_from, j->unique_id);
	
	insert_tail_job_list(to_append_to, new);
}

/* Copy a job, delete it from list 1 and add it in tail of list 2 in sorted by decreasing file size order. */
void copy_delete_insert_job_list_sorted_by_file_size(struct Job_List* to_delete_from, struct Job_List* to_append_to, struct Job* j)
{
	if (to_delete_from == NULL)
    {
		printf("Error list empty.\n");
        exit(EXIT_FAILURE);
    }
    
    /* New copy from j */
	struct Job* new = (struct Job*) malloc(sizeof(struct Job));
	new->next = NULL;
	new->unique_id = j->unique_id;
	new->subtime = j->subtime;
	new->delay = j->delay;
	new->walltime = j->walltime;
	new->cores = j->cores;
	new->data = j->data;
	new->data_size = j->data_size;
	new->index_node_list = j->index_node_list;
	new->start_time = j->start_time;
	new->end_time = j->end_time;
	new->end_before_walltime = j->end_before_walltime;
	//~ new->node_used = j->node_used;
	//~ new->cores_used = j->cores_used;
	new->transfer_time = j->transfer_time;
	new->waiting_for_a_load_time = j->waiting_for_a_load_time;
	new->workload = j->workload;
	new->start_time_from_history = j->start_time_from_history;
	new->node_from_history = j->node_from_history;

	new->node_used = (struct Node*) malloc(sizeof(struct Node));
	new->node_used = j->node_used;
	
	new->cores_used = malloc(new->cores*sizeof(int));
	new->cores_used = j->cores_used;
new->user = j->user;
	/* Delete */
	delete_job_linked_list(to_delete_from, j->unique_id);
	
	/* Add in new list */
	//~ insert_tail_job_list(to_append_to, new);
	insert_job_sorted_by_decreasing_file_size(&to_append_to->head, new);
}

void get_length_job_list(struct Job* head, int* nb_jobs_in_queue, int* nb_cores_in_queue, int* nb_cores_from_workload_1_in_queue)
{
	*nb_jobs_in_queue = 0;
	*nb_cores_in_queue = 0;
	*nb_cores_from_workload_1_in_queue = 0;
	struct Job* j = head;
	while (j != NULL)
	{
		*nb_jobs_in_queue += 1;
		*nb_cores_in_queue += j->cores;
		if (j->workload == 1)
		{
			*nb_cores_from_workload_1_in_queue += j->cores;
		}
		j = j->next;
	}
}

void free_next_time_linked_list(struct Next_Time** head_ref)
{
	/* deref head_ref to get the real head */
	struct Next_Time* current = *head_ref;
	struct Next_Time* next;
 
   while (current != NULL)
   {
       next = current->next;
       free(current);
       current = next;
   }
   
   /* deref head_ref to affect the real head back
      in the caller. */
   *head_ref = NULL;
}

void free_cores_in_a_hole(struct Core_in_a_hole** head_ref)
{
	/* deref head_ref to get the real head */
	struct Core_in_a_hole* current = *head_ref;
	struct Core_in_a_hole* next;
 
   while (current != NULL)
   {
       next = current->next;
       free(current);
       current = next;
   }
   
   /* deref head_ref to affect the real head back
      in the caller. */
   *head_ref = NULL;
}

void freelist(struct Interval* headNode){
    struct Interval* currentNode;
    while (headNode != NULL){
        currentNode = headNode;
        headNode = headNode->next;
        free(currentNode);
    }
}

void free_interval_linked_list(struct Interval** head_ref, struct Interval** tail_ref)
{
 /* deref head_ref to get the real head */
   struct Interval* current = *head_ref;
   struct Interval* next;
 
   while (current != NULL)
   {
       next = current->next;
       free(current);
       current = next;
   }
   
   /* deref head_ref to affect the real head back
      in the caller. */
   *head_ref = NULL;
   *tail_ref = NULL;
}

void free_time_or_data_already_checked_nb_of_copy_linked_list(struct Time_or_Data_Already_Checked_Nb_of_Copy** head_ref)
{
	/* deref head_ref to get the real head */
	struct Time_or_Data_Already_Checked_Nb_of_Copy* current = *head_ref;
	struct Time_or_Data_Already_Checked_Nb_of_Copy* next;
 
	while (current != NULL)
	{
		next = current->next;
		free(current);
		current = next;
	}
   
	/* deref head_ref to affect the real head back in the caller. */
	*head_ref = NULL;
}

void increment_time_or_data_nb_of_copy_specific_time_or_data(struct Time_or_Data_Already_Checked_Nb_of_Copy_List* liste, int time_or_data_to_increment)
{
	struct Time_or_Data_Already_Checked_Nb_of_Copy* current = liste->head;
	if (current == NULL)
	{
		//~ printf("Need to create.\n"); fflush(stdout);
		create_and_insert_head_time_or_data_already_checked_nb_of_copy_list(liste, time_or_data_to_increment, 1);
		//~ printf("Create ok.\n"); fflush(stdout);
		return;
	}
	while (current->time_or_data != time_or_data_to_increment && current != NULL)
	{
		//~ printf("Current is %d.\n", current->time_or_data); fflush(stdout);
		current = current->next;
		//~ printf("NExt.\n"); fflush(stdout);
		if (current == NULL)
		{
			create_and_insert_head_time_or_data_already_checked_nb_of_copy_list(liste, time_or_data_to_increment, 1);
			return;
		}
	}
	//~ printf("End.\n");  fflush(stdout);
	if (current == NULL)
	{
		perror("Error in increment_time_or_data_nb_of_copy_specific_time_or_data.\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}
	//~ printf("++\n");  fflush(stdout);
	//~ printf("On %d.\n", current->time_or_data);  fflush(stdout);
	current->nb_of_copy += 1;
}

void sort_cores_by_available_time_in_specific_node(struct Node* n)
{
	for (int step = 0; step < 20 - 1; step++)
	{
		for (int i = 0; i < 20 - step - 1; ++i)
		{
			if (n->cores[i]->available_time > n->cores[i + 1]->available_time)
			{
				struct Core* temp = n->cores[i];
				n->cores[i] = n->cores[i+1];
				n->cores[i + 1] = temp;
			}
			else if (n->cores[i]->available_time == n->cores[i + 1]->available_time && n->cores[i]->unique_id > n->cores[i + 1]->unique_id)
			{
				struct Core* temp = n->cores[i];
				n->cores[i] = n->cores[i+1];
				n->cores[i + 1] = temp;
			}
		}
	}
}

void sort_cores_by_unique_id_in_specific_node(struct Node* n)
{
	for (int step = 0; step < 20 - 1; step++)
	{
		for (int i = 0; i < 20 - step - 1; ++i)
		{
			if (n->cores[i]->unique_id > n->cores[i + 1]->unique_id)
			{
				struct Core* temp = n->cores[i];
				n->cores[i] = n->cores[i+1];
				n->cores[i + 1] = temp;
			}
		}
	}
}

// Function to insert a given node at its correct sorted position into a given
// list sorted in decreasing order
void insert_job_sorted_by_decreasing_file_size(struct Job** head, struct Job* newNode)
{
    struct Job dummy;
    struct Job* current = &dummy;
    dummy.next = *head;
 
    while (current->next != NULL && current->next->data_size >= newNode->data_size) {
        current = current->next;
    }
    newNode->next = current->next;
    current->next = newNode;
    *head = dummy.next;
}
 
// Given a list, change it to be in decreasing sorted order (using `insert_job_sorted_by_decreasing_file_size()`).
void sort_job_list_by_file_size(struct Job** head)
{
    struct Job* result = NULL;     // build the answer here
    struct Job* current = *head;   // iterate over the original list
    struct Job* next;
 
    while (current != NULL)
    {
        // tricky: note the next pointer before we change it
        next = current->next;
 
        insert_job_sorted_by_decreasing_file_size(&result, current);
        current = next;
    }
	*head = result;
}

void free_and_copy_data_and_intervals_in_temp_data(struct Node_List** head_node, int t)
{
	#ifdef DATA_PERSISTENCE
	
	#ifdef PRINT
	printf("Copy in temp data\n");
	#endif
	
	int i = 0;
	for (i = 0; i < 3; i++)
	{
		struct Node* n = head_node[i]->head;
		//~ if (n->data->head != NULL) { printf("Copy data %d start time %d into temp.\n", n->data->head->unique_id, n->data->head->start_time);	}
		while (n != NULL)
		{
			struct Data* current = n->temp_data->head;
			struct Data* next;
		   while (current != NULL)
		   {
			   next = current->next;
			   free(current);
			   current = next;
		   }
		   /* deref head_ref to affect the real head back
			  in the caller. */
		   //~ *head_ref = NULL;
		   //~ n->temp_data->head = NULL;
			n->temp_data = (struct Data_List*) malloc(sizeof(struct Data_List));
			n->temp_data->head = NULL;
			n->temp_data->tail = NULL;

			struct Data* d = n->data->head;
			while (d != NULL)
			{				
				#ifdef PRINT
				printf("Copy data %d start time %d into temp.\n", d->unique_id, d->start_time);	
				#endif
				
				struct Data* new = (struct Data*) malloc(sizeof(struct Data));
				
				new->next = NULL;
				new->unique_id = d->unique_id;
				new->start_time = d->start_time;
				new->end_time = d->end_time;				
				new->intervals = (struct Interval_List*) malloc(sizeof(struct Interval_List));
				new->intervals->head = NULL;
				new->intervals->tail = NULL;
				new->size = d->size;

						create_and_insert_tail_interval_list(new->intervals, t);
						
						#ifdef PRINT
						printf("Start time is %d.\n", new->start_time);
						#endif
						
						if (new->start_time < t)
						{
							#ifdef PRINT
							printf("Adding t\n");
							#endif
							
							create_and_insert_tail_interval_list(new->intervals, t);
						}
						else
						{
							#ifdef PRINT
							printf("Adding start\n");
							#endif
							
							create_and_insert_tail_interval_list(new->intervals, new->start_time);
						}
						create_and_insert_tail_interval_list(new->intervals, new->end_time);
				insert_tail_data_list(n->temp_data, new);			
				d = d->next;
			}
			n = n->next;
		}
	}
	#else
	printf("This function ins not supported without data persistence def.\n");
	exit(1);
	#endif
}
