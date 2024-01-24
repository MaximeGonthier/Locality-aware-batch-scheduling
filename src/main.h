//~ #define PRINT
//~ #define PRINT_GANTT_CHART
//~ #define PRINT_DISTRIBUTION_QUEUE_TIMES
//~ #define PRINT_CLUSTER_USAGE
//~ #define PRINT_SCORES_DATA

#include <stdio.h>
#include <string.h>
//~ #include <stdint.h>
//~ #include <limits.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <limits.h>

/* Global variables */
extern int first_node_size_to_choose_from;
extern int last_node_size_to_choose_from;

extern char* input_job_file;

#ifdef NB_HOUR_MAX
extern int nb_h_scheduled;
#endif

extern int planned_or_ratio; /* O = planned, 1 = ratio */
extern int constraint_on_sizes;
extern int nb_cores;
extern int nb_job_to_evaluate;
extern int finished_jobs;
extern int total_number_jobs;
extern int total_number_nodes;
extern struct Job_List* job_list; /* All jobs not available yet */
extern struct Job_List* new_job_list; /* New available jobs */
extern struct Job_List* job_list_to_start_from_history; /* With -2 and before start */
extern struct Job_List* scheduled_job_list; /* Scheduled or available */
//~ extern struct Job_List* new_job_list; /* Scheduled or available */
extern struct Job_List* running_jobs; /* Started */
extern struct Node_List** node_list;
extern struct To_Print_List* jobs_to_print_list;
//~ /** For mixed decreasing strategy **/
//~ extern struct Data_List* data_list;
//~ /** For mixed decreasing strategy **/
extern int running_cores;
extern int running_nodes;
extern int nb_job_to_schedule; /* Jobs ready but not running */
extern int nb_cores_to_schedule; /* Cores ready but not running */
#ifdef PRINT_CLUSTER_USAGE
extern int running_nodes_workload_1;
//~ extern int nodes_loading_a_file;
extern int mixed_mode;
extern int running_cores_from_workload_1;
#endif
extern int total_queue_time;
extern int first_subtime_day_0;
extern char* scheduler;
extern char* output_file;
extern long long Planned_Area[3][3];
extern int number_node_size[3];
extern int busy_cluster; /* 0 = no, 1 = yes. Used for fcfs with a score adaptative multiplier that modify multiplier based on cluster contention. */
extern int backfill_mode; /* 0 = no, 1 = yes. Used for fcfs with a score adaptative multiplier that modify multiplier based on cluster contention. */
#ifdef PLOT_STATS
extern int number_of_backfilled_jobs;
extern int number_of_tie_breaks_before_computing_evicted_files_fcfs_score;
extern int total_number_of_scores_computed;
extern int data_persistence_exploited;
#endif
extern int biggest_hole;
extern int biggest_hole_unique_id;
extern int global_nb_non_available_cores_at_time_t;
extern int nb_data_reuse;
extern int busy_cluster_threshold;
//~ extern int on_a_resume;

extern int nb_call_finished_jobs;
extern int nb_call_new_jobs;

/* For area_filling. This is the allocated area updated in start jobs. It corresponds to the area of jobs of size x that
 * were started on nodes of size x+y, y>0. I use it as a global variable to update it in start_jobs. In the schedule of
 * area flling I use a temp tab at the beggining of the schedule that copy the values of this global variable
 * because the schedule can change later on with liberated cores.
 */
extern long long Allocated_Area[3][3];

/* To only call these functions when I need it. */
extern struct Next_Time_List* end_times;
extern struct Next_Time_List* start_times; /* TODO try to do that with update at each new scheduled job and reset when reset jobs and reschedule */

//~ int nb_job_to_evaluate_finished;
extern int nb_job_to_evaluate_started;

/* For fcfs with a score. Allow me to not compute multiple time the same amount of copy of a file at a certain time when computing the amount of copy. I also use it to see if a data was already checked in the reduced complexity version. */
struct Time_or_Data_Already_Checked_Nb_of_Copy_List {
	struct Time_or_Data_Already_Checked_Nb_of_Copy* head;
};
struct Time_or_Data_Already_Checked_Nb_of_Copy {
	struct Time_or_Data_Already_Checked_Nb_of_Copy* next;
	int time_or_data;
	int nb_of_copy;
};

struct Next_Time_List {
	struct Next_Time* head;
};

struct Next_Time {
	struct Next_Time* next;
	int time;
};

struct Job_List {
	struct Job* head;
	struct Job* tail;
};

struct Node_List {
	struct Node* head;
	struct Node* tail;
};

struct Data_List {
	struct Data* head;
	struct Data* tail;
};

struct Job {
	struct Job* next;
	int unique_id;
    int subtime;
    int delay;
    int walltime;
    int cores;
    //~ /** For mixed decreasing strategy **/
    int data;
    float data_size;
    //~ /** For mixed decreasing strategy **/
    int index_node_list;
    int start_time;
    int end_time;
    bool end_before_walltime;
    struct Node* node_used;
    int* cores_used; /* list */
    int transfer_time;
    int waiting_for_a_load_time;
    int workload;
    int start_time_from_history;
    int node_from_history;
    //~ /** For mixed decreasing strategy **/
    //~ struct Data* data;
    //~ /** For mixed decreasing strategy **/
    
    #ifdef PLOT_STATS /*pour plot les stats sur la dernière décision prise pour un job */
    int last_choosen_method;
    #endif
    
    int user;
};

struct Node {
	struct Node* next;
    int unique_id;
    int memory;
    float bandwidth;
    struct Data_List* data;
    struct Core** cores;
    int n_available_cores;
    int index_node_list;
    
    /* Pour conservative bf */
    int number_cores_in_a_hole;
    struct Core_in_a_hole_List* cores_in_a_hole;
    //~ int* cores_in_a_hole;
    //~ int* start_time_of_the_hole; /* Temps auquel le trou n'existera plus. Il y a en a 1 par core car 2 jobs aux starts times différents peuvent créer des trou sur une node. */
    
    /* Pour les data qui restent si rien ne les remplacent */
    #ifdef DATA_PERSISTENCE
    int data_occupation; /* From 0 to 20 */
    struct Data_List* temp_data; /* To get local temporary intervals not interferring with start_jobs */
    #endif
    
    #ifdef PRINT_CLUSTER_USAGE
    int nb_jobs_workload_1;
    #endif
    
    int end_of_file_load;
    //~ #endif
};

struct Core_in_a_hole_List {
	struct Core_in_a_hole* head;
	struct Core_in_a_hole* tail;
};

struct Core_in_a_hole {
	struct Core_in_a_hole* next;
	int unique_id;
	int start_time_of_the_hole;
};

struct Data {
	struct Data* next;
    int unique_id;
    int start_time;
    int end_time;
    //~ /** For mixed decreasing strategy **/
    
    #ifndef DATA_PERSISTENCE
    int nb_task_using_it;
    #endif
    
    struct Interval_List* intervals;
    
    float size;
    //~ /** For mixed decreasing strategy **/
    //~ int number_of_task_using_it_not_running;
    //~ /** For mixed decreasing strategy **/
};

struct Core {
    int unique_id;
    int available_time;
    bool running_job;
    int running_job_end;
    
    #ifdef PRINT_CLUSTER_USAGE
    int end_of_file_load; /* Used to then get the total number of cores running a load */
    #endif
};

struct To_Print_List {
	struct To_Print* head;
	struct To_Print* tail;
};

struct To_Print {
	struct To_Print* next;
	int job_unique_id;
    int job_subtime;
    int time;
    int time_used;
    int transfer_time;
    int job_start_time;
    int job_end_time;
    int job_cores;
    int waiting_for_a_load_time;
    float empty_cluster_time;
    int data_type;
    float job_data_size;
    int upgraded; /* Was it on bigger nodes ? 0 or 1 */
    int user;
    int input_file;
};

struct Interval_List {
	struct Interval* head;
	struct Interval* tail;
};

struct Interval {
	struct Interval* next;
	int time;
};

/* Test free intervals */
void freelist(struct Interval* headNode);

/* From read_input_files.c */
void read_cluster(char* input_node_file);
void read_workload(char* input_job_file, int constraint_on_sizes);
int get_nb_job_to_evaluate(struct Job* l);
int get_first_time_day_0(struct Job* l);
int get_first_times_all_day(struct Job* l);
void write_in_file_first_times_all_day(struct Job* l, int first_subtime_day_0);

/* From print_functions.c */
void print_node_list(struct Node_List** list);
void print_job_list(struct Job* list);
void print_single_node(struct Node* n);
void print_decision_in_scheduler(struct Job* j);
void print_cores_in_specific_node(struct Node* n);
void print_time_list(struct Next_Time* list, int end_or_start);
void to_print_job_csv(struct Job* job, int time);
void print_csv(struct To_Print* head_to_print);
void print_data_intervals(struct Node_List** list, int t);
void print_tab_of_int (int arr[], int n);
void print_holes(struct Node_List** head_node);
void print_holes_specific_node(struct Node* n);
void save_state(int t, int old_finished_jobs, int next_submit_time, char* input_job_file);
void resume_state(int* t, int* old_finished_jobs, int* next_submit_time, char* input_job_file);
void print_job_to_print(struct To_Print* tp);

/* From basic_functions.c */
int get_min_EAT(struct Node_List** head_node, int first_node_size_to_choose_from, int last_node_size_to_choose_from, int nb_cores, int t);
void schedule_job_specific_node_at_earliest_available_time(struct Job* j, struct Node* n, int t);
void start_jobs(int t, struct Job* scheduled);
void end_jobs(struct Job* job_list_head, int t);
void add_data_in_node (int data_unique_id, float data_size, struct Node* node_used, int t, int end_time, int* transfer_time, int* waiting_for_a_load_time, int delay, int walltime, int start_time, int cores);
int get_nb_non_available_cores(struct Node_List** n, int t);
int schedule_job_on_earliest_available_cores(struct Job* j, struct Node_List** head_node, int t, int nb_non_available_cores, bool use_bigger_nodes);
void reset_cores(struct Node_List** l, int t);
void remove_data_from_node(struct Job* j, int t);
void get_current_intervals(struct Node_List** head_node, int t);
//~ int is_my_file_on_node_at_certain_time_and_transfer_time(int predicted_time, struct Node* n, int t, int current_data, int current_data_size, bool* is_being_loaded);
float is_my_file_on_node_at_certain_time_and_transfer_time(int predicted_time, struct Node* n, int t, int current_data, float current_data_size, bool* is_being_loaded);
//~ float time_to_reload_percentage_of_files_ended_at_certain_time(int predicted_time, struct Node* n, int current_data, int percentage_occupied);
float time_to_reload_percentage_of_files_ended_at_certain_time(int predicted_time, struct Node* n, int current_data, float percentage_occupied);
int get_nb_valid_copy_of_a_file(int predicted_time, struct Node_List** head_node, int current_data);
int was_time_or_data_already_checked_for_nb_copy(int t_or_d, struct Time_or_Data_Already_Checked_Nb_of_Copy_List* list);
int schedule_job_to_start_immediatly_on_specific_node_size(struct Job* j, struct Node_List* head_node_size_i, int t, int backfill_big_node_mode, int total_queue_time, int nb_finished_jobs, int nb_non_available_cores, bool* result);
int schedule_job_on_earliest_available_cores_specific_sublist_node(struct Job* j, struct Node_List* head_node_size_i, int t, int nb_non_available_cores);
int get_earliest_available_time_specific_sublist_node(int nb_cores_asked, struct Node_List* head_node_size_i, struct Node** choosen_node, int t);
int try_to_start_job_immediatly_without_delaying_j1(struct Job* j, struct Job* j1, struct Node_List** head_node, int nb_running_cores, bool* result, bool use_bigger_nodes, int t);
int schedule_job_on_earliest_available_cores_return_running_cores(struct Job* j, struct Node_List** head_node, int t, int nb_running_cores, bool use_bigger_nodes);
//~ int try_to_start_job_immediatly_fcfs_score_without_delaying_j1(struct Job* j, struct Job* j1, struct Node_List** head_node, int nb_running_cores, bool* result, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy);
void fcfs_with_a_score_scheduler_without_delaying_j1(struct Job* j, struct Job* j1, struct Node_List** head_node, int nb_running_cores, bool* result, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t);
int schedule_job_fcfs_score_return_running_cores(struct Job* j, struct Node_List** head_node, int t, int nb_running_cores, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t);
void sort_tab_of_int_decreasing_order(long long arr[], int n);
void swap(long long* xp, long long* yp);
void schedule_job_on_earliest_available_cores_with_conservative_backfill(struct Job* j, struct Node_List** head_node, int t, int backfill_mode, int* nb_non_available_cores, int* nb_non_available_cores_at_time_t);
void schedule_job_fcfs_score_with_conservative_backfill(struct Job* j, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int adaptative_multiplier, int start_immediately_if_EAT_is_t, int backfill_mode, int* nb_non_available_cores, int* nb_non_available_cores_at_time_t, int mixed_strategy, int* temp_running_nodes);
void get_node_size_to_choose_from(int index, int* first_node_size_to_choose_from, int* last_node_size_to_choose_from);
void get_nb_nodes_and_cores_loading_a_file(struct Node_List** head_node, int t, int* nodes_loading_a_file, int* cores_loading_a_file);

/* From linked_list_functions.c */
void insert_head_job_list(struct Job_List* liste, struct Job* j);
void insert_tail_job_list(struct Job_List* liste, struct Job* j);
void insert_tail_node_list(struct Node_List* liste, struct Node* n);
void insert_tail_data_list(struct Data_List* liste, struct Data* d);
void delete_job_linked_list(struct Job_List* liste, int unique_id_to_delete);
void copy_delete_insert_job_list(struct Job_List* to_delete_from, struct Job_List* to_append_to, struct Job* j);
void copy_delete_insert_job_list_sorted_by_file_size(struct Job_List* to_delete_from, struct Job_List* to_append_to, struct Job* j);
void get_length_job_list(struct Job* head, int* nb_jobs_in_queue, int* nb_cores_in_queue, int* nb_cores_from_workload_1_in_queue);
void insert_next_time_in_sorted_list(struct Next_Time_List* liste, int time_to_insert);
void delete_next_time_linked_list(struct Next_Time_List* liste, int time_to_delete);
void free_next_time_linked_list(struct Next_Time** head_ref);
void insert_tail_to_print_list(struct To_Print_List* liste, struct To_Print* tp);
void insert_job_in_sorted_list(struct Job_List* liste, struct Job* j);
void create_and_insert_tail_interval_list(struct Interval_List* liste, int time_to_insert);
//~ void free_interval_linked_list(struct Interval** head_ref);
void free_interval_linked_list(struct Interval** head_ref, struct Interval** tail_ref);
void create_and_insert_head_time_or_data_already_checked_nb_of_copy_list(struct Time_or_Data_Already_Checked_Nb_of_Copy_List* liste, int time_or_data_to_insert, int nb_of_copy_to_insert);
void free_time_or_data_already_checked_nb_of_copy_linked_list(struct Time_or_Data_Already_Checked_Nb_of_Copy** head_ref);
void increment_time_or_data_nb_of_copy_specific_time_or_data(struct Time_or_Data_Already_Checked_Nb_of_Copy_List* liste, int time_or_data_to_increment);
void sort_cores_by_available_time_in_specific_node(struct Node* n);
void insert_job_sorted_by_decreasing_file_size(struct Job** head, struct Job* newNode);
void sort_job_list_by_file_size(struct Job** head);
//~ void sort_cores_of_a_hole_by_start_time_decreasing_order_in_specific_node(struct Node* n);
void initialize_cores_in_a_hole(struct Core_in_a_hole_List* liste, struct Core_in_a_hole* c);
void insert_cores_in_a_hole_list_sorted_decreasing_order(struct Core_in_a_hole_List* liste, struct Core_in_a_hole* c);
void delete_core_in_hole_from_head(struct Core_in_a_hole_List* liste, int nb_cores_to_delete);
void free_cores_in_a_hole(struct Core_in_a_hole** head_ref);
void delete_core_in_hole_specific_core(struct Core_in_a_hole_List* liste, int unique_id_to_delete);
void delete_specific_data_from_node(struct Data_List* liste, int unique_id_to_delete);
//~ void free_data_list(struct Data** head_ref);
void free_and_copy_data_and_intervals_in_temp_data(struct Node_List** head_node, int t);
void sort_cores_by_unique_id_in_specific_node(struct Node* n);
void insert_cores_in_a_hole_list_sorted_increasing_order(struct Core_in_a_hole_List* liste, struct Core_in_a_hole* c);

/* From scheduler.c */
void get_state_before_day_0_scheduler(struct Job* j, struct Node_List** n, int t);
void fcfs_scheduler(struct Job* head_job, struct Node_List** head_node, int t, bool use_bigger_nodes);
void fcfs_with_a_score_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t, int mixed_strategy);
void fcfs_scheduler_backfill_big_nodes(struct Job* head_job, struct Node_List** head_node, int t, int backfill_big_node_mode, int total_queue_time, int nb_finished_jobs);
//~ void fcfs_scheduler_planned_area_filling(struct Job* head_job, struct Node_List** head_node, int t, long long Planned_Area[3][3]);
void fcfs_scheduler_planned_area_filling(struct Job* head_job, struct Node_List** head_node, int t);
void fcfs_scheduler_ratio_area_filling(struct Job* head_job, struct Node_List** head_node, int t, float Ratio_Area[3][3]);
void fcfs_easybf_scheduler(struct Job* head_job, struct Node_List** head_node, int t, bool use_bigger_nodes);
void fcfs_with_a_score_easybf_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t);
//~ void fcfs_with_a_score_backfill_big_nodes_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int backfill_big_node_mode, int total_queue_time, int finished_jobs);
void fcfs_with_a_score_backfill_big_nodes_95th_percentile_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int number_node_size_128_and_more, int number_node_size_256_and_more, int number_node_size_1024);
void fcfs_with_a_score_area_filling_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int planned_or_ratio, float Ratio_Area[3][3]);
void fcfs_with_a_score_backfill_big_nodes_weighted_random_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy);
void fcfs_with_a_score_area_factor_scheduler (struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int multiplier_area_bigger_nodes, int division_by_planned_area);
void fcfs_with_a_score_backfill_big_nodes_gain_loss_tradeoff_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy);
void locality_scheduler(struct Job* head_job, struct Node_List** head_node, int t);
void eft_scheduler(struct Job* head_job, struct Node_List** head_node, int t);
double fake_eft_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int break_condition_if_not_started_at_t);
double fake_locality_scheduler(struct Job* head_job, struct Node_List** head_node, int t);
double fake_fcfs_with_a_score_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes);
int locality_scheduler_single_job(struct Job* j, struct Node_List** head_node, int t, int nb_non_available_cores, int mode);
int eft_scheduler_single_job(struct Job* j, struct Node_List** head_node, int t, int nb_non_available_cores);
void mixed_if_EAT_is_t_scheduler(struct Job* j, struct Node_List** head_node, int t, int mode);
void fcfs_conservativebf_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int backfill_mode);
void fcfs_with_a_score_conservativebf_scheduler(struct Job* head_job, struct Node_List** head_node, int t, int multiplier_file_to_load, int multiplier_file_evicted, int adaptative_multiplier, int start_immediately_if_EAT_is_t, int backfill_mode, int mixed_strategy);

/* From backfill_functions.c */
bool can_it_get_backfilled (struct Job* j, struct Node* n, int t, int* nb_cores_from_hole, int* nb_cores_from_outside);
void update_cores_for_backfilled_job(struct Job* j, int t, int nb_cores_from_hole, int nb_cores_from_outside);
void fill_cores_minimize_holes (struct Job* j, bool backfill_activated, int backfill_mode, int t, int* nb_non_available_cores);
bool only_check_conservative_backfill(struct Job* j, struct Node_List** head_node, int t, int backfill_mode, int* nb_non_available_cores_at_time_t);
bool only_check_conservative_backfill_with_a_score(struct Job* j, struct Node_List** head_node, int t, int backfill_mode, int* nb_non_available_cores_at_time_t, int multiplier_file_to_load, int multiplier_file_evicted, int start_immediately_if_EAT_is_t);
void get_new_biggest_hole(struct Node_List** head_node);

/* from scheduler_calling.c */
void call_scheduler(char* scheduler, struct Job_List* liste, int t, int use_bigger_nodes, int multiplier_file_to_load, int multiplier_file_evicted, int multiplier_nb_copy, int adaptative_multiplier, int penalty_on_job_sizes, int start_immediately_if_EAT_is_t, int backfill_mode, int number_node_size_128_and_more, int number_node_size_256_and_more, int number_node_size_1024, float (*Ratio_Area)[3], int multiplier_area_bigger_nodes, int division_by_planned_area, int backfill_big_node_mode, int mixed_strategy);
