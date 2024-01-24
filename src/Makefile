all: main.c read_input_files.c print_functions.c basic_functions.c scheduler.c linked_list_functions.c backfill_functions.c scheduler_calling.c
	gcc -std=c99 -Wall -O3 -o main main.c read_input_files.c print_functions.c basic_functions.c scheduler.c linked_list_functions.c backfill_functions.c scheduler_calling.c -I. -lm

save:
	gcc -std=c99 -Wall -O3 -DSAVE -o main main.c read_input_files.c print_functions.c basic_functions.c scheduler.c linked_list_functions.c backfill_functions.c scheduler_calling.c -I. -lm
	
clean:
	rm *.o
