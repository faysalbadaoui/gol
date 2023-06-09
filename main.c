#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include <SDL2/SDL.h>
#include "./game.h"
#include "./logic.h"
#include "./render.h"


void usage(void)
{
	printf("\nUsage: conway [-g] [-w weight] [-h heigt] [-i input_board_file] [-o output_board_file] [-e End_time] [-t ticks] [-c cell_size] \n\n-t\tSet number of ticks in microseconds.\n\t");
	printf("\n -g\tEnable graphical mode.\n\n");
	printf("\n -w\tSet board weight.\n\n");
	printf("\n -h\tSet board height.\n\n");
	printf("\n -i\tInput board file.\n\n");
	printf("\n -o\tOutput board file.\n\n");
	printf("\n -e\tNumber of simulation iterations.\n\n");
	printf("Enter extremely low values at own peril.\n\tRecommended to stay in 30000-100000 range.\n\tDefaults to 50000.\n\n");
	printf("\n -c\tSet cell size to tiny, small, medium or large.\n\tDefaults to small.\n\n");
}

board_t* create_board(int col_num, int row_num) {

    board_t* board = malloc(sizeof(board_t));

    // Ensure successful memory allocation
    if (board == NULL) {
        printf("Error: Unable to allocate memory for the board.\n");
        exit(1);
    }

    board->COL_NUM = col_num;
    board->ROW_NUM = row_num;
    board->game_state = RUNNING_STATE;

    for (int i = 0; i < D_ROW_NUM; i++) {
        for (int j = 0; j < D_COL_NUM; j++) {
            board->cell_state[i][j] = DEAD;
        }
    }

    return board;
}

int compute_new_state(int current_state, int num_alive_neighbors) {
    if (current_state == ALIVE && (num_alive_neighbors < 2 || num_alive_neighbors > 3)) {
        return DEAD;
    } else if (current_state == DEAD && num_alive_neighbors == 3) {
        return ALIVE;
    } else {
        return current_state;
    }
}
void distribute_board(board_t* global_board, board_t* local_board, int rank, int size) {
    // Calculate the number of rows per process.
    int rows_per_process = global_board->ROW_NUM / size;
    int total_cells_per_process = rows_per_process * global_board->COL_NUM;
    unsigned char* flatten_board = NULL;

    if (rank == 0) {
        // Flatten the global board to a 1D array for scattering.
        flatten_board = malloc(global_board->ROW_NUM * global_board->COL_NUM * sizeof(unsigned char));
        for (int i = 0; i < global_board->ROW_NUM; i++) {
            for (int j = 0; j < global_board->COL_NUM; j++) {
                flatten_board[i * global_board->COL_NUM + j] = global_board->cell_state[i][j];
            }
        }
    }

    // Prepare receiving buffer
    unsigned char* recv_buf = malloc(total_cells_per_process * sizeof(unsigned char));

    // Scatter the flatten board to all processes.
    MPI_Scatter(flatten_board, total_cells_per_process, MPI_UNSIGNED_CHAR,
                recv_buf, total_cells_per_process, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Copy from recv_buf to local board cell_state
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < global_board->COL_NUM; j++) {
            local_board->cell_state[i][j] = recv_buf[i * global_board->COL_NUM + j];
        }
    }


    // Copy the common board settings.
    local_board->game_state = global_board->game_state;
    local_board->COL_NUM = global_board->COL_NUM;
    local_board->ROW_NUM = rows_per_process;
    local_board->CELL_WIDTH = global_board->CELL_WIDTH;
    local_board->CELL_HEIGHT = global_board->CELL_HEIGHT;

    // Clean up
    if (rank == 0) {
        free(flatten_board);
    }
    free(recv_buf);
}



void gather_board(board_t* local_board, board_t* global_board, int rank, int size, MPI_Comm comm)
{
    int stripe_height = local_board->ROW_NUM;
    int total_cells = stripe_height * local_board->COL_NUM;
    unsigned char *gathered_cells = NULL;

    if (rank == 0) {
        gathered_cells = malloc(total_cells * size * sizeof(unsigned char));
    }

    MPI_Gather(local_board->cell_state[0], total_cells, MPI_UNSIGNED_CHAR, 
               gathered_cells, total_cells, MPI_UNSIGNED_CHAR, 0, comm);

    // On rank 0, copy the gathered cells back into the global board.
    if (rank == 0) {
        for (int i = 0; i < global_board->ROW_NUM; i++) {
            for (int j = 0; j < global_board->COL_NUM; j++) {
                global_board->cell_state[i][j] = gathered_cells[i*global_board->COL_NUM + j];
            }
        }
        free(gathered_cells);
    }
}


void print_local_board(board_t* local_board) {
    for (int i = 0; i < local_board->ROW_NUM; ++i) {
        for (int j = 0; j < local_board->COL_NUM; ++j) {
            printf("%d ", local_board->cell_state[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)

{
	// Set default rate of ticks.
	int rank, size;
	MPI_Init(&argc, &argv);  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

		// Configuración inicial del juego
	int COL_NUM = D_COL_NUM;
	int ROW_NUM = D_ROW_NUM;

	// Calcular el tamaño de la submatriz para cada proceso
	int submatrix_cols = COL_NUM / size;
	int extra_cols = COL_NUM % size;

	int start_col = rank * submatrix_cols;
	int end_col = start_col + submatrix_cols - 1;

	if (rank < extra_cols) {
		start_col += rank;
		end_col += rank + 1;
	} else {
		start_col += extra_cols;
		end_col += extra_cols;
	}

	// Crear submatriz local para cada proceso
	int local_cols = end_col - start_col + 1;
	int local_rows = ROW_NUM;

	// Asegurar que la submatriz local tenga espacio suficiente para los bordes vecinos
	if (start_col > 0) {
		local_cols += 1;
		start_col -= 1;
	}
	if (end_col < COL_NUM - 1) {
		local_cols += 1;
		end_col += 1;
	}

	printf(" I'm rank, %d : local_cols: %d, local_rows: %d\n", rank, local_cols, local_rows);
	printf(" I'm rank, %d : start_col: %d, end_col: %d\n", rank, start_col, end_col);

	int TICKS = 50000;
	bool LoadFile = false, SaveFile = false; 
	int EndTime=-1;
	char input_file[256],output_file[256];
	
	// Graphics.
	int SCREEN_WIDTH;
	int SCREEN_HEIGHT;
	int PEEPER_SIZE;
	int PEEPER_OFFSET;  
	SDL_Event e;
	SDL_Rect peeper; // In future may take values from event loop.
	SDL_Window *window;
	SDL_Renderer *renderer;

	// Set initial window scaling factor
	float SCALE = 0.5;

	// This will store window dimension information.
	int window_width;
	int window_height;

	board_t *board = (board_t*) malloc(sizeof(board_t));
	if (board==NULL) {
		fprintf(stderr,"Error reserving board memory %lf Kbytes",sizeof(board_t)/1024.0);
		exit(1);
	}
	// Configure board initial state.
	board->game_state = RUNNING_STATE;
	board->CELL_WIDTH = 4; // Reasonable default size
	board->CELL_HEIGHT = 4;
	board->COL_NUM = D_COL_NUM;
	board->ROW_NUM = D_ROW_NUM;

	for (int i = 0; i < board->COL_NUM; i++) {
		for (int j = 0; j < board->ROW_NUM; j++)
		board->cell_state[i][j] = DEAD;
	}

	unsigned char neighbors[D_COL_NUM][D_ROW_NUM] = {DEAD};

	// Command line options.
	int opt;

	while((opt = getopt(argc, argv, "t:c:h:i:o:w:H:e:g")) != -1) {
		switch (opt) {
		case 't':
			TICKS = atoi(optarg);
			break;
		case 'i':
			strcpy(input_file,optarg);
			LoadFile = true;
			break;
		case 'o':
			SaveFile=true;
			strcpy(output_file,optarg);
			printf("Output Board file %s.\n",optarg);
			break;
		case 'w':
			board->COL_NUM = atoi(optarg);
			printf("Board width %d.\n",board->COL_NUM);
			break;  
		case 'h':
			board->ROW_NUM = atoi(optarg);
			printf("Board height %d.\n",board->ROW_NUM);        
			break;    
		case 'e':
			printf("End Time: %s.\n",optarg);
			EndTime = atoi(optarg);
			break;     	
		case 'g':
			Graphical_Mode = true;
			break;
		case 'c':
			if (strcmp(optarg,"tiny") == 0) {
			board->CELL_WIDTH = 2;
			board->CELL_HEIGHT = 2;
			}      
			else if (strcmp(optarg,"small") == 0) {
			board->CELL_WIDTH = 5;
			board->CELL_HEIGHT = 5;
			}      
			else if (strcmp(optarg,"medium") == 0) {
			board->CELL_WIDTH = 10;
			board->CELL_HEIGHT = 10;
			}
			else if (strcmp(optarg,"large") == 0) {
			board->CELL_WIDTH = 25;
			board->CELL_HEIGHT = 25;        
			}
			break;
		case 'H':
			usage();
			exit(EXIT_SUCCESS);
			break;
		case '?':
			if (optopt == 't' || optopt == 's' || optopt == 'c' || optopt == 'i' || optopt == 'o' || optopt == 'w' || optopt == 'h' || optopt == 'e' )
			fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint (optopt))
			fprintf (stderr, "Unknown option `-%c'.\n", optopt);
			else
			fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
			printf("Setting default options.\n");
			usage();
			break;
		default:
			printf("Setting default options.\n");
			usage();
			break;
		}
	}

	if (LoadFile) 	
	{
		printf("Loading Board file %s.\n",input_file);
		life_read(input_file, board);
	}
	else
	{ // Rando, init file
		printf("Init Cells\n");fflush(stdout);
		double prob = 0.20;
		int seed = 123456789;
		life_init(board, prob, &seed);
	}
	if(rank == 0){
		if (Graphical_Mode) 
		{
			// Initialize SDL subsystem
			if (SDL_Init(SDL_INIT_VIDEO) != 0) {
				fprintf(stderr, "Could not initialize SDL2: %s\n", SDL_GetError());
				return EXIT_FAILURE;
			}

			// Grab display dimensions.
			SDL_DisplayMode DM;
			SDL_GetCurrentDisplayMode(0, &DM);

			// Set and scale window dimensions.
			SCREEN_WIDTH = DM.w;
			SCREEN_HEIGHT = DM.h;
			SCREEN_WIDTH = SCREEN_WIDTH * SCALE ;
			SCREEN_HEIGHT = SCREEN_HEIGHT * SCALE ;

			// An SDL_Rect type called peeper whose scale is fed into
			// SDL_RenderSetViewport() very shortly must also include
			// an offset to ensure boundary conditions along x=0 and
			// y=0 are sufficiently out of frame.
			PEEPER_SIZE = 10 * SCREEN_WIDTH; // Should be sufficient.
			PEEPER_OFFSET = PEEPER_SIZE / 4;
			
			PEEPER_SIZE = 10 * SCREEN_WIDTH; // Should be sufficient.
			PEEPER_OFFSET = PEEPER_SIZE / 4;

			// Create window
			window = SDL_CreateWindow("Conway's Game",
													1, 1,
													SCREEN_WIDTH, SCREEN_HEIGHT,
													SDL_WINDOW_SHOWN);
			if (window == NULL) {
				fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
				return EXIT_FAILURE;
			}

			// Create renderer
			renderer = SDL_CreateRenderer(window, -1,
														SDL_RENDERER_ACCELERATED |
														SDL_RENDERER_PRESENTVSYNC);
			if (renderer == NULL) {
				SDL_DestroyWindow(window);
				fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
				return EXIT_FAILURE;
			}
		}
	}

	printf("Start Simulatiom.\n");fflush(stdout);

	bool quit = false;
	int Iteration=0;
	if (rank == 0){
		printf("Process %d has the following global board:\n", rank);
		for(int i = 0; i < board->COL_NUM; i++) {
			for(int j = 0; j < board->ROW_NUM; j++) {
				printf("%d ", board->cell_state[i][j]);
			}
			printf("\n");
		}
	}
	while (quit==false && (EndTime<0 || Iteration<EndTime)) 
	{	
		board_t* local_board = create_board(board->COL_NUM, board->ROW_NUM / size);
		distribute_board(board, local_board, rank, size);
		int rows_per_process = board->ROW_NUM / size;
		// Print the local board of each process
		printf("\nProcess %d has the following local board:\n", rank);
		int x = 0;
		for(int i = 0; i < rows_per_process; i++) {
			for(int j = 0; j < board->COL_NUM; j++) {
				printf("%d ", local_board->cell_state[i][j]);
				x++;
			}
			printf("\n");
		}
		printf("Process %d cells: %d\n", rank, x);

		if (Graphical_Mode && rank == 0)
		{  
			//Poll event and provide event type to switch statement
			while (SDL_PollEvent(&e)) {
				switch (e.type) {
					case SDL_QUIT:
					quit = true;
					break;
					case SDL_KEYDOWN:
					// For keydown events, test for keycode type. See Wiki SDL_Keycode.
					if (e.key.keysym.sym == SDLK_ESCAPE) {
						quit = true;
						break;
					}
					if (e.key.keysym.sym == SDLK_SPACE) {
						if (board->game_state == RUNNING_STATE) {
						board->game_state = PAUSE_STATE;
						printf("Game paused: editing enabled.\n");
						break;
						}
						else if (board->game_state == PAUSE_STATE) {
						board->game_state = RUNNING_STATE;
						printf("Game running.\n");
						break;
						}
					}
					if (e.key.keysym.sym == SDLK_BACKSPACE) {
						for (int i = 0; i < board->COL_NUM; i++) {
						for (int j = 0; j < board->ROW_NUM; j++)
							board->cell_state[i][j] = DEAD;
						}
						break;
					}
					break;
					case SDL_MOUSEBUTTONDOWN:
					click_on_cell(board,
									(e.button.y + PEEPER_OFFSET) / board->CELL_HEIGHT,
									(e.button.x + PEEPER_OFFSET) / board->CELL_WIDTH);
					printf("%d, %d\n", e.button.x, e.button.y);
					break;
					default: {}
				}
			}

			// Assignment to viewport
			peeper.x = -PEEPER_OFFSET;
			peeper.y = -PEEPER_OFFSET;
			peeper.w = PEEPER_SIZE;
			peeper.h = PEEPER_SIZE;
			SDL_RenderSetViewport(renderer, &peeper);

			// printf("Peeper OFFSET: (%d, %d).\n",-PEEPER_OFFSET, -PEEPER_OFFSET); fflush(stdout);     
			// printf("Peeper position: (%d, %d).\n",peeper.x, peeper.y); fflush(stdout); 
			// printf("Peeper size: (%d, %d).\n",peeper.w, peeper.h); fflush(stdout);     

			// Calculate upper left hand corner and then find domain of array used.
			const int origin_x = PEEPER_OFFSET / board->CELL_WIDTH;
			const int origin_y = PEEPER_OFFSET / board->CELL_HEIGHT;
			const int domain_x = D_COL_NUM - origin_x;
			const int domain_y = D_ROW_NUM - origin_y;
			
			// printf("Origin: (%d, %d).\n",origin_x, origin_y); fflush(stdout); 
			// printf("Domain: (%d, %d).\n",domain_x, domain_y); fflush(stdout); 

			// Use cell size to determine maximum possible window size without allowing
			// array overflow. This will be tested against SDL window size polls.
			// There be dragons here.
			const int maximum_width = domain_x * board->CELL_WIDTH;
			const int maximum_height = domain_y * board->CELL_HEIGHT;
			// Get window measurements in real time.
			SDL_GetWindowSize(window, &window_width, &window_height);
		
			// printf("Maximun window: (%d, %d).\n",maximum_width, maximum_height); fflush(stdout); 
			// printf("Window Size: (%d, %d).\n",window_width, window_height); fflush(stdout); 
		
			// Don't allow overflow.
			if (window_width > maximum_width) {
			printf("WARNING: Attempting to exceed max window size in x (win width: %d - win max width: %d).\n",window_width, maximum_width);
			SDL_SetWindowSize(window, maximum_width, window_height);
			}
			if (window_height > maximum_height) {
			printf("WARNING: Attempting to exceed max window size in y (win height: %d - win max height: %d).\n", window_height, maximum_height);
			SDL_SetWindowSize(window, window_width, maximum_height);
			}
			// Draw
			SDL_SetRenderDrawColor(renderer, 40, 40, 40, 1);
			SDL_RenderClear(renderer);
		}
		gather_board(local_board, board, rank, size, MPI_COMM_WORLD);
		if(rank == 0){
			printf("Gathered all results\n");
			render_board(renderer, local_board, neighbors);
		}
		if (Graphical_Mode && rank == 0)  // only the root process (rank 0) does the graphical work
		{ 
			SDL_RenderPresent(renderer);
			usleep(TICKS);
		}
		
		printf("[%05d] Life Game Simulation step.\r",++Iteration); fflush(stdout);   
	}
	printf("\nEnd Simulation.\n");
	if(rank == 0){
		if (Graphical_Mode) 
		{ 
			// Clean up
			SDL_DestroyWindow(window);
			SDL_Quit();
		}
		// Save board
		if (SaveFile) {
			printf("Writting Board file %s.\n",output_file); fflush(stdout);      
			life_write(output_file, board);
		}
	}


	return EXIT_SUCCESS;

	MPI_Finalize();
}
