#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

float fdot(int n, float *v1, float *v2) {
    float fd = 0.0;
    for (int i = 0; i < n; i++) {
        fd += v1[i] * v2[i];
    }
    return fd;
}

void generate_matrix_and_vector(int n, float **m1, float *v2) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            m1[i][j] = (float)(rand() % 10);
        
        v2[i] = (float)(rand() % 10);
    }
}

void scatter_data(int n, float *data, float *local_data, int custom_rank, int custom_size, MPI_Comm custom_comm) {
    int local_n = n / custom_size + (custom_rank < n % custom_size ? 1 : 0);    // numero di elementi locali per ogni processo
    int counts[custom_size], displs[custom_size];
    int offset = 0;

    for(int p = 0; p < custom_size; p++) {
        counts[p] = n / custom_size + (p < n % custom_size ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }

    MPI_Scatterv(data, counts, displs, MPI_FLOAT, local_data, local_n, MPI_FLOAT, 0, custom_comm);
}

void compute_local_results(int n, float **m1, float *v2, float *result, int custom_rank, int custom_size, MPI_Comm custom_comm) {
    int local_n = n / custom_size + (custom_rank < n % custom_size ? 1 : 0);
    float *local_row = (float *)malloc(local_n * sizeof(float));
    float *local_v2 = (float *)malloc(local_n * sizeof(float));
    
    // scatter v2
    scatter_data(n, v2, local_v2, custom_rank, custom_size, custom_comm);

    for(int i = 0; i < n; i++) {
        scatter_data(n, m1[i], local_row, custom_rank, custom_size, custom_comm);
        float local_result = fdot(local_n, local_row, local_v2);
        MPI_Reduce(&local_result, &result[i], 1, MPI_FLOAT, MPI_SUM, 0, custom_comm);
    }

    free(local_row);
    free(local_v2);
}

void write_to_file(const char* filename, int n, double elapsed_time) {
    FILE *file = fopen(filename, "a");
    if (file) {
        fprintf(file, "%d, %f\n", n, elapsed_time);
        fclose(file);
    } else {
        printf("Errore nell'apertura del file.\n");
    }
}

void free_memory(int n, float **m1, float *v2, float *result) {
    for (int i = 0; i < n; i++)
        free(m1[i]);
    free(m1);
    free(v2);
    free(result);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Invalid params");
        return 1;
    }

    int n = atoi(argv[1]); // dimensione matrice NxN passata come argomento
    char* result_file_name = argv[2];

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // creazione gruppo e comunicatore personalizzato
    MPI_Group world_group, custom_group;
    MPI_Comm custom_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int ranks[world_size];
    for (int i = 0; i < world_size; i++)
        ranks[i] = i;

    // crazione gruppo con tutti i rank del comunicatore globale
    MPI_Group_incl(world_group, world_size, ranks, &custom_group);
    MPI_Comm_create(MPI_COMM_WORLD, custom_group, &custom_comm);

    int custom_rank, custom_size;
    if (custom_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(custom_comm, &custom_rank);
        MPI_Comm_size(custom_comm, &custom_size);
    } else {
        MPI_Finalize();
        return 0;
    }

    // allocazione dinamica matrice e vettore
    float **m1 = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++)
        m1[i] = (float *)malloc(n * sizeof(float));
    
    float *v2 = (float *)malloc(n * sizeof(float));

    float local_result = 0.0;
    float *result = (float *)malloc(n * sizeof(float));

    // processo radice genera casualmente m1 e v2
    if (custom_rank == 0)
        generate_matrix_and_vector(n, m1, v2);

    // sincronizza processi prima di iniziare il calcolo
    MPI_Barrier(custom_comm);
    double start_time = MPI_Wtime();

    compute_local_results(n, m1, v2, result, custom_rank, custom_size, custom_comm);

    // sincronizza i processi dopo il calcolo prima di calcolare il tempo
    MPI_Barrier(custom_comm);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    /*
    // stampa risultato
    if(custom_rank == 0) {
        printf("Product of matrix and vector is:\n");
        for (i = 0; i < n; i++)
            printf("%f\n", result[i]);
    }
    */

    // scrittura su file
    if (custom_rank == 0)
        write_to_file(result_file_name, n, elapsed_time);

    // deallocazione memoria
    free_memory(n, m1, v2, result);

    // deallocazione gruppo e comunicatore personalizzato
    MPI_Group_free(&custom_group);
    MPI_Comm_free(&custom_comm);

    MPI_Finalize();
    return 0;
}