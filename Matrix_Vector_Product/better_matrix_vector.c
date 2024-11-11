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

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <dimensione_matrice>\n", argv[0]);
        return 1;
    }

    int i, n = atoi(argv[1]); // dimensione matrice NxN passata come argomento

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // creazione gruppo e comunicatore personalizzato
    MPI_Group world_group, custom_group;
    MPI_Comm custom_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int ranks[world_size];
    for (i = 0; i < world_size; i++)
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
    for (i = 0; i < n; i++)
        m1[i] = (float *)malloc(n * sizeof(float));
    
    float *v2 = (float *)malloc(n * sizeof(float));

    float local_result = 0.0;
    float *result = (float *)malloc(n * sizeof(float));

    // processo radice genera casualmente m1 e v2
    if (custom_rank == 0) {
        srand(time(NULL));
        for (i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                m1[i][j] = (float)(rand() % 10);
            
            v2[i] = (float)(rand() % 10);
        }
    }

    // sincronizza processi prima di iniziare il calcolo
    MPI_Barrier(custom_comm);
    double start_time = MPI_Wtime();

    int local_n = n / custom_size + (custom_rank < n % custom_size ? 1 : 0);    // numero di elementi locali per ogni processo
    float *local_row = (float *)malloc(local_n * sizeof(float));    // riga locale
    float *local_v2 = (float *)malloc(local_n * sizeof(float)); // vettore locale

    // counts e displs per la Scatterv di v2
    int counts_v2[custom_size], displs_v2[custom_size];
    int offset = 0;
    for (int p = 0; p < custom_size; p++) {
        counts_v2[p] = n / custom_size + (p < n % custom_size ? 1 : 0);
        displs_v2[p] = offset;
        offset += counts_v2[p];
    }

    MPI_Scatterv(v2, counts_v2, displs_v2, MPI_FLOAT, local_v2, local_n, MPI_FLOAT, 0, custom_comm);

    // counts e displs per la Scatterv di m1
    for (i = 0; i < n; i++) {
        int counts[custom_size], displs[custom_size];
        offset = 0;
        for (int p = 0; p < custom_size; p++) {
            counts[p] = n / custom_size + (p < n % custom_size ? 1 : 0);
            displs[p] = offset;
            offset += counts[p];
        }

        MPI_Scatterv(m1[i], counts, displs, MPI_FLOAT, local_row, local_n, MPI_FLOAT, 0, custom_comm);

        local_result = fdot(local_n, local_row, local_v2);

        // raccoglie i risultati parziali e somma i risultati nel processo radice
        MPI_Reduce(&local_result, &result[i], 1, MPI_FLOAT, MPI_SUM, 0, custom_comm);
    }

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
    if (custom_rank == 0) {
        FILE *file = fopen("results.csv", "a");
        if (file) {
            fprintf(file, "%d, %f\n", n, elapsed_time);
            fclose(file);
        } else {
            printf("Errore nell'apertura del file.\n");
        }
    }

    // deallocazione memoria
    for (i = 0; i < n; i++)
        free(m1[i]);
    free(m1);
    free(v2);
    free(local_row);
    free(local_v2);
    free(result);

    // deallocazione gruppo e comunicatore personalizzato
    MPI_Group_free(&custom_group);
    MPI_Comm_free(&custom_comm);

    MPI_Finalize();
    return 0;
}
