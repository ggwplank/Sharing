#include <stdio.h>
#include <mpi.h>

float fdot(int n, float *v1, float *v2) {
    float fd = 0.0;
    for (int i = 0; i < n; i++) {
        fd += v1[i] * v2[i];
    }
    return fd;
}

int main(int argc, char** argv) {
    int i, n = 4;
    float m1[4][4] = {{1, 2, 3, 3}, {4, 5, 6, 4}, {7, 8, 9, 6}, {4, 6, 8, 9}};
    float v2[4] = {2, 4, 3, 5};
    float local_result = 0.0;
    float result[4];

    MPI_Init(&argc, &argv);

    int n_proc, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int local_n = n / n_proc + (my_rank < n % n_proc ? 1 : 0);  // Numero di elementi locali
    float local_row[local_n];  // Memoria per i dati locali di una singola riga
    float local_v2[local_n];   // Memoria per il segmento locale di v2

    // Configura counts e displs per la distribuzione di v2
    int counts_v2[n_proc], displs_v2[n_proc];
    int offset = 0;
    for (int p = 0; p < n_proc; p++) {
        counts_v2[p] = n / n_proc + (p < n % n_proc ? 1 : 0);
        displs_v2[p] = offset;
        offset += counts_v2[p];
    }

    // Distribuisci solo le parti necessarie di v2 a ciascun processo
    MPI_Scatterv(v2, counts_v2, displs_v2, MPI_FLOAT, local_v2, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for(i=1;i<n_proc;i++){
        printf("Hello %d, my vector is %f,%f\n", my_rank,local_v2[0],local_v2[1]);
    }


    int j;
    // Distribuisci e calcola il prodotto per ogni riga
    for (i = 0; i < n; i++) {
        int counts[n_proc], displs[n_proc];

        // Configura counts e displs per la riga corrente della matrice m1
        offset = 0;
        for (int p = 0; p < n_proc; p++) {
            counts[p] = n / n_proc + (p < n % n_proc ? 1 : 0);
            displs[p] = offset;
            offset += counts[p];
        }

        // Distribuisci la riga i-esima tra i processi
        MPI_Scatterv(m1[i], counts, displs, MPI_FLOAT, local_row, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        for(j=1;j<n_proc;j++){
            printf("Hello %d, my row is %f,%f\n", my_rank,local_row[0],local_row[1]);
        }


        // Calcola il prodotto scalare tra la parte locale della riga e il segmento di v2
        local_result = fdot(local_n, local_row, local_v2);

        // Riduci i risultati parziali di tutti i processi per ottenere il risultato completo della riga
        MPI_Reduce(&local_result, &result[i], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Il processo 0 stampa il risultato finale
    if (my_rank == 0) {
        printf("Product of matrix and vector is:\n");
        for (i = 0; i < n; i++) {
            printf("%f\n", result[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
