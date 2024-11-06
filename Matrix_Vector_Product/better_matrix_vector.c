#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int i, n = 3;
    float m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float v2[3] = {3, 3, 3};
    float local_result = 0.0;
    float result[3];  
    int tag = 0;

    MPI_Init(&argc, &argv);

    int n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    float local_m1[3];  // Memoria per una riga della matrice

    if (my_rank == 0) {
        // Il processo 0 invia le righe della matrice ai processi (rank > 0)
        for (int proc = 1; proc < n_proc; proc++) {
            MPI_Send(m1[proc - 1], n, MPI_FLOAT, proc, tag, MPI_COMM_WORLD);
        }
        
        // Il processo 0 invia il vettore completo a tutti i processi
        MPI_Bcast(v2, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        // Ogni processo (rank > 0) riceve una riga della matrice
        MPI_Recv(local_m1, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Calcolo del prodotto scalare tra la riga ricevuta e il vettore
        for (i = 0; i < n; i++) {
            local_result += local_m1[i] * v2[i];
        }
    }

    // Usa MPI_Gather per raccogliere i risultati parziali nel processo 0
    MPI_Gather(&local_result, 1, MPI_FLOAT, result, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

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
