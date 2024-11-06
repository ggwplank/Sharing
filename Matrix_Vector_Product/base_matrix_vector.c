#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int i, n = 3;
    float m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; 
    float v2[3] = {3, 3, 3};
    float local_dot;
    float result[3];  
    int tag = 0;

    MPI_Init(&argc, &argv);

    int n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Ogni processo riceve una riga della matrice
    int local_n = (n / n_proc) +1;  // Supponiamo n divisibile per world_size

    float local_m1[3];  // Memoria per una riga della matrice
    float local_result;  // Risultato parziale per ciascun processo

    // Distribuisci le righe della matrice ai vari processi
    MPI_Scatter(m1, n, MPI_FLOAT, local_m1, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Invia il vettore completo a tutti i processi
    MPI_Bcast(v2, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Calcolo del prodotto scalare tra la riga ricevuta e il vettore
    local_result = 0.0;
    for (i = 0; i < n; i++) {
        local_result += local_m1[i] * v2[i];
    }

    // Raccogli i risultati parziali in 'result' nel processo radice
    MPI_Gather(&local_result, 1, MPI_FLOAT, result, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Il processo radice stampa il risultato finale
    if (my_rank == 0) {
        printf("Product of matrix and vector is:\n");
        for (i = 0; i < n; i++) {
            printf("%f\n", result[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
