#include <stdio.h>
#include <mpi.h>

// Funzione che calcola il prodotto scalare tra due vettori
float fdot(int n, float *v1, float *v2) {
    float fd = 0.0;

    for (int i = 0; i < n; i++)
        fd += v1[i] * v2[i];
    
    return fd;
}

int main(int argc, char **argv) {
    int n = 3;
    float m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float v2[3] = {3, 3, 3};
    float result[3];

    MPI_Init(&argc, &argv);

    int n_proc, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        // master
        for (int dest = 1; dest < n_proc; dest++) {
            // invia riga corrispondente della matrice e il vettore a ogni worker
            MPI_Send(m1[dest - 1], n, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(v2, n, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        }

        // riceve risultati parziali dai worker
        for (int src = 1; src < n_proc; src++) {
            float local_result;
            MPI_Recv(&local_result, 1, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result[src - 1] = local_result;
        }

        printf("Product of matrix and vector is:\n");
        for (int i = 0; i < n; i++) {
            printf("%f\n", result[i]);
        }

    } else if (my_rank <= n) {
        // worker
        float local_m1[3];
        float local_result;

        // riceve la propria riga della matrice e il vettore
        MPI_Recv(local_m1, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(v2, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // prodotto scalare tra riga e vettore
        local_result = fdot(n, local_m1, v2);

        // invia risultato parziale al master (che è il processo radice)
        MPI_Send(&local_result, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
