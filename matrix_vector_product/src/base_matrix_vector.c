#include <stdio.h>
#include <mpi.h>

// Write a code that computes the product between a matrix and a vector.

float fdot(int n, float *v1, float *v2)
{
    int i;
    float fd;

    fd = 0.0;
    for (i = 0; i < n; i++)
        fd += v1[i] * v2[i];

    return (fd);
}

int main(int argc, char **argv)
{
    int i, n = 3, tag = 0;
    float m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float v2[3] = {3, 3, 3};

    float local_m1[3]; // ogni processo riceve una riga della matrice
    float local_result; // risultato parziale per ciascun processo
    
    float result[3];
    // float *result = NULL;

    MPI_Init(&argc, &argv);

    int n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // le righe della matrice vengono distribuite ai vari processi
    MPI_Scatter(m1, n, MPI_FLOAT, local_m1, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // il vettore che deve moltiplicare la matrice viene inviato a tutti i processi, visto che è in comune
    MPI_Bcast(v2, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // calcolo del prodotto scalare tra la riga ricevuta e il vettore
    local_result = fdot(n, local_m1, v2);

    // i risultati parziali vengono raccolti in 'result' nel processo radice
    MPI_Gather(&local_result, 1, MPI_FLOAT, result, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        printf("Product of matrix and vector is:\n");
        for (i = 0; i < n; i++)
        {
            printf("%f\n", result[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
