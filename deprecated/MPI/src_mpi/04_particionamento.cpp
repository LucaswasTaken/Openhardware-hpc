#include <mpi.h>
#include <iostream>
#include <cmath>

#include "points.hpp"

int main(int argc, char** argv){

    // Inicializa o ambiente do MPI
    MPI_Init(&argc, &argv);

    // Numero de processos
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Rank do processo (0 a n-1, onde n = numero de processos)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Montagem de dados
    int npoints=128000;
    double *x = new double[npoints];
    double *y = new double[npoints];    
    if(rank==0){
        create_points_on_xaxis(x,y,npoints);
    }

    // Comunicacao de broadcast dos pontos
    MPI_Bcast(x,npoints,MPI_DOUBLE,0, MPI_COMM_WORLD);
    MPI_Bcast(y,npoints,MPI_DOUBLE,0, MPI_COMM_WORLD);

    // Particionamento de tarefas
    int num_elems_proc = (int) npoints/np;
    int start = rank*num_elems_proc;
    if(rank==np-1){
        num_elems_proc = num_elems_proc + npoints%np;
    }
    std::cout << "Rank: " << rank << " Pontos: " << num_elems_proc << std::endl;


    // Processamento paralelo de calculo de distancia dos pontos
    double * distances = new double[num_elems_proc];
    compute_min_distances(x,y,npoints,start,num_elems_proc, distances);


    // Libera memoria
    delete [] x;
    delete [] y;
    delete [] distances;

    // Finaliza o ambiente do MPI
    MPI_Finalize();
}