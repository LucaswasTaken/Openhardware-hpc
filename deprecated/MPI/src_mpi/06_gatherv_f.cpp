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
    float *x = new float[npoints];
    float *y = new float[npoints];    
    if(rank==0){
        create_random_points(x,y,npoints,true);
    }

    // Comunicacao de broadcast dos pontos
    MPI_Bcast(x,npoints,MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Bcast(y,npoints,MPI_FLOAT,0, MPI_COMM_WORLD);

    // Particionamento de tarefas
    int num_elems_proc_gen = (int) npoints/np;
    int num_elems_proc_last = (int) npoints/np + npoints%np;
    int num_elems_proc;        
    if(rank==np-1){
        num_elems_proc = num_elems_proc_last; 
    }else{
        num_elems_proc = num_elems_proc_gen;
    }
    std::cout << "Rank: " << rank << " Pontos: " << num_elems_proc << std::endl;


    // Processamento paralelo de calculo de distancia dos pontos
    float * distances = new float[num_elems_proc];
    int start = rank*num_elems_proc_gen;
    compute_min_distances(x,y,npoints,start,num_elems_proc, distances);


    // Comunicacao de coleta de dados no processo master
    float * distances_global = NULL;
    int * recvcounts = NULL;
    int * displs = NULL;
    if(rank==0){
        distances_global = new float[npoints];
        recvcounts = new int[np];
        displs = new int[np];
        for(int p=0;p<np;p++){
            recvcounts[p] = num_elems_proc_gen;
            displs[p] = p*num_elems_proc_gen;
            if(p==np-1){
                recvcounts[p]= num_elems_proc_last;
            }
        }
    }    
    MPI_Gatherv(distances, num_elems_proc, MPI_FLOAT, distances_global, recvcounts, displs, MPI_FLOAT,0, MPI_COMM_WORLD);


    // Impressao de resultados no processo master
    if(rank==0){
        print_array("min_dist=",distances_global, npoints);    
    }    

    // Libera memoria
    delete [] x;
    delete [] y;
    delete [] distances;
    if(distances_global != NULL){ delete [] distances_global;}
    if(recvcounts != NULL){ delete [] recvcounts;}
    if(displs != NULL){ delete [] displs;}

    // Finaliza o ambiente do MPI
    MPI_Finalize();
}