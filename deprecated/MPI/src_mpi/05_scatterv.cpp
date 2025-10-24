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
    double *x = NULL; 
    if(rank==0){
        x = new double[npoints];
        create_data(x,npoints);
    }

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

    // Comunicacao dos subvetores
    double *x_local = new double[num_elems_proc];

    int * sendcounts = NULL;
    int * displs = NULL;
    if(rank==0){
        sendcounts = new int[np];
        displs = new int[np];
        for(int p=0;p<np;p++){
            sendcounts[p] = num_elems_proc_gen;
            displs[p] = p*num_elems_proc_gen;
            if(p==np-1){
                sendcounts[p]= num_elems_proc_last;
            }
        }
    }    
    MPI_Scatterv(x, sendcounts, displs, MPI_DOUBLE, x_local, num_elems_proc,MPI_DOUBLE,0,MPI_COMM_WORLD);


    // Soma dos subvetores
    double local_sum = sum_values(x_local,num_elems_proc);
    std::cout << "Rank: " << rank << " Soma local: " << local_sum << std::endl;

    // Soma global
    double global_sum=0; 
    MPI_Allreduce(&local_sum, &global_sum,1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Reduce(&local_sum, &global_sum,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);

    // Media
    double average = global_sum/npoints;

    // Impressao no processo master
    std::cout << "Rank: " << rank << " Media: " << average << std::endl;

    // Libera memoria
    if(x != NULL){ delete [] x;}
    delete [] x_local;
    if(sendcounts != NULL){ delete [] sendcounts;}
    if(displs != NULL){ delete [] displs;}


    // Finaliza o ambiente do MPI
    MPI_Finalize();
}