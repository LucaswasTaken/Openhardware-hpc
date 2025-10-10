#include <mpi.h>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {

    // Inicializa o ambiente do MPI
    MPI_Init(&argc, &argv);

    // Numero de processos
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Rank do processo (0 a n-1, onde n = numero de processos)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Montagem de dados
    double pi=0;
    if(rank==0){
    	pi = 3.14159265359;
    }
    std::cout << "Processo " << rank << " [antes] pi=" << std::setprecision(12) << pi << std::endl;

    // Envio de dados 
    int msg_tag=0;   
    if(rank== 0){

        //Processo master (rank=0) envia para demais processos
        for(int i=1;i<np;i++){
            MPI_Send(&pi,1,MPI_DOUBLE,i,msg_tag,MPI_COMM_WORLD);
        }

    }else{

        //Processos slave recebem do processo master 
        MPI_Recv(&pi,1,MPI_DOUBLE,0,msg_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

	std::cout << "Processo " << rank << " [depois] pi=" << std::setprecision(12) << pi << std::endl;

    // Finaliza o ambiente do MPI
    MPI_Finalize();
}