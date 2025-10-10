#include <mpi.h>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {

    // Inicializa o ambiente do MPI
    MPI_Init(&argc, &argv);

    // Numero de processos
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Rank do processo (0 a np-1, onde np = numero de processos)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Montagem de dados
    double pi=0;
    if(rank==0){
    	pi = 3.14159265359;
    }
    std::cout << "Processo " << rank << " [antes] pi=" << std::setprecision(12) << pi << std::endl;

    // Envio de dados    
    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&pi,1,MPI_DOUBLE,0, MPI_COMM_WORLD);
	std::cout << "Processo " << rank << " [depois] pi=" << std::setprecision(12) << pi << std::endl;

    // Finaliza o ambiente do MPI
    MPI_Finalize();
}