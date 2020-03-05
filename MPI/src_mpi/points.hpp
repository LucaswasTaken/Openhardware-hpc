#ifndef POINTS_HPP
#define POINTS_HPP

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <string>

//###############################################################################################################
// Cria numero aleatorio
double random_number(){
    return (double)rand() / (double)RAND_MAX ;
}
float random_number_float(){
    return (float)rand() / (float)RAND_MAX ;
}


// Cria valores em um vetor
void create_data(double *x, int npoints){

    for(int i=0;i<npoints;i++){
        x[i]=random_number();
    }
}
void create_data(float *x, int npoints){

    for(int i=0;i<npoints;i++){
        x[i]=random_number_float();
    }
}


// Cria pontos no eixo x (2D)
void create_points_on_xaxis(double *x, double * y, int npoints){

    for(int i=0;i<npoints;i++){
        x[i] = i;
        y[i] = 0;
    }
}
void create_points_on_xaxis(float *x, float * y, int npoints){

    for(int i=0;i<npoints;i++){
        x[i] = i;
        y[i] = 0;
    }
}


// Cria pontos aleatorios em 2D
void create_random_points(double *x, double * y, int npoints, bool pseudo){

    if(!pseudo){
        srand(time(NULL));    
    }    
    for(int i=0;i<npoints;i++){
        x[i] = random_number();
        y[i] = random_number();
    }
}

void create_random_points(float *x, float * y, int npoints, bool pseudo){

    if(!pseudo){
        srand(time(NULL));    
    }    
    for(int i=0;i<npoints;i++){
        x[i] = random_number_float();
        y[i] = random_number_float();
    }
}



//###############################################################################################################
double sum_values(double *x, int npoints){
    double sum =0;
    for(int i=0;i<npoints;i++){
        sum = sum + x[i];
    }
    return sum;
}

//###############################################################################################################
// Calcula a menor distancia para cada ponto (em um dado intervalo para os vetores x,y)
void compute_min_distances(double *x, double * y, int npoints, int start, int num_elems_proc, double *distances){


    for(int i=start; i<start+ num_elems_proc; i++){

        double xi = x[i];
        double yi = y[i];
        double min_distance;
        int cont = 0;

        for(int j=0;j<npoints;j++){
            if(i!=j){

                double xj = x[j];
                double yj = y[j];

                double dist = sqrt( (xj-xi)*(xj-xi) + (yj-yi)*(yj-yi) );

                if(cont==0){
                    min_distance = dist;
                }
                if(dist< min_distance){
                    min_distance = dist;
                }

                cont++;
            }
        }

        distances[i-start] = min_distance;
    }

}

void compute_min_distances(float *x, float * y, int npoints, int start, int num_elems_proc, float *distances){


    for(int i=start; i<start+ num_elems_proc; i++){

        float xi = x[i];
        float yi = y[i];
        float min_distance;
        int cont = 0;

        for(int j=0;j<npoints;j++){
            if(i!=j){

                float xj = x[j];
                float yj = y[j];

                float dist = sqrt( (xj-xi)*(xj-xi) + (yj-yi)*(yj-yi) );

                if(cont==0){
                    min_distance = dist;
                }
                if(dist< min_distance){
                    min_distance = dist;
                }

                cont++;
            }
        }

        distances[i-start] = min_distance;
    }

}

//###############################################################################################################
// Imprime vetor
void print_array(std::string prefix, double *vec, int npoints){

    std::cout << prefix;
    std::cout << "[" << std::endl;
    for(int i=0;i<npoints;i++){
        std::cout << std::fixed << std::setprecision(5) << vec[i];
        if(i!=npoints-1){ std::cout << ", ";}
        if((i+1)%10==0){std::cout << std::endl;}
    }
    std::cout << "]";
    std::cout << std::endl;
}
void print_array(std::string prefix, float *vec, int npoints){

    std::cout << prefix;
    std::cout << "[" << std::endl;
    for(int i=0;i<npoints;i++){
        std::cout << std::fixed << std::setprecision(5) << vec[i];
        if(i!=npoints-1){ std::cout << ", ";}
        if((i+1)%10==0){std::cout << std::endl;}
    }
    std::cout << "]";
    std::cout << std::endl;
}


#endif