#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <string>

#include "points.hpp"
#include "readfile.hpp"
#include "argv_parser.hpp"


//########################################################################################################
void error_check(int code, int expected, std::string message){

	if(code!=expected){
		std::cout << message << std::endl;
		exit(1);
	}
}

void error_check_null(void* err, std::string message){
	if(err==NULL){
		std::cout << message << std::endl;
		exit(1);		
	}
}

//########################################################################################################
int main(int argc, char** argv){

	// PREPARACAO DE AMBIENTE --------------------------------------
	ArgvParser cmdl(argc,argv);

	// Selecao de plataforma e dispositivo
	cl_platform_id platform_ids[2];
	cl_platform_id platform_id;
	cl_uint ret_num_platforms;
	cl_device_id device_id;
	int gpu = cmdl.gpu;
	cl_int ret = clGetPlatformIDs(2, &platform_ids[0], &ret_num_platforms);

	if (cmdl.first_platform) {
		platform_id = platform_ids[0];
	}else {
		platform_id = platform_ids[1];
	}
	int err;
	if (gpu) {
		err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	}else {
		err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	}
    error_check(err,CL_SUCCESS,"Error: Failed to create a device group!");

	//Nome do dispositivo
	char * device_name;
	size_t valueSize;
	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0,NULL,&valueSize);
	device_name = new char[valueSize];
	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, device_name, NULL);
	std::cout << device_name << std::endl;

    // Contexto de calculo
	cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);    
	error_check_null(context,"Error: Failed to create a compute context!");

	// Fila de comandos
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	error_check_null(commands,"Error: Failed to create a command queue!");

	// Cria programa para dado codigo fonte
	std::string code = read_file("./cl/min_distance.cl");
	const char * kernel_source = code.c_str();
	if(cmdl.show_source){std::cout << code << std::endl;}

	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
	error_check_null(program,"Error: Failed to create compute program!");

	// Compilacao do codigo fonte do programa e log de compilacao (caso falhe)
	int errb = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errb != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    error_check(errb,CL_SUCCESS,"Error: Failed to build program executable!");

    // Cria kernel para o programa compilado    
    cl_kernel kernel = clCreateKernel(program, "min_distance", &err);
	error_check_null(kernel,"Error: Failed to create compute kernel!");

	// CONTROLE DE DADOS -------------------------------------------
	unsigned int count = 128000;
	float *x = new float[count];
	float *y = new float[count];
	float *distances = new float[count];
	create_random_points(x,y,count,true);

    cl_mem x_device = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    cl_mem y_device = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
	
	error_check_null(x_device,"Error: Failed to allocate device memory!");
	error_check_null(y_device,"Error: Failed to allocate device memory!");
	error_check_null(output,"Error: Failed to allocate device memory!");

	// Escreve dados na memoria do dispositivo
    err = clEnqueueWriteBuffer(commands, x_device, CL_TRUE, 0, sizeof(float) * count, x, 0, NULL, NULL);
    error_check(err,CL_SUCCESS,"Error: Failed to write to source array!");
    err = clEnqueueWriteBuffer(commands, y_device, CL_TRUE, 0, sizeof(float) * count, y, 0, NULL, NULL);
    error_check(err,CL_SUCCESS,"Error: Failed to write to source array!");

    // Atribuicao dos argumentos para o kernel
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_device);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &y_device);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
    }
    error_check(err,CL_SUCCESS,"Error: Failed to set kernel arguments!");

    // EXECUCAO ----------------------------------------------------

    // Informacoes do grupo de trabalho
	size_t local;    
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    error_check(err,CL_SUCCESS,"Error: Failed to retrieve kernel work group info!");

    // Chamada do kernel no dispositivo
    size_t global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    error_check(err,CL_SUCCESS,"Error: Failed to execute kernel!");

	// Leitura de resultados	
	err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * count, distances, 0, NULL, NULL );  
	error_check(err,CL_SUCCESS,"Error: Failed to read output array!");

	if(cmdl.verbose){
		print_array("distances=",distances,count);	
	}	

	// FINALIZACAO DE AMBIENTE -------------------------------------
	delete [] x;
	delete [] y;
	delete [] distances;
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

}


