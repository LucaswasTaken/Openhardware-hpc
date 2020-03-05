#ifndef ARGVPARSER_HPP
#define ARGVPARSER_HPP

#include <cstdlib>
#include <string>

class ArgvParser{

public:
	ArgvParser(int argc, char** argv){
		Defaults();
		for(int i=0;i<argc;i++){
			std::string arg = argv[i];
			if(arg=="--gpu"){ gpu=1;}
			if(arg=="--cpu"){ gpu=0;}
			if(arg=="--quiet"){ verbose=false;}
			if(arg=="--source"){ show_source=true;}
			if (arg == "--first_platform") { first_platform = true; }
			if (arg == "--second_platform") { first_platform = false; }
			if(arg=="--help"){ 
				std::cout << "--help" << std::endl;
				std::cout << "--gpu or --cpu" << std::endl;
				std::cout << "--quiet" << std::endl;
				std::cout << "--source" << std::endl;
				exit(EXIT_SUCCESS);
			}
		}
	}

	void Defaults(){
		gpu=0;
		verbose= true;
		show_source= false;
		first_platform = true;
	}

public:
	int gpu;
	bool verbose;
	bool show_source;
	bool first_platform;
};


#endif