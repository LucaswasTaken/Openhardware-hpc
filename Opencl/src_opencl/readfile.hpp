#ifndef READFILE_HPP
#define READFILE_HPP

#include <string>
#include <fstream>
#include <sstream>

std::string read_file(std::string filename){

    std::ifstream t(filename);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

#endif