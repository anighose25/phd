#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif
#ifndef KERNELARGUMENT_H
#define KERNELARGUMENT_H
class KernelArgument {
  public:
    KernelArgument(int position, DataType data_type, std::string value) {
        pos = position;
        type = data_type;
        val = value;
        switch (type) {
        case DataType::Char:
            *variable_char = value.c_str()[0];
            break;
        case DataType::Int:
            *variable_int = std::stoi(value);
            break;
        case DataType::Float:
            *variable_float = std::stof(value);
            break;
        case DataType::Double:
            *variable_double = std::stod(value);
            break;
        default:
            *variable_int = std::stoi(value);
            break;
        }
    }

    void *get_var(DataType datatype);
    void print_arg_info();
    int get_pos();
    size_t get_size();
    DataType get_type();
    ~KernelArgument() {
        delete variable_char;
        delete variable_int;
        delete variable_float;
        delete variable_double;
    }

    char *variable_char = new char;
    int *variable_int = new int;
    float *variable_float = new float;
    double *variable_double = new double;
    int pos;
    std::string val;
    DataType type;
};

class LocalMemory {
  public:
    LocalMemory(int position, DataType data_type, std::string size) {
        pos = position;
        type = data_type;
        num_elements = std::stoi(size);
        switch (type) {
        case DataType::Char:
            this->size = sizeof(char) * num_elements;
            break;
        case DataType::Int:
            this->size = sizeof(int) * num_elements;
            break;
        case DataType::Float:
            this->size = sizeof(float) * num_elements;
            break;
        case DataType::Double:
            this->size = sizeof(double) * num_elements;
            break;
        default:
            this->size = sizeof(float) * num_elements;
            break;
        }
    }

    void print_arg_info();
    int get_pos();
    size_t get_size();
    DataType get_type();

    int num_elements;
    int pos;
    size_t size;
    DataType type;
};
#endif
