#include "KernelArgument.h"
void *KernelArgument::get_var(DataType datatype) {

    switch (datatype) {
    case DataType::Char:
        return (void *)variable_char;
        break;
    case DataType::Int:
        return (void *)variable_int;
        break;
    case DataType::Float:
        return (void *)variable_float;
        break;
    case DataType::Double:
        return (void *)variable_double;
        break;
    default:
        return (void *)variable_int;
        break;
    }
}

void KernelArgument::print_arg_info() {
    std::cout << "pos: " << pos << " value: " << val << "\n";
}

int KernelArgument::get_pos() { return pos; }

DataType KernelArgument::get_type() { return type; }

size_t KernelArgument::get_size() {
    switch (type) {
    case DataType::Char:
        return sizeof(char);
        break;
    case DataType::Int:
        return sizeof(int);
        break;
    case DataType::Float:
        return sizeof(float);
        break;
    case DataType::Double:
        return sizeof(double);
        break;
    default:
        return sizeof(float);
        break;
    }
}

void LocalMemory::print_arg_info() {
    std::cout << "pos: " << pos << " num_elements: " << num_elements << "\n";
}

int LocalMemory::get_pos() { return pos; }

DataType LocalMemory::get_type() { return type; }

size_t LocalMemory::get_size() { return size; }
