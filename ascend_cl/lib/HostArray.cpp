#include "HostArray.h"

void *HostArray::get_data() { return data; }

void HostArray::print() {

    switch (datatype) {
    case DataType::Char: {
        char *cast_array = (char *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            std::cout << cast_array[i] << " ";
        std::cout << "\n";
        break;
    }
    case DataType::Int: {
        int *cast_array = (int *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            std::cout << cast_array[i] << " ";
        std::cout << "\n";
        break;
    }
    case DataType::Float: {
        float *cast_array = (float *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            std::cout << cast_array[i] << " ";
        std::cout << "\n";
        break;
    }
    case DataType::Double: {
        double *cast_array = (double *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            std::cout << cast_array[i] << " ";
        std::cout << "\n";
        break;
    }
    default: {
        float *cast_array = (float *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            std::cout << cast_array[i] << " ";
        std::cout << "\n";
        break;
    }
    }
}

void HostArray::randomize() {
    switch (datatype) {

    case DataType::Int: {
        int *cast_array = (int *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = (int)(rand() % 1000);
        break;
    }
    case DataType::Float: {
        float *cast_array = (float *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = (float)(rand() % 1000);
        break;
    }
    case DataType::Double: {
        double *cast_array = (double *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = (double)(rand() % 1000);
        break;
    }

    default: {
        float *cast_array = (float *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = (float)(rand() % 1000);
        break;
    }
    }
}

void HostArray::ones() {
    switch (datatype) {

    case DataType::Char: {
        char *cast_array = (char *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = '1';
        break;
    }
    case DataType::Int: {
        int *cast_array = (int *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = 1;
        break;
    }
    case DataType::Float: {
        float *cast_array = (float *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = 1.0;
        break;
    }
    case DataType::Double: {
        double *cast_array = (double *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = 1.0;
        break;
    }

    default: {
        float *cast_array = (float *)data;
        for (unsigned int i = 0; i < num_elements; i++)
            cast_array[i] = 1.0;
        break;
    }
    }
}

int HostArray::get_id() { return id; }
