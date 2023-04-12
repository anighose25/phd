#ifndef HOSTARRAY_H
#define HOSTARRAY_H
#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif

#include "FreeTreeAllocator.h"

static int identifier = 0;
extern bool static_hostarray_allocation;
extern FreeTreeAllocator allocator;

class HostArray {

  public:
    HostArray(DataType data_type, std::string size_array) {
        datatype = data_type;
        num_elements = std::stoi(size_array);

        switch (datatype) {
        case DataType::Char:
            if (static_hostarray_allocation == false)
                data = new char[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(char),
                                          sizeof(char));
            break;
        case DataType::Int:
            if (static_hostarray_allocation == false)
                data = new int[num_elements];
            else
                data =
                    allocator.Allocate(num_elements * sizeof(int), sizeof(int));
            break;
        case DataType::Float:
            if (static_hostarray_allocation == false)
                data = new float[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(float),
                                          sizeof(float));
            break;
        case DataType::Double:
            if (static_hostarray_allocation == false)
                data = new double[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(double),
                                          sizeof(double));
            break;
        default:
            if (static_hostarray_allocation == false)
                data = new float[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(float),
                                          sizeof(float));
            break;
        }

        ones();
        id = identifier++;
    }

    HostArray(DataType data_type, std::string size_array, std::string file_name)
        : HostArray(data_type, size_array) {

        // auto Bin = std::fstream(file_name, std::ios::binary | std::ios::in);
        auto Bin = std::fstream(file_name, std::ios::in);
        if (!Bin.is_open()) {
            std::cout << "Error opening the file" << std::endl;
        }
//        std::cout << "READING FROM FILE" << std::endl;
        datatype = data_type;
        num_elements = std::stoi(size_array);

        switch (datatype) {
        case DataType::Char:
            if (static_hostarray_allocation == false)
                data = new char[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(char),
                                          sizeof(char));
            break;
        case DataType::Int:
            if (static_hostarray_allocation == false)
                data = new int[num_elements];
            else
                data =
                    allocator.Allocate(num_elements * sizeof(int), sizeof(int));
            break;
        case DataType::Float:
            if (static_hostarray_allocation == false)
                data = new float[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(float),
                                          sizeof(float));
            break;
        case DataType::Double:
            if (static_hostarray_allocation == false)
                data = new double[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(double),
                                          sizeof(double));
            break;
        default:
            if (static_hostarray_allocation == false)
                data = new float[num_elements];
            else
                data = allocator.Allocate(num_elements * sizeof(float),
                                          sizeof(float));
            break;
        }

        std::string tp;
        int i = 0;
        // float *A = (float *)data;
        while (getline(Bin, tp)) {
            ((float *)data)[i++] = stof(tp);
        }

        // Bin.read((char *)data, Size);
        // float *A = (float *)data;
        // for(int i=0;i<10;++i)
        // 	std::cout << A[i] << " ";
        // std::cout << std::endl;
    }

    void *get_data();

    void randomize();

    void ones();
    void print();

    int get_id();

    ~HostArray() {
        switch (datatype) {
        case DataType::Char:
            if (static_hostarray_allocation == false)
                delete[](char *) data;
            else
                allocator.Deallocate(data);
            break;
        case DataType::Int:
            if (static_hostarray_allocation == false)
                delete[](int *) data;
            else
                allocator.Deallocate(data);
            break;
        case DataType::Float:
            if (static_hostarray_allocation == false)
                delete[](float *) data;
            else
                allocator.Deallocate(data);
            break;
        case DataType::Double:
            if (static_hostarray_allocation == false)
                delete[](double *) data;
            else
                allocator.Deallocate(data);
            break;
        default:
            if (static_hostarray_allocation == false)
                delete[](float *) data;
            else
                allocator.Deallocate(data);
            break;
        }
    }

    void *data;
    int num_elements;
    DataType datatype;
    int id;
};

#endif
