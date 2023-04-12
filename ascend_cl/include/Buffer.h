#ifndef BUFFER_H
#define BUFFER_H
#include "HostArray.h"
#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif
#include "Events.h"
#include "GpuBufferManager.h"

static int buffer_identifier = 0;

struct BufferFlags {
    bool create = false;
    bool write = false;
    bool read = false;
    bool link = false;

    BufferFlags(bool create, bool write, bool read, bool link) {
        this->create = create;
        this->write = write;
        this->read = read;
        this->link = link;
    }

    void print_buffer_flag_info() {
        std::cout << "Create: " << create << " Write: " << write
                  << " Read: " << read << " Link: " << link << "\n";
    }
};

class Buffer {
  public:
    enum BufferType { Input, Output, IO };
    /*
        Buffer(int position, DataType type, std::string size_buf, bool
    write_buf, bool read_buf, bool allocate_host_array) { id =
    buffer_identifier++; pos = position; data_type = type; num_elements =
    std::stoi(size_buf); this->size_buf = size_buf; this->allocate_host_array =
    allocate_host_array; switch (type) { case DataType::Char: size =
    num_elements * sizeof(char); break; case DataType::Int: size = num_elements
    * sizeof(int); break; case DataType::Float: size = num_elements *
    sizeof(float); break; case DataType::Double: size = num_elements *
    sizeof(double); break; default: size = num_elements * sizeof(float); break;
            }
            write = write_buf;
            read = read_buf;
            if (allocate_host_array)
                host_array = new HostArray(type, size_buf);
    #if FILE_LOGGER
            fp = std::fopen("BufferLogs.txt", "w");
    #endif
        }
    */
    Buffer(int position, DataType type, std::string size_buf, bool write_buf,
           bool read_buf, bool allocate_host_array,
           std::string read_file_name = "") {
        id = buffer_identifier++;
        pos = position;
        data_type = type;
        num_elements = std::stoi(size_buf);

        this->file_name = read_file_name;
        this->size_buf = size_buf;
        this->allocate_host_array = allocate_host_array;
        switch (type) {
        case DataType::Char:
            size = num_elements * sizeof(char);
            break;
        case DataType::Int:
            size = num_elements * sizeof(int);
            break;
        case DataType::Float:
            size = num_elements * sizeof(float);
            break;
        case DataType::Double:
            size = num_elements * sizeof(double);
            break;
        default:
            size = num_elements * sizeof(float);
            break;
        }
        write = write_buf;
        read = read_buf;
        if (allocate_host_array) {
            if (read_file_name.size() == 0) {
                host_array = new HostArray(type, size_buf);
            } else {
                host_array = new HostArray(type, size_buf, read_file_name);
            }
        }
#if FILE_LOGGER
        fp = std::fopen("BufferLogs.txt", "w");
#endif
    }

    ~Buffer() {

        // printf("Releasing host array\n");
        if (allocate_host_array) {

            delete host_array;
            //    printf("Released host array\n");
        } else
            host_array = nullptr;

        if (destructor)
            for (auto x : data) {
                if (vendor_to_bufferflag_map[x.first]->create) {
                    //          printf("Releasing buffer object\n");
                    clReleaseMemObject(x.second);
                }
            }

#if FILE_LOGGER
        fclose(fp);
#endif
    }

    void print_buffer_info();

    void print_buffer_flags(Vendor p);

    void print_buffer_linkage_info(Vendor platform);

    void print_host();

    void allocate_host_array_for_buffer();

    void initialize_buffer_flags(Vendor p);

    void setup_buffer_flags(Vendor p, bool create, bool write, bool read,
                            bool link);

    void set_create(Vendor p, bool value);
    void set_write(Vendor p, bool value);
    void set_read(Vendor p, bool value);
    void set_link(Vendor p, bool value);

    cl_mem &get_buffer(Vendor p);

    void create_buffer(Vendor p, cl_context &ctx, GpuBufferManager *GpuBufMan);

    void write_buffer(Vendor p, cl_command_queue &cmd_q);

    void write_buffer_synchronous(Vendor p, cl_command_queue &cmd_q);

    void read_buffer(Vendor p, cl_command_queue &cmd_q);

    cl_event write_buffer(Vendor p, cl_command_queue &cmd_q, cl_event &dep);

    cl_event read_buffer(Vendor p, cl_command_queue &cmd_q, cl_event &dep);

    void associate_host_array_for_buffer(HostArray *h);

    int get_pos();

    int get_id();

    void link_buffer(Vendor p);

    void reset(Vendor p);

    std::string get_pos_string();

    int id;
    int pos;
    int num_elements;
    bool write;
    bool read;
    bool create = true;
    bool isolated = true;
    bool destructor = true;
    size_t size;
    DataType data_type;
    HostArray *host_array = nullptr;
    Buffer *link = nullptr;
    std::string size_buf;
    std::unordered_map<Vendor, cl_mem> data;
    cl_mem placeholder;
    std::unordered_map<Vendor, bool> buffer_created;
    std::unordered_map<Vendor, BufferFlags *> vendor_to_bufferflag_map;
    bool allocate_host_array;
    bool associate_host_array = false;
    std::string file_name;
    std::atomic<int> life_cycle{0};
    bool persistent = false;
#if FILE_LOGGER
    FILE *fp;
#endif
};
#endif
