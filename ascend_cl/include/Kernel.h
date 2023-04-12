#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif
#ifndef KERNEL_H
#define KERNEL_H

#include "Buffer.h"
#include "Events.h"
#include "GpuBufferManager.h"
#include "KernelArgument.h"
extern std::string database_dir;
static std::map<std::string, std::map<Vendor, cl_program>> cache;
class Kernel {
  public:
    Kernel(const char *kernel_info_file_name, bool allocate_host_array) {
        this->allocate_host_array = allocate_host_array;
        std::ifstream kernel_info_file(kernel_info_file_name);
        json json_object;
        kernel_info_file >> json_object;
        name = json_object["name"];
        src = json_object["src"];
        if (json_object.contains("partition")) {
            int partition = json_object["partition"].get<int>();
            this->set_preferred(partition);
        }
        workDimension = json_object["workDimension"].get<int>();
#if KERNEL_LOG
        std::cout << "Processing input buffers\n";
#endif
        if (json_object.contains("inputBuffers")) {
            for (json &o : json_object["inputBuffers"])
                inputBuffers.push_back(process_buffer(o, Buffer::Input));
        }
#if KERNEL_LOG
        std::cout << "Processing output buffers\n";
#endif
        if (json_object.contains("outputBuffers")) {
            for (json &o : json_object["outputBuffers"])
                outputBuffers.push_back(process_buffer(o, Buffer::Output));
        }

#if KERNEL_LOG
        std::cout << "Processing io buffersi\n";
#endif
        if (json_object.contains("ioBuffers")) {
            for (json &o : json_object["ioBuffers"])
                ioBuffers.push_back(process_buffer(o, Buffer::IO));
        }
#if KERNEL_LOG
        std::cout << "Processing variable arguments\n";
#endif
        if (json_object.contains("varArguments")) {
            for (json &o : json_object["varArguments"])
                varArguments.push_back(process_argument(o));
        }

        if (json_object.contains("localArguments")) {
            for (json &o : json_object["localArguments"])
                localArguments.push_back(process_local_memory_argument(o));
        }

#if KERNEL_LOG
        std::cout << "Processing global worksize arguments\n";
#endif
        if (json_object.contains("globalWorkSize")) {
            std::string gws = json_object["globalWorkSize"];
            gws = gws.substr(1, gws.size() - 2);
            splitstring liststring((char *)gws.c_str());
            std::vector<std::string> ssublist = liststring.split(',');
            int it = 0;
            for (auto s : ssublist)
                globalWorkSize[it++] = std::stoi(s);
        }

#if KERNEL_LOG
        std::cout << "Processing local worksize arguments\n";
#endif
        if (json_object.contains("localWorkSize")) {
            local_worksize = true;
            std::string lws = json_object["localWorkSize"];
            lws = lws.substr(1, lws.size() - 2);
            splitstring liststring((char *)lws.c_str());
            std::vector<std::string> ssublist = liststring.split(',');
            int it = 0;
            for (auto s : ssublist)
                localWorkSize[it++] = std::stoi(s);
        }
        if (json_object.contains("cpu_profile"))
            obtain_cpu_profile(json_object);
        if (json_object.contains("gpu_profile"))
            obtain_gpu_profile(json_object);
        if (json_object.contains("cpu_profile") &&
            json_object.contains("gpu_profile"))
            this->set_preferred();

#if KERNEL_LOG
        std::cout << "Completed kernel initialization\n";
#endif
#if FILE_LOGGER
        // fp = fopen("Kernel_Logs.txt", "w");
        std::string filename = name + ".txt";
        fp = fopen(filename.c_str(), "w");

#endif
    }

    Kernel(const char *kernel_info_file_name, bool allocate_host_array,
           int kernel_id, void *dag_ptr = nullptr)
        : Kernel(kernel_info_file_name, allocate_host_array) {
        id = kernel_id;
        this->create_buffer_map();
        this->dag_ptr = dag_ptr;
    }

    ~Kernel() {
#if DESTRUCTOR_LOG
        std::cout << "Destroying kernel with id " << id << ": " << name << "\n";
        std::cout << "Destroying input buffers\n";
#endif
        for (auto buf : inputBuffers)
            delete buf;

#if DESTRUCTOR_LOG
        std::cout << "Destroying output buffers\n";
#endif
        for (auto buf : outputBuffers)
            delete buf;

#if DESTRUCTOR_LOG
        std::cout << "Destroying input/output buffers\n";
#endif
        for (auto buf : ioBuffers)
            delete buf;

#if DESTRUCTOR_LOG
        std::cout << "Destroying variable arguments\n";
#endif
        for (auto arg : varArguments)
            delete arg;
#if FILE_LOGGER
        fclose(fp);
#endif
    }

    std::string get_id() { return std::to_string(id); }
    std::string get_dag_id() { return std::to_string(dag_id); }
    std::string get_instance_id() { return std::to_string(instance_id); }
    void print_kernel_info();

    void print_kernel_info_with_buffer_linkage(Vendor p);

    Buffer *process_buffer(json &buf, Buffer::BufferType buf_type);

    KernelArgument *process_argument(json &arg);

    LocalMemory *process_local_memory_argument(json &arg);

    void obtain_cpu_profile(json &arg);
    void obtain_gpu_profile(json &arg);

    cl_program cl_compile_program(std::string source_file, cl_context &ctx);

    void initialize_buffer_flags(Vendor p);
    void setup_buffer_flags(Vendor p);

    void setup_buffers(cl_context &ctx, Vendor ven,
                       GpuBufferManager *GpuBufMan);
    void setup_linkage(Vendor ven);
    void reset_linkage(Vendor ven);

    void build_kernel(
        std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
        std::unordered_map<Vendor, cl_context> &ctxs,
        std::vector<Vendor> &platforms);

    void cl_set_kernel_args(Vendor p);

    void
    setup(std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
          std::unordered_map<Vendor, cl_context> &ctxs,
          std::vector<Vendor> &platforms, GpuBufferManager *GpuBufMan);

    void setup(Vendor p, cl_context &ctx, GpuBufferManager *GpuBufMan);

    void copy_persistent_buffers(
        std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
        std::unordered_map<Vendor, cl_context> &ctxs,
        std::vector<Vendor> &platforms,
        std::unordered_map<Vendor, std::vector<cl_command_queue>> &cmd_qs);

    void initiate_dispatch(Vendor platform, int device_id);
    void launch_kernel(Vendor p, cl_command_queue &cmd_q);
    cl_event launch_kernel(Vendor p, cl_command_queue &cmd_q, cl_event &dep);
    void dispatch_synchronous(Vendor p, cl_command_queue &cmd_q);
    cl_event dispatch_with_events_callback(Vendor p, cl_context &ctx,
                                           cl_command_queue &cmd_q);

    cl_event h2d_copy(Vendor p, cl_command_queue &cmd_q, cl_event &dep);
    cl_event h2d_copy(Vendor p, cl_command_queue &cmd_q);
    cl_event d2h_copy(Vendor p, cl_command_queue &cmd_q, cl_event &dep);

    const char *get_name();
    void print_input();
    void print_result();
    void update_edgeinfo(EdgeInfo *);
    void create_buffer_map();
    bool is_ready_for_dispatch();
    Buffer *get_buffer_from_argument_position(int position);
    void set_preferred();
    void set_preferred(Vendor p);
    void set_preferred(int p);
    std::string get_preferred_type();
    void print_profiling_times();
    void print_static_device_profiles();
    void reset_kernel_parameters();
    void reset_buffer_parameters();
#if MULTIKERNEL
    void set_multiple_implementations(std::string &info_cpu,
                                      std::string &info_gpu);
#endif
    void *dag_ptr;
    void *task_ptr;
    int num_parents = 0;
    std::atomic<int> finished_parents{0};
    std::vector<Buffer *> inputBuffers;
    std::vector<Buffer *> outputBuffers;
    std::vector<Buffer *> ioBuffers;

    // Assigned variables at runtime
    Vendor platform;
    Vendor preferred;
    int device_id;
    bool is_finished = false;
    bool in_frontier = false;
    bool has_dispatched = false;
    int task_id = -1;
    std::vector<EdgeInfo *> edge_info;
    std::map<int, Buffer *> pos_to_buffer_map;
    int id;
    HostEvents host_events;
    OpenCLEvents command_events;
    long long int cpu_time;
    long long int gpu_time;
    long long int h2d_time;
    long long int d2h_time;
    float rank = 0;
    float wcet_rank = 0;
    int level=0;
    int max_num_ancestors=0;
    int required_predecessors=0;

    int chain_id = -1;
    int dag_id;
    int instance_id;
    bool multiple_implementations = false;
#if MULTIKERNEL
    std::unordered_map<Vendor, std::string> source_names;
    std::unordered_map<Vendor, size_t *> globalWorkSizes;
#endif

  private:
    std::string name;
    std::string src;
    bool allocate_host_array;
    int workDimension;
    size_t globalWorkSize[3] = {1, 1, 1};
    size_t localWorkSize[3] = {1, 1, 1};
    bool local_worksize = false;
    std::vector<KernelArgument *> varArguments;
    std::vector<LocalMemory *> localArguments;
    std::unordered_map<Vendor, cl_program> programs;
    std::unordered_map<Vendor, cl_kernel> kernel_objects;

#if FILE_LOGGER
    FILE *fp;
#endif
};
#endif
