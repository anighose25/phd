#include "Kernel.h"
void Kernel::print_kernel_info() {
    std::cout << "Kernel Name: " << name << "\n";
    std::cout << "Kernel Source file: " << src << "\n";
    std::cout << "workDimension: " << workDimension << "\n";
    std::cout << "globalWorksize: ";
    for (auto g : globalWorkSize)
        std::cout << g << " ";
    std::cout << "\n";
    std::cout << "Input Buffer Information\n";
    for (auto buf : inputBuffers)
        buf->print_buffer_info();
    std::cout << "Output Buffer Information\n";
    for (auto buf : outputBuffers)
        buf->print_buffer_info();
    std::cout << "Input/Output Buffer Information\n";
    for (auto buf : ioBuffers)
        buf->print_buffer_info();
    std::cout << "Kernel Argument Information\n";
    for (auto arg : varArguments)
        arg->print_arg_info();
    std::cout << "Required Buffer Dependency information \n";
    for (auto e : edge_info) {
        e->print_edge();
    }
}

void Kernel::print_kernel_info_with_buffer_linkage(Vendor p) {
    std::cout << "Kernel Name: " << name << "\n";
    std::cout << "Task id: " << task_id << "\n";
    std::cout << "Kernel Source file: " << src << "\n";
    std::cout << "workDimension: " << workDimension << "\n";
    std::cout << "globalWorksize: ";
    for (auto g : globalWorkSize)
        std::cout << g << " ";
    std::cout << "\n";
    std::cout << "Input Buffer Information\n";
    for (auto buf : inputBuffers)
        buf->print_buffer_linkage_info(p);
    std::cout << "Output Buffer Information\n";
    for (auto buf : outputBuffers)
        buf->print_buffer_linkage_info(p);
    std::cout << "Input/Output Buffer Information\n";
    for (auto buf : ioBuffers)
        buf->print_buffer_linkage_info(p);
    std::cout << "Kernel Argument Information\n";
    for (auto arg : varArguments)
        arg->print_arg_info();
    std::cout << "Required Buffer Dependency information \n";
    for (auto e : edge_info) {
        e->print_edge();
    }
}

Buffer *Kernel::process_buffer(json &buf, Buffer::BufferType buf_type) {

    bool write_buf = false;
    bool read_buf = false;
    std::string read_file_name = "";
    switch (buf_type) {
    case Buffer::Input:
        if (buf.contains("file_name")) {
            std::cout << "FILE READ FROM KJSF  " << buf["file_name"]
                      << std::endl;
            read_file_name = buf["file_name"];
        }
        write_buf = true;
        break;
    case Buffer::Output:
        read_buf = true;
        break;
    case Buffer::IO:
        write_buf = true;
        read_buf = true;
        break;
    default:
        write_buf = true;
        read_buf = true;
        break;
    }
    // std::cout << "Getting position\n";

    int position = buf["pos"].get<int>();
    // std::cout << "Getting size\n";
    std::string size = buf["size"];
    //  std::cout << "Getting buffer datatype\n";
    std::string json_datatype = buf["type"];
    DataType datatype;
    if (json_datatype == "float")
        datatype = Float;
    else if (json_datatype == "int")
        datatype = Int;
    Buffer *new_buffer =
        new Buffer(position, datatype, size, write_buf, read_buf,
                   allocate_host_array, read_file_name);

    if (buf.contains("persistence")) {
        int persistence = buf["persistence"].get<int>();
        if (persistence == 1)
            new_buffer->persistent = true;
    }
    //    Buffer *new_buffer = new Buffer(position, datatype, size, write_buf,
    //                                  read_buf, allocate_host_array);
    return new_buffer;
}

void Kernel::update_edgeinfo(EdgeInfo *e) { edge_info.push_back(e); }
KernelArgument *Kernel::process_argument(json &arg) {

    int position = arg["pos"].get<int>();
    std::string value = arg["value"];
    std::string json_datatype = arg["type"];
    DataType datatype;
    if (json_datatype == "float")
        datatype = Float;
    else if (json_datatype == "int")
        datatype = Int;
    KernelArgument *new_argument =
        new KernelArgument(position, datatype, value);
    return new_argument;
}

LocalMemory *Kernel::process_local_memory_argument(json &arg) {
    int position = arg["pos"].get<int>();
    std::string size = arg["size"];
    std::string json_datatype = arg["type"];
    DataType datatype;
    if (json_datatype == "float")
        datatype = Float;
    else if (json_datatype == "int")
        datatype = Int;
    LocalMemory *new_argument = new LocalMemory(position, datatype, size);
    return new_argument;
}
void Kernel::obtain_cpu_profile(json &arg) {

    long long int start =
        arg["cpu_profile"]["ndrange"]["device_start"].get<long long int>();
    long long int end =
        arg["cpu_profile"]["ndrange"]["device_end"].get<long long int>();
    this->cpu_time = end - start;
}

void Kernel::obtain_gpu_profile(json &arg) {
    long long int nd_start =
        arg["gpu_profile"]["ndrange"]["device_start"].get<long long int>();
    long long int nd_end =
        arg["gpu_profile"]["ndrange"]["device_end"].get<long long int>();
    long long int read_start =
        arg["gpu_profile"]["read"]["device_start"].get<long long int>();
    long long int read_end =
        arg["gpu_profile"]["read"]["device_end"].get<long long int>();

    long long int write_start =
        arg["gpu_profile"]["write"]["device_start"].get<long long int>();
    long long int write_end =
        arg["gpu_profile"]["write"]["device_end"].get<long long int>();

    this->gpu_time = nd_end - nd_start;
    this->h2d_time = write_end - write_start;
    this->d2h_time = read_end - read_start;
}

cl_program Kernel::cl_compile_program(std::string source_file,
                                      cl_context &ctx) {

    cl_int status;

    std::string cl_file_name = database_dir + "kernels/" + source_file;
    LOG("Compiling %s\n", cl_file_name.c_str());
    std::ifstream f(cl_file_name.c_str());
    std::stringstream sbuffer;
    sbuffer << f.rdbuf();
    std::string kernel_file_src = sbuffer.str();
    const char *program_src = kernel_file_src.c_str();
    cl_program program = clCreateProgramWithSource(
        ctx, 1, (const char **)&program_src, NULL, &status);
    check(status, "Creating Program With Source");
    return program;
}

void Kernel::initialize_buffer_flags(Vendor p) {

    for (auto buf : inputBuffers) {
        buf->initialize_buffer_flags(p);
    }
    for (auto buf : outputBuffers) {

        buf->initialize_buffer_flags(p);
    }
    for (auto buf : ioBuffers) {
        buf->initialize_buffer_flags(p);
    }
}

void Kernel::setup_buffer_flags(Vendor p) {

    for (auto buf : inputBuffers) {
        buf->setup_buffer_flags(p, true, true, false, false);
    }
    for (auto buf : outputBuffers) {

        buf->setup_buffer_flags(p, true, false, true, false);
    }
    for (auto buf : ioBuffers) {
        buf->setup_buffer_flags(p, true, true, true, false);
    }
}

void Kernel::setup_buffers(cl_context &ctx, Vendor ven,
                           GpuBufferManager *GpuBufMan) {

    host_events.record_start("create_buffer");
    for (auto buf : inputBuffers) {

        buf->create_buffer(ven, ctx, GpuBufMan);
        LOG("\tCreated Input Buffer\n");
    }
    for (auto buf : outputBuffers) {
        buf->create_buffer(ven, ctx, GpuBufMan);
        LOG("\tCreated Output Buffer\n");
    }
    for (auto buf : ioBuffers) {
        buf->create_buffer(ven, ctx, GpuBufMan);
        LOG("\tCreated Input/Output Buffer\n");
    }

    host_events.record_end("create_buffer");
}

void Kernel::setup_linkage(Vendor ven) {

    for (auto buf : inputBuffers) {

        buf->link_buffer(ven);
        LOG("\tCreated Input Buffer\n");
    }
    for (auto buf : outputBuffers) {
        buf->link_buffer(ven);
        LOG("\tCreated Output Buffer\n");
    }
    for (auto buf : ioBuffers) {
        buf->link_buffer(ven);
        LOG("\tCreated Input/Output Buffer\n");
    }
}

void Kernel::reset_linkage(Vendor ven) {

    for (auto buf : inputBuffers) {

        buf->reset(ven);
        LOG("\tCreated Input Buffer\n");
    }
    for (auto buf : outputBuffers) {
        buf->reset(ven);
        LOG("\tCreated Output Buffer\n");
    }
    for (auto buf : ioBuffers) {
        buf->reset(ven);
        LOG("\tCreated Input/Output Buffer\n");
    }
}

void Kernel::build_kernel(
    std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
    std::unordered_map<Vendor, cl_context> &ctxs,
    std::vector<Vendor> &platforms) {

    LOG("build_kernel: BEGIN \n");

    cl_int status;

    for (auto p : platforms) {

        cl_device_id *dev = &(all_devices[p][0]);
        int num_dev = all_devices[p].size();
#if MULTIKERNEL
        std::string kernel_file;
        if (!source_names.empty())
            kernel_file = source_names[p];
        else
            kernel_file = src;
#else

        std::string kernel_file = src;
#endif
        LOG("Building %s program\n", name.c_str());

        if (cache.find(kernel_file.c_str()) == cache.end()) {
            //            std::cout << "NEW KERNEL: " << kernel_file<<std::endl;
            programs[p] = cl_compile_program(kernel_file.c_str(), ctxs[p]);

            LOG("Compiled Programs: %s\n", kernel_file.c_str());
            status =
                clBuildProgram(programs[p], num_dev, dev, NULL, NULL, NULL);
            check(status, "Building Program");
            LOG("\tBuilt %s program\n", kernel_file.c_str());

            std::map<Vendor, cl_program> x;
            x.emplace(p, programs[p]);
            cache.emplace(kernel_file.c_str(), x);
        } else if (cache[kernel_file.c_str()].find(p) ==
                   cache[kernel_file.c_str()].end()) {
            //          std::cout << "SAME KERNEL, NEW PLATFORM "<< kernel_file
            //          << std::endl;
            programs[p] = cl_compile_program(kernel_file.c_str(), ctxs[p]);

            LOG("Compiled Programs: %s\n", kernel_file.c_str());
            status =
                clBuildProgram(programs[p], num_dev, dev, NULL, NULL, NULL);
            check(status, "Building Program");
            LOG("\tBuilt %s program\n", name.c_str());

            cache[kernel_file.c_str()].emplace(p, programs[p]);
        } else {
            //        std::cout << "USING CACHE for " <<kernel_file<< std::endl;
            programs[p] = cache[kernel_file.c_str()][p];
        }

        // programs[p] = cl_compile_program(src.c_str(), ctxs[p]);
        // LOG("Compiled Programs: %s\n", src.c_str());
        // status = clBuildProgram(programs[p], num_dev, dev, NULL, NULL, NULL);
        // check(status, "Building Program");
        // LOG("\tBuilt %s program\n", name.c_str());

        kernel_objects[p] = clCreateKernel(programs[p], name.c_str(), &status);
        check(status, "\nCreating Kernel");
        LOG("\tBuilt %s kernel\n", name.c_str());
    }
    LOG("build_kernel: END \n");
}

void Kernel::cl_set_kernel_args(Vendor p) {

    LOG("cl_set_kernel_args: BEGIN\n");

    cl_int status;

    for (auto buffer : inputBuffers) {
        LOG("\tSetting Input Buffer Argument at position %d\n",
            buffer->get_pos());

        //        printf("\tSetting Input Buffer Argument at position %d\n",
        //             buffer->get_pos());

        status = clSetKernelArg(kernel_objects[p], buffer->get_pos(),
                                sizeof(cl_mem), &(buffer->get_buffer(p)));
        check(status, "Setting Input Buffer Kernel Argument");
    }

    for (auto buffer : outputBuffers) {
        LOG("\tSetting Output Buffer Argument at position %d",
            buffer->get_pos());
        status = clSetKernelArg(kernel_objects[p], buffer->get_pos(),
                                sizeof(cl_mem), &(buffer->get_buffer(p)));
        check(status, "Setting Output Buffer Kernel Argument\n");
    }

    for (auto buffer : ioBuffers) {
        LOG("\tSetting Input/Output Buffer Argument at position %d\n",
            buffer->get_pos());
        status = clSetKernelArg(kernel_objects[p], buffer->get_pos(),
                                sizeof(cl_mem), &(buffer->get_buffer(p)));
        check(status, "Setting Input Output Buffer Kernel Argument");
    }

    for (auto var : varArguments) {
        LOG("\tSetting Variable Argument at position %d\n", var->get_pos());
        status = clSetKernelArg(kernel_objects[p], var->get_pos(),
                                var->get_size(), var->get_var(var->get_type()));
        check(status, "Setting Variable Kernel Argument");
    }

    for (auto var : localArguments) {
        LOG("\tSetting Local Memory Argument at position %d\n", var->get_pos());
        status = clSetKernelArg(kernel_objects[p], var->get_pos(),
                                var->get_size(), NULL);
        check(status, "Setting Local Memory Argument");
    }

    LOG("cl_set_kernel_args: END\n");
}

void Kernel::setup(
    std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
    std::unordered_map<Vendor, cl_context> &ctxs,
    std::vector<Vendor> &platforms, GpuBufferManager *GpuBufMan) {
    build_kernel(all_devices, ctxs, platforms);
    for (auto p : platforms) {
        LOG("Creating Buffers\n");
        initialize_buffer_flags(p);
        setup_buffer_flags(p);
        setup_buffers(ctxs[p], p, GpuBufMan);
        LOG("Setting Kernel Arguments\n");
        cl_set_kernel_args(p);
    }
}

void Kernel::setup(Vendor p, cl_context &ctx, GpuBufferManager *GpuBufMan) {
    if(profile_engine) FunctionTimer("Setup->CreateKernel");
    initialize_buffer_flags(p);
    setup_buffer_flags(p);
    setup_buffers(ctx, p, GpuBufMan);
    cl_set_kernel_args(p);
}

void Kernel::copy_persistent_buffers(
    std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
    std::unordered_map<Vendor, cl_context> &ctxs,
    std::vector<Vendor> &platforms,
    std::unordered_map<Vendor, std::vector<cl_command_queue>> &cmd_qs) {
    //printf("Copying for kernel %d\n", this->id);
    for (auto p : platforms) {

        for (auto buf : inputBuffers) {
            if (buf->persistent) {
      //          std::cout << "Writing buffer at position "
        //                  << buf->get_pos_string() << "\n";
                buf->write_buffer_synchronous(p, cmd_qs[p][0]);
            }
            LOG("\tCopying Persistent Input Buffer\n");
        }
        for (auto buf : outputBuffers) {
            if (buf->persistent) {
          //      std::cout << "Writing buffer at position "
            //              << buf->get_pos_string() << "\n";
                buf->write_buffer_synchronous(p, cmd_qs[p][0]);
            }

            LOG("\tCopying Persistent Output Buffer\n");
        }
        for (auto buf : ioBuffers) {
            if (buf->persistent) {
              //  std::cout << "Writing buffer at position "
                //          << buf->get_pos_string() << "\n";
                buf->write_buffer_synchronous(p, cmd_qs[p][0]);
            }

            LOG("\tCopying Persistent Input/Output Buffer\n");
        }
    }
}

void Kernel::launch_kernel(Vendor p, cl_command_queue &cmd_q) {
    cl_int status;
    LOG("Enqueueing ndrange command\n");
    if (local_worksize)
        status = clEnqueueNDRangeKernel(cmd_q, kernel_objects[p], workDimension,
                                        0, globalWorkSize, localWorkSize, 0,
                                        NULL, NULL);
    else
        status = clEnqueueNDRangeKernel(cmd_q, kernel_objects[p], workDimension,
                                        0, globalWorkSize, NULL, 0, NULL, NULL);
    check(status, "Launching Kernel");
}

void Kernel::initiate_dispatch(Vendor platform, int device_id) {

    this->in_frontier = false;
    this->has_dispatched = true;
    this->platform = platform;
    this->device_id = device_id;
}

cl_event Kernel::launch_kernel(Vendor p, cl_command_queue &cmd_q,
                               cl_event &dep) {

    cl_event ev;
    LOG("Enqueueing ndrange command\n");
    cl_int status;
#if MULTIKERNEL
    if (globalWorkSizes.empty())
        globalWorkSizes.emplace(p, globalWorkSize);

    if (local_worksize) {
        if (dep != NULL) {
            status = clEnqueueNDRangeKernel(
                cmd_q, kernel_objects[p], workDimension, 0, globalWorkSizes[p],
                localWorkSize, 1, &dep, &ev);

#if ADAS
            std::string name = "ndrange_" + this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "ndrange";
#endif
            command_events.associate(name, ev);

        } else {

            status = clEnqueueNDRangeKernel(
                cmd_q, kernel_objects[p], workDimension, 0, globalWorkSizes[p],
                localWorkSize, 0, NULL, &ev);
            // std::string name = "ndrange_" + std::string{get_device_type(p)};

#if ADAS
            std::string name = "ndrange_" + this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "ndrange";
#endif

            command_events.associate(name, ev);
        }
    } else {
        if (dep != NULL) {
            status = clEnqueueNDRangeKernel(
                cmd_q, kernel_objects[p], workDimension, 0, globalWorkSizes[p],
                NULL, 1, &dep, &ev);
            // std::string name = "ndrange_" + std::string{get_device_type(p)};
#if ADAS
            std::string name = "ndrange_" + this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "ndrange";
#endif
            command_events.associate(name, ev);

        } else {
            status = clEnqueueNDRangeKernel(
                cmd_q, kernel_objects[p], workDimension, 0, globalWorkSizes[p],
                NULL, 0, NULL, &ev);
            // std::string name = "ndrange_" + std::string{get_device_type(p)};
#if ADAS
            std::string name = "ndrange_" + this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "ndrange";
#endif
            command_events.associate(name, ev);
        }
    }
#else
    if (local_worksize) {
        if (dep != NULL) {
            status = clEnqueueNDRangeKernel(cmd_q, kernel_objects[p],
                                            workDimension, 0, globalWorkSize,
                                            localWorkSize, 1, &dep, &ev);
            std::string name = "ndrange";
            command_events.associate(name, ev);

        } else {

            status = clEnqueueNDRangeKernel(cmd_q, kernel_objects[p],
                                            workDimension, 0, globalWorkSize,
                                            localWorkSize, 0, NULL, &ev);
            std::string name = "ndrange";
            command_events.associate(name, ev);
        }
    } else {
        if (dep != NULL) {
            status =
                clEnqueueNDRangeKernel(cmd_q, kernel_objects[p], workDimension,
                                       0, globalWorkSize, NULL, 1, &dep, &ev);
            std::string name = "ndrange";
            command_events.associate(name, ev);

        } else {
            status =
                clEnqueueNDRangeKernel(cmd_q, kernel_objects[p], workDimension,
                                       0, globalWorkSize, NULL, 0, NULL, &ev);
            std::string name = "ndrange";
            command_events.associate(name, ev);
        }
    }

#endif
    check(status, "Launching Kernel");
    return ev;
}
cl_event Kernel::h2d_copy(Vendor p, cl_command_queue &cmd_q) {
    std::vector<cl_event> events;
    host_events.record_start("write");
    for (auto buf : inputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent) {
            buf->write_buffer(p, cmd_q);
#if ADAS
            std::string name = "write-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "write-" + buf->get_pos_string();
#endif

            command_events.associate(name, events.back());
        }
    }

    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent) {
            buf->write_buffer(p, cmd_q);
#if ADAS
            std::string name = "write-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "write-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    host_events.record_end("write");
    return NULL;
}


cl_event Kernel::h2d_copy(Vendor p, cl_command_queue &cmd_q, cl_event &dep) {
    std::vector<cl_event> events;
    events.push_back(dep);
    host_events.record_start("write");
    for (auto buf : inputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent) {
            events.push_back(buf->write_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "write-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "write-" + buf->get_pos_string();
#endif

            command_events.associate(name, events.back());
        }
    }

    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent) {
            events.push_back(buf->write_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "write-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "write-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    host_events.record_end("write");
    return events.back();
}

cl_event Kernel::d2h_copy(Vendor p, cl_command_queue &cmd_q, cl_event &dep) {
    std::vector<cl_event> events;
    events.push_back(dep);

    for (auto buf : outputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->read && !buf->persistent) {
            events.push_back(buf->read_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "read-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "read-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->read && !buf->persistent) {
            events.push_back(buf->read_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "read-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "read-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    if (events.size() > 1)
        return events.back();
    else
        return nullptr;
}

void Kernel::dispatch_synchronous(Vendor p, cl_command_queue &cmd_q) {
    for (auto buf : inputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent)
            buf->write_buffer(p, cmd_q);
    }
    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent)
            buf->write_buffer(p, cmd_q);
    }
    launch_kernel(p, cmd_q);
    for (auto buf : outputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->read && !buf->persistent)
            buf->read_buffer(p, cmd_q);
    }

    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->read && !buf->persistent)
            buf->read_buffer(p, cmd_q);
    }
    cl_int status;
    status = clFinish(cmd_q);
    check(status, "Trying to finish command queue\n");
}

cl_event Kernel::dispatch_with_events_callback(Vendor p, cl_context &ctx,
                                               cl_command_queue &cmd_q) {
    if(profile_engine) FunctionTimer("Dispatch->Kernel");
    std::vector<cl_event> events;
    cl_event ev = clCreateUserEvent(ctx, NULL);
    events.push_back(ev);

    host_events.record_start("write");
    for (auto buf : inputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent) {
            events.push_back(buf->write_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "write-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "write-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->write && !buf->persistent) {
            events.push_back(buf->write_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "write-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "write-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    host_events.record_end("write");
    events.push_back(launch_kernel(p, cmd_q, events.back()));

    for (auto buf : outputBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->read && !buf->persistent) {
            events.push_back(buf->read_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "read-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "read-" + buf->get_pos_string();
#endif

            command_events.associate(name, events.back());
        }
    }

    for (auto buf : ioBuffers) {
        if (buf->vendor_to_bufferflag_map[p]->read && !buf->persistent) {
            events.push_back(buf->read_buffer(p, cmd_q, events.back()));
#if ADAS
            std::string name = "read-" + buf->get_pos_string() + "_" +
                               this->get_dag_id() + "_" +
                               this->get_instance_id() + "_" + this->get_id() +
                               "_" + get_device_type(p);
#else
            std::string name = "read-" + buf->get_pos_string();
#endif
            command_events.associate(name, events.back());
        }
    }

    clSetUserEventStatus(events.front(), CL_COMPLETE);
    return events.back();
}

void Kernel::print_input() {
    for (auto buf : inputBuffers)
        if (buf->allocate_host_array) {
            std::cout << "Input Buffer at position " << buf->get_pos() << "\n";
            buf->print_host();
        }

    for (auto buf : ioBuffers)
        if (buf->allocate_host_array) {
            std::cout << "Input/Output Buffer at position " << buf->get_pos()
                      << "\n";
            buf->print_host();
        }
}

void Kernel::print_result() {
    for (auto buf : ioBuffers)

        if (buf->allocate_host_array) {
            std::cout << "Input/Output Buffer at position " << buf->get_pos()
                      << "\n";
            buf->print_host();
        }

    for (auto buf : outputBuffers)
        if (buf->allocate_host_array) {
            std::cout << "Output Buffer at position " << buf->get_pos() << "\n";
            buf->print_host();
        }
    std::cout << "print_result finished\n";
}

void Kernel::print_static_device_profiles() {
    std::cout << "CPU TIME: " << cpu_time << "\n";
    std::cout << "GPU TIME: " << gpu_time << "\n";
    std::cout << "H2D TIME: " << h2d_time << "\n";
    std::cout << "D2H TIME: " << d2h_time << "\n";
}

void Kernel::create_buffer_map() {
    for (auto buf : inputBuffers)
        pos_to_buffer_map[buf->get_pos()] = buf;
    for (auto buf : outputBuffers)
        pos_to_buffer_map[buf->get_pos()] = buf;
    for (auto buf : ioBuffers)
        pos_to_buffer_map[buf->get_pos()] = buf;
}

Buffer *Kernel::get_buffer_from_argument_position(int position) {

    return pos_to_buffer_map[position];
}

bool Kernel::is_ready_for_dispatch() {
    return num_parents == finished_parents.load();
}
const char *Kernel::get_name() { return name.c_str(); }

void Kernel::set_preferred()
{
    if(this->cpu_time < this->gpu_time+this->h2d_time+this->d2h_time)
        set_preferred(Vendor::ARM_CPU);
    else
        set_preferred(Vendor::ARM_GPU);
}

void Kernel::set_preferred(Vendor p) { this->preferred = p; }

void Kernel::set_preferred(int p) {
    if (p < 5)
        set_preferred(Vendor::ARM_CPU);
    else

        set_preferred(Vendor::ARM_GPU);
}

std::string Kernel::get_preferred_type() {
    switch (preferred) {
    case Vendor::ARM_CPU:
        return "cpu";
    case Vendor::ARM_GPU:
        return "gpu";
    }
}

void Kernel::print_profiling_times() {
    std::cout << id << " : " << name << "\n";
    command_events.print();
}

void Kernel::reset_kernel_parameters() {
    this->is_finished = false;
    this->has_dispatched = false;
    this->finished_parents = 0;
    this->task_id = -1;
    this->task_ptr = NULL;
    // this->instance_id++;
}

void Kernel::reset_buffer_parameters() {
    setup_buffer_flags(Vendor::ARM_CPU);
    setup_buffer_flags(Vendor::ARM_GPU);
    for (auto x : pos_to_buffer_map) {
        Buffer *buf = x.second;
        buf->link = nullptr;
    }
}

#if MULTIKERNEL

void Kernel::set_multiple_implementations(std::string &info_cpu,
                                          std::string &info_gpu) {
    std::ifstream kernel_info_file_cpu(info_cpu);
    std::ifstream kernel_info_file_gpu(info_gpu);

    json json_cpu, json_gpu;
    kernel_info_file_cpu >> json_cpu;
    kernel_info_file_gpu >> json_gpu;

    source_names[Vendor::ARM_CPU] = json_cpu["src"];
    source_names[Vendor::ARM_GPU] = json_gpu["src"];

    auto gws_cpu = new size_t[3];
    auto gws_gpu = new size_t[3];

    globalWorkSizes.emplace(Vendor::ARM_CPU, gws_cpu);
    globalWorkSizes.emplace(Vendor::ARM_GPU, gws_gpu);

    if (json_cpu.contains("globalWorkSize")) {
        std::string gws = json_cpu["globalWorkSize"];
        gws = gws.substr(1, gws.size() - 2);
        splitstring liststring((char *)gws.c_str());
        std::vector<std::string> ssublist = liststring.split(',');
        int it = 0;
        for (auto s : ssublist)
            globalWorkSizes[Vendor::ARM_CPU][it++] = std::stoi(s);
    }

    if (json_gpu.contains("globalWorkSize")) {
        std::string gws = json_gpu["globalWorkSize"];
        gws = gws.substr(1, gws.size() - 2);
        splitstring liststring((char *)gws.c_str());
        std::vector<std::string> ssublist = liststring.split(',');
        int it = 0;
        for (auto s : ssublist)
            globalWorkSizes[Vendor::ARM_GPU][it++] = std::stoi(s);
    }
}
#endif
