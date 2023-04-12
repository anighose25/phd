#include "ScheduleEngine.h"
void Frontier::push(Kernel *new_value) {
    if (profile_engine)
        FunctionTimer("Frontier->PushGlobal");
    std::lock_guard<std::mutex> lk(mut);
    // std::cout << "PUSHING TO FRONTIER: " << new_value->id << "\n";
    q.push(new_value);
    cond.notify_all();
}
void Frontier::push_local(Kernel *new_value) {
    if (profile_engine)
        FunctionTimer("Frontier->PushLocal");
    std::lock_guard<std::mutex> lk(mut);
    // std::cout << "PUSHING TO LOCAL FRONTIER: " << new_value->id << "\n";
    q.push(new_value);
    cond.notify_all();
}

void Frontier::wait_and_peek(Kernel **value) {
    //    std::cout << "Entered Frontier wait and peek with
    //    "<<completed_jobs.load() <<" jobs completed and queue empty?
    //    "<<q.empty()<<"\n";
    *value = NULL;
    // std::cout << "FRONTIER: Entered wait_and_peek\n";
    std::unique_lock<std::mutex> lk(mut);
    if (completed_jobs.load() < num_jobs && !q.empty()) {
        //   std::cout << "Returning a kernel\n";
        *value = q.top();
        //  std::cout << "FRONTIER: Peeked " << (*value)->id << "\n";

    } else {
        *value = NULL;
        //    std::cout << "Peek function returning NULL\n";
    }

    // std::cout << "FRONTIER: Exited wait_and_peek\n";
}

void Frontier::wait_and_remove(Kernel *value) {
    // std::cout << "FRONTIER: Entered wait and remove for kernel " << value->id
    //              << "\n";
    if (profile_engine)
        FunctionTimer("Frontier->WaitAndRemove");
    std::unique_lock<std::mutex> lk(mut);
    // std::cout << "FRONTIER: Exited wait and remove for kernel " << value->id
    //         << "\n";

    q.remove(value);
}

void Frontier::wait_and_pop(Kernel **value) {
    // std::cout << "FRONTIER: Entered Frontier wait and pop\n";
    if (profile_engine)
        FunctionTimer("Frontier->WaitAndPopGlobal");
    *value = NULL;
    std::unique_lock<std::mutex> lk(mut);
    cond.wait(lk, [this] {
        return (!q.empty() || completed_jobs.load() == num_jobs);
    });
    if (completed_jobs.load() < num_jobs) {
        //   std::cout << "Returning a kernel\n";
        *value = q.top();
        //  std::cout << "FRONTIER: Returning kernel " << (*value)->id << "\n";
        q.pop();

    } else {
        *value = NULL;
        // std::cout << "FRONTIER: All jobs finished. Returning NULL\n";
    }
}
void Frontier::wait_and_pop_local(Kernel **value) {
    // std::cout << "LOCAL_FRONTIER: Entered wait and pop for local frontier\n";
    if (profile_engine)
        FunctionTimer("Frontier->WaitAndPopLocal");
    *value = NULL;
    std::unique_lock<std::mutex> lk(mut);
    cond.wait(lk, [this] { return (!q.empty()); });
    *value = q.top();
    // std::cout << "LOCAL_FRONTIER: Popping from local frontier" <<
    // (*value)->id
    //        << "\n";
    q.pop();
}

bool Frontier::empty() const {
    std::lock_guard<std::mutex> lk(mut);
    return q.empty();
}

void Frontier::notify() {
    // std::cout << "Frontier: notify_all()\n";
    cond.notify_all();
}

void ReadyQueue::push(Vendor p, int d) {
    // std::cout << "READY_QUEUE: Pushing" << p << " " << d << "\n";
    if (profile_engine)
        FunctionTimer("ReadyQueue->Push");
    std::lock_guard<std::mutex> lk(mut);
    q[p].push_back(d);
    // cond.notify_all();
}

void ReadyQueue::wait_and_pop(Vendor p, int *d) {

    if (profile_engine)
        FunctionTimer("ReadyQueue->WaitAndPop");
    // std::cout << "READY_QUEUE: Entered Ready queue wait and pop for " << p
    //        << "; is ready queue empty?: " << this->empty(p) << "\n";
    std::unique_lock<std::mutex> lk(mut);
    cond.wait(lk, [this, p] { return !q[p].empty(); });
    *d = q[p].front();
    q[p].pop_front();
    // std::cout << "READY_QUEUE: Exited Ready queue wait and pop for " << p
    //        << "\n";
}

void ReadyQueue::wait_and_remove(Vendor p, int device_id, int *d) {
    if (profile_engine)
        FunctionTimer("ReadyQueue->WaitAndRemove");
    // std::cout
    //  << "READY_QUEUE: Entered Ready queue wait and remove for removing " << p
    // << " " << device_id << "\n";
    std::unique_lock<std::mutex> lk(mut);
    cond.wait(lk, [this, p] { return !q[p].empty(); });
    for (int i = 0; i < q[p].size(); i++) {
        if (q[p][i] == device_id) {
            *d = q[p][i];
            q[p].erase(q[p].begin() + i);
        }
    }
    // std::cout << "READY_QUEUE: Got device id " << *d << "\n";
}

void ReadyQueue::notify() {
    // std::cout << "READY_QUEUE: notify_all\n";
    cond.notify_all();
}

bool ReadyQueue::device_available() const {
    std::lock_guard<std::mutex> lk(mut);
    bool available = false;
    for (auto p : q)
        available = available || !p.second.empty();
    return available;
}

bool ReadyQueue::empty(Vendor p) {
    std::lock_guard<std::mutex> lk(mut);
    return q[p].empty();
}

int ReadyQueue::get_available_devices(
    std::map<Vendor, std::deque<int>> &available_devices) {
    std::lock_guard<std::mutex> lk(mut);
    int num_available_devices = 0;
    for (auto x : q) {
        Vendor device_type = x.first;
        num_available_devices += x.second.size();
        for (auto device_id : x.second)
            available_devices[device_type].push_back(device_id);
    }
    return num_available_devices;
}
void TaskComponent::print_task_info() {
    // std::cout << "Constituent Kernels"
    //        << "\n";
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        std::cout << v << " :" << k->get_name() << "\n";
        k->print_kernel_info_with_buffer_linkage(this->platform);
    }
}

bool TaskComponent::is_finished() {
    return (num_kernels == completed_kernels.load());
}

void TaskComponent::add_free_kernel(Kernel *k) {
#if TASKDAG_LOG
    std::cout << "Adding kernel " << k->id
              << " to local frontier for task component" << task_id << "\n";
#endif
    local_frontier.push_local(k);
}

bool TaskComponent::has_free_kernels() { return !local_frontier.empty(); }

void TaskComponent::initialize_all_buffer_flags() {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        k->initialize_buffer_flags(this->platform);
        k->setup_buffer_flags(this->platform);
    }
}

void TaskComponent::initialize_free_kernels() {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        if (k->is_ready_for_dispatch())
            add_free_kernel(k);
    }
}

void TaskComponent::prepare_kernel(int kernel_id) {

    Kernel *source_kernel = dag_ptr->id_to_kernel_map[kernel_id];

// Preparing internal dependents
#if TASKDAG_LOG
    std::cout << "Preparing internal dependents of kernel " << kernel_id
              << "\n";
#endif
    for (auto s : dag_ptr->successors[kernel_id]) {
#if TASKDAG_LOG
        std::cout << " Preparing successor " << s << "\n";
#endif
        Kernel *destination_kernel = dag_ptr->id_to_kernel_map[s];
#if TASKDAG_LOG
        std::cout << " Obtained successor kernel pointer\n";
#endif
        if (destination_kernel->task_id == source_kernel->task_id) {
#if TASKDAG_LOG
            std::cout << "Kernel and successor are part of same component\n";
#endif

            for (auto e : destination_kernel->edge_info) {
                if (e->source_kernel == kernel_id) {

#if TASKDAG_LOG
                    std::cout << "Obtained source buffer\n";
#endif
                    Buffer *source_buffer =
                        source_kernel->get_buffer_from_argument_position(
                            e->source_buffer);
                    source_buffer->set_read(platform, false);

                    Buffer *destination_buffer =
                        destination_kernel->get_buffer_from_argument_position(
                            e->destination_buffer);
#if TASKDAG_LOG
                    std::cout << "Obtained destination buffer\n";
#endif
                    destination_buffer->set_create(platform, false);
                    destination_buffer->set_write(platform, false);
                    destination_buffer->set_link(platform, true);
                    destination_buffer->link = source_buffer;
                }
            }
        }
    }
// Preparing external dependents
#if TASKDAG_LOG
    std::cout << "Preparing external dependents of kernel " << kernel_id
              << "\n";
#endif
    for (auto s : dag_ptr->successors[kernel_id]) {
#if TASKDAG_LOG
        std::cout << " Preparing successor " << s << "\n";
#endif
        Kernel *destination_kernel = dag_ptr->id_to_kernel_map[s];
        if (destination_kernel->task_id != source_kernel->task_id) {

#if TASKDAG_LOG
            std::cout
                << "Kernel and successor are not part of the same component\n";
#endif
            for (auto e : destination_kernel->edge_info) {
                if (e->source_kernel == kernel_id) {
                    Buffer *source_buffer =
                        source_kernel->get_buffer_from_argument_position(
                            e->source_buffer);
#if TASKDAG_LOG
                    std::cout << "Obtained source buffer\n";
#endif
                    source_buffer->set_read(platform, true);
#if TASKDAG_LOG
                    std::cout << "Set buffer flags of source_buffer\n";
#endif
                }
            }
        }
    }
}
void TaskComponent::prepare_kernels() {
    for (auto v : kernel_ids)
        this->prepare_kernel(v);
}

void TaskComponent::setup_buffers(cl_context &ctx,
                                  GpuBufferManager *GpuBufMan) {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        k->setup_buffers(ctx, platform, GpuBufMan);
    }
}

void TaskComponent::setup_linkage(cl_context &ctx,
                                  GpuBufferManager *GpuBufMan) {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        k->setup_linkage(platform);
    }
}

void TaskComponent::set_kernel_args() {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
#if TASKDAG_LOG
        std::cout << "setting arguments of kernel " << k->id << " "
                  << k->get_name() << "\n";
#endif
        k->cl_set_kernel_args(platform);
    }
}

void TaskComponent::initiate_dispatch(Vendor p, int device_id) {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        k->in_frontier = false;
        k->has_dispatched = true;
        k->device_id = device_id;
        k->platform = p;
    }
    this->device_id = device_id;
    this->platform = p;
}

void TaskComponent::reset_tc_parameters() {
    for (auto v : kernel_ids) {
        Kernel *k = dag_ptr->id_to_kernel_map[v];
        k->reset_kernel_parameters();
        k->reset_linkage(platform);
        // k->reset_buffer_parameters();
    }
    this->id_to_event_map.clear();
}
void TaskComponent::dispatch_single(cl_context &ctx, cl_command_queue &cmd_q,
                                    ScheduleEngine *se) {
    if (profile_engine)
        FunctionTimer("Dispatch->TaskComponent");
    std::vector<cl_event> user_events;
#if TASKDAG_LOG
    std::cout << "Dispatch single called for" << task_id
              << " where queue empty? " << local_frontier.empty() << "\n";
#endif

    while (!local_frontier.empty()) {
        bool increment_flag = true;
        Kernel *k = nullptr;
#if TASKDAG_LOG
        std::cout << "Processing local frontier of task component " << task_id
                  << "\n";
#endif
        local_frontier.wait_and_pop_local(&k);
        if (k != nullptr) {
            // user_events.push_back(clCreateUserEvent(ctx, NULL));
            cl_event h2d_dep;
            h2d_dep = NULL;
            k->h2d_copy(platform, cmd_q, h2d_dep);
            cl_event exec_dep;
            //            if (id_to_event_map.find(k->id) !=
            //            id_to_event_map.end())
            //              exec_dep = id_to_event_map[k->id];
            //        else
            //          exec_dep = NULL;
            exec_dep = NULL;
            cl_event kernel_event = k->launch_kernel(platform, cmd_q, exec_dep);
            cl_event d2h_ev = k->d2h_copy(platform, cmd_q, kernel_event);
            if (d2h_ev != nullptr) {

#if TASKDAG_LOG
                std::cout << "Setting callback for kernel d2h event" << k->id
                          << " of task component " << task_id << "\n";
#endif
                UserArgs *user_data = new UserArgs(k, se);

                // std::cout << __FILE__ << " " << __LINE__
                //        << " Setting callback for d2h event of kernel "
                //      << k->id << " of task component " << this->task_id
                //    << "\n";
                clSetEventCallback(d2h_ev, CL_COMPLETE, &taskdag_callback,
                                   (void *)user_data);
//                clFlush(cmd_q);
                increment_flag = false;
            }

            /*else {
#if TASKDAG_LOG
                std::cout << "Incrementing completed kernels of task component "
                             "and dag for kernel "
                          << k->id << "\n";
#endif
                completed_kernels++;
                dag_ptr->completed_kernels++;
            }*/
            for (auto s : dag_ptr->successors[k->id]) {
#if TASKDAG_LOG
                std::cout << "dispatch_single: investigating successor " << s
                          << " of kernel " << k->id << "\n";
#endif
                Kernel *succ_kernel = dag_ptr->id_to_kernel_map[s];
                if (succ_kernel->task_id == k->task_id) {
#if TASKDAG_LOG
                    std::cout << "Successor belongs to same task component \n";
#endif
                    if (d2h_ev == nullptr) {
#if TASKDAG_LOG
                        std::cout << "Incrementing finished parents\n";
#endif
                        succ_kernel->finished_parents++;
                    }
                    if (succ_kernel->is_ready_for_dispatch()) {
#if TASKDAG_LOG
                        std::cout << "Adding successor " << s
                                  << " to local frontier\n";
#endif
                        add_free_kernel(succ_kernel);
                    }
                    id_to_event_map[succ_kernel->id] = kernel_event;
                } else { // successor belongs to different task component -->
#if TASKDAG_LOG
                    std::cout
                        << "Successor belongs to different task component \n";
#endif
#if TASKDAG_LOG
                    std::cout << "Setting callback for kernel ndrange event "
                              << k->id << " of task component " << task_id
                              << "\n";
#endif
                    // callback needed here!
                    UserArgs *user_data = new UserArgs(k, se);
                    if (d2h_ev == nullptr) {
                        // std::cout
                        //  << __FILE__ << " " << __LINE__
                        // << " Setting callback for ndrange event of kernel "
                        // << k->id << " of task component " << this->task_id
                        // << "\n";
                        clSetEventCallback(kernel_event, CL_COMPLETE,
                                           &taskdag_callback,
                                           (void *)user_data);
//                        clFlush(cmd_q);
                        increment_flag = false;
                    }
                }
            }
            if ((dag_ptr->successors[k->id]).size() ==
                0) { // no successor callback needed for sync at the end
#if TASKDAG_LOG
                std::cout << "Kernel " << k->id << "has no successors\n";
#endif
#if TASKDAG_LOG
                std::cout << "Setting callback for kernel ndrange event "
                          << k->id << " of task component " << task_id << "\n";
#endif
                UserArgs *user_data = new UserArgs(k, se);
                if (d2h_ev == nullptr) {

                    // std::cout
                    //   << __FILE__ << " " << __LINE__
                    // << " Setting callback for ndrange event of kernel "
                    //<< k->id << " of task component " << this->task_id
                    //<< "\n";
                    clSetEventCallback(kernel_event, CL_COMPLETE,
                                       &taskdag_callback, (void *)user_data);
//                    clFlush(cmd_q);
                    increment_flag = false;
                }
            }
            if (increment_flag) {
#if TASKDAG_LOG
                std::cout << "Incrementing completed kernels of task component "
                             "and dag for kernel "
                          << k->id << "\n";
#endif
                completed_kernels++;
                dag_ptr->completed_kernels++;
            }
        }
//        clFlush(cmd_q);
    }
    {
        if(profile_engine) FunctionTimer("OCLDelay->Flush");
        clFlush(cmd_q);
    }
    //    for (auto u : user_events)
    //      clSetUserEventStatus(u, CL_COMPLETE);
    // user_events.clear();
}

void ScheduleEngine::get_all_devices() {

    LOG("get_all_devices(): BEGIN \n");

    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0) {
#if FILE_LOGGER
        fprintf(fp, "\tFound 0 platforms!\n");
#endif
        exit(EXIT_FAILURE);
    }
    LOG("Obtained all platforms\n");
    cl_device_type device_type;
    cl_platform_id platforms[numPlatforms];
    for (unsigned int i = 0; i < numPlatforms; i++) {
        err = clGetPlatformIDs(numPlatforms, platforms, NULL);
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                             &numDevices);
        LOG("Obtained %u devices for platform %u\n", numDevices, i);
        cl_device_id devices[numDevices];

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices,
                             devices, NULL);
        if (numDevices == 0) {
#if FILE_LOGGER
            fprintf(fp, "\nNo device\n");
#endif
            exit(EXIT_FAILURE);
        }

        err = clGetDeviceInfo(devices[0], CL_DEVICE_TYPE, sizeof(device_type),
                              &device_type, NULL);
        if (device_type == CL_DEVICE_TYPE_GPU) {

            for (cl_device_id d : devices) {

                all_devices[platform_gpu].push_back(d);
            }
        } else if (device_type == CL_DEVICE_TYPE_CPU) {

            const cl_device_partition_property properties[3] = {
                CL_DEVICE_PARTITION_EQUALLY, num_cpu_cores_per_device, 0};
            cl_uint num_sub_devices;
            err = clCreateSubDevices(devices[0], properties, 0, NULL,
                                     &num_sub_devices);

            cl_device_id sub_devices[num_sub_devices];

            err = clCreateSubDevices(devices[0], properties,
                                     num_sub_devices * sizeof(cl_device_id),
                                     sub_devices, NULL);
            int count = 0;
            for (cl_device_id d : sub_devices) {
                if (count >= num_cpu_subdevices / 2) {
                    all_devices[platform_cpu].push_back(d);
                }
                count++;
            }
        }
    }

    LOG("get_all_devices(): END\n");
}

std::vector<cl_command_queue>
ScheduleEngine::create_command_queue_for_each(cl_device_id *devs, int num_devs,
                                              cl_context ctx) {

    LOG("create_command_queue_for_each(): BEGIN \n");

    cl_int status;
    std::vector<cl_command_queue> cmd_queues;
    for (unsigned int i = 0; i < num_devs; ++i) {
        cmd_queues.push_back(clCreateCommandQueue(
            ctx, devs[i], CL_QUEUE_PROFILING_ENABLE, &status));
        check(status, "Creating Command Queue\n");
    }
    LOG("Created %d Command Queues.\n", num_devs);
    LOG("create_command_queue_for_each(): END \n");

    return cmd_queues;
}

void ScheduleEngine::host_initialize() {
    LOG("host_initialize(): BEGIN\n");
    get_all_devices();
    cl_int status;
    cl_device_id *gpu = &(all_devices[platform_gpu][0]);
    cl_device_id *cpu = &(all_devices[platform_cpu][0]);
    int num_gpus = all_devices[platform_gpu].size();
    int num_cpus = all_devices[platform_cpu].size();
    cl_platform_id gpu_platform;
    clGetDeviceInfo(all_devices[platform_gpu][0], CL_DEVICE_PLATFORM,
                    sizeof(gpu_platform), &gpu_platform, NULL);
    cl_context_properties gps[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties)gpu_platform, 0};
    ctxs[platform_gpu] =
        clCreateContext(gps, num_gpus, gpu, NULL, NULL, &status);
    check(status, "Creating GPU Context\n");

    cl_platform_id cpu_platform;
    clGetDeviceInfo(all_devices[platform_cpu][0], CL_DEVICE_PLATFORM,
                    sizeof(cpu_platform), &cpu_platform, NULL);
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties)cpu_platform, 0};
    ctxs[platform_cpu] =
        clCreateContext(cps, num_cpus, cpu, NULL, NULL, &status);
    check(status, "Creating CPU Context\n");

    LOG("Created  Contexts Successfully \n");

    LOG("Creating command queues for GPU\n");
    cmd_qs[platform_gpu] =
        create_command_queue_for_each(gpu, num_gpus, ctxs[platform_gpu]);

    LOG("Creating command queues for CPU\n");
    cmd_qs[platform_cpu] =
        create_command_queue_for_each(cpu, num_cpus, ctxs[platform_cpu]);

    LOG("host_initialize(): END \n");
}

void ScheduleEngine::host_synchronize() {
    for (auto x : cmd_qs) {
        for (auto q : x.second) {
            clFinish(q);
        }
    }
}

void ScheduleEngine::create_gpu_buffer_manager(
    const std::vector<std::pair<std::size_t, int>> &sizes) {
    gpu_buf_man = new GpuBufferManager(ctxs[platform_gpu], sizes);
}

void ScheduleEngine::print_device_info(cl_device_id device_id, int i, int j) {
    cl_int err;
    cl_char platform_name[STR_LENGTH] = {0};
    cl_char vendor_name[STR_LENGTH] = {0};
    cl_char device_name[STR_LENGTH] = {0};
    cl_bool device_available;
    cl_uint device_freq;
    cl_uint no_of_parallel_cores;
    cl_platform_id platform_id;
    cl_device_type device_type;

    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name),
                          &device_name, NULL);
    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type),
                          &device_type, NULL);
    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name),
                          &vendor_name, NULL);
    err = clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(platform_id),
                          &platform_id, NULL);
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
                            sizeof(platform_name), &platform_name, NULL);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                          sizeof(device_freq), &device_freq, NULL);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(no_of_parallel_cores), &no_of_parallel_cores,
                          NULL);
    err = clGetDeviceInfo(device_id, CL_DEVICE_AVAILABLE,
                          sizeof(device_available), &device_available, NULL);

    if (device_type == CL_DEVICE_TYPE_GPU)
        printf("\nDeviceID_%d-%d: %u,\nDeviceName: %s, \nType: GPU, "
               "\nPlatform: %s, \nVendor: %s, \nMaxClockFrequency: %d, "
               "\nMaxParallelCore: %d, \nAvailability: %d\n",
               i, j, device_id, device_name, platform_name, vendor_name,
               device_freq, no_of_parallel_cores, device_available);
    if (device_type == CL_DEVICE_TYPE_CPU)
        printf("\nDeviceID_%d-%d: %u,\nDeviceName: %s, \nType: CPU, "
               "\nPlatform: %s, \nVendor: %s, \nMaxClockFrequency: %d, "
               "\nMaxParallelCore: %d, \nAvailability: %d\n",
               i, j, device_id, device_name, platform_name, vendor_name,
               device_freq, no_of_parallel_cores, device_available);
}

void ScheduleEngine::print_all_device_info() {
#if FILE_LOGGER
    fprintf(fp, "\nDEVICE INFO:\n");
#endif
    int i = 0, j = 0;
    for (auto x : all_devices) {
        for (auto device : x.second) {

            print_device_info(device, i, j);
            j++;
        }
        i++;
    }
}

void ScheduleEngine::print_all_kernel_info() {
    for (auto x : Kernels) {
        std::cout << "Kernel info filename: " << x.first << "\n";
        auto kernel = x.second;
        kernel->print_kernel_info();
    }
}
void ScheduleEngine::print_all_dag_info() {
    for (auto x : DAGs) {
        std::cout << "DAG filename: " << x.first << "\n";
        auto dag = x.second;
        dag->print_dag_info();
    }
}

void ScheduleEngine::print_dag_arrival_timestamps() {
    for (auto x : DAGs) {
        std::cout << "DAG id: " << x.first << "\n";
        auto dag = x.second;
        dag->print_arrival_time();
    }
}

void ScheduleEngine::initialize_kernels(
    std::vector<std::string> &kernel_names) {
    for (auto info : kernel_names) {
        Kernel *k = new Kernel(info.c_str(), true);
        Kernels[info] = k;
        frontier.push(k);
        num_jobs++;
    }
    frontier.init_num_jobs(num_jobs);
}

Kernel *ScheduleEngine::get_first_kernel() { return Kernels.begin()->second; }

Kernel *ScheduleEngine::get_kernel_by_name(const char *name) {
    std::string kernel_name = std::string(name);
    return Kernels[kernel_name];
}

void ScheduleEngine::initialize_dags(std::vector<std::string> &dag_names) {
    for (auto dag : dag_names) {
        DAG *d = new DAG(dag.c_str());
        DAGs[dag] = d;
        id_to_dag_map[d->id] = d;
        dag_to_task_components_map.emplace(d, std::vector<TaskComponent *>());
        for (auto v : d->vertices) {
            Kernel *k = d->id_to_kernel_map[v];
            if (k->is_ready_for_dispatch()) {
                frontier.push(k);
                k->in_frontier = true;
                // printf("Pushing kernel %s to frontier\n", k->get_name());
            }
        }
        num_jobs++;
    }
    frontier.init_num_jobs(num_jobs);
}

void ScheduleEngine::initialize_adas_dags(const char *config_file) {

    std::vector<std::string> dag_names;
    std::ifstream stream(config_file);
    std::string line;
    while (std::getline(stream, line)) {
        dag_names.push_back(line);
    }

    for (auto dag : dag_names) {
        DAG *d = new DAG(dag.c_str());
        DAGs[dag] = d;
        id_to_dag_map[d->id] = d;
        dag_to_task_components_map.emplace(d, std::vector<TaskComponent *>());
    }
}

void ScheduleEngine::initialize_online_dags(const char *filename) {
    std::ifstream stream(filename);
    std::string line;
    while (std::getline(stream, line)) {
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(':');
        DAG *d = new DAG(ssublist[0].c_str());
        DAGs[d->get_id()] = d;
        float arrival_time = std::stoll(ssublist[1]);
        d->set_arrival_time(arrival_time);
        arriving_dags.push_back(d);
        dag_to_task_components_map.emplace(d, std::vector<TaskComponent *>());
        num_jobs++;
    }
    frontier.init_num_jobs(num_jobs);
}

void ScheduleEngine::setup_kernels() {
    for (auto x : Kernels) {
        std::string kernel_name = x.first;
        Kernel *k = x.second;
        LOG("Setting up kernel %s\n", kernel_name.c_str());
        k->setup(all_devices, ctxs, platforms, gpu_buf_man);
    }
}

void ScheduleEngine::setup_kernel(int kernel_id, cl_context &ctx, Vendor p,
                                  DAG *dag) {
    Kernel *k = dag->id_to_kernel_map[kernel_id];
    k->initialize_buffer_flags(p);
    k->setup_buffer_flags(p);
    k->setup_buffers(ctx, p, gpu_buf_man);
    k->cl_set_kernel_args(p);
    k->preferred = p;
}

void ScheduleEngine::setup_dags() {
    printf("Setting up DAGs\n");
    for (auto x : DAGs) {
        std::string dag_name = x.first;
        DAG *d = x.second;
        LOG("Setting up DAG %s\n", dag_name.c_str());
        d->setup(all_devices, ctxs, platforms, gpu_buf_man);
    }
}

void ScheduleEngine::copy_persistent_buffers() {
    printf("Copying persistent buffers\n");
    for (auto x : DAGs) {
        std::string dag_name = x.first;
        DAG *d = x.second;
        LOG("Copying Persistent Buffers for kernels in DAG %s\n",
            dag_name.c_str());
        //        printf("Copying persistent buffers for DAG %s\n",
        //        dag_name.c_str());
        d->copy_persistent_buffers(all_devices, ctxs, platforms, cmd_qs);
    }

    for (auto x : Kernels) {
        std::string kernel_name = x.first;
        Kernel *k = x.second;
        LOG("Copying Persistent Buffers for kernel %s\n", kernel_name.c_str());
        k->copy_persistent_buffers(all_devices, ctxs, platforms, cmd_qs);
    }
}
void ScheduleEngine::build_kernels_of_all_dags() {
    //    printf("Building kernels of all DAGs\n");
    for (auto x : DAGs) {
        std::string dag_name = x.first;
        DAG *d = x.second;
        LOG("Building kernels for DAG %s\n", dag_name.c_str());
        d->build_kernels(all_devices, ctxs, platforms);
    }
}
void ScheduleEngine::test_kernel_dispatch() {
    for (auto p : platforms) {
        for (auto x : Kernels) {
            std::string kernel_name = x.first;
            Kernel *k = x.second;
            LOG("Dispatching kernel %s for platform %d\n", kernel_name.c_str(),
                p);
            std::cout << "Printing input buffers\n";
            k->print_input();
            k->dispatch_synchronous(p, cmd_qs[p][0]);
            std::cout << "Printing output buffers\n";
            k->print_result();
        }
    }
}

void ScheduleEngine::schedule_kernels() {
    // std::cout << "Initiating Scheduling Loop\n";
    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running scheduler thread %lu on core %d \n", pthread_self(),
           sched_getcpu());
    while (completed_jobs.load() < num_jobs) {
        Kernel *k = NULL;
        int device_position = -1;
        Vendor platform;
#if SCHEDULE_LOG
        std::cout << "Frontier empty? " << frontier.empty() << "\n";
#endif
        frontier.wait_and_pop(&k);
        if (k == NULL)
            continue;
#if SCHEDULE_LOG
        std::cout << "Obtaining device\n";
#endif
        /* for (auto p : platforms) */
        /* if (!ready_queue.empty(p)) { */
        /* ready_queue.wait_and_pop(p, &device_position); */
        /* platform = p; */
        /* break; */
        /* } */
        platform = k->preferred;
        ready_queue.wait_and_pop(platform, &device_position);
#if SCHEDULE_LOG
        std::cout << "Obtained device " << device_position << " of platform "
                  << platform << "\n";
#endif
        if (device_position == -1) {
            frontier.push(k);
            continue;
        }
        k->in_frontier = false;
        k->platform = platform;
        k->device_id = device_position;
        UserArgs *user_data = new UserArgs(k, this);
        cl_event ev = k->dispatch_with_events_callback(
            platform, ctxs[platform], cmd_qs[platform][device_position]);

        k->has_dispatched = true;
#if SCHEDULE_LOG
        std::cout << "Dispatched kernel "
                  << "\n";
#endif
        clSetEventCallback(ev, CL_COMPLETE, &kernel_callback,
                           (void *)user_data);
#if SCHEDULE_LOG
        std::cout << "Callback Set"
                  << "\n";
#endif
        clFlush(cmd_qs[platform][device_position]);
    }

    // std::cout << "Scheduling of kernels finished\n";
    // for (auto x : Kernels) {
    //    Kernel *k = x.second;
    //    k->print_result();
    // }
}

void ScheduleEngine::schedule_dags() {
    std::cout << "Initiating Scheduling Loop\n";

    while (completed_jobs.load() < num_jobs) {
#if SCHEDULE_LOG
        std::cout << "Number of DAGS: " << num_jobs
                  << " Completed DAGs: " << completed_jobs << "\n";
#endif
        // std::cout << "Fronter Empty? " << !frontier.empty() << "\n";
        // std::cout << "Device available? " << device_available() << "\n";
        std::cout << "Obtaining kernel from frontier\n";

        int device_position = -1;
        Vendor platform;
        Kernel *k = NULL;

        frontier.wait_and_pop(&k);
        if (k == NULL)
            continue;
#if SCHEDULE_LOG
        std::cout << "Obtained kernel from frontier\n";
        std::cout << "Obtaining device from ready_queue\n";
#endif
        for (auto p : platforms)
            if (!ready_queue.empty(p)) {
                ready_queue.wait_and_pop(p, &device_position);
                platform = p;
                break;
            }
#if SCHEDULE_LOG
        std::cout << "Obtained device from ready_queue\n";
#endif
        if (device_position == -1) {
            frontier.push(k);
            continue;
        }

        k->in_frontier = false;
        k->has_dispatched = true;
        k->platform = platform;
        k->device_id = device_position;

        UserArgs *user_data = new UserArgs(k, this);
        cl_event ev = k->dispatch_with_events_callback(
            platform, ctxs[platform], cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
        std::cout << "Dispatched kernel "
                  << "\n";
#endif
        clSetEventCallback(ev, CL_COMPLETE, &dag_callback, (void *)user_data);
#if SCHEDULE_LOG
        std::cout << "Callback Set"
                  << "\n";
#endif
        clFlush(cmd_qs[platform][device_position]);
    }

    std::cout << "Scheduling of dags finished\n";
}

void CL_CALLBACK kernel_callback(cl_event ev, cl_int event_command_exec_status,
                                 void *user_data) {

    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running callback thread %lu on core %d \n", pthread_self(),
           sched_getcpu());
#if CALLBACK_LOG
    std::cout << "Kernel Callback initiated\n";
#endif
    // struct FunctionTimer ft("Callback");
    UserArgs *u = (UserArgs *)user_data;
    Kernel *k = u->k;
    ScheduleEngine *se = u->se;

    k->is_finished = true;

    se->ready_queue.push(k->platform, k->device_id);

    completed_jobs++;
    se->frontier.notify();
}

void static_deallocation(GpuBufferManager *GpuBufMan, Kernel *k, DAG *d) {
    // Deleting the Input Buffers of the finished kernel
    for (auto buf : k->inputBuffers) {
        for (auto x : buf->data) {
            if (buf->vendor_to_bufferflag_map[x.first]->create) {
                if (x.first == Vendor::ARM_CPU)
                    clReleaseMemObject(x.second);
                else
                    GpuBufMan->deallocate_buffer(x.second);
            }
        }
        // If the buffer is isolated we can delete the host_array as well
        if (buf->isolated) {
            if (buf->allocate_host_array) {
                delete buf->host_array;
                buf->allocate_host_array =
                    false; // setting the allocate_host_array flag as false so
                           // that destructor does not delete it again
            }
        }
    }

    // Deleting the Output Buffers of the finished kernel
    for (auto buf : k->outputBuffers) {
        for (auto x : buf->data) {
            if (buf->vendor_to_bufferflag_map[x.first]->create) {
                if (x.first == Vendor::ARM_CPU)
                    clReleaseMemObject(x.second);
                else
                    GpuBufMan->deallocate_buffer(x.second);
            }
        }
    }

    // Checking for Life Cycle Variable for dependent buffers
    for (auto e : k->edge_info) {
        if (e->source_kernel != stoi(k->get_id())) {
            Kernel *source_kernel = d->id_to_kernel_map[e->source_kernel];
            Buffer *source_buffer =
                source_kernel->get_buffer_from_argument_position(
                    e->source_buffer);

            source_buffer->life_cycle--;
            if (source_buffer->allocate_host_array) {
                if (source_buffer->life_cycle == 0) {
                    delete source_buffer->host_array;
                    source_buffer->allocate_host_array = false;
                }
            }
        }
    }
}

void CL_CALLBACK dag_callback(cl_event ev, cl_int event_command_exec_status,
                              void *user_data) {
    // cpu_set_t cpuset;
    // pthread_t thread_id_scheduler = pthread_self();
    // CPU_ZERO(&cpuset);
    // CPU_SET(2, &cpuset); // TODO: Set according to your platform
    // pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    // printf("Running callback thread %lu on core %d \n", pthread_self(),
    //        sched_getcpu());
    // struct FunctionTimer f("DAG Callback");
    if (profile_engine)
        FunctionTimer("Callback->OnlyKernel");
    UserArgs *u = (UserArgs *)user_data;
    Kernel *k = u->k;
    ScheduleEngine *se = u->se;
    DAG *d = (DAG *)k->dag_ptr;

#if CALLBACK_LOG
    std::cout << "DAG CALLBACK initiated for kernel " << k->id << ": "
              << k->get_name() << "\n";
#endif
#if SCHEDULE_LOG
    std::cout << "DAG callback: Finished Kernel " << k->get_id() << " of "
              << d->get_id() << " " << d->current_instance << " on device "
              << k->platform << " " << k->device_id << "\n";
    // Event e(ev, "read");
    // std::cout << "Event status ";
    // e.print_status();
#endif
    k->is_finished = true;
#if CALLBACK_LOG
    std::cout << "Updating number of completed kernels for kernel " << k->id
              << ": " << k->get_name() << "\n";
#endif
    // update number of completed kernels for dag
    d->completed_kernels++;
    // std::cout << "Number of completed kernels: " << d->completed_kernels
    //         << "\n";
    //    std::cout << "Updated number of completed kernels for kernel " <<
    //    k->id
    //            << ": " << k->get_name() << "\n";
    // update number of finished parents for successors of current kernel
    // that has finished

#if CALLBACK_LOG
    std::cout << "Updating number of finished parents for successors of "
                 "current kernel  "
              << k->id << ": " << k->get_name() << "\n";
#endif
    for (auto s : d->successors[k->id]) {
        Kernel *k_s = d->id_to_kernel_map[s];
        k_s->finished_parents++;
    }

#if CALLBACK_LOG
    std::cout << "Updated number of finished parents for successors of "
                 "current kernel  "
              << k->id << ": " << k->get_name() << "\n";
#endif
    // update frontier with successors of current kernels that are ready for
    // dispatch( a kernel is ready for dispatch if all parents of that
    // kernel have finished )

#if CALLBACK_LOG
    std::cout << "Updating frontier with successors of current kernel " << k->id
              << ": " << k->get_name() << "\n";
#endif
    std::vector<TaskComponent *> task_components_to_be_dispatched;
    std::vector<Kernel *> kernels_to_be_dispatched;
    for (auto s : d->successors[k->id]) {
        Kernel *k_s = d->id_to_kernel_map[s];
        if (k_s->is_ready_for_dispatch()) {

            // std::cout << "Pushing kernel id " << s << "\n";
            if (!k_s->has_dispatched) {
                k_s->in_frontier = true;
                se->frontier.push(k_s);
                se->frontier.notify();
            } else {
                if (k_s->task_id != -1) {
                    TaskComponent *tc_s = (TaskComponent *)k_s->task_ptr;
                    tc_s->add_free_kernel(k_s);
                    task_components_to_be_dispatched.push_back(tc_s);
                } else {
                    kernels_to_be_dispatched.push_back(k_s);
                }
            }
        }
    }

#if CALLBACK_LOG
    std::cout << "Updated frontier with successors of current kernel " << k->id
              << ": " << k->get_name() << "\n";
#endif
    // update number of completed kernels for dag
    if (d->completed_kernels.load() == d->num_kernels) {
        // std::cout << "Finished DAG: " << d->get_id() << " "
        //        << d->current_instance << "\n";
        completed_jobs++;
#if ADAS
        if (!d->instance_queue.empty()) {
            InstanceConfig *ic;
            d->instance_queue.wait_and_pop(&ic);
            //            printf("Callback thread ");
            se->setup_new_instance(d, ic);
        } else {
            d->running = false;
        }
#endif

        se->frontier.notify();
    }

#if SCHEDULE_LOG
    std::cout << "TD CALLBACK finished for kernel " << k->id << "of DAG "
              << d->get_id() << " executing on device " << k->platform << " "
              << k->device_id << "\n";
#endif
#if CALLBACK_LOG
    std::cout << "Updating Ready queue for kernel " << k->id << ": "
              << k->get_name() << "\n";
#endif
    // update ready queue
    se->ready_queue.push(k->platform, k->device_id);
    se->ready_queue.notify();
#if CALLBACK_LOG
    std::cout << "Updated Ready queue for kernel " << k->id << ": "
              << k->get_name() << "\n";
#endif
    // Deallocating The Buffer Memory which will not be needed now
    // dynamic_deallocation(k, d);

    // if Successor is ready for dispatch and belongs to a task component that
    // has already been dispatched by the main scheduler loop
    for (auto t : task_components_to_be_dispatched) {
        t->dispatch_single(se->ctxs[t->platform],
                           se->cmd_qs[t->platform][t->device_id], se);
    }
    // if successor is ready for dispatch and has already been dispatched by the
    // main scheduler loop
    for (auto k_tbd : kernels_to_be_dispatched) {
        auto platform = k_tbd->platform;
        auto device_position = k_tbd->device_id;
        UserArgs *user_data = new UserArgs(k_tbd, se);
        cl_event ev = k_tbd->dispatch_with_events_callback(
            platform, se->ctxs[platform],
            se->cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
        std::cout << "Dispatched kernel with id" << k_tbd->id << "to device"
                  << device_position << "of platform " << platform << "\n";
#endif
        clSetEventCallback(ev, CL_COMPLETE, &dag_callback, (void *)user_data);
#if SCHEDULE_LOG
        std::cout << "Callback Set\n";
#endif
        {
            if(profile_engine) FunctionTimer("OCLDelay->Flush");
            clFlush(se->cmd_qs[platform][device_position]);
        }
    }
}

TaskComponent *
ScheduleEngine::create_task_component(std::vector<int> &kernel_ids,
                                      cl_context &ctx, Vendor p, DAG *dag)

{
    // struct FunctionTimer f("Task Component Creation");
    TaskComponent *tc = new TaskComponent(kernel_ids, p, dag);
    tc->setup_linkage(ctx, gpu_buf_man);
    tc->set_kernel_args();
    dag_to_task_components_map[dag].push_back(tc);
    return tc;
}
TaskComponent *ScheduleEngine::create_task_component_from_chain(Chain *c,
                                                                cl_context &ctx,
                                                                DAG *dag)

{
    if (profile_engine)
        FunctionTimer("Setup->CreateTC");
    Vendor p = c->platform;
    std::vector<int> kernel_ids;
    if (c->successor)
        for (int i = 0; i < c->chain_kernels.size(); i++)
            kernel_ids.push_back(c->chain_kernels[i]->id);
    if (c->predecessor)
        for (int i = c->chain_kernels.size() - 1; i >= 0; i--)
            kernel_ids.push_back(c->chain_kernels[i]->id);
    TaskComponent *tc = new TaskComponent(kernel_ids, p, dag);
    tc->setup_buffers(ctx, gpu_buf_man);
    tc->set_kernel_args();
    dag_to_task_components_map[dag].push_back(tc);
    return tc;
}

void ScheduleEngine::print_task_component_info() {
    for (auto x : DAGs) {
        std::cout << "Task Component Information for DAG " << x.first << "\n";
        for (auto tc : dag_to_task_components_map[x.second])
            tc->print_task_info();
    }
}

void ScheduleEngine::static_schedule_taskdags() {
    std::cout << "Initiating Scheduling Loop\n";

    while (completed_jobs.load() < num_jobs) {
#if SCHEDULE_LOG
        std::cout << "Number of DAGS: " << num_jobs
                  << " Completed DAGs: " << completed_jobs << "\n";
        std::cout << "Obtaining kernel from frontier\n";
#endif

        int device_position = -1;
        Vendor platform;
        Kernel *k = NULL;
        TaskComponent *tc = NULL;

        frontier.wait_and_pop(&k);
        if (k == NULL)
            continue;

        if (k->task_id != -1) {
            tc = (TaskComponent *)k->task_ptr;
            platform = tc->preferred;
        } else {
            platform = k->preferred;
        }
#if SCHEDULE_LOG
        std::cout << "Obtained kernel from frontier\n";
        std::cout << "Obtaining device from ready_queue\n";
#endif
        ready_queue.wait_and_pop(platform, &device_position);
#if SCHEDULE_LOG
        std::cout << "Obtained device from ready_queue\n";
#endif
        if (k->task_id == -1) {
            k->initiate_dispatch(platform, device_position);
            UserArgs *user_data = new UserArgs(k, this);
            cl_event ev = k->dispatch_with_events_callback(
                platform, ctxs[platform], cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
            std::cout << "Dispatched kernel with id" << k->id << "to device"
                      << device_position << "of platform " << platform << "\n";
#endif
            clSetEventCallback(ev, CL_COMPLETE, &dag_callback,
                               (void *)user_data);
#if SCHEDULE_LOG
            std::cout << "Callback Set\n";
#endif
            clFlush(cmd_qs[platform][device_position]);

        } else {
            tc->initiate_dispatch(platform, device_position);
            tc->dispatch_single(ctxs[platform],
                                cmd_qs[platform][device_position], this);
#if SCHEDULE_LOG
            std::cout << "Dispatched task component with id" << tc->task_id
                      << "to device" << device_position << "of platform "
                      << platform << "\n";
#endif
        }
    }
    std::cout << "Scheduling of dags finished\n";
}

void CL_CALLBACK taskdag_callback(cl_event ev, cl_int event_command_exec_status,
                                  void *user_data) {
    // struct FunctionTimer f("TaskDAG Callback");
    // cpu_set_t cpuset;
    // pthread_t thread_id_scheduler = pthread_self();
    // CPU_ZERO(&cpuset);
    // CPU_SET(2, &cpuset); // TODO: Set according to your platform
    // pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    // printf("Running callback thread %lu on core %d \n", pthread_self(),
    //       sched_getcpu());
    if (profile_engine)
        FunctionTimer("Callback->KernelInTC");
    UserArgs *u = (UserArgs *)user_data;
    Kernel *k = u->k;
    ScheduleEngine *se = u->se;
    DAG *d = (DAG *)k->dag_ptr;
    TaskComponent *tc = (TaskComponent *)k->task_ptr;
    // std::cout << __FILE__ << " " << __LINE__
    //         << "TD callback called for Task_Component " << tc->task_id
    //       << "\n";
#if CALLBACK_LOG
    std::cout << "TASKDAG CALLBACK initiated for kernel " << k->id
              << "of task component " << tc->task_id << "\n";
#endif
#if SCHEDULE_LOG
    std::cout << "TD:Finished Kernel " << k->get_id() << " of " << d->get_id()
              << " " << d->current_instance << " on device " << k->platform
              << " ";
//    Event e(ev, "read");
//    std::cout << "Event status ";
//    e.print_status();
#endif

    k->is_finished = true;
#if CALLBACK_LOG
    std::cout << "Updating number of completed kernels for kernel " << k->id
              << ": " << k->get_name() << "\n";
    // update number of completed kernels for dag

    std::cout << "Updating number of finished parents for successors of "
                 "current kernel  "
              << k->id << ": " << k->get_name() << "\n";
#endif
    for (auto s : d->successors[k->id]) {
        Kernel *k_s = d->id_to_kernel_map[s];
        // if (k_s->task_id != tc->task_id)
        k_s->finished_parents++;
    }

#if CALLBACK_LOG
    std::cout << "Updated number of finished parents for successors of "
                 "current kernel (only if successor belongs to a separate task "
                 "component) "
              << k->id << ": " << k->get_name() << "\n";
#endif
    // update frontier with successors of current kernels that are ready for
    // dispatch( a kernel is ready for dispatch if all parents of that
    // kernel have finished ) and do not belong to any task component that has
    // been dispatched

#if CALLBACK_LOG
    std::cout << "Updating frontier with successors of current kernel (that "
                 "have not been dispatched as part of some task component)"
              << k->id << ": " << k->get_name() << "\n";
#endif
    for (auto s : d->successors[k->id]) {
        Kernel *k_s = d->id_to_kernel_map[s];
        if (k_s->is_ready_for_dispatch() && !k_s->has_dispatched) {
            k_s->in_frontier = true;
/* if (k_s->task_id != -1) { */
/* TaskComponent *tc_yet_to_be_dispatched = */
/* (TaskComponent *)k_s->task_ptr; */
/* tc_yet_to_be_dispatched->add_free_kernel(k_s); */
/* } else */
#if CALLBACK_LOG
            std::cout << "Successor added: " << k_s->id << "  "
                      << k_s->get_name() << "\n ";
#endif

            se->frontier.push(k_s);
            se->frontier.notify();
        }
    }

#if CALLBACK_LOG
    std::cout << "Updated frontier with successors of current kernel " << k->id
              << ": " << k->get_name() << "\n";
#endif
    tc->completed_kernels++;
    d->completed_kernels++;
#if CALLBACK_LOG
    std::cout << "Number of completed kernels: " << d->completed_kernels
              << "\n";
#endif

    // update number of completed kernels for dag
    if (d->completed_kernels.load() == d->num_kernels) {
#if CALLBACK_LOG
        std::cout << "DAG Finished!\n";
#endif
        completed_jobs++;
#if ADAS
        if (!d->instance_queue.empty()) {
            InstanceConfig *ic;
            d->instance_queue.wait_and_pop(&ic);
            se->setup_new_instance(d, ic);
        } else {
            d->running = false;
        }
#endif

        se->frontier.notify();
    }

#if CALLBACK_LOG
    std::cout << "Updating Ready queue for kernel " << k->id << ": "
              << k->get_name() << "\n";
#endif

    // update ready queue
    // std::cout << __FILE__ << " " << __LINE__ << "Task_Component " <<
    // tc->task_id
    //        << ": kernel " << k->id
    //      << " finished; Total vs Completed: " << tc->num_kernels << ","
    //    << tc->completed_kernels.load() << "\n";
    if (tc->is_finished()) {
#if SCHEDULE_LOG
        std::cout << "TD Callback finished for task component " << tc->task_id
                  << " of DAG " << d->id << "\n";
#endif
        se->ready_queue.push(k->platform, k->device_id);
        se->ready_queue.notify();
    }

#if CALLBACK_LOG
    std::cout << "Updated Ready queue for kernel " << k->id << ": "
              << k->get_name() << "\n";
#endif
    std::vector<TaskComponent *> task_components_to_be_dispatched;
    std::vector<Kernel *> kernels_to_be_dispatched;
    for (auto s : d->successors[k->id]) {
        Kernel *k_s = d->id_to_kernel_map[s];
        if (k_s->is_ready_for_dispatch() && k_s->has_dispatched &&
            k_s->task_id != -1) {
            TaskComponent *tc_s = (TaskComponent *)k_s->task_ptr;

            tc_s->add_free_kernel(k_s);
            task_components_to_be_dispatched.push_back(tc_s);
        }
        if (k_s->is_ready_for_dispatch() && k_s->has_dispatched &&
            k_s->task_id == -1)
            kernels_to_be_dispatched.push_back(k_s);
    }

    for (auto t : task_components_to_be_dispatched) {
#if CALLBACK_LOG

        std::cout << "Dispatched from callback task component with id "
                  << tc->task_id;
        std::cout << "containing kernels ";
        for (auto kid : tc->kernel_ids)
            std::cout << kid << " ";
        std::cout << "the successor of kernel " << k->id << "to device"
                  << k->device_id << "of platform " << k->platform << "\n";
#endif

        t->dispatch_single(se->ctxs[t->platform],
                           se->cmd_qs[t->platform][t->device_id], se);
    }
    // if successor is ready for dispatch and has already been dispatched by the
    // main scheduler loop
    for (auto k_tbd : kernels_to_be_dispatched) {
        auto platform = k_tbd->platform;
        auto device_position = k_tbd->device_id;
        UserArgs *user_data = new UserArgs(k_tbd, se);
        cl_event ev = k_tbd->dispatch_with_events_callback(
            platform, se->ctxs[platform],
            se->cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
        std::cout << "Dispatched kernel with id" << k_tbd->id << "to device"
                  << device_position << "of platform " << platform << "\n";
#endif
        clSetEventCallback(ev, CL_COMPLETE, &dag_callback, (void *)user_data);
#if SCHEDULE_LOG
        std::cout << "Callback Set\n";
#endif
        {
           if(profile_engine) FunctionTimer("OCLDelay->Flush");
            clFlush(se->cmd_qs[platform][device_position]);
        }
    }

#if CALLBACK_LOG
    std::cout << "DAG CALLBACK finished for kernel " << k->id << ": "
              << k->get_name() << "\n";
#endif
}

void ScheduleEngine::list_schedule() {
    // std::cout << "List Scheduling: Initiating Scheduling Loop\n";

    while (completed_jobs.load() < num_jobs) {
#if SCHEDULE_LOG
        std::cout << "Number of DAGS: " << num_jobs
                  << " Completed DAGs: " << completed_jobs << "\n";
        std::cout << "Obtaining kernel from frontier\n";
#endif

        int device_position = -1;
        Vendor platform;
        Kernel *k = NULL;

        frontier.wait_and_pop(&k);
        if (k == NULL)
            continue;

        platform = k->preferred;
#if SCHEDULE_LOG
        std::cout << "Obtained kernel from frontier " << k->id << " "
                  << k->get_name() << "\n";
        std::cout << "Obtaining device from ready_queue of " << platform
                  << "\n";
#endif
        ready_queue.wait_and_pop(platform, &device_position);
#if SCHEDULE_LOG
        std::cout << "Obtained device from ready_queue " << platform << ": "
                  << device_position << "\n";
#endif
        k->initiate_dispatch(platform, device_position);
        UserArgs *user_data = new UserArgs(k, this);
        k->setup(platform, ctxs[platform], gpu_buf_man);
        cl_event ev = k->dispatch_with_events_callback(
            platform, ctxs[platform], cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
        std::cout << "Dispatched kernel with id " << k->id << "to device "
                  << device_position << "of platform " << platform << "\n";
#endif
        clSetEventCallback(ev, CL_COMPLETE, &dag_callback, (void *)user_data);
#if SCHEDULE_LOG
        std::cout << "Callback Set\n";
#endif

        {
           if(profile_engine) FunctionTimer("OCLDelay->Flush");
            clFlush(cmd_qs[platform][device_position]);
        }
    }
    //    std::cout << "Scheduling of dags finished\n";
}

void dynamic_deallocation(Kernel *k, DAG *d) {
    // Deleting the Input Buffers of the finished kernel
    for (auto buf : k->inputBuffers) {
        for (auto x : buf->data) {
            if (buf->vendor_to_bufferflag_map[x.first]->create) {
                buf->destructor = false;
                clReleaseMemObject(x.second);
            }
        }
        // If the buffer is isolated we can delete the host_array as well
        if (buf->isolated) {
            if (buf->allocate_host_array) {
                delete buf->host_array;
                buf->allocate_host_array =
                    false; // setting the allocate_host_array flag as false so
                           // that destructor does not delete it again
            }
        }
    }

    // Deleting the Output Buffers of the finished kernel
    for (auto buf : k->outputBuffers) {
        for (auto x : buf->data) {
            if (buf->vendor_to_bufferflag_map[x.first]->create) {
                buf->destructor = false;
                clReleaseMemObject(x.second);
            }
        }
    }

    // Checking for Life Cycle Variable for dependent buffers
    for (auto e : k->edge_info) {
        if (e->source_kernel != std::stoi(k->get_id())) {
            Kernel *source_kernel = d->id_to_kernel_map[e->source_kernel];
            Buffer *source_buffer =
                source_kernel->get_buffer_from_argument_position(
                    e->source_buffer);

            source_buffer->life_cycle--;
            if (source_buffer->allocate_host_array) {
                if (source_buffer->life_cycle == 0) {
                    delete source_buffer->host_array;
                    source_buffer->allocate_host_array = false;
                }
            }
        }
    }
}

void ScheduleEngine::profile_kernels(std::string &&filename = "timing.json") {
    json timestamps;
    std::map<Vendor, unsigned long long int> ref;
    for (auto x : Kernels) {
        Kernel *k = x.second;
        Vendor p = k->platform;
        unsigned long long int min_start_time =
            k->command_events.get_min_start_time();
        if (ref.find(p) == ref.end())
            ref[p] = min_start_time;
        else
            ref[p] = std::min(ref[p], min_start_time);
    }
    for (auto x : Kernels) {
        Kernel *k = x.second;
        Vendor p = k->platform;
        int device_id = k->device_id;
        std::string kernel = k->get_name();

        timestamps[kernel] = nullptr;
        timestamps[kernel]["dev_type"] = p;
        timestamps[kernel]["device_id"] = device_id;

        timestamps[kernel]["write"] = nullptr;
        timestamps[kernel]["ndrange"] = nullptr;
        timestamps[kernel]["read"] = nullptr;

        timestamps[kernel]["write"]["host_queued_start"] =
            k->host_events.get_timestamp("write", "start");
        timestamps[kernel]["write"]["host_queued_end"] =
            k->host_events.get_timestamp("write", "end");

        timestamps[kernel]["write"]["device_queued"] =
            k->command_events.get_min_timestamp("write", "queued");
        timestamps[kernel]["write"]["device_start"] =
            k->command_events.get_min_timestamp("write", "start");
        timestamps[kernel]["write"]["device_end"] =
            k->command_events.get_max_timestamp("write", "end");

        timestamps[kernel]["ndrange"]["device_start"] =
            k->command_events.get_min_timestamp("ndrange", "start");
        timestamps[kernel]["ndrange"]["device_end"] =
            k->command_events.get_max_timestamp("ndrange", "end");

        timestamps[kernel]["read"]["device_start"] =
            k->command_events.get_min_timestamp("read", "start");
        timestamps[kernel]["read"]["device_end"] =
            k->command_events.get_max_timestamp("read", "end");

        std::cout << "  " << k->get_id() << " : " << k->get_name()
                  << " executing on platform " << p << " and device "
                  << device_id << "\n";

        std::cout << "    host queued events --> Start: "
                  << timestamps[kernel]["write"]["host_queued_start"]
                  << " End: " << timestamps[kernel]["write"]["host_queued_end"]
                  << "\n";

        std::cout << "    write events --> Start: "
                  << timestamps[kernel]["write"]["device_start"]
                  << " End: " << timestamps[kernel]["write"]["device_end"]
                  << "\n";
        std::cout << "    ndrange events --> Start: "
                  << timestamps[kernel]["ndrange"]["device_start"]
                  << " End: " << timestamps[kernel]["ndrange"]["device_end"]
                  << "\n";
        std::cout << "    read Events --> Start: "
                  << timestamps[kernel]["read"]["device_start"]
                  << " End: " << timestamps[kernel]["read"]["device_end"]
                  << "\n";
        auto start_time =
            timestamps[kernel]["ndrange"]["device_start"].get<long long int>();
        auto end_time =
            timestamps[kernel]["ndrange"]["device_end"].get<long long int>();
        std::cout << "Device time:" << end_time - start_time << "\n";
    }
    std::ofstream o(filename.c_str());
    o << timestamps << std::endl;
}
void ScheduleEngine::calculate_delays() {

    json timestamps;
    unsigned long long int min_timestamp = ULONG_LONG_MAX, max_timestamp = 0;
    for (auto d : DAGs) {
        DAG *dag = d.second;
        //        std::cout << dag->get_id() << " : " << dag->get_name() <<
        //        "\n";
        for (auto v : dag->vertices) {
            Kernel *k = dag->id_to_kernel_map[v];
            int kernel_id = k->id;
            int dag_id = dag->id;
            Vendor p = k->platform;
            switch (p) {
            case Vendor::ARM_CPU: {
                auto cpu_enqueue_time =
                    k->command_events.get_min_timestamp("ndrange", "queued");
                auto cpu_submit_time =
                    k->command_events.get_min_timestamp("ndrange", "submit");
                auto cpu_start_time =
                    k->command_events.get_min_timestamp("ndrange", "start");

                if (cpu_start_time > 0) {
                    auto delay_queue_submit =
                        (cpu_submit_time - cpu_enqueue_time) * 1e-6;
                    auto delay_submit_start =
                        (cpu_start_time - cpu_submit_time) * 1e-6;
                    printf("Delay->CPU_NDRANGE_Queued_Submit(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_queue_submit);
                    printf("Delay->CPU_NDRANGE_Submit_Start(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_submit_start);
                }
                break;
            }
            case Vendor::ARM_GPU: {
                auto gpu_write_enqueue_time =
                    k->command_events.get_min_timestamp("write", "queued");
                auto gpu_write_submit_time =
                    k->command_events.get_min_timestamp("write", "submit");
                auto gpu_write_start_time =
                    k->command_events.get_min_timestamp("write", "start");

                if (gpu_write_start_time > 0) {
                    auto delay_queue_submit =
                        (gpu_write_submit_time - gpu_write_enqueue_time) * 1e-6;
                    auto delay_submit_start =
                        (gpu_write_start_time - gpu_write_submit_time) * 1e-6;
                    printf("Delay->GPU_Write_Queued_Submit(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_queue_submit);
                    printf("Delay->GPU_Write_Submit_Start(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_submit_start);
                }
                auto gpu_ndrange_enqueue_time =
                    k->command_events.get_min_timestamp("ndrange", "queued");
                auto gpu_ndrange_submit_time =
                    k->command_events.get_min_timestamp("ndrange", "submit");
                auto gpu_ndrange_start_time =
                    k->command_events.get_min_timestamp("ndrange", "start");

                if (gpu_ndrange_start_time > 0) {
                    auto delay_queue_submit =
                        (gpu_ndrange_submit_time - gpu_ndrange_enqueue_time) * 1e-6;
                    auto delay_submit_start =
                        (gpu_ndrange_start_time - gpu_ndrange_submit_time) * 1e-6;
                    printf("Delay->GPU_NDRANGE_Queued_Submit(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_queue_submit);
                    printf("Delay->GPU_NDRANGE_Submit_Start(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_submit_start);
                }
                auto gpu_read_enqueue_time =
                    k->command_events.get_min_timestamp("read", "queued");
                auto gpu_read_submit_time =
                    k->command_events.get_min_timestamp("read", "submit");
                auto gpu_read_start_time =
                    k->command_events.get_min_timestamp("read", "start");

                if (gpu_read_start_time > 0) {
                    auto delay_queue_submit =
                        (gpu_read_submit_time - gpu_read_enqueue_time) * 1e-6;
                    auto delay_submit_start =
                        (gpu_read_start_time - gpu_read_submit_time) * 1e-6;
                    printf("Delay->GPU_Read_Queued_Submit(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_queue_submit);
                    printf("Delay->GPU_Read_Submit_Start(%d,%d): %lf ms\n",kernel_id,dag_id,
                           delay_submit_start);
                }



                break;
            }
            }
        }
    }
}

void ScheduleEngine::calculate_makespan() {

    json timestamps;
    unsigned long long int min_timestamp = ULONG_LONG_MAX, max_timestamp = 0;
    for (auto d : DAGs) {
        DAG *dag = d.second;
        //        std::cout << dag->get_id() << " : " << dag->get_name() <<
        //        "\n";
        for (auto v : dag->vertices) {
            Kernel *k = dag->id_to_kernel_map[v];
            Vendor p = k->platform;
            switch (p) {
            case Vendor::ARM_CPU: {
                auto cpu_start_time =
                    k->command_events.get_min_timestamp("ndrange", "start");
                if (cpu_start_time > 0)
                    min_timestamp = std::min(min_timestamp, cpu_start_time);
                auto cpu_end_time =
                    k->command_events.get_max_timestamp("ndrange", "end");
                if (cpu_end_time > 0)
                    max_timestamp = std::max(max_timestamp, cpu_end_time);
                break;
            }
            case Vendor::ARM_GPU: {
                auto gpu_start_time =
                    k->command_events.get_min_timestamp("write", "start");
                if (gpu_start_time > 0)
                    min_timestamp = std::min(min_timestamp, gpu_start_time);
                auto gpu_end_time =
                    k->command_events.get_max_timestamp("read", "end");
                if (gpu_end_time > 0)
                    max_timestamp = std::max(max_timestamp, gpu_end_time);
                break;
            }
            }
        }
    }
    printf("Makespan: %llu\n", max_timestamp - min_timestamp);
}

void ScheduleEngine::profile_dags(std::string &&filename = "timing.json") {
    json timestamps;

    std::map<Vendor, unsigned long long int> ref;
    for (auto d : DAGs) {
        DAG *dag = d.second;
        for (auto v : dag->vertices) {
            Kernel *k = dag->id_to_kernel_map[v];
            Vendor p = k->platform;
            unsigned long long int min_start_time =
                k->command_events.get_min_start_time();
            if (ref.find(p) == ref.end())
                ref[p] = min_start_time;
            else
                ref[p] = std::min(ref[p], min_start_time);
        }
    }
    for (auto d : DAGs) {
        DAG *dag = d.second;
//        std::cout << dag->get_id() << " : " << dag->get_name() << "\n";
        for (auto v : dag->vertices) {
            Kernel *k = dag->id_to_kernel_map[v];
            Vendor p = k->platform;
            int device_id = k->device_id;
            std::string kernel =
                dag->get_id() + "_" + k->get_id() + "_" + k->get_name();
            timestamps[kernel] = nullptr;
            timestamps[kernel]["dev_type"] = std::string(get_device_type(p));
            timestamps[kernel]["device_id"] = device_id;

            timestamps[kernel]["write"] = nullptr;
            timestamps[kernel]["ndrange"] = nullptr;
            timestamps[kernel]["read"] = nullptr;

            timestamps[kernel]["write"]["host_queued_start"] =
                k->host_events.get_timestamp("write", "start");
            timestamps[kernel]["write"]["host_queued_end"] =
                k->host_events.get_timestamp("write", "end");

            timestamps[kernel]["write"]["device_queued"] =
                k->command_events.get_min_timestamp("write", "queued");
            timestamps[kernel]["write"]["device_start"] =
                k->command_events.get_min_timestamp("write", "start");
            timestamps[kernel]["write"]["device_end"] =
                k->command_events.get_max_timestamp("write", "end");

            timestamps[kernel]["ndrange"]["device_start"] =
                k->command_events.get_min_timestamp("ndrange", "start");
            timestamps[kernel]["ndrange"]["device_end"] =
                k->command_events.get_max_timestamp("ndrange", "end");

            timestamps[kernel]["read"]["device_start"] =
                k->command_events.get_min_timestamp("read", "start");
            timestamps[kernel]["read"]["device_end"] =
                k->command_events.get_max_timestamp("read", "end");

#if SCHEDULE_LOG
            std::cout << "  " << k->get_id() << " : " << k->get_name()
                      << " executing on platform " << p << " and device "
                      << device_id << "\n";

            std::cout << "    host queued events --> Start: "
                      << timestamps[kernel]["write"]["host_queued_start"]
                      << " End: "
                      << timestamps[kernel]["write"]["host_queued_end"] << "\n";

            std::cout << "    write events --> Start: "
                      << timestamps[kernel]["write"]["device_start"]
                      << " End: " << timestamps[kernel]["write"]["device_end"]
                      << "\n";
            std::cout << "    ndrange events --> Start: "
                      << timestamps[kernel]["ndrange"]["device_start"]
                      << " End: " << timestamps[kernel]["ndrange"]["device_end"]
                      << "\n";
            std::cout << "    read Events --> Start: "
                      << timestamps[kernel]["read"]["device_start"]
                      << " End: " << timestamps[kernel]["read"]["device_end"]
                      << "\n";
#endif
        }
    }
    std::ofstream o(filename);
    o << timestamps << std::endl;
}

void ScheduleEngine::profile_task_component(
    DAG *dag, TaskComponent *tc, std::string &&filename = "timing.json") {
    json timestamps;
    std::ifstream input_file;
    input_file.open(filename);
    if (input_file) {
        input_file >> timestamps;
        input_file.close();
    }
    std::map<Vendor, unsigned long long int> ref;
    for (auto kernel_id : tc->kernel_ids) {
        Kernel *k = dag->id_to_kernel_map[kernel_id];
        Vendor p = k->platform;
        unsigned long long int min_start_time =
            k->command_events.get_min_start_time();
        if (ref.find(p) == ref.end())
            ref[p] = min_start_time;
        else
            ref[p] = std::min(ref[p], min_start_time);
    }
    for (auto kernel_id : tc->kernel_ids) {
        Kernel *k = dag->id_to_kernel_map[kernel_id];
        Vendor p = k->platform;
        int device_id = k->device_id;
        std::string kernel = tc->get_kernel_ids_string() + "_" + k->get_id();

        timestamps[kernel] = nullptr;
        timestamps[kernel]["dev_type"] = p;
        timestamps[kernel]["device_id"] = device_id;

        timestamps[kernel]["write"] = nullptr;
        timestamps[kernel]["ndrange"] = nullptr;
        timestamps[kernel]["read"] = nullptr;

        timestamps[kernel]["write"]["host_queued_start"] =
            k->host_events.get_timestamp("write", "start");
        timestamps[kernel]["write"]["host_queued_end"] =
            k->host_events.get_timestamp("write", "end");

        timestamps[kernel]["write"]["device_queued"] =
            k->command_events.get_min_timestamp("write", "queued");
        timestamps[kernel]["write"]["device_start"] =
            k->command_events.get_min_timestamp("write", "start");
        timestamps[kernel]["write"]["device_end"] =
            k->command_events.get_max_timestamp("write", "end");

        timestamps[kernel]["ndrange"]["device_start"] =
            k->command_events.get_min_timestamp("ndrange", "start");
        timestamps[kernel]["ndrange"]["device_end"] =
            k->command_events.get_max_timestamp("ndrange", "end");

        timestamps[kernel]["read"]["device_start"] =
            k->command_events.get_min_timestamp("read", "start");
        timestamps[kernel]["read"]["device_end"] =
            k->command_events.get_max_timestamp("read", "end");

        std::cout << "  " << k->get_id() << " : " << k->get_name()
                  << " executing on platform " << p << " and device "
                  << device_id << "\n";

        std::cout << "    host queued events --> Start: "
                  << timestamps[kernel]["write"]["host_queued_start"]
                  << " End: " << timestamps[kernel]["write"]["host_queued_end"]
                  << "\n";

        std::cout << "    write events --> Start: "
                  << timestamps[kernel]["write"]["device_start"]
                  << " End: " << timestamps[kernel]["write"]["device_end"]
                  << "\n";
        std::cout << "    ndrange events --> Start: "
                  << timestamps[kernel]["ndrange"]["device_start"]
                  << " End: " << timestamps[kernel]["ndrange"]["device_end"]
                  << "\n";
        std::cout << "    read Events --> Start: "
                  << timestamps[kernel]["read"]["device_start"]
                  << " End: " << timestamps[kernel]["read"]["device_end"]
                  << "\n";
        auto start_time =
            timestamps[kernel]["ndrange"]["device_start"].get<long long int>() -
            ref[p];
        auto end_time =
            timestamps[kernel]["ndrange"]["device_end"].get<long long int>() -
            ref[p];

        if (p == Vendor::ARM_GPU) {
            auto write_stime = timestamps[kernel]["write"]["device_start"]
                                   .get<long long int>() -
                               ref[p];
            auto write_etime =
                timestamps[kernel]["write"]["device_end"].get<long long int>() -
                ref[p];
            auto read_stime = timestamps[kernel]["read"]["device_start"]
                                  .get<long long int>() -
                              ref[p];
            auto read_etime =
                timestamps[kernel]["read"]["device_end"].get<long long int>() -
                ref[p];
            std::cout << "PROFILE_TC " << tc->get_kernel_ids_string() << "_"
                      << k->get_id() << " Write: " << write_stime << "-->"
                      << write_etime << ": " << write_etime - write_stime
                      << "\n";
            std::cout << "PROFILE_TC " << tc->get_kernel_ids_string() << "_"
                      << k->get_id() << " Read: " << read_stime << "-->"
                      << read_etime << ": " << read_etime - read_stime << "\n";
        }

        std::cout << "PROFILE_TC " << tc->get_kernel_ids_string() << "_"
                  << k->get_id() << " Execute:" << start_time << "-->"
                  << end_time << ": " << end_time - start_time << "\n";
        k->command_events.clear();
        k->host_events.clear();
    }
    std::ofstream o(filename.c_str());
    o << timestamps << std::endl;
}

void ScheduleEngine::initialize_frontier_dag(DAG *d) {
    for (auto v : d->vertices) {
        Kernel *k = d->id_to_kernel_map[v];
        if (k->is_ready_for_dispatch()) {
            frontier.push(k);
            k->in_frontier = true;
#if SCHEDULE_LOG

            printf("Pushing kernel %d of DAG %s instance %d to frontier\n",
                   k->id, d->get_name().c_str(), k->instance_id);
#endif
        }
    }
}

void ScheduleEngine::dag_stream() {
    // std::cout << "Initiating dag stream\n";
    int current_dag_index = 0;
    while (current_dag_index < arriving_dags.size() &&
           !arriving_dags[current_dag_index]->arrival_time) {
        initialize_frontier_dag(arriving_dags[current_dag_index++]);
    }
    while (current_dag_index < arriving_dags.size()) {
        DAG *d = arriving_dags[current_dag_index++];
        clock_nanosleep(CLOCK_REALTIME, 0, &d->arrival_timestamp, NULL);
        initialize_frontier_dag(d);
    }
#if SCHEDULE_LOG
    std::cout << "Finished adding dags\n";
#endif
}

void ScheduleEngine::dispatch_task_component(TaskComponent *tc,
                                             Vendor platform) {
    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running scheduler thread %lu on core %d \n", pthread_self(),
           sched_getcpu());

    tc->initiate_dispatch(platform, 0);
    tc->dispatch_single(ctxs[platform], cmd_qs[platform][0], this);

    while (!tc->is_finished()) {
        sleep(1);
        printf("Waiting for tc to be finished\n");
    }
}
void ScheduleEngine::dispatch_sim_instance(SimInstance *sim_instance,
                                           Vendor platform) {
    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running scheduler thread %lu on core %d \n", pthread_self(),
           sched_getcpu());
    if (sim_instance->task_component.size() > 1) {
        TaskComponent *tc = this->create_task_component(
            sim_instance->task_component, this->ctxs[sim_instance->platform],
            sim_instance->platform, this->id_to_dag_map[sim_instance->dag_id]);
        tc->initialize_free_kernels();
        tc->initiate_dispatch(platform, 0);
        tc->dispatch_single(ctxs[platform], cmd_qs[platform][0], this);
    } else {
        auto k = this->id_to_dag_map[sim_instance->dag_id]
                     ->id_to_kernel_map[sim_instance->task_component[0]];
        k->initiate_dispatch(platform, 0);
        UserArgs *user_data = new UserArgs(k, this);
        cl_event ev = k->dispatch_with_events_callback(platform, ctxs[platform],
                                                       cmd_qs[platform][0]);
        clSetEventCallback(ev, CL_COMPLETE, &dag_callback, (void *)user_data);
        clFlush(cmd_qs[platform][0]);
    }
}

void ScheduleEngine::dispatch_task_component_with_inteference(
    TaskComponent *tc, Vendor platform, Kernel *k,
    Vendor platform_interference) {
    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running scheduler thread %lu on core %d \n", pthread_self(),
           sched_getcpu());

    tc->initiate_dispatch(platform, 0);
    tc->dispatch_single(ctxs[platform], cmd_qs[platform][0], this);
    k->initiate_dispatch(platform_interference, 0);
    UserArgs *user_data = new UserArgs(k, this);
    cl_event ev = k->dispatch_with_events_callback(platform, ctxs[platform],
                                                   cmd_qs[platform][0]);
    clSetEventCallback(ev, CL_COMPLETE, &kernel_callback, (void *)user_data);
    clFlush(cmd_qs[platform_interference][0]);

    while (!tc->is_finished()) {
        sleep(1);
        printf("Waiting for tc to be finished\n");
    }
}

void ScheduleEngine::cluster_schedule(int lookahead_depth) {
    // std::cout << "Initiating cluster scheduler\n";
    /*
    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running scheduler thread %lu on core %d \n", pthread_self(),
           sched_getcpu());
           */
    while (completed_jobs.load() < num_jobs) {

        Kernel *k = NULL;
        Vendor platform;
        int device_position;
        bool schedule = true;
        frontier.wait_and_pop(&k);
        if (k == NULL)
            continue;
        if (k->has_dispatched) {
            schedule = false;
            // frontier.wait_and_pop(&k);
            k->in_frontier = false;
        } else
            schedule = true;
        if (schedule) {
            std::map<Vendor, std::deque<int>> available_devices;
            int num_available_devices;
            {
                if (profile_engine)
                    FunctionTimer("ReadyQueue->GetAvailableDevices");
                num_available_devices =
                    ready_queue.get_available_devices(available_devices);
            }

            std::vector<Chain *> chain_set;

            DAG *d = (DAG *)k->dag_ptr;
#if SCHEDULE_LOG
            std::cout << "Available devices from schedule main loop\n";
            d->print_available_devices(available_devices);
#endif
            //          printf("NUM AVAILABLE DEVICES: %d REQUIRED
            //          PREDECESSOR CHAIN AVAILABILITYL
            //          %d\n",num_available_devices,k->required_predecessors);
            if (num_available_devices > k->required_predecessors)
                d->construct_chains(k->id, chain_set, available_devices,
                                    lookahead_depth);
            if (chain_set.size() == 0) {

                // frontier.wait_and_pop(&k);
                platform = k->preferred;
                // std::cout << __FILE__ << " " << __LINE__
                //         << ": Searching for device of platform " <<
                //         platform
                //       << "\n";

                ready_queue.wait_and_pop(platform, &device_position);
#if SCHEDULE_LOG
                std::cout << "Obtained device from ready_queue " << platform
                          << ": " << device_position << "\n";
#endif
                k->initiate_dispatch(platform, device_position);
                UserArgs *user_data = new UserArgs(k, this);
                k->setup(platform, ctxs[platform], gpu_buf_man);
                cl_event ev = k->dispatch_with_events_callback(
                    platform, ctxs[platform],
                    cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
                std::cout << "Dispatched kernel with id " << k->id << "of DAG "
                          << d->id << " to device " << device_position
                          << "of platform " << platform << "\n";
#endif
                clSetEventCallback(ev, CL_COMPLETE, &dag_callback,
                                   (void *)user_data);
#if SCHEDULE_LOG
                std::cout << "Callback Set\n";
#endif
      
                {
                   if(profile_engine) FunctionTimer("OCLDelay->Flush");
                    clFlush(cmd_qs[platform][device_position]);
                }
            } else {
                for (auto c : chain_set) {
                    if(profile_engine) c->print_information();
                    if (c->is_singleton()) {

                        platform = c->platform;
                        device_position = c->device_id;
                        Kernel *k_prime = c->chain_kernels[0];
                        if (k_prime->in_frontier && k_prime != k)
                            frontier.wait_and_remove(k_prime);
                        ready_queue.wait_and_remove(platform, c->device_id,
                                                    &device_position);
#if SCHEDULE_LOG
                        std::cout << "Obtained device from ready_queue "
                                  << platform << ": " << device_position
                                  << "\n";
#endif

                        k_prime->initiate_dispatch(platform, device_position);
                        k_prime->setup(platform, ctxs[platform], gpu_buf_man);
                        if (k_prime->is_ready_for_dispatch()) {
                            UserArgs *user_data = new UserArgs(k_prime, this);
                            cl_event ev =
                                k_prime->dispatch_with_events_callback(
                                    platform, ctxs[platform],
                                    cmd_qs[platform][device_position]);
#if SCHEDULE_LOG
                            std::cout << "Dispatched kernel with id (singleton "
                                         "cluster)"
                                      << k_prime->id << " of DAG " << d->id
                                      << " to device" << device_position
                                      << "of platform " << platform << "\n";
#endif
                            clSetEventCallback(ev, CL_COMPLETE, &dag_callback,
                                               (void *)user_data);
#if SCHEDULE_LOG
                            std::cout << "Callback Set\n";
#endif
                            {
                                if(profile_engine) FunctionTimer("OCLDelay->Flush");
                                clFlush(cmd_qs[platform][device_position]);
                            }
                        }

                    } else {
                        platform = c->platform;
                        device_position = c->device_id;
#if SCHEDULE_LOG
                        std::cout << "Obtained device from ready_queue "
                                  << platform << ": " << device_position
                                  << "\n";
#endif
                        for (auto kernel_in_chain : c->chain_kernels) {
                            if (kernel_in_chain->in_frontier &&
                                kernel_in_chain != k)
                                frontier.wait_and_remove(kernel_in_chain);
                        }
                        ready_queue.wait_and_remove(platform, c->device_id,
                                                    &device_position);
                        TaskComponent *tc = create_task_component_from_chain(
                            c, ctxs[platform], d);
                        tc->initiate_dispatch(platform, device_position);
                        if (tc->has_free_kernels()) {
                            tc->dispatch_single(
                                ctxs[platform],
                                cmd_qs[platform][device_position], this);
#if SCHEDULE_LOG

                            std::cout << "Dispatched task component with id "
                                      << tc->task_id << " of DAG " << d->id
                                      << " ";
                            std::cout << "containing kernels ";
                            for (auto kid : tc->kernel_ids)
                                std::cout << kid << " ";
                            std::cout << "to device" << device_position
                                      << "of platform " << platform << "\n";
#endif
                        }
                    }
                    delete c;
                }
            }
        }
    }
    //    std::cout << "Finished cluster scheduler\n";
}

void ScheduleEngine::parse_arrival_times(const char *filename) {
    std::string file{filename};
    std::ifstream stream(file);
    std::cout << "Opening " << file << "\n";
    std::string line;
    while (std::getline(stream, line)) {
        // std::cout << "Parsing " << line << "\n";
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(':');
        long long int arrival_time = std::stoll(ssublist[0]);
        struct timespec arrival_timestamp;
        arrival_timestamp.tv_sec = 0;
        arrival_timestamp.tv_nsec = 0;
        int sec, nanosec;
        sec = (int)((float)arrival_time / 1000000000);
        if (sec > 0)
            nanosec = arrival_time - 1000000000 * sec;
        else
            nanosec = arrival_time;
        arrival_timestamp.tv_nsec = (int)nanosec;
        arrival_timestamp.tv_sec = (int)sec;

        // std::cout << "Arrival Time: " << arrival_time << "\n";
        sleep_times.push_back(arrival_timestamp);
        splitstring dag_instances_string((char *)ssublist[1].c_str());
        std::vector<std::string> dag_instances =
            dag_instances_string.split(' ');
        std::vector<std::pair<int, int>> dag_instance_id_pairs;
        for (auto x : dag_instances) {
            //  std::cout << x << "\n";
            splitstring ids_string((char *)x.c_str());
            std::vector<std::string> ids = ids_string.split(',');
            //    std::cout << ids[0] << " " << ids[1] << "\n";
            int dag_id = std::stoi(ids[0]);
            int instance_id = std::stoi(ids[1]);
            dag_instance_id_pairs.push_back(
                std::make_pair(dag_id, instance_id));
        }
        arriving_dag_instances.push_back(dag_instance_id_pairs);
    }
}
void ScheduleEngine::parse_deadlines(int num_jobs, const char *filename) {
    std::string file{filename};
    std::ifstream stream(file);
    //    std::cout << "Opening " << file << "\n";
    std::string line;
    for (unsigned int i = 0; i < num_jobs; i++) {
        this->dag_instance_deadlines.emplace(i, std::map<int, float>());
    }
    int total_num_instances = 0;
    while (std::getline(stream, line)) {
        //      std::cout<<"Parsing line "<<line<<"\n";
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(' ');
        int dag_id = std::stoi(ssublist[0]);
        int instance_id = std::stoi(ssublist[1]);
        float deadline = std::stof(ssublist[2]);
        this->dag_instance_deadlines[dag_id][instance_id] = deadline;
        total_num_instances++;
    }
    frontier.init_num_jobs(total_num_instances);
    this->num_jobs = total_num_instances;
}
void ScheduleEngine::print_arrival_info() {
    for (int i = 0; i < sleep_times.size(); i++) {
        std::cout << "Sleep: " << sleep_times[i].tv_sec << " "
                  << sleep_times[i].tv_nsec << "\n";
        for (auto x : arriving_dag_instances[i]) {
            std::cout << x.first << " " << x.second << "\n";
        }
    }
}

void ScheduleEngine::print_dag_instance_mapping_info() {
    printf("Task Component Mapping Decisions for all DAG Instances\n");
    for (auto x : dag_instance_config_map) {
        printf("-----------------------------------\n");
        std::cout << "DAG " << x.first << "\n";
        for (auto y : x.second) {
            std::cout << "Instance " << y.first << "\n";
            auto ic = y.second;
            printf("================================\n");
            ic->print_instance_info();
            printf("================================\n");
        }
        printf("------------------------------------\n");
    }
}

void ScheduleEngine::print_deadlines_of_dag_instances() {
    printf("Deadline Information\n");
    for (auto x : dag_instance_deadlines) {
        auto deadline_instance = x.second;
        for (auto y : deadline_instance)
            std::cout << x.first << " " << y.first << " " << y.second << "\n";
    }
}

void ScheduleEngine::adas_stream() {
    cpu_set_t cpuset;
    pthread_t thread_id_scheduler = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset); // TODO: Set according to your platform
    pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);
    printf("Running adas streaming thread %lu on core %d \n", pthread_self(),
           sched_getcpu());
    int arrival_index = 0;
    int num_arrival_instances = arriving_dag_instances.size();
    while (arrival_index < num_arrival_instances) {
        clock_nanosleep(CLOCK_REALTIME, 0, &sleep_times[arrival_index], NULL);
        for (auto x : this->arriving_dag_instances[arrival_index]) {
            auto dag_id = x.first;
            auto instance_id = x.second;
            InstanceConfig *ic =
                this->dag_instance_config_map[dag_id][instance_id];
            auto dag = id_to_dag_map[dag_id];
            //     std::cout << "Arrived: " << dag_id << " " << instance_id
            //     <<
            //     "\n";
            if (!dag->running) {
                //         printf("Arrival Stream ");
                setup_new_instance(dag, ic);
            } else {
                dag->instance_queue.push(ic);
                dag->deadlines_missed++;
            }
        }
        arrival_index++;
    }
}

void ScheduleEngine::setup_new_instance(DAG *d, InstanceConfig *ic) {

    // struct FunctionTimer f("Setup instance");
    d->reset_dag_parameters();
    // std::cout << "Instance setup " << ic->dag_id << " " <<
    // ic->instance_id
    //         << "\n";
    for (unsigned int i = 0; i < ic->task_components.size(); i++) {
        Vendor p = ic->devices[i];
        if (ic->task_components[i].size() > 1)
            create_task_component(ic->task_components[i], this->ctxs[p], p, d);
        else {
            int kid = ic->task_components[i][0];
            d->id_to_kernel_map[kid]->set_preferred(
                p); // set assigned device as well -->schedule_adas_dags()
        }
    }
    d->update_instance_id_for_kernels(ic->instance_id);
    d->update_local_deadline(
        dag_instance_deadlines[ic->dag_id][ic->instance_id]);
    initialize_frontier_dag(d);
    d->running = true;
}

void ScheduleEngine::initialize_adas_instance_config(int num_dags,
                                                     const char *filename) {
    for (int i = 0; i < num_dags; i++) {
        this->dag_instance_config_map.emplace(
            i, std::map<int, InstanceConfig *>());
    }
    std::ifstream stream(filename);
    std::string line;
    while (std::getline(stream, line)) {
#if DAG_LOG
        std::cout << "parsing " << line << "\n";
#endif
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(' ');
        int dag_id = std::stoi(ssublist[0]);
        int instance_id = std::stoi(ssublist[1]);
        if (dag_instance_config_map[dag_id].find(instance_id) ==
            dag_instance_config_map[dag_id].end()) {
            // Not Present --> Initialize InstanceConfig
            InstanceConfig *ic = new InstanceConfig(dag_id, instance_id,
                                                    ssublist[2], ssublist[3]);
            dag_instance_config_map[dag_id].emplace(instance_id, ic);
        } else {
            // Present --> Add to existing InstanceConfig
            dag_instance_config_map[dag_id][instance_id]->add(ssublist[2],
                                                              ssublist[3]);
        }
    }
}

void ScheduleEngine::parse_dispatch_decisions(const char *filename) {
    std::ifstream stream(filename);
    std::string line;
    while (std::getline(stream, line)) {
#if DAG_LOG
        std::cout << "parsing " << line << "\n";
#endif
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(' ');
        int dag_id = std::stoi(ssublist[0]);
        int instance_id = std::stoi(ssublist[1]);
        SimInstance *sim_instance =
            new SimInstance(dag_id, instance_id, ssublist[2], ssublist[3]);
        sim_instances.push_back(sim_instance);
        break;
    }
}
void ScheduleEngine::parse_speedup_table(const char *filename) {
    this->speedup_to_frequency_map.emplace(Vendor::ARM_CPU,
                                           std::map<float, unsigned int>());
    this->speedup_to_frequency_map.emplace(Vendor::ARM_GPU,
                                           std::map<float, unsigned int>());
    std::ifstream stream(filename);
    std::string line;
    while (std::getline(stream, line)) {
#if DAG_LOG
        std::cout << "parsing " << line << "\n";
#endif
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(',');
        int platform_id = std::stoi(ssublist[0]);
        unsigned int freq = std::stoi(ssublist[1]);
        float speedup = std::stof(ssublist[2]);
        if (platform_id == 0)
            this->speedup_to_frequency_map[Vendor::ARM_GPU][speedup] = freq;
        else
            this->speedup_to_frequency_map[Vendor::ARM_CPU][speedup] = freq;
    }
}

void ScheduleEngine::print_sim_instance_info() {
    for (auto i : sim_instances)
        i->print_info();
}
void ScheduleEngine::schedule_adas_dags() {
    std::cout << "Initiating ADAS Scheduling Loop\n";

    while (completed_jobs.load() < num_jobs) {
#if SCHEDULE_LOG
        std::cout << "Number of DAGS: " << num_jobs
                  << " Completed DAGs: " << completed_jobs << "\n";
//        std::cout << "Obtaining kernel from frontier\n";
#endif

        int device_position = -1;
        Vendor platform;
        Kernel *k = NULL;
        TaskComponent *tc = NULL;

        frontier.wait_and_pop(&k);
        if (k == NULL)
            continue;

        if (k->task_id != -1) {
            tc = (TaskComponent *)k->task_ptr;
            platform = tc->preferred;
        } else {
            platform = k->preferred;
        }
#if SCHEDULE_LOG
        //      std::cout << "Obtained kernel from frontier\n";
        //     std::cout << "Obtaining device from ready_queue\n";
#endif
        ready_queue.wait_and_pop(platform, &device_position);
#if SCHEDULE_LOG
        //   std::cout << "Obtained device from ready_queue\n";
#endif
        if (k->task_id == -1) {
            k->initiate_dispatch(platform, device_position);
            UserArgs *user_data = new UserArgs(k, this);
            cl_event ev = k->dispatch_with_events_callback(
                platform, ctxs[platform], cmd_qs[platform][device_position]);
            clSetEventCallback(ev, CL_COMPLETE, &dag_callback,
                               (void *)user_data);
#if SCHEDULE_LOG
            DAG *d = (DAG *)k->dag_ptr;
            std::cout << "Dispatched kernel with id " << k->id << "of DAg "
                      << d->get_id() << " " << d->current_instance
                      << " to device " << device_position << "of platform "
                      << platform << "\n";
#endif
#if SCHEDULE_LOG
            // std::cout << "Callback Set\n";

#endif
            clFlush(cmd_qs[platform][device_position]);

        } else {
            tc->initiate_dispatch(platform, device_position);
            tc->initialize_free_kernels();
            tc->dispatch_single(ctxs[platform],
                                cmd_qs[platform][device_position], this);
#if SCHEDULE_LOG
            std::cout << "Dispatched task component with id" << tc->task_id
                      << "to device " << device_position << "of platform "
                      << platform << "\n";
#endif
        }
    }
    std::cout << "Scheduling of dags finished\n";
}

void ScheduleEngine::profile_adas_dags(std::string &&filename = "timing.json") {
    std::map<Vendor, unsigned long long int> ref;
    for (auto d : id_to_dag_map) {
        DAG *dag = d.second;
        for (auto v : dag->vertices) {
            Kernel *k = dag->id_to_kernel_map[v];
            std::cout << "Profiling for " << k->get_id() << " of DAG "
                      << dag->get_id() << "\n";
            k->command_events.dump_json(filename);
        }
    }
}

void ScheduleEngine::change_frequency(unsigned int frequency, Vendor platform) {

    char cmd[200];
    int ret;
    switch (platform) {
    case Vendor::ARM_GPU: {
        // devn="GPU";
        char line[1024];
        unsigned int freq;
        FILE *fp;
        fp = fopen(
            "/sys/devices/platform/11800000.mali/devfreq/devfreq0/cur_freq",
            "r"); // read mode
        if (fp == NULL)
            exit(EXIT_FAILURE);
        char *tmp;
        while (fgets(line, 1024, fp)) {
            // tmp = strdup(line);
            tmp = (char *)calloc(strlen(line) + 1, 1);
            strcpy(tmp, line);
            // printf("change_frequency cur: %s freq: %u\n",tmp,frequency);
        }
        fclose(fp);
        char *tok;
        tok = strtok(tmp, "\n");
        freq = (unsigned int)atoi(tok);
        if (frequency >= freq) {
            sprintf(cmd,
                    "echo %u > "
                    "/sys/devices/platform/11800000.mali/devfreq/devfreq0/"
                    "max_freq ",
                    frequency);
            ret = system(cmd);

            sprintf(cmd,
                    "echo %u > "
                    "/sys/devices/platform/11800000.mali/devfreq/devfreq0/"
                    "min_freq ",
                    frequency);
            ret = system(cmd);
        } else {
            sprintf(cmd,
                    "echo %u > "
                    "/sys/devices/platform/11800000.mali/devfreq/devfreq0/"
                    "min_freq ",
                    frequency);
            ret = system(cmd);

            sprintf(cmd,
                    "echo %u > "
                    "/sys/devices/platform/11800000.mali/devfreq/devfreq0/"
                    "max_freq ",
                    frequency);
            ret = system(cmd);
        }
    }
    case Vendor::ARM_CPU: {
        int core = 4;
        char line[1024];
        unsigned int freq;
        FILE *fp;
        std::string str = "/sys/devices/system/cpu/cpu";
        str = str + std::to_string(core) + "/cpufreq/scaling_cur_freq";
        fp = fopen(str.c_str(), "r"); // read mode
        if (fp == NULL)
            exit(EXIT_FAILURE);
        char *tmp;
        while (fgets(line, 1024, fp)) {
            tmp = (char *)calloc(strlen(line) + 1, 1);
            strcpy(tmp, line);
        }
        fclose(fp);

        char *tok;
        tok = strtok(tmp, "\n");
        freq = (unsigned int)atoi(tok);

        if (freq != frequency) {
            sprintf(cmd, "cpufreq-set -c %d --max %uHz --min %uHz ", core,
                    frequency, frequency);
            ret = system(cmd);
        }
    }
    }
}
