#include "DAG.h"
extern std::string database_dir;

bool InstanceQueue::empty() {
    std::lock_guard<std::mutex> lk(mut);
    return q.empty();
}

void InstanceQueue::push(InstanceConfig *new_value) {
    std::lock_guard<std::mutex> lk(mut);
    q.push(new_value);
    cond.notify_all();
}

void InstanceQueue::wait_and_pop(InstanceConfig **value) {
    *value = NULL;
    std::unique_lock<std::mutex> lk(mut);
    cond.wait(lk, [this] { return (!q.empty()); });
    *value = q.front();
    q.pop();
}

void Chain::calculate_gain(Kernel *k, bool merge = false) {
    assert(chain_kernels.size() > 0);
    if (k->get_preferred_type() == "cpu")
        expected_baseline = baseline + k->cpu_time;
    else
        expected_baseline = baseline + k->gpu_time + k->h2d_time + k->d2h_time;

    if (successor) {

        expected_total_cpu_time = total_cpu_time + k->cpu_time;
        expected_total_gpu_time =
            total_gpu_time + k->gpu_time + k->d2h_time -
            chain_kernels[chain_kernels.size() - 1]->d2h_time;
    }

    if (predecessor && !merge) {

        expected_total_cpu_time = total_cpu_time + k->cpu_time;
        expected_total_gpu_time =
            total_gpu_time + k->gpu_time + k->h2d_time -
            chain_kernels[chain_kernels.size() - 1]->h2d_time;
    }

    if (predecessor && merge) {

        expected_total_cpu_time = total_cpu_time + k->cpu_time;
        expected_total_gpu_time = total_gpu_time + k->gpu_time + k->d2h_time -
                                  chain_kernels[0]->d2h_time;
    }

    expected_gain_cpu = expected_baseline / expected_total_cpu_time;
    expected_gain_gpu = expected_baseline / expected_total_gpu_time;
}

bool Chain::contains_kernel(Kernel *k) {
    if (std::find(chain_kernels.begin(), chain_kernels.end(), k) !=
        chain_kernels.end())
        return true;
    else
        return false;
}

void Chain::add_kernel(Kernel *k) {

    assert(contains_kernel(k) != true);
    chain_kernels.push_back(k);
    total_cpu_time = expected_total_cpu_time;
    total_gpu_time = expected_total_gpu_time;
    baseline = expected_baseline;
    gain_cpu = expected_gain_cpu;
    gain_gpu = expected_gain_gpu;
    k->chain_id = id;
}

bool Chain::contains_frontier_kernel() {
    for (auto k : chain_kernels)
        if (k->in_frontier)
            return true;
    return false;
}
void Chain::remove_last_kernel() {
#if CHAIN_LOG
    std::cout << "REVERTING: Removing from chain " << this->id
              << chain_kernels[chain_kernels.size() - 1]->get_id() << "\n";
#endif
    chain_kernels[chain_kernels.size() - 1]->chain_id = -1;
    chain_kernels.pop_back();
}
bool Chain::is_singleton() { return chain_kernels.size() == 1; }

bool Chain::assign_preferred_device(
    std::map<Vendor, std::deque<int>> &available_devices) {
    bool device_available = false;
    assert(chain_kernels.size() > 0);
    Kernel *k = chain_kernels[0];
    for (auto x : available_devices) {
        if (x.first == k->preferred)
            if (x.second.size() > 0) {
                device_available = true;
                platform = x.first;
                device_id = x.second.front();
                x.second.pop_front();
                break;
                device_assigned = true;
            }
    }
    return device_available;
}

void Chain::assign_device(
    std::pair<Vendor, int> device,
    std::map<Vendor, std::deque<int>> &available_devices) {

    platform = device.first;
    device_id = device.second;
    device_assigned = true;
}

bool is_there_any_device(std::map<Vendor, std::deque<int>> &available_devices) {
    bool flag = false;
    for (auto x : available_devices) {
        if (x.second.size() > 0)
            flag = flag | true;
    }
    if (!flag)
        std::cout << "No devices are available\n";
    return flag;
}

int get_number_of_devices(
    std::map<Vendor, std::deque<int>> &available_devices) {
    int num_devices = 0;
    for (auto x : available_devices) {
        num_devices += x.second.size();
    }
    return num_devices;
}

std::pair<Vendor, int> Chain::get_assigned_device() {
    assert(device_assigned);
    return std::make_pair(platform, device_id);
}

void Chain::print_information() {
    printf("=============================\n");
    std::cout << "Chain identifier: " << id << "\n";
    assert(successor != predecessor);
    if (successor)
        std::cout << "Chain Type: "
                  << "successor chain\n";
    if (predecessor)
        std::cout << "Chain Type: "
                  << "predecessor chain\n";
    std::cout << "Constituent kernels: ";
    for (auto k : chain_kernels)
        std::cout << k->id << " ";
    printf("\n");
    if (device_id == -1)
        std::cout << "Device not assigned\n";
    else
        std::cout << "Device assigned-> Platform: " << platform
                  << " Device: " << device_id << "\n";

    printf("=============================\n");
}
void DAG::print_dag_info() {
    std::cout << "Kernel in DAG\n";
    for (auto v : vertices) {
        std::cout << "kernel id " << v << "\n";
        id_to_kernel_map[v]->print_kernel_info();
    }
    std::cout << "edges in DAG\n";
    std::cout << "successor list\n";
    for (auto vertex : successors) {
        std::cout << vertex.first << ": ";
        for (auto v : vertex.second)
            std::cout << v << " ";
        std::cout << "\n";
    }
    std::cout << "predecessor list\n";
    for (auto vertex : predecessors) {
        std::cout << vertex.first << ": ";
        for (auto v : vertex.second)
            std::cout << v << " ";
        std::cout << "\n";
    }
}

void DAG::create_edges() {
    for (auto v : vertices) {
        successors.emplace(v, std::vector<int>());
        predecessors.emplace(v, std::vector<int>());
    }
    for (auto e : edges) {
        successors[e->source_kernel].push_back(e->destination_kernel);
        predecessors[e->destination_kernel].push_back(e->source_kernel);
        Kernel *destination_kernel = id_to_kernel_map[e->destination_kernel];
        destination_kernel->update_edgeinfo(e);
    }
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        k->num_parents = predecessors[v].size();
    }
}

void DAG::sort_neighbours() {

    for (auto &x : predecessors) {
        sort(x.second.begin(), x.second.end(),
             [this](const int &a, const int &b) -> bool {
                 return id_to_kernel_map[a]->rank > id_to_kernel_map[b]->rank;
             });
    }
    for (auto &x : successors) {
//        std::cout<<"Sorting for successors of "<<x.first<<"\n";
  //      for (auto y : x.second)
    //        std::cout << y << " ";
      //  printf("\n");

        sort(x.second.begin(), x.second.end(),
             [this](const int &a, const int &b) -> bool {
                 return id_to_kernel_map[a]->rank > id_to_kernel_map[b]->rank;
             });
        //for (auto y : x.second)
          //  std::cout << y << " ";
       // printf("\n");
    }
}
void DAG::precompute_ancestor_info() {
    std::queue<int> q;
    std::vector<bool> visited(vertices.size());
    for (auto x : predecessors) {
        if (x.second.size() == 0) {

            visited[x.first] = true;
            q.push(x.first);
        }
    }

    while (!q.empty()) {

        int v = q.front();
        q.pop();
        Kernel *k = id_to_kernel_map[v];
        for (auto s : successors[v]) {
            if (!visited[s]) {
                q.push(s);
                Kernel *s_k = id_to_kernel_map[s];
                if (predecessors[s].size() > 1) {
                    s_k->max_num_ancestors +=
                        predecessors[s].size() - 1 + k->max_num_ancestors;
                } else
                    s_k->max_num_ancestors = 1;

                k->required_predecessors =
                    std::max(k->required_predecessors, s_k->max_num_ancestors);
                visited[s] = true;
                //            printf("MAX PRED COUNT of kernels(k,s) (%d,%d) is
                //            (%d,%d)\n",k->id,s_k->id,k->required_predecessors,s_k->max_num_ancestors);
            }
        }
    }
}
void DAG::bind_host_array() {
    std::queue<int> q;
    std::vector<bool> visited(vertices.size());
#if DAG_LOG
    for (auto x : predecessors) {
        std::cout << x.first << ": ";
        for (auto y : x.second)
            std::cout << y << " ";
        std::cout << "\n";
    }
#endif

    for (auto x : predecessors) {
        if (x.second.size() == 0) {

            visited[x.first] = true;
            q.push(x.first);
        }
    }
    while (!q.empty()) {

        int v = q.front();
        q.pop();

        for (auto s : successors[v]) {
            if (!visited[s]) {
                q.push(s);
                Kernel *s_k = id_to_kernel_map[s];
                s_k->level++;
            }
            visited[s] = true;
            Kernel *destination_kernel = id_to_kernel_map[s];
            Kernel *source_kernel = id_to_kernel_map[v];
            for (auto e : destination_kernel->edge_info) {
                if (e->source_kernel == v) {
                    Buffer *source_buffer =
                        source_kernel->get_buffer_from_argument_position(
                            e->source_buffer);
                    if (!source_buffer->allocate_host_array)
                        source_buffer->allocate_host_array_for_buffer();
                    Buffer *destination_buffer =
                        destination_kernel->get_buffer_from_argument_position(
                            e->destination_buffer);
                    destination_buffer->associate_host_array_for_buffer(
                        source_buffer->host_array);
                    source_buffer->isolated = false;
                    destination_buffer->isolated = false;
                    source_buffer->life_cycle++;
                }
            }
        }
    }

    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        for (auto x : k->pos_to_buffer_map) {
            Buffer *buf = x.second;
            if (buf->isolated)
                buf->allocate_host_array_for_buffer();
        }
    }
#if DAG_LOG
    printf("Binding host arrays done\n");
#endif
}
void DAG::setup(
    std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
    std::unordered_map<Vendor, cl_context> &ctxs,
    std::vector<Vendor> &platforms, GpuBufferManager *GpuBufMan) {

    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        k->setup(all_devices, ctxs, platforms, GpuBufMan);
    }
}
void DAG::copy_persistent_buffers(
    std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
    std::unordered_map<Vendor, cl_context> &ctxs,
    std::vector<Vendor> &platforms,
    std::unordered_map<Vendor, std::vector<cl_command_queue>> &cmd_qs) {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        k->copy_persistent_buffers(all_devices, ctxs, platforms, cmd_qs);
    }
}

void DAG::build_kernels(
    std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
    std::unordered_map<Vendor, cl_context> &ctxs,
    std::vector<Vendor> &platforms) {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        k->build_kernel(all_devices, ctxs, platforms);
    }
}

void DAG::set_arrival_time(long long int arrival_time) {
    this->arrival_time = arrival_time;
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
    this->arrival_timestamp = arrival_timestamp;
}

void DAG::print_output() {
    for (auto v : vertices) {
        if (successors[v].size() == 0) {
            Kernel *output_kernel = id_to_kernel_map[v];
            output_kernel->print_result();
        }
    }
}

void DAG::print_all_inputs() {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        std::cout << "========================================\n";
        std::cout << "Inputs for kernel " << k->id << "\n";
        k->print_input();
        std::cout << "========================================\n";
    }
}

void DAG::print_all_outputs() {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        std::cout << "========================================\n";
        std::cout << "Inputs for kernel " << k->id << "\n";
        k->print_result();
        std::cout << "========================================\n";
    }
}

void DAG::print_buffer_host_information() {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];

        std::cout << "========================================\n";
        std::cout << "Kernel ID: " << k->id << "\n";
        for (auto b : k->pos_to_buffer_map) {
            Buffer *buf = b.second;
            std::cout << "Buffer at Position " << buf->get_pos_string()
                      << " has id " << buf->get_id()
                      << " and host array with id " << buf->host_array->get_id()
                      << "\n";
        }

        std::cout << "========================================\n";
    }
}

void DAG::print_buffer_host_information(Vendor p) {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];

        std::cout << "========================================\n";
        std::cout << "Kernel ID: " << k->id << "\n";
        for (auto b : k->pos_to_buffer_map) {
            Buffer *buf = b.second;
            std::cout << "Buffer at Position " << buf->get_pos_string()
                      << " has id " << buf->get_id()
                      << " and host array with id " << buf->host_array->get_id()
                      << "\n";
            buf->print_buffer_flags(p);
        }

        std::cout << "========================================\n";
    }
}

void DAG::print_profiling_information() {
    for (auto v : vertices) {
        Kernel *k = id_to_kernel_map[v];
        k->print_profiling_times();
    }
}

void DAG::print_arrival_time() {
    std::cout << arrival_timestamp.tv_sec << "s, " << arrival_timestamp.tv_nsec
              << "ns\n";
}

void DAG::topological_sort(std::vector<int> &topological_order) {
    std::vector<int> in_degree(vertices.size(), 0);
    for (auto v : vertices)
        in_degree[v] = predecessors[v].size();

    std::queue<int> q;
    for (int i = 0; i < in_degree.size(); i++)
        if (in_degree[i] == 0)
            q.push(i);
    int num_visited = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        topological_order.push_back(u);

        for (auto v : successors[u])
            if (--in_degree[v] == 0)
                q.push(v);

        num_visited++;
    }
    assert(num_visited == vertices.size());
}

void DAG::compute_bottom_level_rank() {
    std::vector<int> topological_order;
    topological_sort(topological_order);
    for (int v = topological_order.size() - 1; v >= 0; v--) {
        if (successors[v].size() == 0) {

            auto kernel_id = topological_order[v];
            Kernel *k = id_to_kernel_map[kernel_id];
            if (k->cpu_time <= k->gpu_time + k->h2d_time + k->d2h_time)
                k->rank = k->cpu_time;
            else
                k->rank = k->gpu_time;
        }
    }

    for (int v = topological_order.size() - 1; v >= 0; v--) {
        if (successors[v].size() > 0) {
            long long int max_rank = 0;
            auto kernel_id = topological_order[v];
            Kernel *k = id_to_kernel_map[kernel_id];
            if (k->cpu_time <= k->gpu_time + k->h2d_time + k->d2h_time)
                k->rank = k->cpu_time;
            else
                k->rank = k->gpu_time;
            max_rank = k->rank;
            for (auto s : successors[v]) {
                Kernel *succ = id_to_kernel_map[s];
                if (succ->rank + k->rank > max_rank)
                    max_rank = succ->rank + k->rank;
            }
            k->rank = max_rank;
        }
    }
}

void DAG::print_rank_information() {
    std::cout << "Ranks of kernels\n";
    for (auto x : id_to_kernel_map) {
        Kernel *k = x.second;
        std::cout << x.first << ": " << k->wcet_rank << "\n";
    }
}

void DAG::print_wcet_time() {
    float extime = 0.0;
    for (auto x : id_to_kernel_map) {
        Kernel *k = x.second;
        if (k->cpu_time > k->gpu_time + k->h2d_time + k->d2h_time)
            extime += k->cpu_time;
        else
            extime += k->gpu_time + k->h2d_time + k->d2h_time;
    }
    std::cout << "WCET time of " << this->get_name() << "is " << extime << "\n";
}

bool DAG::contains_device(
    std::string device_type,
    std::map<Vendor, std::deque<int>> &available_devices) {
    for (auto &x : available_devices)
        if (device_type == get_device_type(x.first) && x.second.size() > 0)
            return true;
    return false;
}

void DAG::add_device(std::pair<Vendor, int> device,
                     std::map<Vendor, std::deque<int>> &available_devices) {
    std::cout << "Adding device " << device.first << " " << device.second
              << "\n";

    available_devices[device.first].push_back(device.second);
}

std::pair<Vendor, int>
DAG::remove_device(std::string device_type,
                   std::map<Vendor, std::deque<int>> &available_devices) {
#if CHAIN_LOG
    std::cout << "Searching for " << device_type << "\n";
#endif
    for (auto &x : available_devices)
        if (device_type == get_device_type(x.first))
            if (!x.second.empty()) {
                int device_id = x.second.front();
                x.second.pop_front();
#if CHAIN_LOG
                std::cout << "Found Device " << device_type << " " << device_id
                          << "\n";
#endif
                return std::make_pair(x.first, device_id);
            }
}
void DAG::print_available_devices(
    std::map<Vendor, std::deque<int>> &available_devices) {
    std::cout << "Available devices\n";
    for (auto x : available_devices) {
        std::cout << "Platform" << x.first << ": ";
        for (auto d : x.second)
            std::cout << d << " ";
        std::cout << "\n";
    }
}
bool DAG::chain_growth(Chain *c, Kernel *successor_kernel,
                       std::vector<Chain *> &chain_set,
                       std::map<Vendor, std::deque<int>> &available_devices) {

    bool task_growth = false;
    if (successor_kernel->has_dispatched)
        return false;
    c->calculate_gain(successor_kernel);
    auto gain_cpu = c->expected_gain_cpu;
    auto gain_gpu = c->expected_gain_gpu;
#if CHAIN_LOG
    print_available_devices(available_devices);
    std::cout << " Gains (CPU,GPU): " << gain_cpu << " " << gain_gpu << "\n";
#endif
    if (gain_cpu < 1 && gain_gpu < 1)
        return false;
    if (gain_cpu > 1 && gain_gpu > 1) {
        if (gain_cpu > gain_gpu)
            task_growth = device_assignable(c, successor_kernel, chain_set,
                                            available_devices, "cpu");
        else

            task_growth = device_assignable(c, successor_kernel, chain_set,
                                            available_devices, "gpu");
    }

    if (!task_growth) {
        if (gain_cpu > 1)
            task_growth = device_assignable(c, successor_kernel, chain_set,
                                            available_devices, "cpu");

        if (gain_gpu > 1)
            task_growth = device_assignable(c, successor_kernel, chain_set,
                                            available_devices, "gpu");
    }
    if (task_growth) {
#if CHAIN_LOG
        std::cout << "Chain growth for chain " << c->id
                  << "is possible with kernel id " << successor_kernel->id
                  << "\n";

        print_available_devices(available_devices);
#endif

        c->add_kernel(successor_kernel);
    }
    return task_growth;
}

bool DAG::device_assignable(
    Chain *c, Kernel *successor_kernel, std::vector<Chain *> &chain_set,
    std::map<Vendor, std::deque<int>> &available_devices,
    const char *device_type) {

    bool task_growth = false;
    // Case 1: Device not assigned to chain c and available_devices contains
    // device of required type

    bool device_type_present = contains_device(device_type, available_devices);
    if (!c->device_assigned && device_type_present) {
        auto device = remove_device(device_type, available_devices);
#if CHAIN_LOG
        std::cout << "Case 1A: Beginning: ";
        print_available_devices(available_devices);
#endif
        // 1a) c is a successor chain
        if (c->successor) {
            if (predecessor_assignment(successor_kernel, chain_set,
                                       available_devices))
                task_growth = true;
        } else // 1b) c is a predecessor chain
            task_growth = true;

        if (task_growth)
            c->assign_device(device, available_devices);
        else
            add_device(device, available_devices);
#if CHAIN_LOG
        std::cout << "Case 1A: End:  ";
        print_available_devices(available_devices);
#endif

    } else if (c->device_assigned) // Case 2: Device assigned to chain c
    {
        auto device_prime = c->get_assigned_device();
#if CHAIN_LOG
        std::cout << "Case 2A: Beginning:  ";
        print_available_devices(available_devices);
#endif

        if (get_device_type(device_prime.first) !=
            device_type) { // 2a) assigned device is not of required type

            if (device_type_present) { // Verify that device is indeed present

                auto device = remove_device(device_type, available_devices);
                add_device(device_prime, available_devices);

                if (c->successor) { // 2a) i) c is a successor chain
                    if (predecessor_assignment(successor_kernel, chain_set,
                                               available_devices))
                        task_growth = true;
                } else // 2a ii) c is a predecessor chain
                    task_growth = true;

                if (task_growth) {
                    c->assign_device(device, available_devices);
                    add_device(device_prime, available_devices);
                }
            }
        } else {
            // 2b) assigned device is of requried type
#if CHAIN_LOG
            std::cout << "Case 2B: Beginning:  ";
            print_available_devices(available_devices);
#endif

            if (c->successor) { // 2b) i) c is a successor chain
                if (predecessor_assignment(successor_kernel, chain_set,
                                           available_devices))
                    task_growth = true;
            } else // 2b) ii) c is a predecessor chain
                task_growth = true;
        }
    }

    return task_growth;
}

void DAG::revert_clusters(std::vector<Chain *> &mergeable_chains,
                          std::vector<Kernel *> predecessors_merged) {

    for (auto k : predecessors_merged) {
        for (auto c : mergeable_chains) {
            if (c->contains_kernel(k)) {
                c->remove_last_kernel();
                break;
            }
        }
    }
}
bool DAG::predecessor_assignment(
    Kernel *successor_kernel, std::vector<Chain *> &chain_set,
    std::map<Vendor, std::deque<int>> &available_devices) {

    int kernel_id = successor_kernel->id;

    std::vector<Kernel *> current_predecessors_merged_in_chains;
    std::vector<Chain *> mergeable_chains;
    std::vector<Chain *> new_chains;
    // Predecessor belongs to the successor chain under consideration
#if CHAIN_LOG
    std::cout << "Number of predecessors of kernel " << kernel_id << " is "
              << predecessors[kernel_id].size() << "\n";
#endif
    if (predecessors[kernel_id].size() == 1)
        return true;
#if CHAIN_LOG
    std::cout << "predecessor assignment called for kernel "
              << successor_kernel->id << "\n";
#endif
    std::stack<Kernel *> ps;
    for (auto p : predecessors[kernel_id]) {
        Kernel *predecessor_kernel = id_to_kernel_map[p];
#if CHAIN_LOG
        std::cout << "Predecessor " << p << " Chain ID "
                  << predecessor_kernel->chain_id << " Has been dispatched? "
                  << predecessor_kernel->has_dispatched << "\n";
#endif
        if (predecessor_kernel->chain_id == -1 &&
            !predecessor_kernel->has_dispatched) {
#if CHAIN_LOG
            std::cout << "pushing to predecessor stack  "
                      << predecessor_kernel->id << "\n";
#endif

            ps.push(predecessor_kernel);
        }
    }
    while (!ps.empty()) {
        bool task_merge = false;
        Kernel *k = ps.top();
#if CHAIN_LOG
        std::cout << "Popping kernel " << k->id << "\n";
#endif
        ps.pop();
        for (auto p : predecessors[k->id]) {
#if CHAIN_LOG
            std::cout << "Investigating predecessor " << p << "\n";
#endif
            Kernel *predecessor = id_to_kernel_map[p];
            if (predecessor->chain_id != -1 && !predecessor->has_dispatched)
                if (id_to_chain_map[predecessor->chain_id]->predecessor) {
                    Chain *c = id_to_chain_map[predecessor->chain_id];

#if CHAIN_LOG
                    std::cout << "Selected chain " << c->id << "\n";
#endif

                    c->calculate_gain(predecessor);
                    auto gain_cpu = c->expected_gain_cpu;
                    auto gain_gpu = c->expected_gain_gpu;

                    if (gain_cpu > 1 && get_device_type(c->platform) == cpu) {
                        c->add_kernel(k);
                        task_merge = true;
                        current_predecessors_merged_in_chains.push_back(
                            predecessor);
                        mergeable_chains.push_back(c);
                    }
                    if (gain_gpu > 1 && get_device_type(c->platform) == gpu) {
                        c->add_kernel(k);
                        task_merge = true;
                        current_predecessors_merged_in_chains.push_back(
                            predecessor);
                        mergeable_chains.push_back(c);
                    }
                    if (task_merge) {
#if CHAIN_LOG
                        std::cout << "Merged predecessor " << p
                                  << " with chain  " << c->id << "\n";
#endif

                        for (auto p_prime : predecessors[k->id]) {
                            Kernel *other_predecessor =
                                id_to_kernel_map[p_prime];
                            if (other_predecessor->chain_id == -1 &&
                                !other_predecessor->has_dispatched)
                                ps.push(other_predecessor);
                        }
                        break;
                    }
                }
        }
        if (!task_merge) {

            Chain *c_prime = new Chain(k, "pred");
#if CHAIN_LOG
            std::cout << "New predecessor chain formed with id " << c_prime->id
                      << "starting with kernel " << k->id << "\n";
#endif
            id_to_chain_map[c_prime->id] = c_prime;
            bool predecessor_growth = false;
            while (!k->in_frontier) { // !k->has_dispatched
                assert(k->has_dispatched != true);
                Kernel *predecessor_kernel;
                // Check if there are any non-chain predecessor that has not
                // been dispatched. if there aren't any stop cluster chain
                // process here
                bool has_predecessors = false;
                for (auto p : predecessors[k->id]) {
                    predecessor_kernel = id_to_kernel_map[p];
                    if (predecessor_kernel->chain_id == -1 &&
                        !predecessor_kernel->has_dispatched)
                        has_predecessors = has_predecessors | true;
                }
                if (has_predecessors == false)
                    break;
                for (auto p : predecessors[k->id]) {
                    predecessor_kernel = id_to_kernel_map[p];
                    if (predecessor_kernel->chain_id == -1 &&
                        !predecessor_kernel->has_dispatched)
                        predecessor_growth =
                            chain_growth(c_prime, predecessor_kernel, chain_set,
                                         available_devices);
                    if (predecessor_growth) {
                        // c_prime->add_kernel(predecessor_kernel);
#if CHAIN_LOG
                        std::cout << "Predecessor growth possible for "
                                  << c_prime->id << "\n";
#endif

                        for (auto p_prime : predecessors[k->id]) {
                            Kernel *other_predecessor =
                                id_to_kernel_map[p_prime];
                            if (other_predecessor->chain_id == -1 &&
                                !other_predecessor->has_dispatched)
                                ps.push(other_predecessor);
                        }
                        k = predecessor_kernel;
                        break;
                    }
                }
                // if predecessor growth is not possible with any non-chain
                // predecessor that has not been dispatched return false

                if (!predecessor_growth) {
                    revert_clusters(mergeable_chains,
                                    current_predecessors_merged_in_chains);
                    delete c_prime;
                    new_chains.clear();
                    return false;
                }
                // break;
            }
            if (c_prime->is_singleton()) {
#if CHAIN_LOG
                std::cout << "predecessor chain with id " << c_prime->id
                          << "starting with kernel " << k->id
                          << " is singleton \n";
#endif

                bool device_available =
                    c_prime->assign_preferred_device(available_devices);
                if (device_available) {
                    // chain_set.push_back(c_prime);
                    new_chains.push_back(c_prime);
                } else {
                    // revert clusters
#if CHAIN_LOG
                    std::cout << "Revert called for singleton predecessor "
                                 "chain\n";
#endif
                    revert_clusters(mergeable_chains,
                                    current_predecessors_merged_in_chains);
                    delete c_prime;
                    new_chains.clear();

                    return false;
                }
            } else if (predecessor_growth) {
                // chain_set.push_back(c_prime);
                new_chains.push_back(c_prime);
            } else {
                // revert clusters
#if CHAIN_LOG
                std::cout << "Revert called for non-singleton predecessor "
                             "chain\n";
#endif
                revert_clusters(mergeable_chains,
                                current_predecessors_merged_in_chains);
                new_chains.clear();
                delete c_prime;
                return false;
            }
        }
    }

    for (auto c : new_chains) {
#if CHAIN_LOG
        std::cout << "Adding to chain set predecessor chain " << c->id << "\n";
#endif

        chain_set.push_back(c);
    }
    return true;
}

void DAG::construct_chains(int kernel_id, std::vector<Chain *> &chain_set,
                           std::map<Vendor, std::deque<int>> &available_devices,
                           int d = 2) {
    static int construct_chain_invocation = 0;
    if (profile_engine)
        FunctionTimer("Algorithm->ConstructChains");
    construct_chain_invocation++;
    // FunctionTimer("CONSTRUCT CHAIN function");
    /*
    if (is_there_any_device(available_devices) == false) {
        // std::cout << "CONSTRUCT CHAIN function " <<
        // construct_chain_invocation
        //         << " : no chain created due to no "
        //          "device being present\n";
        return;
    }
*/
    Chain *c = new Chain(id_to_kernel_map[kernel_id], "succ");
    id_to_chain_map[c->id] = c;
    int lookahead_depth = 0;
#if CHAIN_LOG
    // std::cout << "Constructing successor chain with  kernel " << kernel_id
    //        << "\n";
    print_available_devices(available_devices);
#endif

    while (lookahead_depth++ < d) {
        bool task_growth = false;
        for (auto s : successors[kernel_id]) {
#if CHAIN_LOG
            std::cout << "Checking chain growth with kernel " << s << "\n";
#endif
            Kernel *successor_kernel = id_to_kernel_map[s];
#if CHAIN_LOG
            std::cout << "chain id of successor kernel is "
                      << successor_kernel->chain_id << "\n";
#endif
            if (successor_kernel->required_predecessors >
                get_number_of_devices(available_devices)) {
                task_growth = false;
                break;
            }
            if (successor_kernel->chain_id == -1 &&
                !successor_kernel->has_dispatched)
                task_growth = chain_growth(c, successor_kernel, chain_set,
                                           available_devices);
            if (task_growth == true) {
                kernel_id = s;
                break;
            }
        }
        if (task_growth == false)
            break;
    }
    if (c->is_singleton()) {

        bool device_available = c->assign_preferred_device(available_devices);
        if (device_available)
            chain_set.push_back(c);
        else
            delete c;
    } else // Note, device is assigned inside chain growth function itself for
           // this case.
    {
        chain_set.push_back(c);
    }
#if SCHEDULE_LOG
    std::cout << "Chains created in construct chains function: "
              << "\n";
    for (auto c : chain_set)
        c->print_information();
    std::cout << "Exiting construct chains function: "
              << "\n";
#endif
    /*
    if (chain_set.size() == 0)
        std::cout << "CONSTRUCT CHAIN function " << construct_chain_invocation
                  << " did not produce any chains\n";
    else if (chain_set.size() == 1) {

        if (chain_set[0]->is_singleton())
            std::cout << "CONSTRUCT CHAIN function "
                      << construct_chain_invocation
                      << " produced a singleton chain\n";
        else
            std::cout << "CONSTRUCT CHAIN function "
                      << construct_chain_invocation
                      << " produced a single cluster chain\n";

    } else {

        std::cout << "CONSTRUCT CHAIN function " << construct_chain_invocation
                  << " produced multiple chains\n";
    }
    */
}

void DAG::update_instance_id_for_kernels(int instance_identifier) {
    for (auto x : id_to_kernel_map) {
        Kernel *k = x.second;
        k->instance_id = instance_identifier;
    }
    current_instance = instance_identifier;
}
void DAG::reset_dag_parameters() {
    for (auto x : id_to_kernel_map) {
        Kernel *k = x.second;
        k->reset_kernel_parameters();
    }
    this->completed_kernels = 0;
}

void DAG::update_local_deadline(float deadline) {
    for (auto x : id_to_kernel_map) {
        int kid = x.first;
        Kernel *k = x.second;
        if (successors[kid].size() == 0)
            k->rank = deadline;
        else
            k->rank = deadline - k->wcet_rank;
    }
}
