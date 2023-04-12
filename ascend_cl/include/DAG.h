#include "GpuBufferManager.h"
#include "Kernel.h"
#include "core.h"
extern std::string database_dir;
static int dag_identifier = 0;
static int chain_identifier = 0;
bool is_there_any_device(std::map<Vendor, std::deque<int>> &available_devices);

struct InstanceConfig {
    std::vector<std::vector<int>> task_components;
    std::vector<Vendor> devices;
    int dag_id;
    int instance_id;

    InstanceConfig(int dag_id, int instance_id, std::string &tc_string,
                   std::string &platform) {
        this->dag_id = dag_id;
        this->instance_id = instance_id;
        this->add(tc_string, platform);
    }
    void add(std::string &tc_string, std::string &platform) {
        splitstring liststring((char *)tc_string.c_str());
        std::vector<std::string> ssublist = liststring.split(',');
        std::vector<int> task_component;
        for (auto x : ssublist)
            task_component.push_back(std::stoi(x));
        this->task_components.push_back(task_component);
        this->devices.push_back(get_vendor(platform));
    }

    void print_instance_info() {
        for (int i = 0; i < this->task_components.size(); i++) {
            for (auto x : task_components[i])
                std::cout << x << " ";
            std::cout << get_device_type(devices[i]) << "\n";
        }
    }
};

class InstanceQueue {

  public:
    InstanceQueue() {}

    InstanceQueue(InstanceQueue const &other) {
        std::lock_guard<std::mutex> lk(mut);
        q = other.q;
    }
    void push(InstanceConfig *ic);
    void wait_and_pop(InstanceConfig **ic);
    bool empty();

  private:
    mutable std::mutex mut;
    std::queue<InstanceConfig *> q;
    std::condition_variable cond;
};

class Chain {
  public:
    Chain(Kernel *source, const char *chain_type) {
        chain_kernels.push_back(source);
        if (schain == chain_type)
            successor = true;
        if (pchain == chain_type)
            predecessor = true;
        id = chain_identifier++;
        source->chain_id = id;
        this->total_cpu_time = source->cpu_time;
        this->total_gpu_time =
            source->gpu_time + source->h2d_time + source->d2h_time;
        if (source->preferred == Vendor::ARM_CPU)
            this->baseline = source->cpu_time;
        else
            this->baseline =
                source->gpu_time + source->h2d_time + source->d2h_time;
        gain_cpu = baseline / this->total_cpu_time;
        gain_gpu = baseline / this->total_gpu_time;
    }

    ~Chain() {
        //        std::cout << "Deleting chain " << this->id << "\n";
        for (auto k : chain_kernels) {
            //          std::cout << "Setting chain id of kernel " << k->id <<
            //          "to -1\n";
            k->chain_id = -1;
        }
        chain_kernels.clear();
    }

    bool contains_kernel(Kernel *k);
    void add_kernel(Kernel *k);
    void remove_last_kernel();
    bool contains_frontier_kernel();
    void assign_device(Vendor p, int device_id);

    void calculate_gain(Kernel *k, bool merge);

    bool is_singleton();
    bool assign_preferred_device(
        std::map<Vendor, std::deque<int>> &available_devices);

    void assign_device(std::pair<Vendor, int> device,
                       std::map<Vendor, std::deque<int>> &available_devices);

    std::pair<Vendor, int> get_assigned_device();
    void print_information();

    std::vector<Kernel *> chain_kernels;
    bool successor = false;
    bool predecessor = false;
    bool device_assigned = false;
    Vendor platform;
    int device_id = -1;
    int id;
    std::map<Vendor, float> current_gain;
    std::string schain = "succ";
    std::string pchain = "pred";
    float gain_cpu = 0.0;
    float gain_gpu = 0.0;
    float baseline = 0.0;
    float total_cpu_time = 0.0;
    float total_gpu_time = 0.0;

    float expected_baseline = 0.0;
    float expected_gain_cpu = 0.0;
    float expected_gain_gpu = 0.0;
    float expected_total_cpu_time = 0.0;
    float expected_total_gpu_time = 0.0;
};

class DAG {
  public:
    DAG(const char *graph_folder) {
        this->id = dag_identifier++;
        name = graph_folder;
        std::string graph_file =
            database_dir + "dags/" + graph_folder + "/dag.graph";
        std::ifstream stream(graph_file.c_str());
        std::string line;
        while (std::getline(stream, line)) {
#if DAG_LOG
            std::cout << "parsing " << line << "\n";
#endif
            if (line == "---")
                break;
            splitstring liststring((char *)line.c_str());
            std::vector<std::string> ssublist = liststring.split(' ');
            int kernel_id = std::stoi(ssublist[0]);
#if DAG_LOG
            std::cout << "initializing kernel object\n";
#endif

            std::string info = database_dir + "dags/" + graph_folder +
                               "/output/" + ssublist[1];
            vertices.push_back(kernel_id);
            id_to_kernel_map.emplace(
                kernel_id,
                new Kernel(info.c_str(), false, kernel_id, (void *)this));

#if MULTIKERNEL
            std::string info_prime = database_dir + "dags/" + graph_folder +
                                     "/output/" + ssublist[1];
            id_to_kernel_map[kernel_id]->set_multiple_implementations(
                info, info_prime);

#endif
            id_to_kernel_map[kernel_id]->dag_id = this->id;
            id_to_kernel_map[kernel_id]->instance_id = 0;
        }
        while (std::getline(stream, line)) {
#if DAG_LOG
            std::cout << "parsing " << line << "\n";
#endif
            if (line == "---")
                continue;
            splitstring edgestring((char *)line.c_str());

            std::vector<std::string> ssublist = edgestring.split('-');
            EdgeInfo *e = new EdgeInfo(ssublist[0], ssublist[1]);
            edges.push_back(e);
        }
#if DAG_LOG
        printf("Creating edges\n");
#endif
        create_edges();
#if DAG_LOG
        printf("Binding host arrays\n");
#endif
        bind_host_array();
        num_kernels = vertices.size();
        precompute_ancestor_info();

#if ADAS
        std::string rank_file = database_dir + "dags/" + graph_folder + "/" +
                                graph_folder + ".rank";
        // std::cout<<"Opening file "<<rank_file<<"\n";
        std::ifstream rank_file_contents(rank_file.c_str());
        std::string rline;
        while (std::getline(rank_file_contents, rline)) {
            //  std::cout<<"Parsing line "<<rline<<"\n";
            splitstring rankstring((char *)rline.c_str());
            std::vector<std::string> rsublist = rankstring.split(':');
            int kid = std::stoi(rsublist[0]);
            float rank_value = std::stof(rsublist[1]);
            id_to_kernel_map[kid]->wcet_rank = rank_value;
        }

#endif
    }

    ~DAG() {
        for (auto x : id_to_kernel_map)
            delete x.second;
    }
    std::string get_id() { return std::to_string(id); }
    std::string get_name() { return name; }
    Kernel *get_kernel(int kid) { return id_to_kernel_map[kid]; }
    void print_dag_info();
    void print_all_inputs();
    void print_all_outputs();
    void print_output();
    void print_buffer_host_information();
    void print_buffer_host_information(Vendor p);
    void print_profiling_information();
    void create_edges();
    void bind_host_array();
    void topological_sort(std::vector<int> &topological_order);
    void compute_bottom_level_rank();
    void print_rank_information();
    void print_wcet_time();
    void
    setup(std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
          std::unordered_map<Vendor, cl_context> &ctxs,
          std::vector<Vendor> &platforms, GpuBufferManager *GpuBufMan);
    void copy_persistent_buffers(
        std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
        std::unordered_map<Vendor, cl_context> &ctxs,
        std::vector<Vendor> &platforms,
        std::unordered_map<Vendor, std::vector<cl_command_queue>> &cmd_qs);

    void build_kernels(
        std::unordered_map<Vendor, std::vector<cl_device_id>> &all_devices,
        std::unordered_map<Vendor, cl_context> &ctxs,
        std::vector<Vendor> &platforms);

    void set_arrival_time(long long int arrival_time);

    void print_arrival_time();

    bool contains_device(std::string device_type,
                         std::map<Vendor, std::deque<int>> &available_devices);
    void add_device(std::pair<Vendor, int> device,
                    std::map<Vendor, std::deque<int>> &available_devices);
    std::pair<Vendor, int>
    remove_device(std::string device_type,
                  std::map<Vendor, std::deque<int>> &available_devices);
    void print_available_devices(
        std::map<Vendor, std::deque<int>> &available_devices);

    bool chain_growth(Chain *c, Kernel *successor,
                      std::vector<Chain *> &chain_set,
                      std::map<Vendor, std::deque<int>> &available_devices);
    bool device_assignable(Chain *c, Kernel *successor_kernel,
                           std::vector<Chain *> &chain_set,
                           std::map<Vendor, std::deque<int>> &available_devices,
                           const char *device_type);
    bool predecessor_assignment(
        Kernel *successor_kernel, std::vector<Chain *> &chain_set,
        std::map<Vendor, std::deque<int>> &available_devices);

    void construct_chains(int kernel_id, std::vector<Chain *> &chain_set,
                          std::map<Vendor, std::deque<int>> &available_devices,
                          int lookahead_depth);

    void revert_clusters(std::vector<Chain *> &mergeable_chains,
                         std::vector<Kernel *> predecessors_merged);
    void update_instance_id_for_kernels(int instance_identifier);
    void reset_dag_parameters();
    void update_local_deadline(float deadline);
    void precompute_ancestor_info();
    void sort_neighbours();
    long long int arrival_time;
    struct timespec arrival_timestamp;
    std::vector<int> vertices;
    std::vector<EdgeInfo *> edges;
    std::map<int, Kernel *> id_to_kernel_map;
    std::map<int, Chain *> id_to_chain_map;
    std::map<int, std::vector<int>> successors;
    std::map<int, std::vector<int>> predecessors;
    std::atomic<int> completed_kernels{0};
    std::atomic<bool> running{false};
    InstanceQueue instance_queue;
    int num_kernels = 0;
    int id;
    int current_instance = 0;
    int deadlines_missed = 0;
    std::string name;
    std::string cpu = "cpu";
    std::string gpu = "gpu";
    std::vector<int> levelwise_max_pred_count;
};
