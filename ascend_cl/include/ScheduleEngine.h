#ifndef SCHEDULEENGINE_H
#define SCHEDULEENGINE_H

#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif

#include "DAG.h"
#include "Events.h"
#include "GpuBufferManager.h"

static int task_identifier = 0;
std::atomic<int> completed_jobs{0};
struct SimInstance {
    std::vector<int> task_component;
    Vendor platform;
    int dag_id;
    int instance_id;

    SimInstance(int dag_id, int instance_id, std::string &tc_string,
                std::string &platform) {
        this->dag_id = dag_id;
        this->instance_id = instance_id;
        this->platform = get_vendor(platform);
        splitstring liststring((char *)tc_string.c_str());
        std::vector<std::string> ssublist = liststring.split(',');
        for (auto x : ssublist)
            task_component.push_back(std::stoi(x));
    }

    ~SimInstance() { task_component.clear(); }

    void print_info() {
        std::cout << dag_id << " " << instance_id << "--> ";
        for (auto k : task_component)
            std::cout << k << " ";
        std::cout << get_device_type(platform) << "\n";
    }
};

struct CmpKernelPtrs {
    bool operator()(const Kernel *lhs, const Kernel *rhs) const {
        return (lhs->rank < rhs->rank);
    }
};

template <typename T, class Container = std::vector<T>,
          class Compare = std::less<typename Container::value_type>>
class custom_priority_queue
    : public std::priority_queue<T, Container, Compare> {
  public:
    bool remove(const T &value) {
        auto it = std::find(this->c.begin(), this->c.end(), value);
        if (it != this->c.end()) {
            this->c.erase(it);
            std::make_heap(this->c.begin(), this->c.end(), this->comp);
            return true;
        } else {
            return false;
        }
    }
};

class Frontier {
  public:
    Frontier(){};
    Frontier(Frontier const &other) {
        std::lock_guard<std::mutex> lk(mut);
        q = other.q;
    }
    void push(Kernel *new_value);
    void push_local(Kernel *new_value);
    void wait_and_pop(Kernel **value);
    void wait_and_remove(Kernel *value);
    void wait_and_remove(std::vector<Kernel *> values);
    void wait_and_peek(Kernel **value);
    void wait_and_pop_local(Kernel **value);
    bool empty() const;
    void init_num_jobs(int num_jobs) { this->num_jobs = num_jobs; }
    void notify();

  private:
    mutable std::mutex mut;
    custom_priority_queue<Kernel *, std::vector<Kernel *>, CmpKernelPtrs> q;
    std::condition_variable cond;
    int num_jobs;
};

class ReadyQueue {

  public:
    ReadyQueue() {}

    void initialize(std::vector<Vendor> platforms) {
        for (auto p : platforms)
            q.emplace(p, std::deque<int>());
    }
    ReadyQueue(ReadyQueue const &other) {
        std::lock_guard<std::mutex> lk(mut);
        q = other.q;
    }
    void push(Vendor p, int d);
    void wait_and_pop(Vendor p, int *d);
    void notify();
    void wait_and_remove(Vendor p, int device_id, int *d);
    bool device_available() const;
    bool empty(Vendor p);

    int
    get_available_devices(std::map<Vendor, std::deque<int>> &available_devices);

  private:
    mutable std::mutex mut;
    std::unordered_map<Vendor, std::deque<int>> q;
    std::condition_variable cond;
};

class TaskComponent;

class ScheduleEngine {
  public:
    ScheduleEngine(Vendor gpu, Vendor cpu, int num_cpu_cores,
                   int num_subdevices) {
        // FunctionTimer("ScheduleEngine Constructor");
        num_platforms = 0;
        platform_gpu = gpu;
        platform_cpu = cpu;
        platforms.push_back(platform_gpu);
        platforms.push_back(platform_cpu);

        if (platform_gpu != -1)
            num_platforms++;
        if (platform_cpu != -1)
            num_platforms++;

        num_cpu_cores_per_device = num_cpu_cores;
        num_cpu_subdevices = num_subdevices;
        all_devices.emplace(platform_cpu, std::vector<cl_device_id>());
        all_devices.emplace(platform_gpu, std::vector<cl_device_id>());
        cmd_qs.emplace(platform_cpu, std::vector<cl_command_queue>());
        cmd_qs.emplace(platform_gpu, std::vector<cl_command_queue>());
        ready_queue.initialize(platforms);

#if FILE_LOGGER
        fp = std::fopen("ScheduleEngineLogs.txt", "w");
#endif
     //   printf("Before host init\n");
        host_initialize();
       // printf("After host init\n");
        gpu_buf_man = NULL;

        for (auto p : platforms) {
            int idx = cmd_qs[p].size() - 1;
            for (int i = 0; i < cmd_qs[p].size(); i++) {
                ready_queue.push(p, i);
                break;
            } // idx-- if lower powerful device first or i if more powerful
              // device first (applicable for odroid xu4)
        }
    }

    ~ScheduleEngine() {
#if DESTRUCTOR_LOG
        std::cout << "Deleting Schedule Engine\n";
        LOG("Destructor: Releasing Command Queues\n");
        std::cout << "Deleting command queues\n";
#endif
        for (auto x : cmd_qs) {
            for (auto cmd_q : x.second) {
                cl_int status = clReleaseCommandQueue(cmd_q);
                check(status, "Releasing Command Queue ");
            }
        }
        cmd_qs.clear();
#if DESTRUCTOR_LOG
        std::cout << "Deleting context\n";
        LOG("Destructor: Releasing Context\n");
#endif
        for (auto x : ctxs) {
            cl_int status = clReleaseContext(x.second);
            check(status, "Releasing Context");
        }
        ctxs.clear();

#if DESTRUCTOR_LOG
        std::cout << "Deleting devices\n";
#endif
        for (auto x : all_devices) {
            auto device = x.second;
            device.clear();
            device.shrink_to_fit();
        }
        all_devices.clear();
#if DESTRUCTOR_LOG
        std::cout << "Deleting kernels\n";
#endif
        for (auto x : Kernels) {
            auto kernel = x.second;
#if MEMORY_LOG
            std::cout << "ScheduleEngine.h -> Destroying kernels\n";
            std::cout << "ScheduleEngine.h -> Kernel name = " << x.first << "\n";
#endif
            delete kernel;
        }
#if DESTRUCTOR_LOG
        std::cout << "Deleting DAGs\n";
#endif
        for (auto x : DAGs) {
            auto dag = x.second;
#if MEMORY_LOG
            std::cout << "ScheduleEngine.h -> Destroying dags\n";
            std::cout << "ScheduleEngine.h -> DAG name = " << x.first << "\n";
#endif
            delete dag;
        }
#if FILE_LOGGER
        std::fclose(fp);
#endif
        if (gpu_buf_man) {
#if DESTRUCTOR_LOG
            std::cout << "Deleting GPU Buffer Manager\n";
#endif
            delete gpu_buf_man;
        }
	task_identifier = 0;
        completed_jobs.store(0,std::memory_order_relaxed);
#if DESTRUCTOR_LOG
        std::cout << "Completed Deletion of jobs\n";
#endif
    }
    void get_all_devices();
    std::vector<cl_command_queue>
    create_command_queue_for_each(cl_device_id *devs, int num_devs,
                                  cl_context ctx);
    void host_initialize();
    void host_synchronize();
    void create_gpu_buffer_manager(
        const std::vector<std::pair<std::size_t, int>> &sizes);
    void print_device_info(cl_device_id device_id, int i, int j);
    void print_all_device_info();
    void initialize_kernels(std::vector<std::string> &kernel_names);
    Kernel *get_first_kernel();
    Kernel *get_kernel_by_name(const char *name);
    void initialize_dags(std::vector<std::string> &dag_names);
    void initialize_adas_dags(const char *config_name);
    void initialize_frontier_dag(DAG *d);
    void print_all_kernel_info();
    void print_all_dag_info();
    void print_dag_arrival_timestamps();

    void print_task_component_info();
    void print_sim_instance_info();
    void setup_kernels();

    void setup_kernel(int kernel_id, cl_context &ctx, Vendor p, DAG *dag);
    void setup_dags();
    void copy_persistent_buffers();
    void test_kernel_dispatch();
    void build_kernels_of_all_dags();
    void list_schedule();
    void custom_cluster_schedule(std::unordered_map<std::string, std::vector<std::pair<Vendor, std::vector<int> > > > &dag_tc_map, int memory_budget);
    void dispatch_task_component(TaskComponent *tc, Vendor platform);
    void dispatch_task_component_with_inteference(TaskComponent *tc,
                                                  Vendor platform, Kernel *k,
                                                  Vendor platform_inteference);
    void dispatch_sim_instance(SimInstance *sim_instance, Vendor platform);
    void cluster_schedule(int lookahead_depth);
    void calculate_makespan();
    void calculate_delays();
    void profile_dags(std::string &&filename);
    void profile_adas_dags(std::string &&filename);
    void profile_kernels(std::string &&filename);
    void profile_task_component(DAG *dag, TaskComponent *tc,
                                std::string &&filename);

    void initialize_online_dags(const char *filename);
    void dag_stream();
    void print_dag_ending_times();
    void calculate_individual_dag_makespan();
    void parse_arrival_times(const char *filename);
    void parse_deadlines(int num_jobs, const char *filename);
    void print_arrival_info();
    void print_deadlines_of_dag_instances();
    void setup_new_instance(DAG *d, InstanceConfig *ic);
    //For generating profile database
    void set_preferred_platform(Vendor p);
    // void profile_tc_data(const std::unordered_map<std::string, std::vector<std::pair<Vendor, std::vector<int> > > > &dag_tc_map, std::string filename);
    //Following 3 functions used for Memory Aware Schedule database
    void save_kernel_edge_info(std::string filename);
    void profile_kernel_memory_requirements(std::string filename);
    void profile_kernel_overhead_requirements(std::string filename, int callback_overhead);
    std::vector<std::vector<std::pair<int, int>>> arriving_dag_instances;
    std::vector<struct timespec> sleep_times;
    int num_jobs = 0;

    Frontier frontier;
    ReadyQueue ready_queue;
    bool device_available();
    void schedule_kernels();
    void schedule_dags();
    void static_schedule_taskdags();
    std::map<std::string, DAG *> DAGs;
    std::vector<DAG *> arriving_dags;
    std::map<DAG *, std::vector<TaskComponent *>> dag_to_task_components_map;
    TaskComponent *create_task_component(std::vector<int> &kernel_ids,
                                         cl_context &ctx, Vendor p, DAG *dag);
    TaskComponent *create_task_component_from_chain(Chain *c, cl_context &ctx,
                                                    DAG *dag);
    TaskComponent *custom_create_task_component(std::vector<int> &kernel_ids, cl_context &ctx, Vendor p, DAG *dag);
    std::unordered_map<Vendor, std::vector<cl_device_id>> all_devices;
    std::unordered_map<Vendor, std::vector<cl_command_queue>> cmd_qs;
    std::unordered_map<Vendor, cl_context> ctxs;

    std::vector<Vendor> platforms;
    std::map<std::string, Kernel *> Kernels;
    std::map<int, DAG *> id_to_dag_map;
    std::map<int, std::map<int, InstanceConfig *>> dag_instance_config_map;
    std::map<int, std::map<int, float>> dag_instance_deadlines;
    std::map<Vendor, std::map<float, unsigned int>> speedup_to_frequency_map;
    std::vector<SimInstance *> sim_instances;
    void print_dag_instance_mapping_info();
    void adas_stream();
    void schedule_adas_dags();
    void initialize_adas_instance_config(int num_dags, const char *filename);
    void parse_dispatch_decisions(const char *filename);
    void parse_speedup_table(const char *filename);
    void change_frequency(unsigned int frequency, Vendor platform);
#if FILE_LOGGER
    FILE *fp;
#endif
    int num_cpu_cores_per_device;
    Vendor platform_cpu;
    Vendor platform_gpu;
    int num_cpu_subdevices;
    int num_platforms;
    struct HostEvents host_events;
    GpuBufferManager *gpu_buf_man;
};

struct UserArgs {
    Kernel *k;
    ScheduleEngine *se;
    UserArgs(Kernel *k_arg, ScheduleEngine *se_arg) {
        k = k_arg;
        se = se_arg;
    }
    ~UserArgs() {
        k = NULL;
        se = NULL;
    }
};


void CL_CALLBACK kernel_callback(cl_event ev, cl_int event_command_exec_status,
                                 void *user_data);
void CL_CALLBACK dag_callback(cl_event ev, cl_int event_command_exec_status,
                              void *user_data);
void CL_CALLBACK taskdag_callback(cl_event ev, cl_int event_command_exec_status,
                                  void *user_data);

void dynamic_deallocation(Kernel *k, DAG *d);
void static_deallocation(GpuBufferManager *GpuBufMan, Kernel *k, DAG *d);

class TaskComponent {
  public:
    TaskComponent(std::vector<int> &kernel_ids, Vendor platform, DAG *dag_ptr) {
        this->kernel_ids = kernel_ids;
        this->preferred = platform;
        this->platform =
            platform; // TODO: Differentiate between static and dynamic
        this->dag_ptr = dag_ptr;
        this->task_id = task_identifier++;
        this->num_kernels = kernel_ids.size();
        for (auto v : kernel_ids) {

            Kernel *k = this->dag_ptr->id_to_kernel_map[v];
            k->task_id = this->task_id;
            k->task_ptr = (void *)this;
        }
        //std::cout << "Initializing buffer flags\n";
        this->initialize_all_buffer_flags();

        //std::cout << "Initialized buffer flags\n";
        //std::cout << "Preparing kernels\n";
        this->prepare_kernels();
        //std::cout << "Prepared kernels\n";
        //std::cout << "Initializing local frontier\n";
#if ADAS == 0
        initialize_free_kernels();
#endif
    }

    ~TaskComponent() { kernel_ids.clear(); }

    void print_task_info();

    void initialize_all_buffer_flags();

    void prepare_kernel(int kernel_id);

    void prepare_kernels();

    void setup_buffers(cl_context &ctx, GpuBufferManager *GpuBufMan);
    void setup_linkage(cl_context &ctx, GpuBufferManager *GpuBufMan);

    void set_kernel_args();

    void dispatch_single(cl_context &ctx, cl_command_queue &cmd_q,
                         ScheduleEngine *se);

    bool is_finished();

    void initialize_free_kernels();

    void add_free_kernel(Kernel *k);

    bool has_free_kernels();

    void initiate_dispatch(Vendor p, int device_id);

    void reset_tc_parameters();

    void profile_dags();

    void check_and_allocate_host_arrays();

    std::string get_kernel_ids_string() {
        std::string component = "";
        for (auto k : kernel_ids)
            component += std::to_string(k) + ",";
        component.pop_back();
        return component;
    }

    std::vector<int> kernel_ids;
    Vendor preferred;
    Vendor platform;
    int device_id;
    DAG *dag_ptr;
    int task_id;
    std::map<int, cl_event> id_to_event_map;
    Frontier local_frontier;
    int num_kernels;
    std::atomic<int> completed_kernels{0};
    std::mutex resource_lock;
};

#endif
