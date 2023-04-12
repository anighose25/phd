#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <pthread.h>
#include <unistd.h>

#include <FreeTreeAllocator.h>

FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));
bool static_hostarray_allocation = true;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Program must have 5 arguments: DAG Name, Device Type, Fusion "
               "Variants File for DAG, Dump "
               "File for Profiling, Mode for Profiling");
        return 0;
    }
    std::string device_type{argv[2]};
    std::string mode{argv[5]};
    std::vector<std::string> dag_names;
    dag_names.push_back(std::string(argv[1]));
    printf("Instatiating ScheduleEngine\n");

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
    SE->initialize_dags(dag_names);

    std::vector<std::string> kernel_names;
    std::string interference_kernel =
        std::string{"./database/info/microkernel.json"};

    Kernel *m = NULL;
    if (mode == "interference") {
        kernel_names.push_back(interference_kernel);
        SE->initialize_kernels(kernel_names);
        printf("Setting up interference kernel\n");
        SE->setup_kernels();
        m = SE->get_kernel_by_name(kernel_names[0].c_str());
        if (device_type == "cpu")
            m->set_preferred(Vendor::ARM_GPU);
        else
            m->set_preferred(Vendor::ARM_CPU);

    }

    std::vector<std::vector<int>> tc_vector;
    std::ifstream stream(argv[3]);
    std::string line;
    while (std::getline(stream, line)) {
        splitstring liststring((char *)line.c_str());
        std::vector<std::string> ssublist = liststring.split(',');
        std::vector<int> v;
        for (auto k : ssublist)
            v.push_back(std::stoi(k));
        tc_vector.push_back(v);
    }

    // tc_vector.push_back(v3);
    SE->build_kernels_of_all_dags();
    SE->setup_dags();
    SE->copy_persistent_buffers();
    Vendor platform, platform_interference;
    if (device_type == "cpu")
    {
        platform = Vendor::ARM_CPU;
        platform_interference = Vendor::ARM_GPU;
    }
    else
    {
        platform = Vendor::ARM_GPU;
        platform_interference = Vendor::ARM_CPU;
    }
    for (auto v : tc_vector) {
        auto k = SE->DAGs[argv[1]]->get_kernel(v[0]);
        k->finished_parents = k->num_parents;
        TaskComponent *tc = SE->create_task_component(
            v, SE->ctxs[platform], platform, SE->DAGs[argv[1]]);
        printf("Created task component %d\n", tc->task_id);

        std::thread scheduler(&ScheduleEngine::dispatch_task_component_with_inteference, SE, tc,
                              platform,m,platform_interference);
        scheduler.join();
        SE->host_synchronize();
        printf("Profiling\n");
        SE->profile_task_component(SE->DAGs[argv[1]], tc, argv[4]);
        SE->DAGs[argv[1]]->reset_dag_parameters();
//        tc->reset_tc_parameters();
        
        if(mode == "interference")
            m->reset_kernel_parameters();
    }

    delete SE;
    printf("Main finished\n");
    return 0;
}
