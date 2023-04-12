#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <pthread.h>
#include <unistd.h>

#include <FreeTreeAllocator.h>

FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));
bool static_hostarray_allocation = false;

int main(int argc, char *argv[]) {

    if (argc < 5) {
        printf("5 Program Arguments Required --> ADAS Graph Configuration "
               "File, Arrival instances file, DAG Instance deadlines file, "
               "Simulator Mapping Configurations File, Profiling Dump JSON "
               "File\n");
        return 0;
    }

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
    printf("Initializing ADAS Dags\n");
    std::vector<std::string> dag_names;
    dag_names.push_back(std::string(argv[1]));
    SE->initialize_dags(dag_names);
    SE->build_kernels_of_all_dags();
    SE->setup_dags();
    SE->copy_persistent_buffers();
    printf("Copied persistent buffers\n");
    printf("Rank Information\n");
    for (auto d : SE->DAGs) {
        d.second->print_rank_information();
    }
   // printf("Parsing Arrival Times\n");
   // SE->parse_arrival_times(argv[2]);
   // SE->print_arrival_info();
   // printf("Parsing Deadlines\n");
   // SE->parse_deadlines(4, argv[3]);
   // SE->print_deadlines_of_dag_instances();
    printf("Parsing Dispatch Decisions\n");
    SE->parse_dispatch_decisions(argv[4]);
    SE->print_sim_instance_info();
    auto sim_instance = SE->sim_instances[0];
    std::thread scheduler(&ScheduleEngine::dispatch_sim_instance, SE,
                          sim_instance, sim_instance->platform);
    scheduler.join();
    SE->host_synchronize();

    printf("Dumping profiling information\n");
    SE->host_synchronize();
    SE->profile_adas_dags(argv[5]);
    delete SE;

    printf("Main finished\n");
    return 0;
}
