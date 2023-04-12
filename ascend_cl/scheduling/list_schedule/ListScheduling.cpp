#include <HostArray.h>
#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <unistd.h>
//#include <FreeTreeAllocator.h>
bool profile_engine;
FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));
// FreeTreeAllocator cpu_buffer_pool(static_cast<std::size_t>(1e12));
bool static_hostarray_allocation = false;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Program must be provided with three arguments (arrival input "
               "file, profile dump file, profile_flag)\n");
        return 0;
    }
    //printf("Instatiating ScheduleEngine\n");
    int profile_flag = atoi(argv[3]);
    if(profile_flag==1)
        profile_engine=true;
    else
        profile_engine=false;
    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
    //SE->print_all_device_info();
    SE->initialize_online_dags(argv[1]);
    SE->build_kernels_of_all_dags();
    for (auto d : SE->DAGs) {
        auto dag = d.second;
        dag->compute_bottom_level_rank();
        dag->sort_neighbours();
    }
    std::thread dstream(&ScheduleEngine::dag_stream, SE);
    SE->list_schedule();
    dstream.join();
    SE->profile_dags(argv[2]);
    /*
    for (auto d : SE->DAGs) {
        auto dag = d.second;
        dag->print_wcet_time();
    }
    */
    SE->calculate_delays();
    SE->calculate_makespan();
    delete SE;
    //printf("Main finished\n");
    return 0;
}
