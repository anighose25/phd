#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
//#include <FreeTreeAllocator.h>
bool profile_engine;

FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));	
bool static_hostarray_allocation = false;
 
int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Program must be provided with four arguments (arrival input "
               "file, profile dump file, lookahead_depth, profile flag)\n");
        return 0;
    }
    int lookahead_depth=atoi(argv[3]);
    int profile_flag=atoi(argv[4]);
    if(profile_flag==1)
        profile_engine=true;
    else
        profile_engine=false;
    //printf("Instatiating ScheduleEngine\n");
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
    SE->cluster_schedule(lookahead_depth);
    dstream.join();
    SE->profile_dags(argv[2]);
    SE->calculate_delays();
    SE->calculate_makespan();
    delete SE;
    //printf("Main finished\n");
    return 0;
}
