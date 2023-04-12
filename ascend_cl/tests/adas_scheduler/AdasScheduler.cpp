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
               "Simulator Mapping Configurations File, Profiling Dump JSON File\n");
        return 0;
    }

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
   // printf("Initializing ADAS Dags\n");
    SE->initialize_adas_dags(argv[1]);
   // printf("Rank Information\n");
    for (auto d : SE->DAGs) {
        d.second->print_rank_information();
    }
   // printf("Parsing Arrival Times\n");
    SE->parse_arrival_times(argv[2]);
    SE->print_arrival_info();
   // printf("Parsing Deadlines\n");
    SE->parse_deadlines(4, argv[3]);
    //SE->print_deadlines_of_dag_instances();
    SE->initialize_adas_instance_config(4, argv[4]);
    //SE->print_dag_instance_mapping_info();
    SE->build_kernels_of_all_dags();
    SE->setup_dags();
    SE->copy_persistent_buffers();
   // printf("Copied persistent buffers\n");
    std::thread streamer(&ScheduleEngine::adas_stream, SE);
    std::thread scheduler(&ScheduleEngine::schedule_adas_dags, SE);
    streamer.join();
    scheduler.join();

   // for (auto x : SE->id_to_dag_map) {
     //   auto dag_id = x.first;
       // auto dag = x.second;
        //std::cout << "Deadlines missed for DAG " << dag_id << " "
          //        << dag->deadlines_missed << "\n";
    //}

    printf("Dumping profiling information\n"); 
    SE->host_synchronize();
    SE->profile_adas_dags(argv[5]);
    delete SE;

    printf("Main finished\n");
    return 0;
}
