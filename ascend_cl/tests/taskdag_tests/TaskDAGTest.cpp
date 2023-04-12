#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <unistd.h>
#include <HostArray.h>
#include <FreeTreeAllocator.h>	


FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));	
bool static_hostarray_allocation = true;

int main() {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
    auto th = pthread_self();
    int rc = pthread_setaffinity_np(th,
                                    sizeof(cpu_set_t), &cpuset);
    if(rc!=0)
    {
        printf("Thread affinity not set\n");
        exit(-1);
    }
    std::vector<std::string> dag_names;
    dag_names.push_back(std::string("fork_join2"));
    printf("Instatiating ScheduleEngine\n");

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
    SE->print_all_device_info();
    SE->initialize_dags(dag_names);
    SE->print_all_dag_info();
    std::vector<int> v1 = {0, 2};
    std::vector<int> v2 = {1, 3};
    std::vector<int> v3 = {4, 5};
    SE->build_kernels_of_all_dags();
//    SE->setup_dags(); 

    
    SE->create_task_component(v1, SE->ctxs[SE->platform_gpu], SE->platform_gpu,
                               SE->DAGs["fork_join2"]);

    SE->create_task_component(v2, SE->ctxs[SE->platform_gpu], SE->platform_gpu,
                               SE->DAGs["fork_join2"]);

    SE->create_task_component(v3, SE->ctxs[SE->platform_cpu], SE->platform_cpu,
                               SE->DAGs["fork_join2"]);

  
    
    SE->static_schedule_taskdags(); 
//    printf("ALL INPUT BUFFERS OF ALL KERNELS OF DAG\n");
//    SE->DAGs["sample"]->print_all_inputs();
//    printf("ALL OUTPUT BUFFERS OF ALL KERNELS OF DAG\n");
//    SE->DAGs["sample"]->print_all_outputs();
    sleep(1);
    printf("Profiling Information\n");
    SE->profile_dags("timing.json");
   
    delete SE;

    printf("Main finished\n");
    return 0;
}
