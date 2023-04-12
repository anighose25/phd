#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <HostArray.h>
#include <FreeTreeAllocator.h>	


FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));	
bool static_hostarray_allocation = true;

int main() {

    std::vector<std::string> dag_names;
    dag_names.push_back(std::string("fork_join1"));
    printf("Instatiating ScheduleEngine\n");

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
    SE->print_all_device_info();
    SE->initialize_dags(dag_names);
    SE->build_kernels_of_all_dags();
    SE->DAGs["fork_join1"]->compute_bottom_level_rank();
    SE->DAGs["fork_join1"]->print_rank_information();
    SE->cluster_schedule();
    delete SE;

    printf("Main finished\n");
    return 0;
}
