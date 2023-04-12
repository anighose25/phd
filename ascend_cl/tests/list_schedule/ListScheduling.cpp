#include <FreeTreeAllocator.h>
#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <unistd.h>

FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));
FreeTreeAllocator cpu_buffer_pool(static_cast<std::size_t>(1e12));
bool static_hostarray_allocation = true;

int main() {

    std::vector<std::string> dag_names;
    dag_names.push_back(std::string("sample"));
    printf("Instatiating ScheduleEngine\n");
    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);

    std::vector<std::pair<std::size_t, int>> sizes;
    sizes.push_back(std::make_pair(262144, 50));
    sizes.push_back(std::make_pair(1024, 50));
    SE->create_gpu_buffer_manager(sizes);

    SE->print_all_device_info();
    SE->initialize_online_dags("arrival.txt");
    SE->print_all_dag_info();
    SE->build_kernels_of_all_dags();
    SE->print_dag_arrival_timestamps();
    std::thread dstream(&ScheduleEngine::dag_stream, SE);
    SE->list_schedule();
    dstream.join();
    delete SE;
    printf("Main finished\n");
    return 0;
}
