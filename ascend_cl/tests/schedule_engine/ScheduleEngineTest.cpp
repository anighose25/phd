#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <HostArray.h>
#include <FreeTreeAllocator.h>
#include <cstdlib>	

FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));
bool static_hostarray_allocation = true;

int main() {

    std::vector<std::string> kernel_names;
    kernel_names.push_back(std::string("atax1.json"));
    printf("Instatiating ScheduleEngine\n");

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);
    SE->print_all_device_info();
    SE->initialize_kernels(kernel_names);
    SE->print_all_kernel_info();
    SE->setup_kernels();
    SE->test_kernel_dispatch();
    delete SE;

    printf("Main finished\n");
    return 0;
}
